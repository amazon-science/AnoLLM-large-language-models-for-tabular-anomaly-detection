'''
Original Copyright (c) 2022 Kathrin Seßler and Vadim Borisov. Licensed under the MIT License.
Part of code is adapted from the GReaT repository (https://github.com/kathrinse/be_great/tree/main)
Modifications Copyright 2025 Amazon.com, Inc. or its affiliates. All Rights Reserved.
'''

import os
import random
import numpy as np
import torch
import typing as tp
from torch.utils.data import DataLoader
from transformers import Trainer
from sklearn import metrics
from anollm.anollm_dataset import AnoLLMDataCollator, AnoLLMDataLoader
from torch.nn import CrossEntropyLoss
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler


class AnoLLMTrainer(Trainer):
	"""
	Overwrites the get_train_dataloader methode of the HuggingFace Trainer to not remove the "unused" columns -
	they are needed later!
	"""

	def get_train_dataloader(self) -> DataLoader:
		if self.train_dataset is None:
			raise ValueError("Trainer: training requires a train_dataset.")

		data_collator = self.data_collator
		train_dataset = (
			self.train_dataset
		)  # self._remove_unused_columns(self.train_dataset, description="training")
		local_rank = int(os.environ["LOCAL_RANK"])
		world_size = dist.get_world_size()
		train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=local_rank, shuffle=False, drop_last=True)

		return DataLoader(
			train_dataset,
			batch_size=self._train_batch_size,
			sampler=train_sampler,
			collate_fn=data_collator,
			drop_last=self.args.dataloader_drop_last,
			num_workers=self.args.dataloader_num_workers,
			pin_memory=self.args.dataloader_pin_memory,
			worker_init_fn=_seed_worker,
		)
	
	# 2025-02-12: Amazon addition. 
	def set_eval_setting(self, n_permutations):
		self.n_permutations = n_permutations

	def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix: str = "eval"):
		eval_dataset = self.eval_dataset if eval_dataset is None else eval_dataset
		# do not use distributed sampler
		dataloader = torch.utils.data.DataLoader(eval_dataset, batch_size=self.args.eval_batch_size, shuffle = False, 
												collate_fn = AnoLLMDataCollator(self.tokenizer))

		
		perplexities = np.zeros((len(eval_dataset), self.n_permutations))
		eval_losses = np.zeros((len(eval_dataset), self.n_permutations))

		loss_fct = CrossEntropyLoss(reduction="none")
		
		# for conditional columns
		comma_id =  eval_dataset.tokenizer.convert_tokens_to_ids(',')
		n_col = eval_dataset.get_n_columns()
		column_names = eval_dataset.get_column_names()

		for perm_idx in range(self.n_permutations):
			start_idx = 0
			eval_dataset.shuffle_column_order()
			for data in dataloader:
				encoded_batch = data["input_ids"].to(self.model.device)
				attn_mask = data["attention_mask"].to(self.model.device)
				end_idx = start_idx + len(encoded_batch)
				labels = encoded_batch 
				
				start_pos_batch = data["feature_value_start"]
				end_pos_batch = data["feature_value_end"]
				col_indices_batch = data["col_indices"]

				with torch.no_grad():
					out_logits = self.model(encoded_batch, attention_mask=attn_mask).logits

				shift_logits = out_logits[..., :-1, :].contiguous()
				shift_labels = labels[..., 1:].contiguous()
				shift_attention_mask_batch = attn_mask[..., 1:].contiguous()
				eval_loss_batch = (loss_fct(shift_logits.transpose(1, 2), shift_labels) * shift_attention_mask_batch).sum(1) / shift_attention_mask_batch.sum(1)
				
				if len(eval_dataset.textual_columns) > 0:
					perplexity_batch = (loss_fct(shift_logits.transpose(1, 2), shift_labels) * shift_attention_mask_batch).cpu().numpy() # batch * (ori_seq_len -1)
					for i in range(len(encoded_batch)):
						perplexity_single = 0
						for j in range(n_col): 
							start_pos = start_pos_batch[i][j]
							end_pos = end_pos_batch[i][j]
							col_idx = col_indices_batch[i][j]
							if column_names[col_idx] in eval_dataset.textual_columns:
								perplexity_single += perplexity_batch[i, start_pos:end_pos].sum() / (end_pos - start_pos)
							else:
								perplexity_single += perplexity_batch[i, start_pos:end_pos].sum()
							if np.isnan(perplexity_single):
								print(start_pos, end_pos, perplexity_batch[i, start_pos:end_pos].sum())
								print(perplexity_batch[i, start_pos:end_pos].sum() / (end_pos - start_pos))
								print(perplexity_single)
						perplexities[start_idx+i, perm_idx] = perplexity_single
				else:
					perplexity_batch = (loss_fct(shift_logits.transpose(1, 2), shift_labels) * shift_attention_mask_batch).sum(1) 
					perplexities[start_idx:end_idx, perm_idx] = perplexity_batch.cpu().numpy()
				
				eval_losses[start_idx:end_idx, perm_idx] = eval_loss_batch.cpu().numpy()
				start_idx = end_idx

		local_rank = int(os.environ["LOCAL_RANK"])
		world_size = dist.get_world_size()
		
		all_perplexity = [None for _ in range(world_size)]
		dist.all_gather_object(all_perplexity, perplexities)
		perplexities = np.concatenate(all_perplexity, axis = 1)
		
		all_eval_loss = [None for _ in range(world_size)]
		dist.all_gather_object(all_eval_loss, eval_losses)
		eval_losses = np.concatenate(all_eval_loss, axis = 1)
		
		labels = eval_dataset.anomaly_labels
		
		mean_perplexity = np.mean(perplexities)
		normal_indices = np.where(labels == 0)[0]
		anomaly_indices = np.where(labels == 1)[0]
		perplexity_normal = np.mean(perplexities[normal_indices])
		eval_loss_normal = np.mean(eval_losses[normal_indices])
		perplexity_anomaly = np.mean(perplexities[anomaly_indices])
		eval_loss_anomaly = np.mean(eval_losses[anomaly_indices])

		#print("is nan:", np.isnan(eval_dataset.anomaly_labels).sum(), np.isnan(perplexities).sum())
		auc_roc = metrics.roc_auc_score(eval_dataset.anomaly_labels, np.mean(perplexities, axis = 1))
		
		metric = {"eval_loss": np.mean(eval_losses), "eval_perplexity": mean_perplexity, "eval_auc_roc": auc_roc, \
						"eval_loss_normal": eval_loss_normal, "eval_perplexity_normal": perplexity_normal,
						"eval_loss_anomaly": eval_loss_anomaly, "eval_perplexity_anomaly": perplexity_anomaly}
		
		if local_rank == 0:
			self.log(metric)
			self._memory_tracker.stop_and_update_metrics(metric)

		return metric
	# End of Amazon addition.

def _seed_worker(_):
	"""
	Helper function to set worker seed during Dataloader initialization.
	"""
	worker_seed = torch.initial_seed() % 2**32
	random.seed(worker_seed)
	np.random.seed(worker_seed)
	torch.manual_seed(worker_seed)
	torch.cuda.manual_seed_all(worker_seed)

