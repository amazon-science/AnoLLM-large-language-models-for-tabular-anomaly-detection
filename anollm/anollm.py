'''
Original Copyright (c) 2022 Kathrin Seßler and Vadim Borisov. Licensed under the MIT License.
Part of code is adapted from the GReaT repository (https://github.com/kathrinse/be_great/tree/main)
Modifications Copyright 2025 Amazon.com, Inc. or its affiliates. All Rights Reserved.
'''
import os
import warnings

import logging
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, AutoConfig
from torch.nn import CrossEntropyLoss
import typing as tp
from transformers import Trainer
from collections import OrderedDict
from pathlib import Path

from anollm.anollm_trainer import AnoLLMTrainer 
from anollm.anollm_utils import _array_to_dataframe
from anollm.anollm_dataset import AnoLLMDataset, AnoLLMDataCollator


class AnoLLM:
	"""AnoLLM Class

	The AnoLLM class handles the whole generation flow. It is used to fine-tune a large language model for tabular data,
	and to sample synthetic tabular data.

	Attributes:
		llm (str): HuggingFace checkpoint of a pretrained large language model, used a basis of our model
		tokenizer (AutoTokenizer): Tokenizer, automatically downloaded from llm-checkpoint
		model (AutoModelForCausalLM): Large language model, automatically downloaded from llm-checkpoint
		experiment_dir (str): Directory, where the training checkpoints will be saved
		batch_size (int): Batch size used for fine-tuning
		train_hyperparameters (dict): Additional hyperparameters added to the TrainingArguments used by the
		 HuggingFaceLibrary, see here the full list of all possible values
		 https://huggingface.co/docs/transformers/main/en/main_classes/trainer#transformers.TrainingArguments
		columns (list): List of all features/columns of the tabular dataset
		num_cols (list): List of all numerical features/columns of the tabular dataset
	"""

	def __init__(
		self,
		llm: str,
		experiment_dir: str = "models",
		batch_size: int = 8,
		efficient_finetuning: str = "",
		max_length_dict: tp.Optional[tp.Dict[str, int]] = None,
		textual_columns: tp.List[str] = [], # columns that needs to be normalized, e.g. text columns
		random_init: bool = False, # if True, the model will be initialized with random weights.
		no_random_permutation: bool = False, # if True, columns will not be permuted randomly
		**train_kwargs,
	):
		"""

		Args:
			llm: HuggingFace checkpoint of a pretrained large language model, used a basis of our model
			experiment_dir:  Directory, where the training checkpoints will be saved
			batch_size: Batch size used for fine-tuning
			efficient_finetuning: if efficient_finetuning is 'lora', the model will be fine-tuned with LoRA
			max_length_dict: Dictionary that contains the maximum length of each textual features. 
			train_kwargs: Additional hyperparameters added to the TrainingArguments used by the HuggingFaceLibrary,
			 see here the full list of all possible values
			 https://huggingface.co/docs/transformers/main/en/main_classes/trainer#transformers.TrainingArguments
		"""
		# Load Model and Tokenizer from HuggingFace
		self.efficient_finetuning = efficient_finetuning
		self.llm = llm
		self.tokenizer = AutoTokenizer.from_pretrained(self.llm)
		self.tokenizer.pad_token = self.tokenizer.eos_token
		if not random_init:
			self.model = AutoModelForCausalLM.from_pretrained(self.llm, torch_dtype=torch.bfloat16)
		else:
			config = AutoConfig.from_pretrained(self.llm)
			self.model = AutoModelForCausalLM.from_config(config)

		if self.efficient_finetuning == "lora":
			# Lazy importing
			try:
				from peft import (
					LoraConfig,
					get_peft_model,
				)
			except ImportError:
				raise ImportError(
					"This function requires the 'peft' package. Please install it with - pip install peft"
				)

			# Define LoRA Config
			lora_config = LoraConfig(
				r=8, 
				lora_alpha=32,
				target_modules=[
					"q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"
				],  # this is specific for smolLM, to be adapted
				lora_dropout=0.1,
				bias="none",
			)
			# add LoRA adaptor
			self.model = get_peft_model(self.model, lora_config)
			self.model.print_trainable_parameters()

		# Set the training hyperparameters
		self.experiment_dir = experiment_dir
		self.batch_size = batch_size
		self.max_length_dict = max_length_dict
		self.textual_columns = textual_columns
		self.no_random_permutation = no_random_permutation
		self.train_hyperparameters = train_kwargs

	def fit(
		self,
		data: tp.Union[pd.DataFrame, np.ndarray],
		column_names: tp.Optional[tp.List[str]] = None,
		resume_from_checkpoint: tp.Union[bool, str] = False,
		use_wandb: bool = False,
		data_val: tp.Union[pd.DataFrame, np.ndarray] = None,
		label_val: np.ndarray = None,
		eval_steps: int = 400,
		processed_data_dir: str = None
		) -> Trainer:
		"""Fine-tune AnoLLM using tabular data.

		Args:
			data: Pandas DataFrame that contains the tabular data
			column_names: If data is Numpy Array, the feature names have to be defined. If data is Pandas
			DataFrame, the value is ignored

		Returns:
			AnoLLM Trainer used for the fine-tuning process
		"""
		df = _array_to_dataframe(data, columns=column_names)

		# Convert DataFrame into HuggingFace dataset object
		logging.info("Convert data into HuggingFace dataset object...")
		dataset = AnoLLMDataset.from_pandas(df, preserve_index=False)
		dataset.set_tokenizer(self.tokenizer)
		dataset.set_textual_columns(self.textual_columns)
		if self.no_random_permutation:
			dataset.fix_column_order()

		processed_data_path = Path(processed_data_dir) / "train_data.pkl" if processed_data_dir is not None else None 
		dataset.prepare(is_eval = False, max_length_dict=self.max_length_dict, 
				  data_path=processed_data_path)
		print("Data 0:", self.tokenizer.decode(dataset[0]['input_ids'] ))
		# Set training hyperparameters
		logging.info("Create AnoLLM Trainer...")
		trainer_args = {}

		if data_val is not None:
			df_val = _array_to_dataframe(data_val, columns=column_names)
			dataset_val = AnoLLMDataset.from_pandas(df_val, preserve_index=False)
			dataset_val.set_tokenizer(self.tokenizer)
			dataset_val.set_anomaly_label(label_val)
			dataset_val.set_textual_columns(self.textual_columns)
			if self.no_random_permutation:
				dataset_val.fix_column_order()
			
			processed_data_path = Path(processed_data_dir) / "val_data.pkl" if processed_data_dir is not None else None 
			dataset_val.prepare(is_eval = True, max_length_dict=self.max_length_dict, 
					   data_path = processed_data_path)

			self.train_hyperparameters["eval_strategy"] = "steps"
			self.train_hyperparameters["eval_steps"] = eval_steps
			trainer_args["eval_dataset"] = dataset_val
		
		if use_wandb:
			self.train_hyperparameters["report_to"] = ["wandb"]
			self.train_hyperparameters["logging_strategy"] = "steps"
			self.train_hyperparameters["logging_dir"] = "./logs"
			self.train_hyperparameters["logging_steps"] = 50
			self.train_hyperparameters["log_level"] = 'info'	
		
		training_args = TrainingArguments(
			self.experiment_dir,
			per_device_train_batch_size=self.batch_size,
			per_device_eval_batch_size=self.batch_size * 2,
			save_strategy = 'no',
			max_grad_norm = 0.7,
			**self.train_hyperparameters,
		)

		#optimizer = bnb.optim.PagedAdamW32bit(self.model.parameters(), betas=(0.9, 0.95), eps=1e-5)
		trainer = AnoLLMTrainer(
			self.model,
			training_args,
			train_dataset=dataset,
			tokenizer=self.tokenizer,
			data_collator=AnoLLMDataCollator(self.tokenizer),
			**trainer_args,
		)

		if data_val is not None:
			trainer.set_eval_setting(n_permutations=1)

		# Start training
		logging.info("Start training...")
		trainer.train(resume_from_checkpoint=resume_from_checkpoint)

		return trainer
	
	def decision_function(
		self, 
		df_test: pd.DataFrame,
		n_permutations: int = 16, 
		batch_size: int = 32,
		device: str = "cuda",
		feature_wise: bool = False,
		) -> np.ndarray:
		''' Obtain anomaly scores for each sample in the test data
		df_test: pandas dataframe of test data
		n_permutations: number of permutations to calculate the anomaly score
		batch_size: batch size for prediction
		device: device to run the model
		feature_wise: get anomaly scores for each features. If True, returns anomaly scores for each feature in the test data. Size: (n_test, n_features, n_permutation)
		# Returns:
		# np.ndarray: Anomaly scores for each sample in the test data. Size: (n_test, n_permutation) or (n_test, n_features, n_permutation) if feature_wise is True
		'''
		# Convert DataFrame into HuggingFace dataset object
		logging.info("Convert data into HuggingFace dataset object...")
		dataset = AnoLLMDataset.from_pandas(df_test, preserve_index=False)
		dataset.set_tokenizer(self.tokenizer)
		dataset.set_textual_columns(self.textual_columns)
		
		if self.no_random_permutation:
			dataset.fix_column_order()
		
		dataset.prepare(is_eval = True, max_length_dict=self.max_length_dict)
		dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle = False, 
												 collate_fn = AnoLLMDataCollator(self.tokenizer))
		
		self.model.to(device)
		comma_id =  self.tokenizer.convert_tokens_to_ids(',')
		n_col = len(df_test.columns)
		column_names = dataset.get_column_names()
		if feature_wise:
			anomaly_scores = np.zeros((len(df_test), n_col, n_permutations))
		else:
			anomaly_scores = np.zeros((len(df_test), n_permutations))

		loss_fct = CrossEntropyLoss(reduction="none")


		for perm_idx in tqdm(range(n_permutations)):
			start_idx = 0
			dataset.shuffle_column_order()
			for data in dataloader:
				encoded_batch = data["input_ids"].to(device)
				attn_mask = data["attention_mask"].to(device)
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

				if feature_wise:
					score_batch = (loss_fct(shift_logits.transpose(1, 2), shift_labels) * shift_attention_mask_batch).cpu().numpy() # batch * (ori_seq_len -1)

					for i in range(len(encoded_batch)):
						for j in range(n_col): 
							start_pos = start_pos_batch[i][j]
							end_pos = end_pos_batch[i][j]
							col_idx = col_indices_batch[i][j]
							anomaly_scores[start_idx+i, col_idx, perm_idx] = score_batch[i, start_pos:end_pos].sum()
				elif len(self.textual_columns) > 0:
					score_batch = (loss_fct(shift_logits.transpose(1, 2), shift_labels) * shift_attention_mask_batch).cpu().numpy() # batch * (ori_seq_len -1)
					for i in range(len(encoded_batch)):
						score_single = 0
						for j in range(n_col): 
							start_pos = start_pos_batch[i][j]
							end_pos = end_pos_batch[i][j]
							col_idx = col_indices_batch[i][j]
							if column_names[col_idx] in self.textual_columns:
								score_single += score_batch[i, start_pos:end_pos].sum() / (end_pos - start_pos)
							else:
								score_single += score_batch[i, start_pos:end_pos].sum()
						anomaly_scores[start_idx+i, perm_idx] = score_single
				else:
					score_batch = (loss_fct(shift_logits.transpose(1, 2), shift_labels) * shift_attention_mask_batch).sum(1) # remove normalization
					anomaly_scores[start_idx:end_idx, perm_idx] = score_batch.cpu().numpy()
				start_idx = end_idx

		return anomaly_scores #(len(df_test), n_permutations)
	
	def save_state_dict(self, path: str):
		"""Save AnoLLM Model 

		Saves the model weights and a configuration file in the given directory.
		Warning: Only works in DDP setting!

		Args:
			path: Path where to save the model
		"""
		directory = os.path.dirname(path)
		# Make directory
		if os.path.isdir(directory):
			warnings.warn(f"Directory {path} already exists and is overwritten now.")
		else:
			os.mkdir(directory)

		state_dict = self.model.state_dict()
		new_state_dict = OrderedDict()
		for k, v in state_dict.items():
			name = k[7:]  # remove `module.`
			new_state_dict[name] = v
		# Save the model with the modified state dict
		torch.save(new_state_dict, path)
	
	def load_from_state_dict(self, path: str):
		"""Load AnoLLM model from state_dict

		Args:
			path: path where AnoLLM model is saved
		"""
		
		if self.efficient_finetuning == 'lora':
			self.model.to('cpu')
			state_dict = torch.load(path, map_location=torch.device('cpu'))
			self.model.load_state_dict(state_dict)
		else:
			self.model.load_state_dict(torch.load(path))
