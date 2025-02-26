import os
from pathlib import Path

import numpy as np
import argparse
import torch.distributed as dist
import torch
import wandb
import time

from anollm import AnoLLM
from src.data_utils import load_data, DATA_MAP, get_text_columns, get_max_length_dict

#run by torchrun --nproc_per_node=8 train_llm.py <args> 

def get_args():
	parser = argparse.ArgumentParser()
	parser.add_argument("--dataset", type = str, default='wine', choices = [d.lower() for d in DATA_MAP.keys()],
					help="Name of datasets in the ODDS benchmark")
	parser.add_argument("--exp_dir", type = str, default=None)
	parser.add_argument("--setting", type = str, default='semi_supervised', choices = ['semi_supervised', 'unsupervised'], help="semi_supervised:an uncontaminated, unsupervised setting; unsupervised:a contaminated, unsupervised setting")
	
	# wandb
	parser.add_argument("--wandb", action='store_true')
	parser.add_argument("--entity", type = str, default = None)
	parser.add_argument("--project", type = str, default = 'AnoLLM')
	
	#dataset hyperparameters
	parser.add_argument("--data_dir", type = str, default='data')
	parser.add_argument("--n_splits", type = int, default=5)
	parser.add_argument("--split_idx", type = int, default=0) # 0 to n_split-1
	parser.add_argument("--train_ratio", type = float, default=0.5)
	parser.add_argument("--seed", type = int, default=42)
	
	# preprocessing
	parser.add_argument("--binning", type = str, choices=['quantile', 'equal_width', 'language', 'none', 'standard'], default='standard')
	parser.add_argument("--n_buckets", type = int, default=10)
	parser.add_argument("--remove_feature_name", action = 'store_true')
	
	#training
	parser.add_argument("--model", type = str, choices = ['gpt2', 'distilgpt2', 'smol', 'smol-360', 'smol-1.7b'], default='smol')
	parser.add_argument("--batch_size", type = int, default=32) # per gpu, eval_batch_size = 2*batch_size
	parser.add_argument("--lr", type = float, default=5e-5)
	parser.add_argument("--lora", action='store_true', default=False)
	parser.add_argument("--max_steps", type = int, default=2000) 
	parser.add_argument("--eval_steps", type = int, default = 1000)
	parser.add_argument("--random_init", action='store_true', default=False)
	parser.add_argument("--no_random_permutation", action='store_true', default=False)

	args = parser.parse_args()
	if args.exp_dir is None:
		args.exp_dir = Path('exp') / args.dataset / args.setting / "split{}".format(args.n_splits) / "split{}".format(args.split_idx)

	if args.model == 'smol':
		args.model = 'HuggingFaceTB/SmolLM-135M'
	elif args.model == 'smol-360':
		args.model = 'HuggingFaceTB/SmolLM-360M'
	elif args.model == 'smol-1.7b':	
		args.model = 'HuggingFaceTB/SmolLM-1.7B'
	
	args.save_dir = Path(args.exp_dir) / 'models' # save to save models
	os.makedirs(args.save_dir, exist_ok = True)

	return args

def get_run_name(args):
	name = 'anollm' 
	name += '_lr{}'.format(args.lr)
	name += '_{}'.format(args.binning)
	
	if args.model == 'HuggingFaceTB/SmolLM-135M': 
		name += '_smolLM'
	elif args.model == 'HuggingFaceTB/SmolLM-360M':
		name += '_smolLM360'
	elif args.model == 'HuggingFaceTB/SmolLM-1.7B':
		name += '_smolLM1.7B'
	else:
		name += '_' + args.model
	
	if args.random_init:
		name += '_random_init'	
	
	if args.no_random_permutation:
		name += '_no_random_permutation'	
	
	if args.lora:
		name += '_lora'
	name += "_test"
	return name


def main():
	# Set CUDA devices for each process
	local_rank = int(os.environ["LOCAL_RANK"])
	torch.cuda.set_device(local_rank)

	args = get_args()
	if dist.get_rank() == 0:
		X_train, X_test, y_train, y_test = load_data(args)
	dist.barrier()
	if dist.get_rank() != 0:
		X_train, X_test, y_train, y_test = load_data(args)
	dist.barrier()
	
	run_name = get_run_name(args)
	efficient_finetuning = 'lora' if args.lora else ''
	model_path = args.save_dir / '{}.pt'.format(run_name)
	dataset_tmp_path = args.save_dir / (run_name + '_data')
	
	os.makedirs(dataset_tmp_path, exist_ok= True)
	print("Model path:", model_path)	
	#if False:
	if os.path.exists(model_path):
		print("Model exists, skip training")
		return

	max_length_dict = get_max_length_dict(args.dataset)
	text_columns = get_text_columns(args.dataset)
	def get_model():
		model = AnoLLM(args.model,
					batch_size=args.batch_size,
					max_steps = args.max_steps,
					efficient_finetuning = efficient_finetuning,
					max_length_dict=max_length_dict, 
					textual_columns = text_columns,
					random_init=args.random_init,
					no_random_permutation=args.no_random_permutation,
					bf16=True,
					adam_beta2=0.99,
					adam_epsilon=1e-7,
					learning_rate=args.lr,
				)
		return model 
	# Initialize the LLM 
	if dist.get_rank() == 0:
		anollm = get_model()
	dist.barrier()
	if dist.get_rank() != 0:
		anollm = get_model()
	dist.barrier()
	# Move the model to the appropriate GPU
	anollm.model.to(local_rank)  

	# Wrap the model for distributed training
	anollm.model = torch.nn.parallel.DistributedDataParallel(
		anollm.model, device_ids=[local_rank], output_device=local_rank
	)
	if args.wandb and dist.get_rank() == 0: 
		run = wandb.init(
			entity=args.entity,
			project=args.project,
			name = "{}_splits{}_{}_{}".format(args.dataset, args.split_idx, args.n_splits, run_name),
		)
	if len(X_test) > 3000:
		np.random.seed(args.seed)
		X_test.reset_index(drop = True, inplace = True)
		indices = np.random.choice(len(X_test), 3000, replace = False)
		X_test = X_test.loc[indices].reset_index(drop = True)
		y_test = y_test[indices]
	if not args.wandb:
		X_test, y_test = None, None
	
	# Train the model
	start_time = time.time()
	trainer = anollm.fit(X_train, X_train.columns.to_list(), 
					  use_wandb = args.wandb, 
					  data_val=X_test, 
					  label_val = y_test,
					  eval_steps = args.eval_steps,
					  processed_data_dir = dataset_tmp_path,
			)
	end_time = time.time()

	# Save the model only from rank 0 process
	if dist.get_rank() == 0:
		
		print("Training time:", end_time - start_time)
		run_time_dir = args.exp_dir / "run_time" / "train"
		os.makedirs(run_time_dir, exist_ok = True)
		run_time_path = run_time_dir / "{}.txt".format(run_name)
		with open(run_time_path, 'w') as f:
			f.write(str(end_time - start_time))

		print("Save model to ", model_path)
		anollm.save_state_dict(model_path)
		
		
	dist.destroy_process_group()

if __name__ == "__main__":
	# Initialize the distributed process group
	dist.init_process_group(backend="nccl") 
	main()