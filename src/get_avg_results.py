import argparse

import numpy as np
import pandas as pd

from data_utils import  DATA_MAP, MIXED, ODDS
from get_results import get_metrics, aggregate_results, filter_results

def get_args():
	parser = argparse.ArgumentParser()
	parser.add_argument("--dataset", type = str, default='all', choices = ['all', 'mixed', 'odds']) # subset: 10 datasets that contain feature names
	parser.add_argument("--setting", type = str, default='semi_supervised', choices = ['semi_supervised', 'unsupervised'])
	parser.add_argument("--exp_dir", type = str, default=None)
	parser.add_argument("--metric", type = str, choices =["AUC-ROC", "F1", "AUC-PR"], default='AUC-ROC')
	#dataset hyperparameters
	parser.add_argument("--data_dir", type = str, default='data')
	parser.add_argument("--n_splits", type = int, default=5)
	parser.add_argument("--only_normalized", action='store_true', default=False)
	parser.add_argument("--only_ordinal", action='store_true', default=False)
	args = parser.parse_args()
	
	return args

def main():
	args = get_args()
	roc_scores = {}
	ranking_dict = {}
	std_scores = {}
	if args.dataset == 'all':
		DATASETS = [k for k in DATA_MAP.keys()]
	elif args.dataset == 'odds':
		DATASETS = ODDS
	elif args.dataset == 'mixed':
		DATASETS = MIXED
	# sorted datasets by alphabetic order
	DATASETS = sorted(DATASETS)
	all_rocs = {}
	for dataset_idx, dataset in enumerate(DATASETS):
		try:
			print("*"*100)
			print(dataset)
			args.split_idx = None
			args.dataset = dataset
			L = []
			for i in range(args.n_splits):
				args.split_idx = i
				args.exp_dir = None
				results = get_metrics(args, only_normalized=args.only_normalized, only_ordinal=args.only_ordinal)
				L.append(results)
			metrics, rankings = aggregate_results(L)
		except:
			print("Error in dataset: ", dataset)
			continue
		
		metrics = filter_results(metrics)
		for k in metrics.keys():
			
			if k not in all_rocs:
				all_rocs[k] = np.zeros((len(DATASETS), args.n_splits))
			
			for i in range(args.n_splits):
				all_rocs[k][dataset_idx] = metrics[k][args.metric] 
		
		roc_scores[dataset] = { k: np.mean(metrics[k][args.metric]) for k in metrics.keys()} 
		ranking_dict[dataset] = {k: int(rankings[args.metric][idx]) for idx, k in enumerate(metrics.keys())}
		std_scores[dataset] = { k: np.std(metrics[k][args.metric]) for k in metrics.keys()}

	df = pd.DataFrame(roc_scores).T
	avg_row = df.mean(axis=0)
	df.loc['avg'] = avg_row
	df = df.round(3)
	print(df)
	df.to_csv('exp/{}_avg_{}.csv'.format(args.setting, args.metric))

	'''
	ranking_df = pd.DataFrame(ranking_dict).T
	avg_row = ranking_df.mean(axis=0)
	ranking_df.loc['avg'] = avg_row
	ranking_df = ranking_df.round(3) 
	print(ranking_df)
	ranking_df.to_csv('exp/{}_avg_ranking.csv'.format(args.setting))
	'''
	# std
	std_df = pd.DataFrame(std_scores).T
	avg_std = []
	for c in std_df.columns:
		if c not in all_rocs:
			avg_std.append(0)
			continue
		std = np.std( np.mean(all_rocs[c], axis=0))
		avg_std.append(std)
	std_df.loc['avg'] = avg_std
	std_df = std_df.round(3)
	print(std_df)
	std_df.to_csv('exp/{}_std_{}.csv'.format(args.setting,args.metric))

if __name__ == '__main__':
	main()
