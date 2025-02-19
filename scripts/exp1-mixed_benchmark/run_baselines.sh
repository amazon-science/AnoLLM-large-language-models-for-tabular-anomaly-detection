#!/bin/bash
n_splits=5
setting='semi_supervised'
for dataset in 'vifd' 'fraudecom' 'lymphography' 'seismic' 'fakejob'; do
	expdir=exp/$dataset/$setting/split$n_splits
	for ((split_idx = 0 ; split_idx < $n_splits ; split_idx++ )); do  
		CUDA_VISIBLE_DEVICES=0 python evaluate_baselines.py --dataset $dataset --n_splits $n_splits --normalize  --setting $setting --split_idx $split_idx & 
	done
	wait
	expdir=exp/$dataset/$setting/split$n_splits
	python -u src/get_results.py --dataset $dataset --n_splits $n_splits --setting $setting | tee $expdir/evaluate.log
done

for ((idx = 0 ; idx < 5 ; idx++ )); do  
	dataset=20news-$idx
	expdir=exp/$dataset/$setting/split$n_splits
	for ((split_idx = 0 ; split_idx < $n_splits ; split_idx++ )); do  
		CUDA_VISIBLE_DEVICES=$split_idx python evaluate_baselines.py --dataset $dataset --n_splits $n_splits --normalize  --setting $setting --split_idx $split_idx & 
	done
	wait
	expdir=exp/$dataset/$setting/split$n_splits
	python -u src/get_results.py --dataset $dataset --n_splits $n_splits --setting $setting | tee $expdir/evaluate.log
done