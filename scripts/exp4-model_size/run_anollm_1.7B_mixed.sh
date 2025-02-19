#!/bin/bash
n_splits=5
setting=semi_supervised
TRAIN_GPUS="0,1,2,3"
INFERENCE_GPUS="1,2,3"
n_train_node=4
n_test_node=3
n_permutations=21

for model in 'smol-1.7b'; do
    batch_size=32
    eval_batch_size=$((batch_size*2))
    for dataset in 'vifd' 'fraudecom' 'lymphography'; do
        expdir=exp/$dataset/$setting/split$n_splits
        wandb online
        CUDA_VISIBLE_DEVICES=$TRAIN_GPUS torchrun --nproc_per_node=$n_train_node train_anollm.py --dataset $dataset --n_splits $n_splits --split_idx 0 --setting $setting --max_steps 2000 \
                                                    --batch_size $batch_size --model $model --binning standard --wandb --lora
        CUDA_VISIBLE_DEVICES=$INFERENCE_GPUS torchrun --nproc_per_node=$n_test_node evaluate_anollm.py --dataset $dataset --n_splits $n_splits --split_idx 0  --setting $setting\
                                                --batch_size $eval_batch_size  --n_permutations $n_permutations --model $model --binning standard   --lora
        wandb offline
        for ((split_idx = 1 ; split_idx < $n_splits ; split_idx++ )); do    
            CUDA_VISIBLE_DEVICES=$TRAIN_GPUS torchrun --nproc_per_node=$n_train_node train_anollm.py --dataset $dataset --n_splits $n_splits --split_idx $split_idx  --setting $setting --max_steps 2000\
                                                        --batch_size $batch_size --model $model --binning standard --lora 
            CUDA_VISIBLE_DEVICES=$INFERENCE_GPUS torchrun --nproc_per_node=$n_test_node evaluate_anollm.py --dataset $dataset --n_splits $n_splits --split_idx $split_idx  --setting $setting\
                                                    --batch_size $eval_batch_size  --n_permutations $n_permutations --model $model --binning standard --lora  
        done
        python -u src/get_results.py --dataset $dataset --n_splits $n_splits --setting $setting | tee $expdir/evaluate.log

    done

    batch_size=16
    eval_batch_size=$((batch_size*2))
    for dataset in 'seismic'; do
        expdir=exp/$dataset/$setting/split$n_splits
        wandb online
        CUDA_VISIBLE_DEVICES=$TRAIN_GPUS torchrun --nproc_per_node=$n_train_node train_anollm.py --dataset $dataset --n_splits $n_splits --split_idx 0 --setting $setting --max_steps 2000 \
                                                    --batch_size $batch_size --model $model --binning standard  --wandb --lora
        CUDA_VISIBLE_DEVICES=$INFERENCE_GPUS torchrun --nproc_per_node=$n_test_node evaluate_anollm.py --dataset $dataset --n_splits $n_splits --split_idx 0  --setting $setting\
                                                --batch_size $eval_batch_size  --n_permutations $n_permutations --model $model --binning standard  --lora   
        wandb offline
        for ((split_idx = 1 ; split_idx < $n_splits ; split_idx++ )); do    
            CUDA_VISIBLE_DEVICES=$TRAIN_GPUS torchrun --nproc_per_node=$n_train_node train_anollm.py --dataset $dataset --n_splits $n_splits --split_idx $split_idx  --setting $setting --max_steps 2000\
                                                        --batch_size $batch_size --model $model --binning standard  --lora
            CUDA_VISIBLE_DEVICES=$INFERENCE_GPUS torchrun --nproc_per_node=$n_test_node evaluate_anollm.py --dataset $dataset --n_splits $n_splits --split_idx $split_idx  --setting $setting\
                                                    --batch_size $eval_batch_size  --n_permutations $n_permutations --model $model --binning standard  --lora   
        done
        python -u src/get_results.py --dataset $dataset --n_splits $n_splits --setting $setting | tee $expdir/evaluate.log
    done

    batch_size=4
    eval_batch_size=$((batch_size*2))
    for dataset in 'fakejob'; do
        expdir=exp/$dataset/$setting/split$n_splits
        wandb online
        CUDA_VISIBLE_DEVICES=$TRAIN_GPUS torchrun --nproc_per_node=$n_train_node train_anollm.py --dataset $dataset --n_splits $n_splits --split_idx 0 --setting $setting --max_steps 20000 \
                                                    --batch_size $batch_size --model $model --binning standard  --wandb --lora --lr 1e-3
        CUDA_VISIBLE_DEVICES=$INFERENCE_GPUS torchrun --nproc_per_node=$n_test_node evaluate_anollm.py --dataset $dataset --n_splits $n_splits --split_idx 0  --setting $setting\
                                                --batch_size $eval_batch_size  --n_permutations $n_permutations --model $model --binning standard  --lr 1e-3 --lora 
        wandb offline
        for ((split_idx = 1 ; split_idx < $n_splits ; split_idx++ )); do    
            CUDA_VISIBLE_DEVICES=$TRAIN_GPUS torchrun --nproc_per_node=$n_train_node train_anollm.py --dataset $dataset --n_splits $n_splits --split_idx $split_idx  --setting $setting --max_steps 20000\
                                                        --batch_size $batch_size --model $model --binning standard  --lora --lr 1e-3
            CUDA_VISIBLE_DEVICES=$INFERENCE_GPUS torchrun --nproc_per_node=$n_test_node evaluate_anollm.py --dataset $dataset --n_splits $n_splits --split_idx $split_idx  --setting $setting\
                                                    --batch_size $eval_batch_size  --n_permutations $n_permutations --model $model --binning standard  --lr 1e-3 --lora 
        done
        python -u src/get_results.py --dataset $dataset --n_splits $n_splits --setting $setting | tee $expdir/evaluate.log
    done
    
    batch_size=4
    eval_batch_size=$((batch_size*2))
    for ((idx = 0 ; idx < 6 ; idx++ )); do  
        dataset=20news-$idx
        expdir=exp/$dataset/$setting/split$n_splits
        wandb offline
        for ((split_idx = 1 ; split_idx < $n_splits ; split_idx++ )); do    
            CUDA_VISIBLE_DEVICES=$TRAIN_GPUS torchrun --nproc_per_node=$n_train_node train_anollm.py --dataset $dataset --n_splits $n_splits --split_idx $split_idx  --setting $setting --max_steps 2000\
                                                        --batch_size $batch_size --model $model --binning standard  --lr 0.0005 --lora 
            CUDA_VISIBLE_DEVICES=$INFERENCE_GPUS torchrun --nproc_per_node=$n_test_node evaluate_anollm.py --dataset $dataset --n_splits $n_splits --split_idx $split_idx  --setting $setting\
                                                    --batch_size $eval_batch_size  --n_permutations $n_permutations --model $model --binning standard  --lr 0.0005 --lora 
        done
        python -u src/get_results.py --dataset $dataset --n_splits $n_splits --setting $setting | tee $expdir/evaluate.log
    done

done