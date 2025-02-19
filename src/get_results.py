import os
from pathlib import Path
import argparse

from sklearn import metrics
import numpy as np
import pandas as pd
from data_utils import load_data, DATA_MAP


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type = str, default='wine', choices = [d.lower() for d in DATA_MAP.keys()],
                    help="Name of datasets in the ODDS benchmark")
    parser.add_argument("--exp_dir", type = str, default=None)
    parser.add_argument("--setting", type = str, default='semi_supervised', choices = ['semi_supervised', 'unsupervised'])
    
    #dataset hyperparameters
    parser.add_argument("--data_dir", type = str, default='data')
    parser.add_argument("--n_splits", type = int, default=5)
    parser.add_argument("--split_idx", type = int, default=None) # 0 to n_split-1

    args = parser.parse_args()
    
    return args

def tabular_metrics(y_true, y_score):
    """
    Calculates evaluation metrics for tabular anomaly detection.
    Adapted from  https://github.com/xuhongzuo/DeepOD/blob/main/deepod/metrics/_anomaly_detection.py 
    Args:
    
        y_true (np.array, required): 
            Data label, 0 indicates normal timestamp, and 1 is anomaly.
            
        y_score (np.array, required): 
            Predicted anomaly scores, higher score indicates higher likelihoods to be anomaly.

    Returns:
        tuple: A tuple containing:
        
        - auc_roc (float):
            The score of area under the ROC curve.
            
        - auc_pr (float):
            The score of area under the precision-recall curve.
            
        - f1 (float): 
            The score of F1-score.
        
        - precision (float):
            The score of precision.
        
        - recall (float):  
            The score of recall.

    """
    # F1@k, using real percentage to calculate F1-score
    n_test = len(y_true)
    new_index = np.random.permutation(n_test) # shuffle y to prevent bias of ordering (argpartition may discard entries with same value)
    y_true = y_true[new_index]
    y_score = y_score[new_index]

    #ratio = 100.0 * len(np.where(y_true == 0)[0]) / len(y_true)
    #thresh = np.percentile(y_score, ratio)
    #y_pred = (y_score >= thresh).astype(int)
    
    top_k = len(np.where(y_true == 1)[0]) 
    indices = np.argpartition(y_score, -top_k)[-top_k:]
    y_pred = np.zeros_like(y_true)
    y_pred[indices] = 1

    y_true = y_true.astype(int)
    p, r, f1, support = metrics.precision_recall_fscore_support(y_true, y_pred, average='binary')

    return metrics.roc_auc_score(y_true, y_score), metrics.average_precision_score(y_true, y_score), f1, p, r

def get_metrics(args, only_raw = False, only_normalized = False, only_ordinal = False):
    X_train, X_test, y_train, y_test = load_data(args)
    if isinstance(y_test, pd.Series):
        y_test = np.array(y_test)
    #y_test = y_test.to_numpy()
    if args.exp_dir is None:
        args.exp_dir = Path('exp') / args.dataset / args.setting / "split{}".format(args.n_splits) / "split{}".format(args.split_idx)
    score_dir = args.exp_dir / 'scores'
    if not os.path.exists(score_dir):
        raise ValueError("Score directory {} does not exist".format(score_dir))


    method_dict = {}
    for score_npy in os.listdir(score_dir):
        if '.npy' in score_npy:
            if score_npy.startswith('raw'):
                continue
            if is_baseline(score_npy) and only_normalized:
                if 'normalized' not in score_npy:
                    continue
                
            elif is_baseline(score_npy) and only_ordinal:
                if 'ordinal' not in score_npy:
                    continue


            method = '.'.join(score_npy.split('.')[:-1])
            if method == 'rdp':
                continue
            scores = np.load(score_dir / score_npy)
            if np.isnan(scores).any():
                print("NaNs in scores for {}".format(method))
                method_dict[method] = [0, 0, 0, 0, 0] 
            elif np.isinf(scores).any():
                print("Infs in scores for {}".format(method))
                method_dict[method] = [0, 0, 0, 0, 0] 
            else:
                auc_roc, auc_pr, f1, p, r = tabular_metrics(y_test, scores)
                method_dict[method] = [auc_roc, auc_pr, f1, p, r]
    # get ranking info for all methods
    rankings = []
    method = list(method_dict.keys())[0]
    for i in range(len(method_dict[method])):
        scores = [-method_dict[k][i] for k in method_dict.keys()]
        ranking = np.argsort(scores).argsort() + 1
        rankings.append(ranking)
        
    print("-"*100)
    for idx, (k, v) in enumerate(method_dict.items()):
        #print("{:30s}: AUC-ROC: {:.4f}, AUC-PR: {:.4f}, F1: {:.4f}".format(k, v[0], v[1], v[2]))
        print("{:30s}: AUC-ROC: {:.4f} ({:2d}), AUC-PR: {:.4f} ({:2d}), F1: {:.4f} ({:2d}), P: {:.4f} ({:2d}), R: {:.4f} ({:2d})".format(k, 
            v[0], rankings[0][idx],
            v[1], rankings[1][idx],
            v[2], rankings[2][idx],
            v[3], rankings[3][idx],
            v[4], rankings[4][idx],
        ))

    return method_dict
def is_baseline(s):
    if 'anollm' in s:
        return False
    return True

def filter_results(d:dict):
    d2 = {}
    for k in d.keys():
        new_key = k
        if is_baseline(k):
            #baselines
            d2[k] = d[k]
        else:
            if '_lora' in k:
                temp = k.replace('_lora', '')
                if temp in d:
                    continue
                else:
                    new_key = new_key.replace('_lora', '')
                    d2[new_key] = d[k]
            else:
                d2[new_key] = d[k]
    return d2
                
def aggregate_results(m_dicts):
    aggregate_results = {k: {'AUC-ROC':[], 'AUC-PR': [], 'F1': [], 'P': [], 'R':[]} for k in m_dicts[0].keys()}
    for i in range(len(m_dicts)):
        all_keys = list(m_dicts[0].keys()) 
        for k in all_keys:
            try:
                aggregate_results[k]['AUC-ROC'] += [m_dicts[i][k][0]]
                aggregate_results[k]['AUC-PR'] += [m_dicts[i][k][1]]
                aggregate_results[k]['F1'] += [m_dicts[i][k][2]]
                aggregate_results[k]['P'] += [m_dicts[i][k][3]]
                aggregate_results[k]['R'] += [m_dicts[i][k][4]]
            except:
                print("Incomplete results for ", k)
                if k in aggregate_results:
                    del aggregate_results[k]
                for i in range(len(m_dicts)):
                    if k in m_dicts[i]:
                        del m_dicts[i][k]
                continue

    print("-"*100)
    
    # get ranking info for all methods
    rankings = {}
    key =  list(m_dicts[0].keys())[0] 
    for metric_name in aggregate_results[key].keys():
        scores = [-np.mean(aggregate_results[k][metric_name]) for k in aggregate_results.keys()] 
        ranking = np.argsort(scores).argsort() + 1
        rankings[metric_name] = ranking

    for idx, k in enumerate(aggregate_results.keys()):
        print("{:30s}: AUC-ROC: {:.4f} +- {:.4f} ({:2d}), AUC-PR: {:.4f} +- {:.4f} ({:2d}), F1: {:.4f} +- {:.4f} ({:2d})  P: {:.4f} +- {:.4f} ({:2d})  R: {:.4f} +- {:.4f} ({:2d})".format(k, 
            np.mean(aggregate_results[k]['AUC-ROC']), np.std(aggregate_results[k]['AUC-ROC']), rankings['AUC-ROC'][idx],
            np.mean(aggregate_results[k]['AUC-PR']), np.std(aggregate_results[k]['AUC-PR']), rankings['AUC-PR'][idx],
            np.mean(aggregate_results[k]['F1']), np.std(aggregate_results[k]['F1']), rankings['F1'][idx],
            np.mean(aggregate_results[k]['P']), np.std(aggregate_results[k]['P']), rankings['P'][idx],
            np.mean(aggregate_results[k]['R']), np.std(aggregate_results[k]['R']), rankings['R'][idx],
            ))
    return aggregate_results, rankings 

def main():
    args = get_args()
    if args.split_idx is None:
        L = []
        for i in range(args.n_splits):
            args.split_idx = i
            args.exp_dir = None
            results = get_metrics(args)
            L.append(results)
        aggregate_results(L)
    else:
        print(args) 
        scores = get_metrics(args)


if __name__ == '__main__':
    main()