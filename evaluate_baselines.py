import os
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import time

from deepod.models.tabular import DeepSVDD, REPEN, RDP, RCA, GOAD, NeuTraL, SLAD, DeepIsolationForest
from src.baselines.icl import ICL
from src.baselines.dte import DTECategorical
from pyod.models.ecod import ECOD
from pyod.models.pca import PCA
from pyod.models.knn import KNN
from pyod.models.iforest import IForest
from src.data_utils import load_data, DATA_MAP, df_to_numpy, get_text_columns

DEEPOD_METHODS= {
    'ecod': ECOD(),
    'knn': KNN(),
    'iforest': IForest(),
    'pca': PCA(),
    'deepsvdd': DeepSVDD(),
    'repen': REPEN(),
    'rdp': RDP(),
    'rca': RCA(),
    'goad': GOAD(),
    'neutural': NeuTraL(),
    'icl': ICL(),
    'dif': DeepIsolationForest(),
    'slad': SLAD(),
    'dte': DTECategorical(),
}

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type = str, default='wine', choices = [d.lower() for d in DATA_MAP.keys()],
                    help="Name of datasets in the ODDS benchmark")
    parser.add_argument("--exp_dir", type = str, default=None)
    parser.add_argument("--setting", type = str, default='semi_supervised', choices = ['semi_supervised', 'unsupervised'])
    parser.add_argument("--normalize", action='store_true', default=False) #normalize the numerical features to have zero mean and unit variance 
    parser.add_argument("--text_encoding", type = str, default='word2vec', choices = ['tfidf', 'word2vec', 'bag_of_words'])
    #dataset hyperparameters
    parser.add_argument("--data_dir", type = str, default='data')
    parser.add_argument("--n_splits", type = int, default=5)
    parser.add_argument("--split_idx", type = int, default=None) # 0 to n_split-1
    parser.add_argument("--cat_encoding", type = str, default="one_hot", choices = ["one_hot", "ordinal"])
    args = parser.parse_args()
    
    return args

def get_run_name(args, method):
    if args.normalize:
        run_name = '{}_normalized'.format(method)
    else:
        run_name = method

    if args.cat_encoding == 'ordinal':
        run_name += "_ordinal"
        
    if args.dataset in ['fakejob', 'fakenews']:
        run_name += "_{}".format(args.text_encoding)
    run_name += "_test_run_time"

    return run_name

def benchmark(args):
    X_train, X_test, y_train, y_test = load_data(args)
    
    if args.exp_dir is None:
        args.exp_dir = Path('exp') / args.dataset / args.setting / "split{}".format(args.n_splits) / "split{}".format(args.split_idx)
     
    if not os.path.exists(args.exp_dir):
        os.makedirs(args.exp_dir, exist_ok = True)
        #raise ValueError("Experiment directory {} does not exist".format(args.exp_dir))
    
    score_dir = args.exp_dir / 'scores'
    os.makedirs(score_dir, exist_ok = True)
    
    train_run_time_dir = args.exp_dir / 'run_time' / 'train'
    os.makedirs(train_run_time_dir, exist_ok = True)
    
    test_run_time_dir = args.exp_dir / 'run_time' / 'test'
    os.makedirs(test_run_time_dir, exist_ok = True)

    # transform dataframe to numpy array
    n_train = X_train.shape[0]
    X = pd.concat([X_train, X_test], axis = 0)
    
    textual_columns = get_text_columns(args.dataset)

    X_np = df_to_numpy(X, args.dataset, method = args.cat_encoding, 
                       normalize_numbers=args.normalize, textual_encoding = args.text_encoding,
                       textual_columns = textual_columns)
    X_train = X_np[:n_train] 
    X_test = X_np[n_train:] 
    
    # ordinal encoding for slad:
    X_ord = df_to_numpy(X, args.dataset, method = 'ordinal',
                       normalize_numbers=args.normalize, textual_encoding = args.text_encoding,
                       textual_columns = textual_columns)
    X_train_ord = X_ord[:n_train] 
    X_test_ord = X_ord[n_train:] 
     
        
    for name, clf in DEEPOD_METHODS.items():
        score_path = score_dir / '{}.npy'.format(get_run_name(args, name))
        train_time_path = train_run_time_dir / '{}.txt'.format(get_run_name(args, name))
        test_time_path = test_run_time_dir / '{}.txt'.format(get_run_name(args, name))

        if name == 'icl' and args.dataset in ['mulcross', 'covertype', 'http', 'smtp' ]:
            clf = ICL(epochs=80)
        if name == 'icl' and args.dataset in [ 'covertype', 'http', 'smtp' ]:
            clf = ICL(epochs=40)

        #if not os.path.exists(score_path):
        if True:
            try:
                print("Training {} on {} (Split {})...".format(name, args.dataset, args.split_idx))
                if name == 'slad':
                    start_time = time.time()
                    clf.fit(X_train_ord, y=None)
                    end_time = time.time()
                    train_time = end_time - start_time
                    
                    start_time = time.time()
                    scores = clf.decision_function(X_test_ord)
                    end_time = time.time()
                    test_time = end_time - start_time
                elif name == "icl" or name == "dte":
                    start_time = time.time()
                    clf.fit(X_train_ord, y_train=None)
                    end_time = time.time()
                    train_time = end_time - start_time
                    
                    start_time = time.time()
                    scores = clf.decision_function(X_test_ord)
                    end_time = time.time()
                    test_time = end_time - start_time
                else:
                    start_time = time.time()
                    clf.fit(X_train, y=None)
                    end_time = time.time()
                    train_time = end_time - start_time
                    
                    start_time = time.time()
                    scores = clf.decision_function(X_test)
                    end_time = time.time()
                    test_time = end_time - start_time
                if name == 'pca' and np.isinf(scores).any():
                    print("Inf in training {}".format(name))
                    clf = PCA(n_components = 'mle') 
                    
                    start_time = time.time()
                    clf.fit(X_train, y=None)
                    end_time = time.time()
                    train_time = end_time - start_time
                    
                    start_time = time.time()
                    scores = clf.decision_function(X_test)
                    end_time = time.time()
                    test_time = end_time - start_time
                np.save(score_path, scores)
                
                with open(train_time_path, 'w') as f:
                    f.write(str(train_time))
                
                with open(test_time_path, 'w') as f:
                    f.write(str(test_time))

            except:
                print("Error in training {}".format(name))
                continue

def main():
    args = get_args()
    
    if args.split_idx is None:
        for i in range(args.n_splits):
            args.split_idx = i
            args.exp_dir = None
            benchmark(args)
    else:
        benchmark(args)

if __name__ == '__main__':
    main()