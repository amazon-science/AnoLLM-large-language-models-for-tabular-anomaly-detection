import os
import numpy as np
import pandas as pd
from typing import Optional
from pathlib import Path
import copy
import scipy.io
import pickle as pkl

import ucimlrepo 
import adbench
import pickle
# Preprocessing
import string
from string import ascii_uppercase

import re
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from feature_engine.encoding import RareLabelEncoder
from sklearn.feature_extraction.text import CountVectorizer 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.datasets import fetch_20newsgroups
import gensim.downloader as api

MIXED = [ 'vifd', 'fraudecom',  'fakejob',  'seismic', 'lymphography',  '20news-0', '20news-1','20news-2','20news-3','20news-4','20news-5']
ODDS = ['breastw', 'cardio', 'ecoli', 'lymphography', 'vertebral', 'wbc', 'wine', 'yeast', 'heart', 'arrhythmia', 
		'mulcross', 'annthyroid', 'covertype', 'glass', 'http', 'ionosphere', 'letter_recognition', 'mammography',  'musk', 
		'optdigits', 'pendigits', 'pima', 'satellite', 'satimage-2', 'seismic', 'shuttle', 'smtp', 'speech', 'thyroid', 'vowels']

# Map of dataset names to their corresponding dataset IDs in the UCI ML repository
DATA_MAP ={
	# ucimlrepo
	'breastw':15,
	'cardio':193,
	'ecoli': 39,
	'lymphography': 63,
	'vertebral': 212,
	'wbc':17,
	'wine': 109,
	'yeast':110,
	# fraud detection
	'vifd': None,
	'fraudecom': None,
	'fakejob': None,
	'fakenews': None,
	# without feature names 
	'heart': 96,
	'arrhythmia': None, # download from https://odds.cs.stonybrook.edu/arrhythmia-dataset/
	'mulcross': None, # download from  https://www.openml.org/search?type=data&sort=runs&id=40897&status=active
	# adbench datasets:
	'annthyroid': 2,
	'covertype':31,
	'glass': 14,
	'http': 16,
	'ionosphere': 18,
	'letter_recognition':20,
	'mammography': 23,
	'mulcross': None,
	'musk': 25,
	'optdigits':26,
	'pendigits':28,
	'pima':29,
	'satellite':30,
	'satimage-2':31,
	'seismic': None,
	'shuttle':32,
	'smtp':34,
	'speech':36,
	'thyroid':38,
	'vowels':40,
	#20news:
	'20news-0': None,
	'20news-1': None,
	'20news-2': None,
	'20news-3': None,
	'20news-4': None,
	'20news-5': None,
}

def load_dataset(dataset_name, data_dir):
	dataset_dir = Path(data_dir) / dataset_name
	os.makedirs(dataset_dir, exist_ok = True)
	pkl_file = dataset_dir / 'data.pkl'
	if os.path.exists(pkl_file):
		with open(pkl_file, 'rb') as f:
			X, y= pickle.load(f)
		return X, y
	
	if dataset_name == 'wine':
		dataset_id = DATA_MAP[dataset_name]
		df = ucimlrepo.fetch_ucirepo(id=dataset_id).data['original']
		np_data = load_adbench_data(dataset_name)
		columns = [name.replace('_', ' ') for name in df.columns[:-1] ]

		X = pd.DataFrame(data = np_data['X'], columns = columns)
		y = np_data['y']
	elif dataset_name == 'breastw':
		dataset_id = DATA_MAP[dataset_name]
		df = ucimlrepo.fetch_ucirepo(id=dataset_id).data['original']
		columns = [name.replace('_', ' ') for name in df.columns[1:-1] ]
		np_data = load_adbench_data(dataset_name)

		X = pd.DataFrame(data = np_data['X'], columns = columns)
		y = np_data['y']

	elif dataset_name == 'cardio':
		dataset_id = DATA_MAP[dataset_name]
		uci_dataset = ucimlrepo.fetch_ucirepo(id=dataset_id)
		# get columns descriptions
		var_info = uci_dataset['metadata']['additional_info']['variable_info']
		L = [ k.split(' - ') for k in var_info.split('\n') ]
		column_dict = {}
		for k, v in L:
			column_dict[k] = v.strip('\r')

		df = uci_dataset.data['original']
		df = df[df['NSP'] != 2].reset_index(drop=True)
		y = df['NSP'].map({3:1, 1:0}) # map pathologic to 1, normal to 0
		y = y.to_numpy()

		df.drop(['CLASS','NSP'], inplace = True, axis = 1)
		new_columns = [ column_dict[c] for c in df.columns]
		df.columns = new_columns
		X = df 
	elif dataset_name == 'ecoli':
		dataset_id = DATA_MAP[dataset_name]
		uci_dataset = ucimlrepo.fetch_ucirepo(id=dataset_id)
		columns = uci_dataset['variables']['description'][:8]
		X = uci_dataset.data['original'].drop(['class'], axis = 1)
		X.columns = columns
		X = X.drop(X.columns[0], axis=1)# drop id column
		y = uci_dataset.data['original']['class'].map({'omL':1,'imL':1,'imS':1, 'cp':0, 'im':0, 'pp':0, 'imU':0, 'om':0})
		y = y.to_numpy()
	elif dataset_name == 'lymphography':
		dataset_id = DATA_MAP[dataset_name]
		uci_dataset = ucimlrepo.fetch_ucirepo(id=dataset_id)
		df = uci_dataset.data['original']
		y = df['class'].map({1:1,2:0,3:0,4:1}) # 142 normal, 6 anomalies
		y = y.to_numpy()

		df.drop('class', inplace = True, axis = 1)
		df.drop('no. of nodes in', inplace = True, axis = 1)

		var_info = uci_dataset['metadata']['additional_info']['variable_info']
		df['lymphatics'] = df['lymphatics'].map({1:'normal', 2:'arched', 3:'deformed', 4:'displaced'}).astype('object')
		df['defect in node'] = df['defect in node'].map({1:'no',2:'lacunar', 3:'lac. marginal', 4:'lac. central'}).astype('object')
		df['changes in lym'] = df['changes in lym'].map({1:'bean',2:'oval', 3:'round'}).astype('object')
		df['changes in node'] = df['changes in node'].map({1:'no',2:'lacunar', 3:'lac. marginal', 4:'lac. central'}).astype('object')
		df['changes in stru'] = df['changes in stru'].map({1:'no',2:'grainy', 3:'drop-like', 4:'coarse', 5:'diluted', 6: 'reticular', 7:'stripped', 8:'faint'}).astype('object')
		df['special forms'] = df['special forms'].map({1:'no',2:'chalices', 3:'vesicles'}).astype('object')
		
		for k in ['block of affere', 'bl. of lymph. c', 'bl. of lymph. s', 'by pass', 'extravasates', 'regeneration of', 'early uptake in', 'dislocation of', 'exclusion of no']:
			df[k] = df[k].map({1:'no',2:'yes'}).astype('object')
		
		X = df
	
	elif dataset_name == 'vertebral':
		dataset_id = DATA_MAP[dataset_name]
		uci_dataset = ucimlrepo.fetch_ucirepo(id=dataset_id)
		df = uci_dataset.data['original']
		
		df_anomaly = df[df['class'] == 'Normal'] # 100 normal data is treated as abnormal
		df_normal = df[df['class'] != 'Normal'] # 210
		df_anomaly = df_anomaly.sample(n=30, random_state = 42)
		df = pd.concat([df_anomaly, df_normal], axis = 0, ignore_index=True)
	
		y = df['class'].map({'Spondylolisthesis':0, 'Normal':1, 'Hernia': 0}) # 210 normal, 30 anomalies
		y = y.to_numpy()
		df.drop('class', inplace = True, axis = 1)
		df.columns = [name.replace('_', ' ') for name in df.columns ]
		X = df
	elif dataset_name == 'covertype':
		dataset_id = DATA_MAP[dataset_name]
		uci_dataset = ucimlrepo.fetch_ucirepo(id=dataset_id)
		df = uci_dataset.data['original']
		
		for column in df.columns:
			if 'Soil' in column or 'Wilderness' in column:
				df.drop(column, axis =1 , inplace = True)
		df_normal = df[df['Cover_Type'] == 2]
		df_anomaly = df[df['Cover_Type'] == 4]
		df = pd.concat([df_anomaly, df_normal], axis = 0, ignore_index=True)
		
		y = df['Cover_Type'].map({2:0, 4:1})
		y = y.to_numpy()
		df.drop('Cover_Type', inplace = True, axis = 1)
		
		df.columns = [name.replace('_', ' ') for name in df.columns ]
		X = df
	elif dataset_name == 'heart':
		dataset_id = DATA_MAP[dataset_name]
		uci_dataset = ucimlrepo.fetch_ucirepo(id=dataset_id)
		df = uci_dataset.data['original']
		
		y = df['diagnosis'] 
		y = y.to_numpy()
		
		X = uci_dataset.data['original'].drop(['diagnosis'], axis = 1)

	elif dataset_name == 'wbc':
		dataset_id = DATA_MAP[dataset_name]
		uci_dataset = ucimlrepo.fetch_ucirepo(id=dataset_id)
		df = uci_dataset.data['original']
		# downsample anomaly to 21 samples
		df_anomaly = df[df['Diagnosis'] == 'M']
		df_normal = df[df['Diagnosis'] == 'B']
		df_anomaly = df_anomaly.sample(n=21, random_state = 42)
		df = pd.concat([df_anomaly, df_normal], axis = 0, ignore_index=True)
	
		y = df['Diagnosis'].map({'M':1, 'B':0}) # 142 normal, 6 anomalies
		y = y.to_numpy()
		df.drop('Diagnosis', inplace = True, axis = 1)
		df.drop('ID', inplace = True, axis = 1)

		X = df
	elif dataset_name == 'yeast':
		# the split is different than the one in the ADbench
		dataset_id = DATA_MAP[dataset_name]
		uci_dataset = ucimlrepo.fetch_ucirepo(id=dataset_id)
		df = uci_dataset.data['original']
		columns = [ s.rstrip('.') for s in uci_dataset['variables']['description'][1:9] ]
	
		y = df['localization_site'].map({'CYT':0, 'NUC':0, 'MIT':0,'ME3':0, 'ME2':1, 'ME1':1, 'EXC':0, 'VAC':0, 'POX':0, 'ERL':0}) 
		y = y.to_numpy()
		df.drop('localization_site', inplace = True, axis = 1)
		df.drop('Sequence_Name', inplace = True, axis = 1)
		df.columns = columns

		X = df

	elif dataset_name == 'vifd':
		# dataset can be downloaded from https://www.kaggle.com/datasets/khusheekapoor/vehicle-insurance-fraud-detection/data

		df = pd.read_csv( Path(data_dir) / 'vifd'/ 'carclaims.csv')
		y = df['FraudFound'].map({"Yes":1, "No":0})
		y = y.to_numpy()

		df.drop('FraudFound', axis = 1, inplace = True)
		def split_on_uppercase(s):
			return ''.join(' ' + i if i.isupper() else i for i in s).lower().strip()
		columns = [ split_on_uppercase(c) for c in df.columns]
   
		df.columns = columns
		X = df

	elif dataset_name == 'arrhythmia':
		data_path = Path(data_dir) / 'arrhythmia' / 'arrhythmia.mat'
		if not os.path.exists(data_path):
			print("Please download the dataset from https://odds.cs.stonybrook.edu/arrhythmia-dataset/ and put it to data/arrhythmia")
			raise ValueError('arrhythmia.mat is not found in {}'.format(data_path))
		data = scipy.io.loadmat(data_path)
		X_np, y = data['X'], data['y']
		X = convert_np_to_df(X_np)

	elif dataset_name == 'mulcross':
		data_path = Path(data_dir) / 'mulcross' / 'mulcross.arff'
		if not os.path.exists(data_path):
			print("Please download the dataset from https://www.openml.org/search?type=data&sort=runs&id=40897&status=active and put it to data/mulcross")
			raise ValueError('mulcross.arff is not found in {}'.format(data_path))	
		data, meta = scipy.io.arff.loadarff(data_path)
		X = [ [x[i] for i in range(4)] for x in data]
		X_np = np.array(X)
		y = [ x[4] for x in data]
		y = [ 0 if y == b'Normal' else 1 for y in y]
		y = np.array(y)
		X = convert_np_to_df(X_np)
	elif dataset_name == 'seismic':
		# downloaded from https://archive.ics.uci.edu/ml/machine-learning-databases/00266/seismic-bumps.arff
		data_path = Path(data_dir) / 'seismic' / 'seismic-bumps.arff'
		if not os.path.exists(data_path):
			print("Please dwnload the dataset from https://archive.ics.uci.edu/ml/machine-learning-databases/00266/seismic-bumps.arff and put it to data/seismic")
			raise ValueError('mulcross.arff is not found in {}'.format(data_path))	
		data, meta = scipy.io.arff.loadarff(data_path)
		df = pd.DataFrame(data)

		column_replacement = {
			'seismic': 'result of shift seismic hazard assessment in the mine working obtained by the seismic method',
			'seismoacoustic': 'result of shift seismic hazard assessment in the mine working obtained by the seismoacoustic method',
   			'shift': 'information about type of a shift',
			'genergy': 'seismic energy recorded within previous shift by the most active geophone (GMax) out of geophones monitoring the longwall',
			'gpuls': 'a number of pulses recorded within previous shift by GMax',
			'gdenergy': 'a deviation of energy recorded within previous shift by GMax from average energy recorded during eight previous shifts',
			'gdpuls': 'a deviation of a number of pulses recorded within previous shift by GMax from average number of pulses recorded during eight previous shifts',
			'ghazard': 'result of shift seismic hazard assessment in the mine working obtained by the seismoacoustic method based on registration coming from GMax only',
			'nbumps': 'the number of seismic bumps recorded within previous shift',
			'nbumps2': 'the number of seismic bumps (in energy range [10^2,10^3)) registered within previous shift',
			'nbumps3': 'the number of seismic bumps (in energy range [10^3,10^4)) registered within previous shift',
			'nbumps4': 'the number of seismic bumps (in energy range [10^4,10^5)) registered within previous shift',
			'nbumps5': 'the number of seismic bumps (in energy range [10^5,10^6)) registered within the last shift',
			'nbumps6': 'the number of seismic bumps (in energy range [10^6,10^7)) registered within previous shift',
			'nbumps7': 'the number of seismic bumps (in energy range [10^7,10^8)) registered within previous shift',
			'nbumps89': 'the number of seismic bumps (in energy range [10^8,10^10)) registered within previous shift',
			'energy': 'total energy of seismic bumps registered within previous shift',
			'maxenergy': 'the maximum energy of the seismic bumps registered within previous shift',
		}
		# take log on magnitude columns
		df['maxenergy'] = np.log(df['maxenergy'].replace(0, 1e-6))
		df['energy'] = np.log(df['energy'].replace(0, 1e-6))
		# Rename the columns
		df.rename(columns=column_replacement, inplace=True)

		# Replace categorical values in the columns
		df['result of shift seismic hazard assessment in the mine working obtained by the seismic method'] = df['result of shift seismic hazard assessment in the mine working obtained by the seismic method'].replace({b'a': 'lack of hazard', b'b': 'low hazard', b'c': 'high hazard', b'd': 'danger state'})
		df['result of shift seismic hazard assessment in the mine working obtained by the seismoacoustic method'] = df['result of shift seismic hazard assessment in the mine working obtained by the seismoacoustic method'].replace({b'a': 'lack of hazard', b'b': 'low hazard', b'c': 'high hazard', b'd': 'danger state'})
		df['result of shift seismic hazard assessment in the mine working obtained by the seismoacoustic method based on registration coming from GMax only'] = \
			df['result of shift seismic hazard assessment in the mine working obtained by the seismoacoustic method based on registration coming from GMax only'].replace({b'a': 'lack of hazard', b'b': 'low hazard', b'c': 'high hazard', b'd': 'danger state'})
		df['information about type of a shift'] = df['information about type of a shift'].replace({'W': 'coal-getting', 'N': 'preparation shift'})
			
		y = df['class'].map({b'0':0,b'1':1}) 
		y = y.to_numpy()

		df.drop('class', inplace = True, axis = 1)
		X = df
		
	elif dataset_name == 'fraudecom':
		# data downloaded from https://www.kaggle.com/datasets/vbinh002/fraud-ecommerce/data
		# add one index for device id that only appears once
		# preprocessing code adapted from https://www.kaggle.com/code/pa4494/catch-the-bad-guys-with-feature-engineering
		# remove device id
		import calendar

		data_path = Path(data_dir) / 'fraudecom'
		dataset = pd.read_csv(data_path / "Fraud_Data.csv")              # Users information
		IP_table = pd.read_csv(data_path / "IpAddress_to_Country.csv")   # Country from IP in 

		IP_table.upper_bound_ip_address.astype("float")
		IP_table.lower_bound_ip_address.astype("float")
		dataset.ip_address.astype("float")

		# function that takes an IP address as argument and returns country associated based on IP_table

		def IP_to_country(ip) :
			try :
				return IP_table.country[(IP_table.lower_bound_ip_address < ip)                            
										& 
										(IP_table.upper_bound_ip_address > ip)].iloc[0]
			except IndexError :
				return "Unknown"     
			
		# To affect a country to each IP :
		dataset["IP_country"] = dataset.ip_address.apply(IP_to_country)
		# We convert signup_time and purchase_time en datetime
		#dataset = pd.read_csv(data_path / "Fraud_data_with_country.csv")
		dataset.signup_time = pd.to_datetime(dataset.signup_time, format = '%Y-%m-%d %H:%M:%S')
		dataset.purchase_time = pd.to_datetime(dataset.purchase_time, format = '%Y-%m-%d %H:%M:%S')

		# --- 2 ---
		# Column month
		dataset["month_purchase"] = dataset.purchase_time.apply(lambda x: calendar.month_name[x.month])

		# --- 3 ---
		# Column week
		dataset["weekday_purchase"] = dataset.purchase_time.apply(lambda x: calendar.day_name[x.weekday()])
		# --- 4 ---
		# map the device id that appears only once to 0
		device_duplicates = pd.DataFrame(dataset.groupby(by = "device_id").device_id.count())  # at this moment, index column name and first column name both are equal to "device_id"
		device_duplicates.rename(columns={"device_id": "freq_device"}, inplace=True)           # hence we need to replace the "device_id" column name
		device_duplicates.reset_index(level=0, inplace= True)                                  # and then we turn device_id from index to column

		dataset = dataset.merge(device_duplicates, on= "device_id")
		indices = dataset[dataset.freq_device == 1].index
		dataset.loc[indices, "device_id"]= "0"

		le = LabelEncoder()
		dataset['device_id'] = le.fit_transform(dataset['device_id']).astype('object')
		for column in ['user_id', 'signup_time', 'purchase_time', 'ip_address', 'freq_device']:
			dataset.drop(column, axis=1, inplace = True)

		dataset.columns = [name.replace('_', ' ') for name in dataset.columns ]
		y = dataset['class'].to_numpy()
		X = dataset.drop("class", axis = 1)
		X = dataset.drop("device id", axis = 1)

	elif dataset_name == 'fakejob':
		# data download link: https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction?select=fake_job_postings.csv
		df = pd.read_csv( Path(data_dir) / 'fakejob'/ 'fake_job_postings.csv')

		# deal with Nan values
		df['location'].fillna('Unknown', inplace=True)
		df['department'].fillna('Unknown', inplace=True)
		df['salary_range'].fillna('Not Specified', inplace=True)
		df['employment_type'].fillna('Not Specified', inplace=True)
		df['required_experience'].fillna('Not Specified', inplace=True)
		df['required_education'].fillna('Not Specified', inplace=True)
		df['industry'].fillna('Not Specified', inplace=True)
		df['function'].fillna('Not Specified', inplace=True)
		df.drop('job_id', inplace=True, axis=1)

		text_columns = ['title', 'company_profile', 'description', 'requirements', 'benefits']
		df[text_columns] = df[text_columns].fillna('NaN')
		
		y = df['fraudulent'].to_numpy()
		X = df.drop('fraudulent', axis=1)
		X.columns = [name.replace('_', ' ') for name in X.columns ]
	
	
	elif dataset_name.startswith('20news-'):
		def data_generator(subsample=None, target_label=None):
			dataset = fetch_20newsgroups(subset='train')
			groups = [['comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware', 'comp.windows.x'],
				['rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey'],
				['sci.crypt', 'sci.electronics', 'sci.med', 'sci.space'],
				['misc.forsale'],
				['talk.politics.misc', 'talk.politics.guns', 'talk.politics.mideast'],
				['talk.religion.misc', 'alt.atheism', 'soc.religion.christian']]

			def flatten(l):
				return [item for sublist in l for item in sublist]
			label_list = dataset['target_names']
			label = []
			for _ in dataset['target']:
				_ = label_list[_]
				if _ not in flatten(groups):
					raise NotImplementedError
				
				for i, g in enumerate(groups):
					if _ in g:
						label.append(i)
						break
			label = np.array(label)
			print("Number of labels", len(label))
			idx_n = np.where(label==target_label)[0]
			idx_a = np.where(label!=target_label)[0]
			label[idx_n] = 0
			label[idx_a] = 1
			# subsample
			if int(subsample * 0.95) > sum(label == 0):
				pts_n = sum(label == 0)
				pts_a = int(0.05 * pts_n / 0.95)
			else:
				pts_n = int(subsample * 0.95)
				pts_a = int(subsample * 0.05)

			idx_n = np.random.choice(idx_n, pts_n, replace=False)
			idx_a = np.random.choice(idx_a, pts_a, replace=False)
			idx = np.append(idx_n, idx_a)
			np.random.shuffle(idx)

			text = [dataset['data'][i] for i in idx]
			label = label[idx]
			del dataset
	
			text = [_.strip().replace('<br />', '') for _ in text]

			print(f'number of normal samples: {sum(label==0)}, number of anomalies: {sum(label==1)}')

			return text, label
		target_label = int(dataset_name.split('-')[1])
		text, label = data_generator(subsample=10000, target_label=target_label)
		y = label
		X = pd.DataFrame(data = text, columns = ['text'])
		
	elif dataset_name in DATA_MAP.keys():
		# datasets from ADBench
		dataset_root = Path(adbench.__file__).parent.absolute() / "datasets/Classical"
		n = DATA_MAP[dataset_name]
		for npz_file in os.listdir(dataset_root):
			if npz_file.startswith(str(n) + '_'):
				print(dataset_name, npz_file)
				data = np.load(dataset_root / npz_file, allow_pickle=True)
				break
		else: 
			ValueError('{} is not found.'.format(dataset_name))
		X_np, y = data['X'], data['y']
		X = convert_np_to_df(X_np)
	else:
		raise ValueError('Invalid dataset name {}'.format(dataset_name))
			
	assert len(X) == len(y)

	with open(pkl_file, 'wb') as f:
		pickle.dump((X,y), f)
	
	return X, y

def load_adbench_data(dataset):
	dataset_root = Path(adbench.__file__).parent.absolute() / "datasets/Classical"
	if not os.path.exists(dataset_root):
		from adbench.myutils import Utils
		Utils().download_datasets(repo='jihulab')
	
	if dataset == 'cardio':
		return np.load(dataset_root / '6_cardio.npz', allow_pickle=True)

	for npz_file in os.listdir(dataset_root):
		if dataset in npz_file.lower():
			return np.load(dataset_root / npz_file, allow_pickle=True)
	else: 
		ValueError('{} is not found.'.format(dataset))

def split_data(
		X: pd.DataFrame, 
		dataset_name: str, 
		n_splits: int, 
		data_dir: str, 
		train_ratio: Optional[float] = 0.5,
		y: Optional[np.ndarray] = None, # should be provided in semi-supervised settinig
		seed: Optional[int] = 42, 
		setting: Optional[str] = 'semi_supervised'
	) -> tuple: # list of train indices and test indices
	np.random.seed(seed)
	#save path
	split_dir = Path(data_dir) / dataset_name / setting / 'split{}'.format(n_splits) 
	os.makedirs(split_dir, exist_ok = True)
	
	train_indices, test_indices = [], []
	for i in range(n_splits):
		pkl_file = split_dir / 'index{}.pkl'.format(i)
		if os.path.exists(pkl_file):
			with open(pkl_file, 'rb') as f:
				train_index, test_index = pickle.load(f)
		else:
			if setting == 'unsupervised':
				normal_data_indices = np.where(y==0)[0]
				anormal_data_indices = np.where(y==1)[0]
				normal_index = np.random.permutation(normal_data_indices)
				anormal_index = np.random.permutation(anormal_data_indices)
				
				train_index = np.concatenate([normal_index[:int(train_ratio * len(normal_index))], anormal_index[:int(train_ratio * len(anormal_index))]])
				test_index = np.concatenate([normal_index[int(train_ratio * len(normal_index)):], anormal_index[int(train_ratio * len(anormal_index)):]])
			elif setting == 'semi_supervised':
				normal_data_indices = np.where(y==0)[0]
				anormal_data_indices = np.where(y==1)[0]
				data_length = len(normal_data_indices)
				index = np.random.permutation(normal_data_indices)
				
				train_index = index[:int(train_ratio * data_length)] 
				test_index = index[int(train_ratio * data_length):]
				test_index = np.concatenate([test_index, anormal_data_indices])
			else:
				raise ValueError('Invalid setting. Choose either unsupervised or semi_supervised')
			train_index = np.random.permutation(train_index)
			test_index = np.random.permutation(test_index)
			with open(pkl_file, 'wb') as f:
				pickle.dump((train_index, test_index), f)
		train_indices.append(train_index)
		test_indices.append(test_index)
	return train_indices, test_indices 

def convert_np_to_df(X_np):
	n_train, n_cols = X_np.shape
	# Add missing column names
	L = list(string.ascii_uppercase) + [letter1+letter2 for letter1 in string.ascii_uppercase for letter2 in string.ascii_uppercase]
	columns = [ L[i] for i in range(n_cols) ]
	df = pd.DataFrame(data = X_np, columns = columns)
	return df

def load_data(args):
	dataset_dir = Path(args.data_dir) / args.dataset
	X, y = load_dataset(args.dataset, args.data_dir)
	
	if 'binning' in args and args.binning != 'none':
		X = normalize(X, args.binning, args.n_buckets)
	if 'remove_feature_name' in args and args.remove_feature_name:
		print("Removing column names and category names.")
		L = list(ascii_uppercase) + [letter1+letter2 for letter1 in ascii_uppercase for letter2 in ascii_uppercase]
		X.columns = [ L[i] for i in range(len(X.columns))]
		
		categorical_data = X.select_dtypes(include = ['object'])
		categorical_columns = categorical_data.columns.tolist()
		le = LabelEncoder()
		for i in categorical_data.columns:
			categorical_data[i] = le.fit_transform(categorical_data[i])
		
		X_prime = X.drop(categorical_columns, axis = 1)
		X = pd.concat([X_prime, categorical_data], axis = 1)

	if 'train_ratio' not in args:
		args.train_ratio = 0.5
	if 'seed' not in args:
		args.seed = 42
	train_indices, test_indices = split_data(X, args.dataset, args.n_splits, args.data_dir, 
												args.train_ratio, y = y, seed = args.seed, setting = args.setting )
	train_index, test_index = train_indices[args.split_idx], test_indices[args.split_idx]
	X_train, X_test = X.loc[train_index], X.loc[test_index]
	y_train, y_test = y[train_index], y[test_index]
	
	return X_train, X_test, y_train, y_test

def normalize(X, method, n_buckets):
	# method: ['quantile', 'equal_width', 'language', 'none', 'standard'] 
	# n_buckets: 0-100
	X = copy.deepcopy(X)
	def ordinal(n):
		if np.isnan(n):
			return 'NaN'
		n = int(n)
		if 10 <= n % 100 <= 20:
			suffix = 'th'
		else:
			suffix = {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')
		return 'the ' + str(n) + suffix + ' percentile'
	
	word_list = ['Minimal', 'Slight', 'Moderate', 'Noticeable', 'Considerable', 'Significant', 'Substantial', 'Major', 'Extensive', 'Maximum']
	def get_word(n):
		n = int(n)
		if n == 10:
			return word_list[-1]
		return word_list[n]
	
	if method == 'quantile':
		for column in X.columns:
			if X[column].dtype in ['float64', 'int64', 'uint8', 'int16'] and  X[column].nunique() > 1:
				ranks = X[column].rank(method='min')
				X[column] = ranks / len(X[column]) * 100
				X[column] = X[column].apply(ordinal)
					
	elif method == 'equal_width':
		for column in X.columns:
			if X[column].dtype in ['float64', 'int64', 'uint8', 'int16']:
				if X[column].nunique() > 1:
					X[column] = X[column].astype('float64')
					X[column] = (X[column] - X[column].min()) / (X[column].max() - X[column].min()) * n_buckets 
				
				if 10 % n_buckets == 0:
					X[column] = X[column].round(0) / 10
					X[column] = X[column].round(1) 
				else: 
					X[column] = X[column].round(0) / 100
					X[column] = X[column].round(2)
	elif method == 'standard':
		for column in X.columns:
			if X[column].dtype in ['float64', 'int64', 'uint8', 'int16']:
				scaler = StandardScaler()
				scaler.fit(X[column].values.reshape(-1,1))
				X[column] = scaler.transform(X[column].values.reshape(-1,1))
				X[column] = X[column].round(1) 

	elif method == 'language':
		for column in X.columns:
			if X[column].dtype in ['float64', 'int64', 'uint8', 'int16'] and X[column].nunique() > 1:
				X[column] = X[column].astype('float64')
				X[column] = (X[column] - X[column].min()) / (X[column].max() - X[column].min()) * 10
				X[column] = X[column].apply(get_word)
	else:
		raise ValueError('Invalid method. Choose either percentile, language or decimal')
	return X

def get_text_columns(dataset_name):
	text_columns = []
	if dataset_name == 'fakejob':
		text_columns = ['title', 'company profile', 'description', 'requirements', 'benefits']
	elif 'fakenews' == dataset_name:
		text_columns = ['title', 'text']
	elif '20news' in dataset_name:
		text_columns = ['text']
	return text_columns

def get_max_length_dict(dataset_name):
	max_length_dict = {}
	if dataset_name == 'fakejob':
		max_length_dict['title'] = 20
		text_columns = ['company profile', 'description', 'requirements', 'benefits']
		for col in text_columns:	
			max_length_dict[col] = 700
	elif 'fakenews' == dataset_name:
		max_length_dict['title'] = 30
		max_length_dict['text'] = 500
	elif '20news' in dataset_name:
		max_length_dict['text'] = 1000
	return max_length_dict

def df_to_numpy(
		X: pd.DataFrame, 
		dataset_name: Optional[str] = None, 
		method: Optional[str] = 'ordinal',
		normalize_numbers: Optional[bool] = False,
		verbose: Optional[bool] = False,
		textual_encoding: Optional[str] = 'word2vec', # bag_of_words, tfidf, word2vec, or none
		textual_columns: Optional[list] = None
	) -> np.ndarray: 
	if dataset_name == 'ecoli':
		X_np = X.drop(X.columns[0], axis=1).to_numpy()
		return X_np	
	
	numeric_data = X.select_dtypes(include = ['float64', 'int64', 'uint8', 'int16', 'float32'])
	numeric_columns = numeric_data.columns.tolist()
	categorical_data = X.select_dtypes(include = ['object', 'category'])
	categorical_columns = categorical_data.columns.tolist()

	if verbose:
		print("Number of categorical data", len(categorical_columns))
		print("Categorical columns:", categorical_columns)

	# fill na
	if len(numeric_columns) > 0:
		for numeric_col in numeric_columns:
			X[numeric_col] = X[numeric_col].fillna(X[numeric_col].mean())

		if normalize_numbers:
			# normalize it to have zero mean and unit variance
			scaler = StandardScaler()	
			X[numeric_columns] = scaler.fit_transform(X[numeric_columns])
	
	# Handle textual data	
	if textual_encoding == 'none' and len(textual_columns) > 0:
		for col in textual_columns:
			categorical_columns.remove(col)
		X = X.drop(columns = textual_columns)
		textual_columns = []
	
	if len(textual_columns) > 0:
		if textual_encoding == 'word2vec':
			model = api.load('word2vec-google-news-300')
			tmp = X[textual_columns].agg(' '.join, axis=1)
			X_vecs = []
			for i in range(len(X)):
				words = []
				for word in tmp[i].split():
					if word in model.key_to_index:
						words.append(word)
				# Compute the average word embedding
				if words:  # Ensure there are valid words left
					word_vectors = [model[word] for word in words]
					X_vec = np.mean(word_vectors, axis=0)
				else:
					X_vec = np.zeros(model.vector_size)  # Handle the case where no words are in the vocabulary
				X_vecs.append(X_vec)
			X_vecs = np.array(X_vecs)	
			for col in textual_columns:
				categorical_columns.remove(col)

		elif textual_encoding == 'bag_of_words':
			corpus = []
			for col in textual_columns:
				for i in range(len(X)):
					corpus.append(X[col][i])
			vectorization = CountVectorizer(max_features = 300)
			vectorization.fit(corpus)
			tmp = X[textual_columns].agg(' '.join, axis=1)
			X_vecs = vectorization.transform(tmp).todense()

			for col in textual_columns:
				categorical_columns.remove(col)
		
		elif textual_encoding == 'tfidf':
			corpus = []
			for col in textual_columns:
				for i in range(len(X)):
					corpus.append(X[col][i])
			vectorization = TfidfVectorizer(max_features = 300)
			vectorization.fit(corpus)
			tmp = X[textual_columns].agg(' '.join, axis=1)
			X_vecs = vectorization.transform(tmp).todense()

			for col in textual_columns:
				categorical_columns.remove(col)

		else:
			raise ValueError('Invalid textual encoding. Choose either bag_of_words, tf-idf or word2vec')
		X = X.drop(columns = textual_columns)
		X = pd.concat([X, pd.DataFrame(X_vecs)], axis = 1)

	
	if len(categorical_columns) > 0:
		# categorical features:
		# group categories with low frequency into a single category
		encoder = RareLabelEncoder(
			tol=0.01,  # Minimum frequency to be considered as a separate class
			max_n_categories=None,  # Maximum number of categories to keep
			replace_with='Rare',  # Value to replace rare categories with
			variables=categorical_columns , # Columns to encode
			missing_values='ignore',
		)
		X = encoder.fit_transform(X)
		
		# Remove columns that contain identical values 
		X = X.loc[:, (X != X.iloc[0]).any()]
		
		# remove categories that have only one value
		for column in categorical_columns:
			if X[column].nunique() == 1:
				X.drop(column, inplace = True, axis = 1)
		
		if method == 'ordinal':
			le = LabelEncoder()
			for i in categorical_data.columns:
				categorical_data[i] = le.fit_transform(categorical_data[i])
		elif method == 'one_hot':
			enc = OneHotEncoder(handle_unknown='ignore', sparse_output=False, drop='first')
			one_hot_encoded = enc.fit_transform(X[categorical_columns])
			categorical_data = pd.DataFrame(one_hot_encoded, columns=enc.get_feature_names_out(categorical_columns))
		else:
			raise ValueError('Invalid method. Choose either ordinal or one_hot')
		X_prime = X.drop(categorical_columns, axis = 1)
		X = pd.concat([X_prime, categorical_data], axis = 1)
	# remove columns that contain identical values	
	print(X.shape)
	X = X.loc[:, (X != X.iloc[0]).any()]
	X_np = X.to_numpy()
	return X_np

def print_dataset_information(dataset, data_dir):
	print("-"*100) 
	print("Dataset: {}".format(dataset))
	X, y = load_dataset(dataset, data_dir)
	#print(X['company profile'][:3]) 
	print(X.columns)
	train_indices, test_indices = split_data(X, dataset, 5, data_dir, 
												0.5, y = y, seed = 42, setting = 'semi_supervised' )
	print("Dtypes of columns:", X.dtypes)	
	X_np = df_to_numpy(X, dataset_name = dataset, method = 'one_hot', verbose = True, textual_encoding='word2vec')
	print("Number of training samples:", len(train_indices[0]))
	print("Number of testing samples:", len(test_indices[0]))
	print("Number of anomalies: {:f} ({:.2f}%)".format(np.sum(y), np.sum(y)/len(y) * 100))
	print("Number of features:", len(X.columns)) 
	print("Number of feature dimensions", X_np.shape[1])

def filter_anomalies(X_test, y_test):
	X_test = X_test[y_test == 0]
	y_test = y_test[y_test == 0]
	return X_test, y_test	 


if __name__ == '__main__':
	#print_dataset_information('fakejob', 'data')
	print_dataset_information('20news-6', 'data')
	exit()