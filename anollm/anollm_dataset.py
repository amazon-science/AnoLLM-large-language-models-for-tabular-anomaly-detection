import random
import typing as tp
import os 

from datasets import Dataset
from dataclasses import dataclass
from transformers import DataCollatorWithPadding
from torch.utils.data import DataLoader
from tqdm import tqdm
import pickle as pkl
MAX_COL_LENGTH = 128

class AnoLLMDataset(Dataset):
	"""AnoLLM Dataset

	The AnoLLM overwrites the _getitem function of the HuggingFace Dataset Class to include the permutation step.

	Attributes:
		tokenizer (AutoTokenizer): Tokenizer from HuggingFace
	"""

	def set_tokenizer(self, tokenizer):
		"""Set the Tokenizer

		Args:
			tokenizer: Tokenizer from HuggingFace
		"""
		self.tokenizer = tokenizer 
	
	def set_anomaly_label(self, labels):
		assert len(labels) == len(self._data)
		self.anomaly_labels = labels

	def set_textual_columns(self, columns: tp.List[str]):
		col_list = self.get_column_names()
		for col in columns:
			if col not in col_list:
				raise ValueError("Column {} not in the dataset.".format(col))
		self.textual_columns = columns
	
	def get_n_columns(self):
		row = self._data.fast_slice(0, 1)
		return row.num_columns

	def get_column_names(self):
		row = self._data.fast_slice(0, 1)
		return row.column_names
	
	def shuffle_column_order(self):
		# used in evalutaion. the order of the columns is shuffled and then fixed for all data
		row = self._data.fast_slice(0, 1)
		self.shuffle_idx = list(range(row.num_columns))
		random.shuffle(self.shuffle_idx)
	
	def fix_column_order(self):
		# set the column order to be default column order. Do not shuffle the columns.
		row = self._data.fast_slice(0, 1)
		self.shuffle_idx = list(range(row.num_columns))
	
	def prepare(
		self,
		is_eval: bool = True, 
		max_length_dict: tp.Optional[tp.Dict[str, int]] = {},
		data_path = None,
		):
		'''
		Preprocess the data by tokenizing each column and truncating the columns to max_length
		Inputs:
		max_length_dict specifies the maximum length of each column. If None, all columns are truncated to max length
		pad_columns specifies whether to pad the columns to the same length according to max_length of a each column
		'''
		self.is_eval = is_eval
		n_col = self.get_n_columns()
		column_names = self.get_column_names()
		self.processed_data = [] 
		self.tokenized_feature_names = []
		bos_token_id = self.tokenizer.bos_token_id
		
		for col_idx in range(n_col):
			feature_names = ' ' + column_names[col_idx] + ' '
			tokenized_feature_names = self.tokenizer(feature_names)
			tokenized_is = self.tokenizer('is ')
			if bos_token_id and tokenized_feature_names['input_ids'][0] == bos_token_id:
				tokenized_feature_names['input_ids'] = tokenized_feature_names['input_ids'][1:]
				tokenized_is['input_ids'] = tokenized_is['input_ids'][1:]

			self.tokenized_feature_names.append(tokenized_feature_names["input_ids"] + tokenized_is["input_ids"])
		
		if data_path is not None and os.path.exists(data_path):
			self.processed_data = pkl.load(open(data_path, 'rb'))
		else:
			for key in tqdm(range(len(self._data))):
				row = self._data.fast_slice(key, 1)
				tokenized_texts = []
				for col_idx in range(n_col):
					feature_values = str(row.columns[col_idx].to_pylist()[0]).strip()
					if len(feature_values) == 0:
						feature_values = "None"
					data = self.tokenizer(feature_values)
					if bos_token_id and data['input_ids'][0] == bos_token_id:
						data['input_ids'] = data['input_ids'][1:]

					tokenized_texts.append(data["input_ids"])
					if len(data["input_ids"]) == 0:
						print("Warning: tokenized text is empty.", column_names[col_idx],len( feature_values),feature_values)
				self.processed_data.append(tokenized_texts)
			
			# truncate the columns that are too long	
			for col_idx in range(n_col):
				name = column_names[col_idx]
				if name not in max_length_dict:
					max_length = MAX_COL_LENGTH
				else:
					max_length = max_length_dict[name]
				assert isinstance(max_length, int)
				
				for data_idx in range(len(self.processed_data)):
					length = len(self.processed_data[data_idx][col_idx]) + len(self.tokenized_feature_names[col_idx])
					if length >= max_length:
						self.processed_data[data_idx][col_idx] = self.processed_data[data_idx][col_idx][:max_length - len(self.tokenized_feature_names[col_idx])]
			if data_path is not None:
				pkl.dump(self.processed_data, open(data_path, 'wb'))
		print("Preprocessing done.")

	def _getitem(
		self, 
		key: tp.Union[int, slice, str], 
		decoded: bool = True, 
		**kwargs
	) -> tp.Union[tp.Dict, tp.List]:
		"""
		Get one instance of the tabular data, permuted, converted to text and tokenized.
		"""
		row = self._data.fast_slice(key, 1)
		

		# get shuffle_idx
		if "shuffle_idx" in self.__dict__: 
			shuffle_idx = self.shuffle_idx
		else:
			shuffle_idx = list(range(row.num_columns))
			random.shuffle(shuffle_idx)
		
		# get tokenized text
		comma_id =  self.tokenizer.convert_tokens_to_ids(',')
		eos_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.eos_token)
		bos_token_id = self.tokenizer.bos_token_id
		if self.is_eval:
			tokenized_text = {"input_ids": [], "attention_mask": [], "feature_value_start":[],
							"feature_value_end":[],'col_indices':shuffle_idx}
		else:
			tokenized_text = {"input_ids": [], "attention_mask": []}
		if bos_token_id:
			tokenized_text["input_ids"] = [bos_token_id]

		if hasattr(self, "processed_data"):
			start_idx = 0
			for idx, col_idx in enumerate(shuffle_idx):
				tokenized_feature_names = self.tokenized_feature_names[col_idx]
				tokenized_feature_values = self.processed_data[key][col_idx]
				tokenized_col = tokenized_feature_names + tokenized_feature_values 
				if idx == len(shuffle_idx) - 1:
					tokenized_text["input_ids"] += tokenized_col + [eos_id]
				else:
					tokenized_text["input_ids"] += tokenized_col + [comma_id]
				if self.is_eval:
					tokenized_text["feature_value_start"].append(start_idx + len(tokenized_feature_names) -1 )
					tokenized_text["feature_value_end"].append(start_idx + len(tokenized_col) )
				start_idx += len(tokenized_col) + 1
		else:
			raise ValueError("processed_data is not found. Please run prepare function first.")	
		tokenized_text["attention_mask"] += [1] * len(tokenized_text["input_ids"])
		return tokenized_text
	
	def get_item_test(self, key):
		row = self._data.fast_slice(key, 1)
		shuffle_idx = list(range(row.num_columns))
		random.shuffle(shuffle_idx)
		
		shuffled_text = ",".join(
			[
				" %s is %s "
				% (row.column_names[i], str(row.columns[i].to_pylist()[0]).strip() )
				for i in shuffle_idx
			]
		)
		tokenized_text = self.tokenizer(shuffled_text, padding=True)

		return shuffled_text, tokenized_text 
	
	def __getitems__(self, keys: tp.Union[int, slice, str, list]):
		if isinstance(keys, list):
			return [self._getitem(key) for key in keys]
		else:
			return self._getitem(keys)

	#def add_gaussian_noise(self, value):
#		return value + np.random.normal(0, 0.1)

@dataclass
class AnoLLMDataCollator(DataCollatorWithPadding):
	"""

	Overwrites the DataCollatorWithPadding to also pad the labels and not only the input_ids
	"""

	def __call__(self, features: tp.List[tp.Dict[str, tp.Any]]):
		batch = self.tokenizer.pad(
			features,
			padding=self.padding,
			max_length=self.max_length,
			pad_to_multiple_of=self.pad_to_multiple_of,
			return_tensors=self.return_tensors,
		)
		batch["labels"] = batch["input_ids"].clone()
		return batch

class AnoLLMDataLoader(DataLoader):
	'''
	Add set_epoch function so that huggingface trainer can call it 
	'''
	def set_epoch(self, epoch):
		if hasattr(self.sampler, "set_epoch"):
			self.sampler.set_epoch(epoch)
			print("Set epoch", epoch)
