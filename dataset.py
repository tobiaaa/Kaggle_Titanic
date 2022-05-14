import torch as th
import torch.nn.functional as F
import pandas as pd
import numpy as np
import json


class DatasetCSV(th.utils.data.Dataset):
	def __init__(self, path, json_path='data_info.json', test_set=False):
		super().__init__()
		self._json_path = json_path
		self._test_set = test_set
		train_csv_raw = pd.read_csv(path + 'train.csv')
		self.train_table = self.replace_nan(train_csv_raw, 'Cabin')
		self.train_table = self.replace_nan(self.train_table, 'Age')
		self.train_table = self.replace_nan(self.train_table, 'Embarked', 'N')

		self.num_rows = len(self.train_table.index)

		self.make_json()

		self.encoding_dict = self.read_json()

	def make_json(self):
		self.make_encode_class()
		self.make_encode_sex()
		self.make_encode_age()
		self.make_encode_family_sib_sp()
		self.make_encode_family_par_ch()
		self.make_encode_fare()
		self.make_encode_embark()

	def replace_nan(self, table, column, value=0):
		table_copy = table.copy()
		table_copy[column] = table_copy.copy()[column].fillna(value)
		return table_copy

	def _encode(self, row):
		# print(row)
		class_enc = self.encode_class(row['Pclass'])
		sex_enc = self.encode_sex(row['Sex'])
		age_enc = self.encode_age(row['Age'])
		sib_enc = self.encode_family_sib_sp(row['SibSp'])
		par_enc = self.encode_family_par_ch(row['Parch'])
		fare_enc = self.encode_fare(row['Fare'])
		emb_enc = self.encode_embark(row['Embarked'])

		enc_vec = th.concat((class_enc,
		                    sex_enc,
		                    age_enc,
		                    sib_enc,
		                    par_enc,
		                    fare_enc,
		                    emb_enc))

		return enc_vec

	def make_encode_class(self):
		# One-Hot Encoding
		unique_values = self.train_table['Pclass'].unique()
		save_dict = {'num_groups_class': len(unique_values)}
		self.write_json(save_dict)

	def encode_class(self, value):
		num_groups = self.encoding_dict['num_groups_class']
		encoded = F.one_hot(th.tensor(value - 1), num_groups)
		return encoded

	def make_encode_sex(self):
		# One-Hot Encoding
		unique_values = self.train_table['Sex'].unique()
		save_dict = {'groups_sex': {key: i for i, key in enumerate(unique_values)},
		             'num_groups_sex': len(unique_values)}
		self.write_json(save_dict)

	def encode_sex(self, value):
		value_enc = self.encoding_dict['groups_sex'][value]
		num_groups = self.encoding_dict['num_groups_sex']
		value_enc = F.one_hot(th.tensor(value_enc), num_groups)
		return value_enc

	def make_encode_age(self):
		# Rounding, Grouped One-Hot
		unique_values = self.train_table['Age'].unique()
		num_groups = 10
		quantiles = [1 / (num_groups - 1) * i for i in range(1, num_groups - 1)]
		quantile_values = np.quantile(self.train_table['Age'][self.train_table['Age'] != 0.0], quantiles)
		quantile_values = [0.0, *quantile_values]

		save_dict = {
			'num_groups_age': num_groups,
			'groups_age': quantile_values
		}
		self.write_json(save_dict)

	def encode_age(self, value):
		bounds = self.encoding_dict['groups_age']
		num_groups = self.encoding_dict['num_groups_age']
		group = th.bucketize(th.tensor(value), th.tensor(bounds), right=False)
		encoded = F.one_hot(group, num_groups)
		return encoded

	def make_encode_family_sib_sp(self):
		# Z-Norm
		unique_values = self.train_table['SibSp'].unique()
		mean = np.mean(self.train_table['SibSp'])
		std = np.std(self.train_table['SibSp'])
		save_dict = {
			'sib_mean': mean,
			'sib_std': std
		}
		self.write_json(save_dict)

	def encode_family_sib_sp(self, value):
		mean = self.encoding_dict['sib_mean']
		std = self.encoding_dict['sib_std']

		encoded = th.unsqueeze(th.tensor((value - mean) / std), 0)
		return encoded

	def make_encode_family_par_ch(self):
		# Z-Norm
		unique_values = self.train_table['Parch'].unique()
		mean = np.mean(self.train_table['Parch'])
		std = np.std(self.train_table['Parch'])
		save_dict = {
			'par_mean': mean,
			'par_std': std
		}
		self.write_json(save_dict)

	def encode_family_par_ch(self, value):
		mean = self.encoding_dict['par_mean']
		std = self.encoding_dict['par_std']

		encoded = th.unsqueeze(th.tensor((value - mean) / std), 0)
		return encoded

	def make_encode_fare(self):
		# Quantile Group + One-Hot
		unique_values = self.train_table['Fare'].unique()
		num_groups = 10
		quantiles = [1 / (num_groups - 1) * i for i in range(1, num_groups - 1)]
		quantile_values = np.quantile(self.train_table['Fare'], quantiles).tolist()
		save_dict = {'num_groups_fare': num_groups,
		             'groups_fare': quantile_values}
		self.write_json(save_dict)

	def encode_fare(self, value):
		bounds = self.encoding_dict['groups_fare']
		num_groups = self.encoding_dict['num_groups_fare']
		group = th.bucketize(th.tensor(value), th.tensor(bounds), right=False)
		encoded = F.one_hot(group, num_groups)
		return encoded

	def make_encode_embark(self):
		# One-Hot Encoding
		unique_values = self.train_table['Embarked'].unique()
		save_dict = {'num_groups_embark': len(unique_values),
		             'groups_embark': {key: i for i, key in enumerate(unique_values)}}
		self.write_json(save_dict)

	def encode_embark(self, value):
		value_enc = self.encoding_dict['groups_embark'][value]
		num_groups = self.encoding_dict['num_groups_embark']
		value_enc = F.one_hot(th.tensor(value_enc), num_groups)
		print(value_enc)
		return value_enc

	def __getitem__(self, item):
		row = self.train_table.iloc[item]
		encoded_row = self._encode(row)

		if self._test_set:
			return encoded_row

		target = th.unsqueeze(th.tensor(self.train_table['Survived'][item]), 0)

		return encoded_row, target

	def __len__(self):
		return self.num_rows

	def read_json(self):
		with open(self._json_path, 'r') as f:
			return json.load(f)

	def write_json(self, new_dict):
		old_dict = self.read_json()
		old_dict.update(new_dict)
		with open(self._json_path, 'w') as f:
			json.dump(old_dict, f, indent=4, sort_keys=True)


if __name__ == '__main__':
	dataset = DatasetCSV('/Users/tobiasraichle/Downloads/titanic/')
	print(len(dataset[0][0]))
