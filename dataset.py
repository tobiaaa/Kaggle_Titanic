import torch as th
import pandas as pd


class DatasetCSV(th.utils.data.Dataset):
	def __init__(self, path):
		super().__init__()
		train_csv_raw = pd.read_csv(path+'train.csv')
		self.train_table = self.replace_nan(train_csv_raw, 'Cabin')
		print(self.train_table)

	def replace_nan(self, table, column):
		table_copy = table.copy()
		table_copy[column] = table_copy.copy()[column].fillna(0)
		return table_copy

	def _encode(self, row):
		return row

	def __getitem__(self, item):
		print(item)

if __name__ == '__main__':
	dataset = DatasetCSV('/Users/tobiasraichle/Downloads/titanic/')

