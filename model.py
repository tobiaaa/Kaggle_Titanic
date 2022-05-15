import torch as th


class Model(th.nn.Module):
	def __init__(self, c_in):
		super().__init__()
		self.dropout_1 = th.nn.Dropout(0.4)
		self.dense_1 = th.nn.Linear(c_in, 128)
		self.activation_1 = th.nn.ReLU()
		self.dense_2 = th.nn.Linear(128, 256)
		self.activation_2 = th.nn.GLU()
		self.dense_3 = th.nn.Linear(128, 64)
		self.activation_3 = th.nn.ReLU()
		self.dense_4 = th.nn.Linear(64, 16)
		self.activation_4 = th.nn.ReLU()
		self.dense_5 = th.nn.Linear(16, 1)
		self.activation_5 = th.nn.Sigmoid()

	def forward(self, x):
		x = self.dropout_1(x)
		x = self.dense_1(x)
		x = self.activation_1(x)
		x = self.dense_2(x)
		x = self.activation_2(x)
		x = self.dense_3(x)
		x = self.activation_3(x)
		x = self.dense_4(x)
		x = self.activation_4(x)
		x = self.dense_5(x)
		x = self.activation_5(x)
		return x
