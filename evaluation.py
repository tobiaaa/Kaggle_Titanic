import torch as th


class EvaluationTracker:
	def __init__(self, threshold=0.5):
		self._num_calls = 0
		self._num_true_positive = 0
		self._num_false_positive = 0
		self._num_true_negative = 0
		self._num_false_negative = 0
		self._num_true = 0
		self._num_false = 0
		self._threshold = threshold

	def get_results(self):
		return {'accuracy': self.accuracy(),
		        'precision': self.precision(),
		        'recall': self.recall(),
		        'f': self.f_score()}

	def accuracy(self):
		return self._num_true / self._num_calls

	def precision(self, eps=1e-8):
		return self._num_true_positive / (self._num_true_positive + self._num_false_positive + eps)

	def recall(self, eps=1e-8):
		return self._num_true_positive / (self._num_true_positive + self._num_false_negative + eps)

	def f_score(self, eps=1e-8):
		precision = self.precision()
		recall = self.recall()
		return 2*(precision * recall) / (precision + recall + eps)

	def __hard_decision(self, x):
		return x >= self._threshold

	def __call__(self, *, y_pred, y_true):
		y_pred_copy = y_pred.clone().detach()
		y_true_copy = y_true.clone().detach()

		y_pred_copy = self.__hard_decision(y_pred_copy)
		y_true_copy = self.__hard_decision(y_true_copy)

		comp = y_pred_copy == y_true_copy
		self._num_calls += len(comp)

		self._num_true += th.sum(comp)
		self._num_false += len(comp) - th.sum(comp)

		self._num_true_positive += th.sum(th.logical_and(comp, y_true_copy))
		self._num_true_negative += th.sum(th.logical_and(comp, th.logical_not(y_true_copy)))

		self._num_false_positive += th.sum(th.logical_and(th.logical_not(comp),
		                                                  th.logical_not(y_true_copy)))
		self._num_false_negative += th.sum(th.logical_and(th.logical_not(comp), y_true_copy))

	def __repr__(self):
		return f'Accuracy: {self.accuracy()}, F Score: {self.f_score()}'



