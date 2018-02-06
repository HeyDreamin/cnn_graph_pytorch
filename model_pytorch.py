from lib import graph
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import scipy.sparse
import numpy as np
import math
import os, time, shutil

class CGCNN_Net(nn.Module):
	def __init__(self, L, F, K, p, M, 
				filter='chebyshev5', bias='bias1', pool='maxpool', 
				num_epochs=20, learning_rate=0.1, decay_rate=0.95, 
				decay_steps=None, momentum=0.9, regularization=0, dropout=0, 
				batch_size=100, eval_frequency=200, dir_name=''):
		super(CGCNN_Net, self).__init__()
		
		# Verify the consistency w.r.t. the number of layers.
		assert len(L) >= len(F) == len(K) == len(p)
		assert np.all(np.array(p) >= 1)
		p_log2 = np.where(np.array(p) > 1, np.log2(p), 0)
		assert np.all(np.mod(p_log2, 1) == 0)  # Powers of 2.
		assert len(L) >= 1 + np.sum(p_log2)  # Enough coarsening levels for pool sizes.
		
		# Keep the useful Laplacians only. May be zero.
		M_0 = L[0].shape[0]
		j = 0
		self.L = []
		for pp in p:
			self.L.append(L[j])
			j += int(np.log2(pp)) if pp > 1 else 0
		L = self.L

		# Store attributes and bind operations.
		self.L, self.F, self.K, self.p, self.M = L, F, K, p, M
		self.num_epochs, self.learning_rate = num_epochs, learning_rate
		self.decay_rate, self.decay_steps, self.momentum = decay_rate, decay_steps, momentum
		self.regularization, self.dropout = regularization, dropout
		self.batch_size, self.eval_frequency = batch_size, eval_frequency
		self.dir_name = dir_name
		self.filter = getattr(self, filter)
		self.bias = bias
		self.pool = pool

		# Linear layers
		self.linear_chebyshev = nn.ModuleList([nn.Linear(self.K[0], self.F[0])])
		for i in range(1, len(self.p)):
			self.linear_chebyshev.append(nn.Linear(self.F[i-1]*self.K[i], self.F[i]))
		self.fc_hidden = nn.Linear(self.M[0]*self.F[-1], self.M[1])
		self.fc_output = nn.Linear(self.M[1], self.M[2])

	def chebyshev5(self, x, L, K):
		N, M, Fin = x.size()
		N, M, Fin = int(N), int(M), int(Fin)
		# Rescale Laplacian and store as a sparse tensor. 
		# Copy to not modify the shared L.
		L = scipy.sparse.csr_matrix(L)
		L = graph.rescale_L(L, lmax=2)
		L = L.tocoo()
		indices = torch.from_numpy(np.column_stack((L.row, L.col))).long()
		L = torch.sparse.DoubleTensor(indices.t(), 
                                     torch.from_numpy(L.data).double(),
                                     torch.Size(list(L.shape)))
		L = L.to_dense()
		# Transform to Chebyshev basis
		L = Variable(L).cuda()
		x0 = x.permute(1, 2, 0).t()  # M x Fin x N
		x0 = x0.contiguous().view(M, Fin*N)  # M x Fin*N
		x = x0.unsqueeze(0)  # 1 x M x Fin*N
		def concat(x, x_):
			x_ = x_.unsqueeze(0)  # 1 x M x Fin*N
			return torch.cat((x, x_), dim=0)  # K x M x Fin*N
		if K > 1:
			x1 = torch.matmul(L, x0)
			x = concat(x, x1)
		for k in range(2, K):
			x2 = 2 * torch.matmul(L, x1) - x0  # M x Fin*N
			x = concat(x, x2)
			x0, x1 = x1, x2
		x = x.contiguous().view(K, M, Fin, N)  # K x M x Fin x N
		x = x.permute(3, 1, 2, 0)  # N x M x Fin x K
		x = x.contiguous().view(N, M, Fin*K)  # N x M x Fin*K
		# The linear mul is outside for autograd
		return x

	def bias1(self, x):
		"""One bias per filter."""
		N, M, C = x.size()
		b = torch.Tensor(1, 1, int(C))
		nn.init.constant(b, 0.1)
		return b

	def bias2(self, x):
		"""One bias per vertex per filter."""
		N, M, C = x.size()
		b = torch.Tensor(1, int(M), int(C))
		nn.init.constant(b, 0.1)
		return b

	def forward(self, x):
		# shape of x: N x C x H x W
		N, C, M = x.size()
		x = x.view(N, M, C)
		for i in range(len(self.p)):
			# x = self.filter(x, self.L[i], self.F[i], self.K[i])
			x = self.filter(x, self.L[i], self.K[i])
			# if self.bias=='bias1':
			# 	b = bias1(x)
			# elif self.bias=='bias2':
			# 	b = bias2(x)
			x = self.linear_chebyshev[i](x) # N x M x Channels
			x = F.relu(x)
			x = x.permute(0, 2, 1) # N x Chs x M
			if self.p[i] > 1:
				if self.pool=='maxpool':
					x = F.max_pool1d(x, self.p[i])
				elif self.pool=='avgpool':
					x = F.avg_pool1d(x, self.p[i])
			x = x.permute(0, 2, 1) # N x M x Chs
		
		N, M, C = x.size()
		x = x.contiguous().view(N, M*C)
		x = self.fc_hidden(x)
		# self.regularizers.append((self.fc_hidden.weight))
		x = F.relu(x)
		x = F.dropout(x, p=self.dropout, training=self.training)
		# Add a flag of training or evaluate
		x = self.fc_output(x)
		# self.regularizers.append()
		return x
