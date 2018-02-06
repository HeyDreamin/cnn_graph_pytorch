
# coding: utf-8

# In[1]:


# get_ipython().run_line_magic('load_ext', 'autoreload')
# get_ipython().run_line_magic('autoreload', '2')
# get_ipython().run_line_magic('reload_ext', 'autoreload')
# get_ipython().run_line_magic('env', 'CUDA_DEVICE_ORDER=PCI_BUS_ID')
# get_ipython().run_line_magic('env', 'CUDA_VISIBLE_DEVICES=3')

import sys, os
sys.path.insert(0, '..')
from lib import graph, coarsening, utils

import torch
import numpy as np
import time

# print(torch.cuda.device_count())
# torch.cuda.set_device(3)

# get_ipython().run_line_magic('matplotlib', 'inline')


# # Arguments and Graph

# In[2]:


# get_ipython().run_line_magic('reload_ext', 'autoreload')
import argparse
# get_ipython().run_line_magic('xmode', '')
sys.argv=['']
parser = argparse.ArgumentParser(description='Implementation of Graph CNN for MNIST pytorch version')
# Graphs.
parser.add_argument('--number_edges', type=int, default=8,
                    help='Graph: minimum number of edges per vertex. default=8')
# TODO: change cgcnn for combinatorial Laplacians.
parser.add_argument('--metric', type=str, default='euclidean', 
                    help='Graph: similarity measure (between features). default=euclidean')
parser.add_argument('--normalized_laplacian', type=bool, default=True, 
                    help='Graph Laplacian: normalized. default=True')
parser.add_argument('--coarsening_levels', type=int, default=4, 
                    help='Number of coarsened graphs. default=4')
# Directories.
parser.add_argument('--dir_data', type=str, default=os.path.join('..', 'data', 'mnist'), 
                    help='Directory to store data. default=os.path.join(\'..\', \'data\', \'mnist\')')

parser.add_argument('--log-interval', type=int, default=50, metavar='N',
                    help='No. of batches to wait before logging training status')

parser.add_argument('--cuda', type=bool, default=True,
                   help='Run on GPU or not. default=True')

args = parser.parse_args()


def grid_graph(m, corners=False):
    z = graph.grid(m)
    dist, idx = graph.distance_sklearn_metrics(z, k=args.number_edges, metric=args.metric)
    A = graph.adjacency(dist, idx)

    # Connections are only vertical or horizontal on the grid.
    # Corner vertices are connected to 2 neightbors only.
    if corners:
        import scipy.sparse
        A = A.toarray()
        A[A < A.max()/1.5] = 0
        A = scipy.sparse.csr_matrix(A)
        print('{} edges'.format(A.nnz))

    print("{} > {} edges".format(A.nnz//2, args.number_edges*m**2//2))
    return A

t_start = time.process_time()
A = grid_graph(28, corners=False)
A = graph.replace_random_edges(A, 0)
graphs, perm = coarsening.coarsen(A, levels=args.coarsening_levels, self_connections=False)
L = [graph.laplacian(A, normalized=True) for A in graphs]
print('Execution time: {:.2f}s'.format(time.process_time() - t_start))
# graph.plot_spectrum(L)
del A


# # Data

# In[3]:


# get_ipython().run_line_magic('reload_ext', 'autoreload')
from torchvision import datasets, transforms
from torch.autograd import Variable
kwargs ={'num_workers': 1, 'pin_memory': True} if args.cuda else {}

t_start = time.process_time()
print('Coarsening...')
trainset = datasets.MNIST('./data/MNIST', train=True, download=True,
                          transform=transforms.Compose([
                              transforms.ToTensor(),
                              transforms.Normalize((0.1307,), (0.3081,)),
                              lambda x: x.view(1,-1).numpy(),
                              lambda x: coarsening.perm_data(x, perm),
                              torch.from_numpy
                          ]))

testset = datasets.MNIST('./data/MNIST', train=False, 
                         transform=transforms.Compose([
                             transforms.ToTensor(),
                             transforms.Normalize((0.1307,), (0.3081,)),
                             lambda x: x.view(1,-1).numpy(),
                             lambda x: coarsening.perm_data(x, perm),
                             torch.from_numpy
                         ]))
print('Execution time: {:.2f}s'.format(time.process_time() - t_start))
#del perm

train_loader = torch.utils.data.DataLoader(
    trainset, batch_size=100, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    testset, batch_size=1000, shuffle=True, **kwargs)

print('Loader done.')


# # Model init

# In[4]:


from model_pytorch import CGCNN_Net

common = {}
common['dir_name']       = 'mnist/'
# common['num_epochs']     = 20
common['batch_size']     = 100
# common['decay_steps']    = mnist.train.num_examples / common['batch_size']
common['eval_frequency'] = 600 # 30 * common['num_epochs']
common['bias']          = 'bias1'
common['pool']           = 'maxpool'
# Common hyper-parameters for LeNet5-like networks.
common['regularization'] = 5e-4
# common['dropout']        = 0.5
common['learning_rate']  = 0.02
# 0.03 in the paper but sgconv_sgconv_fc_softmax has difficulty to converge
common['decay_rate']     = 0.001
common['momentum']       = 0.9
common['F']              = [32, 64]
common['K']              = [5, 5]
common['p']              = [4, 4]
common['M']              = [L[-1].shape[0],512, 10]

model = CGCNN_Net(L, **common)
model.double().cuda()


# # Training and Test

# In[5]:


from torch.autograd import Variable
from torch.nn import functional as F
from tensorboardX import SummaryWriter

writer = SummaryWriter()
optimizer = torch.optim.SGD(model.parameters(),    
                            lr=model.learning_rate, 
                            momentum=model.momentum,
                            weight_decay=model.decay_rate)

def train(epoch):
    model.train()
    t_process, t_wall = time.process_time(), time.time()
    loader_len = len(train_loader)
    for batch_idx, (data, target) in enumerate(train_loader):
        n_iter = (epoch * loader_len) + batch_idx
        data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)        
        loss.backward()
        optimizer.step()
        writer.add_scalar('Loss', loss.data[0], n_iter)
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, (batch_idx+50) * len(data), len(train_loader.dataset),
                100. * (batch_idx+50) / loader_len, loss.data[0])) 

def test(epoch):
    model.eval()
    test_loss = 0
    correct = 0
    dataset_len = len(test_loader.dataset)
    for data, target in test_loader:
        data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        # sum up batch loss
        test_loss += F.cross_entropy(output, target, size_average=False).data[0]
        # get the index of the max log-probability
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= dataset_len
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, dataset_len, 100. * correct / dataset_len))
    writer.add_scalar('Accuracy', 100. * correct / dataset_len, epoch)
for epoch in range(1, 20 + 1):
    train(epoch)
    test(epoch)

writer.close()

