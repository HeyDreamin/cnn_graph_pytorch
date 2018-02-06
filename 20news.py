
# coding: utf-8

# In[1]:


#get_ipython().run_line_magic('load_ext', 'autoreload')
#get_ipython().run_line_magic('autoreload', '2')
#get_ipython().run_line_magic('env', 'CUDA_DEVICE_ORDER=PCI_BUS_ID')
#get_ipython().run_line_magic('env', 'CUDA_VISIBLE_DEVICES=2')

import sys, os
sys.path.insert(0, '..')
from lib import graph, coarsening, utils

import torch
import numpy as np
import scipy.sparse
import matplotlib.pyplot as plt
import time

# print(torch.cuda.device_count())
# torch.cuda.set_device(2)

#get_ipython().run_line_magic('matplotlib', 'inline')


# # Arguments and Graph

# In[20]:


#get_ipython().run_line_magic('reload_ext', 'autoreload')

import argparse
#get_ipython().run_line_magic('xmode', '')
sys.argv=['']
parser = argparse.ArgumentParser(description='Implementation of Graph CNN for 20news pytorch version')
# Graphs.
parser.add_argument('--number_edges', type=int, default=16,
                    help='Graph: minimum number of edges per vertex. default=8')
# TODO: change cgcnn for combinatorial Laplacians.
parser.add_argument('--metric', type=str, default='cosine', 
                    help='Graph: similarity measure (between features). default=cosine')
parser.add_argument('--normalized_laplacian', type=bool, default=True, 
                    help='Graph Laplacian: normalized. default=True')
parser.add_argument('--coarsening_levels', type=int, default=0, 
                    help='Number of coarsened graphs. default=4')
# Directories.
parser.add_argument('--dir_data', type=str, default=os.path.join('..', 'data', '20news'), 
                    help='Directory to store data. default=os.path.join(\'..\', \'data\', \'mnist\')')

parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='No. of batches to wait before logging training status')

parser.add_argument('--cuda', type=bool, default=True,
                   help='Run on GPU or not. default=True')

args = parser.parse_args()


# # Data

# In[3]:


from torchvision import datasets, transforms
from torch.autograd import Variable

# Fetch dataset. Scikit-learn already performs some cleaning.
remove = ('headers','footers','quotes')  # (), ('headers') or ('headers','footers','quotes')
train = utils.Text20News(data_home=args.dir_data, subset='train', remove=remove)

# Pre-processing: transform everything to a-z and whitespace.
print(train.show_document(1)[:400])
train.clean_text(num='substitute')

# Analyzing / tokenizing: transform documents to bags-of-words.
#stop_words = set(sklearn.feature_extraction.text.ENGLISH_STOP_WORDS)
# Or stop words from NLTK.
# Add e.g. don, ve.
train.vectorize(stop_words='english')
print(train.show_document(1)[:400])



# In[4]:


train.data_info(True)
wc = train.remove_short_documents(nwords=20, vocab='full')
train.data_info()
print('shortest: {}, longest: {} words'.format(wc.min(), wc.max()))
#plt.figure(figsize=(17,5))
#plt.semilogy(wc, '.');

# Remove encoded images.
def remove_encoded_images(dataset, freq=1e3):
    widx = train.vocab.index('ax')
    wc = train.data[:,widx].toarray().squeeze()
    idx = np.argwhere(wc < freq).squeeze()
    dataset.keep_documents(idx)
    return wc
wc = remove_encoded_images(train)
train.data_info()
#plt.figure(figsize=(17,5))
#plt.semilogy(wc, '.');


# In[5]:


# Word embedding
if True:
    train.embed()
else:
    train.embed(os.path.join('..', 'data', 'word2vec', 'GoogleNews-vectors-negative300.bin'))
train.data_info()
# Further feature selection. (TODO)


# In[6]:

# Feature selection.
# Other options include: mutual information or document count.
freq = train.keep_top_words(1000, 20)
train.data_info()
train.show_document(1)
#plt.figure(figsize=(17,5))
#plt.semilogy(freq);

# Remove documents whose signal would be the zero vector.
wc = train.remove_short_documents(nwords=5, vocab='selected')
train.data_info(True)


# In[7]:


train.normalize(norm='l1')
train.show_document(1);


# In[8]:


# Test dataset.
test = utils.Text20News(data_home=args.dir_data, subset='test', remove=remove)
test.clean_text(num='substitute')
test.vectorize(vocabulary=train.vocab)
test.data_info()
wc = test.remove_short_documents(nwords=5, vocab='selected')
print('shortest: {}, longest: {} words'.format(wc.min(), wc.max()))
test.data_info(True)
test.normalize(norm='l1')


# In[9]:


if True:
    train_data = train.data.astype(np.float32)
    test_data = test.data.astype(np.float32)
    train_labels = train.labels
    test_labels = test.labels
else:
    perm = np.random.RandomState(seed=42).permutation(dataset.data.shape[0])
    Ntest = 6695
    perm_test = perm[:Ntest]
    perm_train = perm[Ntest:]
    train_data = train.data[perm_train,:].astype(np.float32)
    test_data = train.data[perm_test,:].astype(np.float32)
    train_labels = train.labels[perm_train]
    test_labels = train.labels[perm_test]

if True:
    graph_data = train.embeddings.astype(np.float32)
else:
    graph_data = train.data.T.astype(np.float32).toarray()

#del train, test


# # Feature graph

# In[10]:


t_start = time.process_time()
dist, idx = graph.distance_sklearn_metrics(graph_data, k=args.number_edges, metric=args.metric)
A = graph.adjacency(dist, idx)
print("{} > {} edges".format(A.nnz//2, args.number_edges*graph_data.shape[0]//2))
A = graph.replace_random_edges(A, 0)
graphs, perm = coarsening.coarsen(A, levels=args.coarsening_levels, self_connections=False)
L = [graph.laplacian(A, normalized=True) for A in graphs]
print('Execution time: {:.2f}s'.format(time.process_time() - t_start))
#graph.plot_spectrum(L)
#del graph_data, A, dist, idx


# In[11]:


t_start = time.process_time()
train_data = scipy.sparse.csr_matrix(coarsening.perm_data(train_data.toarray(), perm))
test_data = scipy.sparse.csr_matrix(coarsening.perm_data(test_data.toarray(), perm))
print('Execution time: {:.2f}s'.format(time.process_time() - t_start))
del perm


# In[12]:


val_data = test_data
val_labels = test_labels
utils.baseline(train_data, train_labels, test_data, test_labels)


# In[14]:


from torch.utils.data import TensorDataset, DataLoader
kwargs ={'num_workers': 1, 'pin_memory': True} if args.cuda else {}

def cus_dataset(L):
    L = L.tocoo()
    indices = torch.from_numpy(np.column_stack((L.row, L.col))).long()
    L = torch.sparse.DoubleTensor(indices.t(), 
                                  torch.from_numpy(L.data).double(), 
                                  torch.Size(list(L.shape)))
    return L.to_dense()

trainset = TensorDataset(cus_dataset(train_data), torch.from_numpy(train_labels))
testset = TensorDataset(cus_dataset(test_data), torch.from_numpy(test_labels))
train_loader = DataLoader(trainset, batch_size=100, shuffle=True, **kwargs)
test_loader = DataLoader(testset, batch_size=1000, shuffle=True, **kwargs)


# # Model init

# In[41]:


from model_pytorch import CGCNN_Net

common = {}
common['dir_name']       = '20news/'
common['num_epochs']     = 80
common['batch_size']     = 100
common['decay_steps']    = len(train_labels) / common['batch_size']
common['eval_frequency'] = 5 * common['num_epochs']
common['filter']         = 'chebyshev5'
common['bias']           = 'bias1'
common['pool']           = 'maxpool'
C = max(train_labels) + 1  # number of classes

name = 'cgconv_fc_softmax'
common['dir_name'] += name
common['regularization'] = 0
common['dropout']        = 0
common['learning_rate']  = 0.1
common['decay_rate']     = 0.001
common['momentum']       = 0
common['F']              = [5]
common['K']              = [15]
common['p']              = [1]
common['M']              = [L[-1].shape[0], 100, int(C)]

model = CGCNN_Net(L, **common)
model.double().cuda()


# # Training and Test

# In[42]:


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
        data = data.unsqueeze(1)
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        writer.add_scalar('Loss', loss.data[0], n_iter)
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, min((batch_idx+10)*len(data),len(train_loader.dataset)), 
                len(train_loader.dataset),
                100. * (batch_idx+10) / loader_len, loss.data[0])) 

def test(epoch):
    model.eval()
    test_loss = 0
    correct = 0
    dataset_len = len(test_loader.dataset)
    for data, target in test_loader:
        data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        data = data.unsqueeze(1)
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
for epoch in range(1, 80 + 1):
    train(epoch)
    test(epoch)

writer.close()

