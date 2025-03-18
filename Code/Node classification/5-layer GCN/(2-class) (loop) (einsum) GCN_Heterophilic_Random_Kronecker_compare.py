# -*- coding: utf-8 -*-
"""
Created on Thu Mar 13 19:33:02 2025

Ref: https://mlabonne.github.io/blog/posts/2022_02_20_Graph_Convolution_Network.html
"""

#%%
import os
import json
import torch
import torch.nn as nn
from torch.nn import Linear
from torch.nn import Sigmoid
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from torch_geometric.utils import to_dense_adj
from torch_geometric.utils import to_networkx
from sklearn.preprocessing import OneHotEncoder
import tracemalloc
import itertools
import random
# import psutil
import time

formatter = ScalarFormatter(useMathText=True)
formatter.set_powerlimits((-1, 1))  # Always use scientific notation

#%%
random.seed(0)
random.random()

torch.manual_seed(0)
torch.rand(4)

np.random.seed(0)
np.random.rand(4)

#%% Generate the Node Labels (2 Classes)

from torch_geometric.data import Data
import random

def fix_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # For CUDA reproducibility (optional if using GPU)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Ensure deterministic behavior (optional but sometimes slows things)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Call it before anything random happens
fix_seed(42)

# Number of nodes
num_nodes = 3000 # Keep increasing by a power of 10

# Generate binary labels (0 and 1)
labels = torch.randint(0, 2, (num_nodes,))

print(torch.bincount(labels))  # Check class distribution

#%% Create a Highly Heterophilic Edge List
# Generate edges with 80% heterophilic ratio
hetero_ratio = 0.8  # 80% heterophilic edges
edges_per_node = 5  # Average number of edges per node
edge_list = []
hetero_edges_per_node = int(edges_per_node * hetero_ratio)
homo_edges_per_node = edges_per_node - hetero_edges_per_node

for node in range(num_nodes):
    node_label = labels[node].item()

    # Opposite label nodes for heterophilic edges
    opposite_label_nodes = (labels != node_label).nonzero(as_tuple=True)[0].tolist()
    if len(opposite_label_nodes) >= hetero_edges_per_node:
        hetero_neighbors = random.sample(opposite_label_nodes, hetero_edges_per_node)
    else:
        hetero_neighbors = opposite_label_nodes  # Fallback

    # Same label nodes for homophilic edges
    same_label_nodes = (labels == node_label).nonzero(as_tuple=True)[0].tolist()
    same_label_nodes.remove(node)  # Remove self-loop
    if len(same_label_nodes) >= homo_edges_per_node:
        homo_neighbors = random.sample(same_label_nodes, homo_edges_per_node)
    else:
        homo_neighbors = same_label_nodes  # Fallback

    # Add hetero edges
    for neighbor in hetero_neighbors:
        edge_list.append([node, neighbor])

    # Add homo edges
    for neighbor in homo_neighbors:
        edge_list.append([node, neighbor])

# Convert edge list to tensor
edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()

# Make edges undirected (optional)
edge_index = torch.cat([edge_index, edge_index.flip([0])], dim=1)

print(edge_index.shape)  # Should be [2, num_edges]

#%% Create Similar Features for the Same Class
# Let's create random node features with 10 dimensions
num_features = 10
# Define two random class centers in the feature space
center_0 = torch.randn(num_features)
center_1 = torch.randn(num_features)

# Standard deviation for noise around each center
std_dev = 0.1  # Smaller means tighter clusters, more similar features

# Initialize feature matrix
x = torch.zeros((num_nodes, num_features))

for i in range(num_nodes):
    if labels[i] == 0:
        x[i] = center_0 + torch.randn(num_features) * std_dev
    else:
        x[i] = center_1 + torch.randn(num_features) * std_dev
        
#%% Build the PyG Data Object
data = Data(x=x, edge_index=edge_index, y=labels)

print(data)

#%% 
from torch_geometric.utils import to_dense_adj

A = (to_dense_adj(data.edge_index)[0].numpy().astype(int) > 0)*1
print(f'A = {A.shape}')
print(A)

#%% Verify Heterophily
def compute_homophily(data):
    edge_index = data.edge_index
    labels = data.y
    same_label_edges = 0

    for i in range(edge_index.size(1)):
        src = edge_index[0, i]
        dst = edge_index[1, i]
        if labels[src] == labels[dst]:
            same_label_edges += 1

    total_edges = edge_index.size(1)
    homophily = same_label_edges / total_edges
    return homophily

homo_ratio = compute_homophily(data)
print(f'Homophily ratio: {homo_ratio:.4f}')  # Should be close to 0!



# #%%
# from torch_geometric.datasets import KarateClub

# # Import dataset from PyTorch Geometric
# dataset = KarateClub()

# # Print information
# print(dataset)
# print('------------')
# print(f'Number of graphs: {len(dataset)}')
# print(f'Number of features: {dataset.num_features}')
# print(f'Number of classes: {dataset.num_classes}')

# #%%
# # Print first element
# print(f'Graph: {dataset[0]}')

# #%%
# data = dataset[0]

# print(f'x = {data.x.shape}')
# print(data.x)

# #%%
# print(f'edge_index = {data.edge_index.shape}')
# print(data.edge_index)

# #%%
# A = to_dense_adj(data.edge_index)[0].numpy().astype(int)
# print(f'A = {A.shape}')
# print(A)

# #%%
# print(f'y = {data.y.shape}')
# print(data.y)

# #%% Change class 3 to 1 and class 2 to 0
# data.y[data.y==3] = 1
# data.y[data.y==2] = 0

# #%%
# print(f'train_mask = {data.train_mask.shape}')
# print(data.train_mask)

# #%%
# print(f'Edges are directed: {data.is_directed()}')
# print(f'Graph has isolated nodes: {data.has_isolated_nodes()}')
# print(f'Graph has loops: {data.has_self_loops()}')

#%% Visualize the Graph
G = to_networkx(data, to_undirected=True)
plt.figure(figsize=(12,12))
plt.axis('off')
nx.draw_networkx(G,
                pos=nx.spring_layout(G, seed=0),
                with_labels=True,
                node_size=800,
                node_color=data.y,
                cmap="hsv",
                vmin=-2,
                vmax=3,
                width=0.8,
                edge_color="grey",
                font_size=14
                )
plt.show()

#%%
def my_loss(output, target):
    loss_true = -sum(torch.from_numpy(target)*torch.log(output).squeeze(1))
    target_mis = 1-target # Wrong classification
    loss_mis = -sum(torch.from_numpy(target_mis)*torch.log(1-output).squeeze(1))
    loss = loss_true + loss_mis
    return loss

#%%
# enc = OneHotEncoder()
# data.y.unsqueeze(1)
# enc.fit(data.y.unsqueeze(1))
# dummy_y = enc.transform(data.y.unsqueeze(1)).toarray()

#%%
LeakyReLU_negative_slope=0.01
ELU_alpha=1.0

#%%
class GCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # self.gcn = GCNConv(dataset.num_features, 3)
        # self.out = Linear(dataset.num_classes, dataset.num_classes)
        # self.gcn = GCNConv(dataset.num_features, dataset.num_classes)
        self.num_features = 20
        self.num_hiddens_1 = 2*1
        self.num_hiddens_2 = 3*1
        self.num_hiddens_3 = 2*1
        self.num_hiddens_4 = 3*1
        self.FCL1 = nn.Linear(num_features, self.num_hiddens_1, bias=False)
        self.FCL2 = nn.Linear(self.num_hiddens_1, self.num_hiddens_2, bias=False)
        self.FCL3 = nn.Linear(self.num_hiddens_2, self.num_hiddens_3, bias=False)
        self.FCL4 = nn.Linear(self.num_hiddens_3, self.num_hiddens_4, bias=False)
        self.FCL5 = nn.Linear(self.num_hiddens_4, 1, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.leakyrelu = nn.LeakyReLU(negative_slope=LeakyReLU_negative_slope)
        self.elu = nn.ELU(alpha=ELU_alpha)
        self.swish = nn.SiLU()

    def forward(self, x, edge_index):
        h0 = x
        # 1
        temp = torch.from_numpy(A).float()@h0
        h1 = self.relu(self.FCL1(temp)) # y = xW'
        # 2
        temp2 = torch.from_numpy(A).float()@h1
        h2 = self.swish(self.FCL2(temp2)) # y = xW'
        # 3
        temp3 = torch.from_numpy(A).float()@h2
        h3 = self.elu(self.FCL3(temp3)) # y = xW'
        # 4
        temp4 = torch.from_numpy(A).float()@h3
        h4 = self.leakyrelu(self.FCL4(temp4)) # y = xW'
        # 5
        temp5 = torch.from_numpy(A).float()@h4
        h5 = self.FCL5(temp5)
        # 6
        z = self.sigmoid(h5)
        h = [h0,h1,h2,h3,h4,h5]
        return h, z
    
#%%
class GCN_Kron(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # self.gcn = GCNConv(dataset.num_features, 3)
        # self.out = Linear(dataset.num_classes, dataset.num_classes)
        # self.gcn = GCNConv(dataset.num_features, dataset.num_classes)
        self.num_features = 20
        self.num_hiddens_1 = 2*1
        self.num_hiddens_2 = 3*1
        self.num_hiddens_3 = 2*1
        self.num_hiddens_4 = 3*1
        # self.FCL1 = nn.Linear(dataset.num_features, self.num_hiddens_1, bias=False)
        # self.FCL2 = nn.Linear(self.num_hiddens_1, self.num_hiddens_2, bias=False)
        # self.FCL3 = nn.Linear(self.num_hiddens_2, self.num_hiddens_3, bias=False)
        # self.FCL4 = nn.Linear(self.num_hiddens_3, self.num_hiddens_4, bias=False)
        # self.FCL5 = nn.Linear(self.num_hiddens_4, dataset.num_classes, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.leakyrelu = nn.LeakyReLU(negative_slope=LeakyReLU_negative_slope)
        self.elu = nn.ELU(alpha=ELU_alpha)
        self.swish = nn.SiLU()

    def forward(self, x, edge_index, W_kron):
        W = []
        for s in range(len(W_kron)):
            W.append(W_kron[s].detach().clone().float().numpy().T)
        h0 = x
        # 1
        temp = torch.from_numpy(A).float()@h0
        ahw1 = temp@W[0]
        h1 = self.relu(ahw1) # y = xW'
        # 2
        temp2 = torch.from_numpy(A).float()@h1
        ahw2 = temp2@W[1]
        h2 = self.swish(ahw2) # y = xW'
        # 3
        temp3 = torch.from_numpy(A).float()@h2
        ahw3 = temp3@W[2]
        h3 = self.elu(ahw3) # y = xW'
        # 4
        temp4 = torch.from_numpy(A).float()@h3
        ahw4 = temp4@W[3]
        h4 = self.leakyrelu(ahw4) # y = xW'
        # 5
        temp5 = torch.from_numpy(A).float()@h4
        ahw5 = temp5@W[4]
        # 6
        z = self.sigmoid(ahw5)
        # h = [h0,h1,h2,h3,h4,h5]
        ah = [temp, temp2, temp3, temp4, temp5]
        ahw = [ahw1, ahw2, ahw3, ahw4, ahw5]
        return ah, ahw, z
#%%
# https://stackoverflow.com/questions/56067643/speeding-up-kronecker-products-numpy
def myKron(a,b):
   na_r = np.shape(a)[0]
   na_c = np.shape(a)[1]
   nb_r = np.shape(b)[0]
   nb_c = np.shape(b)[1]
   return (a[:, None, :, None]*b[None, :, None, :]).reshape(na_r*nb_r,na_c*nb_c)

def unitvector( dimension, position ):
   e = np.zeros( (dimension, 1) )
   e[position-1] = 1
   return e

#test
#ee = unitvector(3, 1)
#ef = unitvector(2, 2)
#E = ee*np.transpose(ef)

def ElementaryMatrix( row, col, position1, position2 ):
   E = unitvector(row, position1)*np.transpose(unitvector(col, position2))
   return E

#test
#E2 = ElementaryMatrix(3,2,1,2)

def PermutationMatrix( row, col ):
    U = np.zeros( (row*col, row*col) )
    for i in range(0, row):
        for j in range(0, col):
            U = U + np.kron(ElementaryMatrix(row,col,i+1,j+1), ElementaryMatrix(col,row,j+1,i+1))
    return U

#test
#U2 = PermutationMatrix(2,3)

def PermutationRelatedMatrix( row, col ):
    U = np.zeros( (row*row, col*col) )
    for i in range(0, row):
        for j in range(0, col):
            U = U + np.kron(ElementaryMatrix(row,col,i+1,j+1), ElementaryMatrix(row,col,i+1,j+1))
    return U

#%%
# def dsigmoid():
#     return lambda x : (np.exp(-x))/(1+np.exp(-x))**2

# def didentity():
#     return lambda x : np.ones(np.shape(x))

# def dswish():
#     return lambda x : (1+np.exp(-x)+x*np.exp(-x))/(np.exp(-x)+1)**2

# def delu(alpha):
#     return lambda x : (x<0)*alpha*np.exp(x) + (x>=0)*1

# def dleakyrelu(alpha):
#     return lambda x : (x<0)*alpha + (x>=0)*1

# def drelu():
#     return lambda x : (x>0)*1

def dsigmoid():
    return lambda x: torch.exp(-x) / (1 + torch.exp(-x))**2

def didentity():
    return lambda x: torch.ones_like(x)

def dswish():
    return lambda x: (1 + torch.exp(-x) + x * torch.exp(-x)) / (1 + torch.exp(-x))**2

def delu(alpha):
    return lambda x: (x < 0).float() * alpha * torch.exp(x) + (x >= 0).float() * 1

def dleakyrelu(alpha):
    return lambda x: (x < 0).float() * alpha + (x >= 0).float() * 1

def drelu():
    return lambda x: (x > 0).float() * 1


#%%
def my_weight_update(output, target, W_kron, AH, AHW):
    d = len(W_kron)

    # Derivative functions
    dSigma = [drelu(), dswish(), delu(ELU_alpha), dleakyrelu(LeakyReLU_negative_slope), didentity(), dsigmoid()]
    dSigma_dplus = dSigma[d]
    dSigma_d = dSigma[d - 1]

    nr, nc = output.shape

    # Compute delta and activation derivative once
    delta = (target - output) / (output * (1 - output))           # (nr, nc)
    # dSigma_dplus_vals = dSigma_dplus(AHW[-1])                     # (nr, nc)
    # dSigma_d_vals = dSigma_d(AHW[-1])                     # (nr, nc) The layer before sigmoid is identity, the input for identity is the same as the input of sigmoid

    # gW = [torch.zeros_like(W_kron[s].T) for s in range(len(W_kron))]
    gW = []

    for s in range(len(W_kron)):

        if s < d-1:
            # Vectorized g(), returns shape (nr, nc, P, Q)
            # g_val = g_vectorized(d-1, W_kron, AH, AHW, dSigma, s)

            # Combine delta, dSigma_dplus and dSigma_d_vals
            # coeff = (delta * dSigma_dplus_vals * dSigma_d_vals).float()                     # (nr, nc)

            # Multiply and sum over i,j
            # gW[s] = -torch.einsum('ij,ijpq->pq', coeff, g_val)
            gW.append(-torch.einsum('ij,ijpq->pq', (delta * dSigma_dplus(AHW[-1]) * dSigma_d(AHW[-1])).float(), g_vectorized(d-1, W_kron, AH, AHW, dSigma, s)))

        else:
            # Simpler case (no recursion)
            # delta_q = (delta * dSigma_dplus_vals * dSigma_d_vals).float()                   # (nr, nc)

            # AH shape: (nr, P), delta_q: (nr, Q)
            # gW[s] = -torch.einsum('ip,iq->pq', AH[-1], delta_q)
            gW.append(-torch.einsum('ip,iq->pq', AH[-1], (delta * dSigma_dplus(AHW[-1]) * dSigma_d(AHW[-1])).float()))


        # print(f"gW[{s}]: {gW[s].shape}")
    return gW

def g_vectorized(d, W, AH, AHW, dSigma, s):
    """
    Vectorized version of g() for all i, j, p, q
    Returns shape (nr, nc, P, Q)
    """
    # print(f"s, d: {s}, {d}")
    # nr = A.shape[0]               # i dimension
    # nc = W[d - 1].shape[1]        # j dimension
    # K = A.shape[1]                # number of features k
    # P = AH[s - 1].shape[1]        # p dimension
    # Q = W[s].shape[0]             # q dimension
    # print("s: " + str(s))

    if s == d-1:
        # Base case
        # dSigma_d = dSigma[d - 1]
        # dSigma_d_vals = dSigma_d(AHW[d-1])        # (K, Q)
        # AH_prev = AH[d-1]                     # (K, P)
        # W_d_plus = W[d].detach().T                              # (Q, nc)

        # Einsum steps:
        # - i -> A[i, k]
        # - k -> dSigma_d_vals[k, q], AH_prev[k, p]
        # - q -> W_s[q, j]
        # t = torch.einsum('ik,kq,kp,qj->ijpq', torch.tensor(A, dtype=torch.float32), dSigma_d_vals, AH_prev, W_d_plus)
        t = torch.einsum('ik,kq,kp,qj->ijpq', torch.tensor(A, dtype=torch.float32), dSigma[d - 1](AHW[d-1]), AH[d-1], W[d].detach().T)
    else:
        # Recursive case
        # g_recursive = g_vectorized(d-1, W, AH, AHW, dSigma, s)  # (nr, nc, P, Q)
        # dSigma_d = dSigma[d - 1]
        # dSigma_d_vals = dSigma_d(AHW[d-1])        # (U, V)
        # W_d_plus = W[d].detach().T                              # (V, nc)

        # The einsum over u, v
        # t = torch.einsum('iu,uv,vj,uvpq->ijpq', torch.tensor(A, dtype=torch.float32), dSigma_d_vals, W_d_plus, g_recursive)
        t = torch.einsum('iu,uv,vj,uvpq->ijpq', torch.tensor(A, dtype=torch.float32), dSigma[d - 1](AHW[d-1]), W[d].detach().T, g_vectorized(d-1, W, AH, AHW, dSigma, s))

    return t  # Shape: (nr, nc, P, Q)

# #%%
# def my_weight_update(output, target, W_kron, AH, AHW):
#   d = len(W_kron)
#   dSigma = [drelu(),dswish(),delu(ELU_alpha),dleakyrelu(LeakyReLU_negative_slope),didentity(),dsigmoid()]
#   dSigma_dplus = dSigma[d]
#   dSigma_d = dSigma[d-1]
  
#   dSigma_dplus = dSigma[d]
#   dSigma_d = dSigma[d-1]

#   nr = np.shape(output)[0]
#   nc = np.shape(output)[1]
  
#   d = len(W_kron)
#   gW = []
#   for s in range(len(W)):
#       gW.append(np.zeros(np.shape(W[s]),dtype="float32"))
#   for s in range(len(W_kron)):
#       print(f"d: {d}")
#       gW[s] = torch.zeros(np.shape(W_kron[s].T))
#       if s < d:
#           for i in range(nr):
#               print(f"i: {i}")
#               for j in range(nc):
#                   print(f"j: {j}")
#                   for p in range(np.shape(W_kron[s].detach().T)[0]):
#                       print(f"p: {p}")
#                       for q in range(np.shape(W_kron[s].detach().T)[1]):
#                           print(f"s,i,j,p,q: {s},{i},{j},{p},{q}")
#                           gW[s][p,q] += (target[i,j]-output[i,j])/(output[i,j]*(1-output[i,j])) * dSigma_dplus(AHW[-1][i,j]) * g(d-2,i,j,p,q,s,W_kron,A,AH,AHW, dSigma_d)
#       else:
#           for i in range(nr):
#               gW[s][p,q] += (target[i,q]-output[i,q])/(output[i,q]*(1-output[i,q])) * dSigma_dplus(AHW[-1][i,q]) * dSigma_d(AHW[-1])[i,q] * AH[i,p]
#   return gW            

# def g(d,i,j,p,q,s,W,A,AH,AHW,dSigma_d):
#     t = 0
#     if s == d:
#         for k in range(np.shape(A)[0]):
#             t += W[d+1].detach().T[q,j] * A[i,k] * dSigma_d(AHW[d])[k,q] * (AH[d])[k,p]
#     elif s < d:
#         for v in range(np.shape(W[d+1].detach().T)[0]):
#             for u in range(np.shape(A)[0]):
#                 t += W[d+1].detach().T[v,j] * A[i,u] * dSigma_d(AHW[d])[u,v] * g(d-1,u,v,p,q,s,W,A,AH,AHW,dSigma_d)
#     return t
#%%
# # Initialize psutil process monitor
# process = psutil.Process(os.getpid())

# # Function to get memory usage (RAM)
# def get_cpu_memory_usage():
#     # rss = Resident Set Size, in bytes
#     mem_in_bytes = process.memory_info().rss
#     mem_in_mb = mem_in_bytes / (1024 ** 2)
#     return mem_in_mb
    
#%%
all_log = []
time_dif = []
mem_dif = []

time_tol1 = []
time_tol2 = []
mem_tol1 = []
mem_tol2 = []

itr = 0
while(1):
# for itr in range(0, 1060):# run 1060 times
    if itr == 1060:
        break
    
    device = torch.device('cpu')
    model = GCN().to(device)
    # print(model)
    
    time1 = []
    time2 = []
    mem1 = []
    mem2 = []
    
    count = 0
    W = []
    W_peak = []
    W_kron = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            W.append(param.data.detach().clone())
            W_kron.append(param.data.detach().clone())
            W_peak.append(param.data)
            # print(name, param.numel())
            count = count + 1
    
    #%%
    # criterion = torch.nn.CrossEntropyLoss()
    lr = 0.00003
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    
    # Calculate accuracy
    def accuracy(pred_y, y):
        return (1*(pred_y>0.5).squeeze(1) == y).sum() / len(y)
    
    # save weights
    W_change = []
    W_temp = []
    for s in range(len(W)):
        W_temp.append(W_peak[s].detach().clone())
    W_change.append(W_temp)
    
    # Data for animations
    embeddings = []
    losses = []
    accuracies = []
    outputs = []
    
    model.eval()
    n_epochs = 10
    nan_flag = False
    
    # Training loop
    for epoch in range(n_epochs):
        # Start tracing Python memory allocations
        tracemalloc.start()
        time_start = time.perf_counter()
        # Clear gradients
        optimizer.zero_grad()
    
        # Forward pass
        h, z = model(data.x, data.edge_index)
    
        # Calculate loss function
        loss = my_loss(z, data.y.numpy())
        
        if  loss.isnan().any():
            nan_flag = True
            break
    
        # Calculate accuracy
        acc = accuracy(z, data.y)
    
        # Compute gradients
        loss.backward()
        # g = my_weight_update(z, dummy_y, A, data.x)
    
        # Tune parameters
        optimizer.step()
        
        # Get the current memory usage and peak memory usage
        current, peak = tracemalloc.get_traced_memory()
        
        # Stop tracing
        tracemalloc.stop()
    
        # Updated weight
        W_temp = []
        for s in range(len(W_peak)):
            W_temp.append(W_peak[s].detach().clone())
        W_change.append(W_temp)
    
        # Store data for animations
        losses.append(loss.detach().numpy())
        accuracies.append(acc)
        outputs.append(1*(z>0.5))
        time_elapsed = (time.perf_counter() - time_start)
        time1.append(time_elapsed)
        mem1.append(peak)
    
        # Print metrics every 10 epochs
        # if epoch % 1 == 0:
        #     print(f'Epoch {epoch:>3} | Loss: {loss:.2f} | Acc: {acc*100:.2f}%')
    if nan_flag:
        continue
    #%%
    # criterion = torch.nn.CrossEntropyLoss()
    mymodel_Kron = GCN_Kron().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    
    W_change_2 = []
    W_temp = []
    for s in range(len(W)):
        W_temp.append(W_kron[s])
    W_change_2.append(W_temp)
    
    # Data for animations
    embeddings = []
    losses_2 = []
    accuracies_2 = []
    outputs_2 = []
    
    model.eval()
    
    # Training loop
    for epoch in range(n_epochs):
        # Start tracing Python memory allocations
        tracemalloc.start()
        time_start = time.perf_counter()
        # Clear gradients
        # optimizer.zero_grad()
    
        # Forward pass
        # h, z = model(data.x, data.edge_index)
        AH, HB, z = mymodel_Kron(data.x, A, W_kron)
    
        # Calculate loss function
        loss = my_loss(z, data.y.numpy())
    
        # Calculate accuracy
        acc = accuracy(z, data.y)
    
        # Compute gradients
        # loss.backward()
        # g = my_weight_update(z, dummy_y, A, data.x)
        gW = my_weight_update(z.detach().numpy(), data.y.unsqueeze(1), W_kron, AH, HB)
        
        # Get the current memory usage and peak memory usage
        current, peak = tracemalloc.get_traced_memory()
        
        # Stop tracing
        tracemalloc.stop()
    
        # Tune parameters
        # optimizer.step()
    
        # Update weight
        for s in range(len(W_kron)):
            W_kron[s] = W_kron[s] - (lr*gW[s]).T
        # W1 = W1 - (lr*g).T
        # for name, param in model.named_parameters():
        #     if param.requires_grad:
        #         print(name, param.numel())
        # param.data = W1.float()
        
        W_temp = []
        for s in range(len(W_kron)):
            W_temp.append(W_kron[s])
        W_change_2.append(W_temp)
    
        # Store data for animations
        losses_2.append(loss.detach().numpy())
        accuracies_2.append(acc)
        outputs_2.append(1*(z>0.5))
        time_elapsed = (time.perf_counter() - time_start)
        time2.append(time_elapsed)
        mem2.append(peak)
    
        # Print metrics every 10 epochs
        # if epoch % 1 == 0:
        #     print(f'Epoch {epoch:>3} | Loss: {loss:.2f} | Acc: {acc*100:.2f}%')
    
    #%%
    W_dif = []
    nan_flag = False
    for i in range(n_epochs+1):
        W_temp = []
        for s in range(len(W_kron)):
            W_temp.append(torch.mean((W_change[i][s]-W_change_2[i][s])**2))
            # check nan
            if  torch.mean((W_change[i][s]-W_change_2[i][s])**2).isnan().any():
                nan_flag = True
                break
        if nan_flag:
            break
        W_dif.append(W_temp)
    if nan_flag:
        continue   
    itr = itr + 1    
        
    all_log.append(W_dif)
    dif_time = np.array(time1) - np.array(time2)
    time_dif.append(dif_time.tolist())
    dif_mem = np.array(mem1) - np.array(mem2)
    mem_dif.append(dif_mem.tolist())
    time_tol1.append(time1)
    time_tol2.append(time2)
    mem_tol1.append(mem1)
    mem_tol2.append(mem2)
    if itr%5 == 0:
        print(itr)

np.save('all_log.npy', np.asarray(all_log))
#%%
# ['#0173b2',
#  '#de8f05',
#  '#029e73',
#  '#d55e00',
#  '#cc78bc',
#  '#ca9161',
#  '#fbafe4',
#  '#949494',
#  '#ece133',
#  '#56b4e9']

# color = itertools.cycle(('#0173b2','#de8f05','#029e73','#d55e00','#cc78bc','#ca9161', '#fbafe4', '#949494', '#ece133', '#56b4e9'))
# marker = itertools.cycle(('v', 'P', '^', 'o', '<', '*', '>',))
# for s in range(len(W)):
#     plt.rcParams["xtick.labelsize"] = 20
#     plt.rcParams["ytick.labelsize"] = 20
#     plt.figure(figsize=(16,8))
#     lines = plt.plot(list(range(n_epochs+1)), np.asarray(all_log).T[s], color=next(color),linewidth=3, marker=next(marker), markersize=20,label='SSE of W' + str(s+1), alpha=0.1)
#     plt.setp(lines[1:], label="_")
#     # Sum of Squared Errors
#     # plt.title('Sum of Squared Errors (SSE) of W' + str(s+1) + ' between \nautograd and our method', fontsize=30)
#     plt.title('W' + str(s+1), fontsize=30)
#     plt.xlabel('iterations', fontsize=30)
#     plt.ylabel('SSE', fontsize=30)
#     plt.legend(loc="best", fontsize=30)
#     plt.grid()
#     plt.rcParams.update(plt.rcParamsDefault)
#     plt.show()

#%%
# ['#0173b2',
#  '#de8f05',
#  '#029e73',
#  '#d55e00',
#  '#cc78bc',
#  '#ca9161',
#  '#fbafe4',
#  '#949494',
#  '#ece133',
#  '#56b4e9']

color = itertools.cycle(('#0173b2','#de8f05','#029e73','#d55e00','#cc78bc','#ca9161', '#fbafe4', '#949494', '#ece133', '#56b4e9'))
marker = itertools.cycle(('v', 'P', '^', 'o', '<', '*', '>',))
for s in range(len(W)):
    plt.rcParams["xtick.labelsize"] = 20
    plt.rcParams["ytick.labelsize"] = 20
    plt.figure(figsize=(9,4))
    lines = plt.plot(list(range(n_epochs+1)), np.asarray(all_log).T[s], color=next(color),linewidth=3, marker=next(marker), markersize=20,label='SSE of W' + str(s+1), alpha=0.1)
    plt.setp(lines[1:], label="_")
    # Sum of Squared Errors
    # plt.title('Sum of Squared Errors (SSE) between automatic differentiation \nand our method using Kronecker product', fontsize=30)
    plt.title('SSE of W' + str(s+1), fontsize=30)
    plt.xlabel('Iterations', fontsize=30)
    plt.ylabel('SSE', fontsize=30)
    plt.yscale("log")
    # plt.legend(loc="best", fontsize=30)
    plt.grid()
    # ax = plt.gca()
    # ax.yaxis.set_major_formatter(formatter)
    plt.rcParams.update(plt.rcParamsDefault)
    plt.show()
    
#%%
# https://stackoverflow.com/questions/34608613/matplotlib-boxplot-calculated-on-log10-values-but-shown-in-logarithmic-scale
for s in range(len(W)):
    plt.rcParams["xtick.labelsize"] = 20
    plt.rcParams["ytick.labelsize"] = 20
    fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(9, 4))
    
    # data = []
    # for box_i in range(len(np.asarray(all_log).T[0])):
    #     data.append()
    
    # plot box plot
    # axs.boxplot(np.log10(np.asarray(all_log).T[0]).tolist())
    axs.boxplot(np.log10(np.asarray(all_log)).T[s].tolist())
    axs.set_title('W' + str(s+1), fontsize=30)
    
    # adding horizontal grid lines
    axs.set_xticks([y + 1 for y in range(len(np.asarray(all_log).T[s]))],
                  labels=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10'])
    # https://stackoverflow.com/questions/34608613/matplotlib-boxplot-calculated-on-log10-values-but-shown-in-logarithmic-scale
    axs.set_xlabel('Iterations', fontsize=30)
    axs.set_ylabel(r'$log_{10}$(SSE)', fontsize=30)
    # axs.set_yticks(np.arange(-18, -6))
    # axs.set_yticklabels(np.arange(1e-18, 1e-6,10))
    # axs.yaxis.grid(True)
    
    axs.yaxis.set_major_formatter(formatter)
    
    # plt.yscale("log")
    # plt.grid()
    plt.rcParams.update(plt.rcParamsDefault)
    plt.show()
    
#%%
# https://stackoverflow.com/questions/34608613/matplotlib-boxplot-calculated-on-log10-values-but-shown-in-logarithmic-scale
for s in range(len(W)):
    plt.rcParams["xtick.labelsize"] = 20
    plt.rcParams["ytick.labelsize"] = 20
    fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(9, 4))
    
    # data = []
    # for box_i in range(len(np.asarray(all_log).T[0])):
    #     data.append()
    
    # plot box plot
    # axs.boxplot(np.log10(np.asarray(all_log).T[0]).tolist())
    axs.boxplot(np.asarray(all_log).T[s].tolist())
    axs.set_title('W' + str(s+1), fontsize=30)
    
    # adding horizontal grid lines
    axs.set_xticks([y + 1 for y in range(len(np.asarray(all_log).T[s]))],
                  labels=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10'])
    # https://stackoverflow.com/questions/34608613/matplotlib-boxplot-calculated-on-log10-values-but-shown-in-logarithmic-scale
    axs.set_xlabel('Iterations', fontsize=30)
    axs.set_ylabel('SSE', fontsize=30)
    # axs.set_yticks(np.arange(-18, -6))
    # axs.set_yticklabels(np.arange(1e-18, 1e-6,10))
    axs.yaxis.grid(True)
    
    axs.yaxis.set_major_formatter(formatter)
    
    # plt.yscale("log")
    # plt.grid()
    plt.rcParams.update(plt.rcParamsDefault)
    plt.show()
    
#%%
# https://stackoverflow.com/questions/34608613/matplotlib-boxplot-calculated-on-log10-values-but-shown-in-logarithmic-scale
for s in range(len(W)):
    plt.rcParams["xtick.labelsize"] = 20
    plt.rcParams["ytick.labelsize"] = 20
    fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(9, 4))
    
    # data = []
    # for box_i in range(len(np.asarray(all_log).T[0])):
    #     data.append()
    
    # plot box plot
    # axs.boxplot(np.log10(np.asarray(all_log).T[0]).tolist())
    axs.boxplot(np.asarray(all_log).T[s].tolist(), showfliers=False) # showfliers=False to remove outliers
    axs.set_title('W' + str(s+1), fontsize=30)
    
    # adding horizontal grid lines
    axs.set_xticks([y + 1 for y in range(len(np.asarray(all_log).T[s]))],
                  labels=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10'])
    # https://stackoverflow.com/questions/34608613/matplotlib-boxplot-calculated-on-log10-values-but-shown-in-logarithmic-scale
    axs.set_xlabel('Iterations', fontsize=30)
    axs.set_ylabel('SSE', fontsize=30)
    # axs.set_yticks(np.arange(-18, -6))
    # axs.set_yticklabels(np.arange(1e-18, 1e-6,10))
    axs.yaxis.grid(True)
    
    axs.yaxis.set_major_formatter(formatter)
    
    # plt.yscale("log")
    # plt.grid()
    plt.rcParams.update(plt.rcParamsDefault)
    plt.show()

#%% Statistics
import pandas as pd

# Assuming all_log is already defined and is a list or array of shape (samples, iterations)
all_log_np = np.asarray(all_log)

# Loop through each "W" (same as in your boxplot code)
for s in range(len(W)):
    data_s = all_log_np.T[s]  # Get data for the s-th W
    
    # Calculate statistics for each iteration
    stats = {
        'Iteration': [],
        'Min': [],
        'Max': [],
        'Mean': [],
        'Median': [],
        'Std': [],
        'Q1': [],
        'Q3': []
    }
    
    for i, iter_values in enumerate(data_s):  # iter_values = data points for iteration i
        stats['Iteration'].append(i)
        stats['Min'].append(np.min(iter_values))
        stats['Max'].append(np.max(iter_values))
        stats['Mean'].append(np.mean(iter_values))
        stats['Median'].append(np.median(iter_values))
        stats['Std'].append(np.std(iter_values))
        stats['Q1'].append(np.percentile(iter_values, 25))
        stats['Q3'].append(np.percentile(iter_values, 75))
    
    # Create a DataFrame
    stats_df = pd.DataFrame(stats)
    
    # Function to convert to custom scientific notation format
    def custom_sci_notation(x):
        if isinstance(x, (int, np.integer)):  # For iteration numbers, return as-is
            return f'{int(x)}'
        elif x == 0:
            return '0'
        else:
            exponent = int(np.floor(np.log10(abs(x))))
            mantissa = x / (10**exponent)
            return f'{mantissa:.2f} x 10^{exponent}'
    
    # Apply formatting to each cell (except the column headers)
    cell_text = stats_df.applymap(custom_sci_notation).values.tolist()
    
    # Console print with formatted strings
    print(f"\nStatistics for W{s+1}\n")
    print(stats_df.applymap(custom_sci_notation).to_string(index=False))  # No index, just table
    
    # Plot the table in matplotlib
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.axis('off')
    ax.axis('tight')
    
    table = ax.table(cellText=cell_text,
                     colLabels=stats_df.columns,
                     loc='center')
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.2)
    
    ax.set_title(f'Statistics Table for W{s+1}', fontsize=20)
    
    plt.show()

    
#%%
# avg_time = np.average(np.array(time_dif))
# print("Average time improvement: ")
# print(avg_time)

# avg_time1 = np.average(np.array(time_tol1))
# print("Average time AD: ")
# print(avg_time1)

# avg_time2 = np.average(np.array(time_tol2))
# print("Average time Ours: ")
# print(avg_time2)

# avg_mem = np.average(np.array(mem_dif))
# print("Average memory improvement: ")
# print(f"Average Peak memory usage improvement: {avg_mem / 1024 / 1024:.2f} MB")

# avg_mem1 = np.average(np.array(mem_tol1))
# print("Average memory AD: ")
# # print(f"Average Peak memory usage AD: {avg_mem1 / 1024 / 1024:.2f} MB")
# print(f"Average Peak memory usage AD: {avg_mem1 / 1024 :.2f} KB")

# avg_mem2 = np.average(np.array(mem_tol2))
# print("Average memory Ours: ")
# print(f"Average Peak memory usage Ours: {avg_mem2 / 1024 / 1024:.2f} MB")
#%% Calculate averages and standard deviations
avg_time = np.average(np.array(time_dif) * 1000)
std_time = np.std(np.array(time_dif) * 1000)

avg_time1 = np.average(np.array(time_tol1) * 1000)
std_time1 = np.std(np.array(time_tol1) * 1000)

avg_time2 = np.average(np.array(time_tol2) * 1000)
std_time2 = np.std(np.array(time_tol2) * 1000)

avg_mem = np.average(mem_dif)
std_mem = np.std(mem_dif)

avg_mem1 = np.average(np.array(mem_tol1) / 1024)
std_mem1 = np.std(np.array(mem_tol1) / 1024)

avg_mem2 = np.average(np.array(mem_tol2) / 1024)
std_mem2 = np.std(np.array(mem_tol2) / 1024)

# Print results
print("=== Time Results ===")
print(f"Average time improvement: {avg_time:.2f} (std: {std_time:.2f}) ms")
print(f"Average time AD: {avg_time1:.2f} (std: {std_time1:.2f}) ms")
print(f"Average time Ours: {avg_time2:.2f} (std: {std_time2:.2f}) ms")

print("\n=== Memory Results ===")
print(f"Average memory improvement: {avg_mem / 1024 / 1024:.2f} MB (std: {std_mem / 1024 / 1024:.2f} MB)")
print(f"Average memory AD: {avg_mem1:.2f} KB (std: {std_mem1:.2f} KB)")
print(f"Average memory Ours: {avg_mem2:.2f} KB (std: {std_mem2:.2f} KB)")

# Save all lists to a JSON file
data_to_save = {
    "time_dif": time_dif,
    "time_tol1": time_tol1,
    "time_tol2": time_tol2,
    "mem_dif": mem_dif,
    "mem_tol1": mem_tol1,
    "mem_tol2": mem_tol2
}

with open(f'results_data{num_nodes}.json', 'w') as f:
    json.dump(data_to_save, f, indent=4)

print("\nLists saved to 'results_data{num_nodes}.json'.")
