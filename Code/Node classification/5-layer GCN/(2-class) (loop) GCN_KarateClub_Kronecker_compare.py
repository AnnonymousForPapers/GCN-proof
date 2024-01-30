# -*- coding: utf-8 -*-
"""
Created on Sat Jan 27 17:45:42 2024

Ref: https://mlabonne.github.io/blog/posts/2022_02_20_Graph_Convolution_Network.html
"""

#%%
import torch
import torch.nn as nn
from torch.nn import Linear
from torch.nn import Sigmoid
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.utils import to_dense_adj
from torch_geometric.utils import to_networkx
from sklearn.preprocessing import OneHotEncoder
import itertools
import random

#%%
random.seed(0)
random.random()

torch.manual_seed(0)
torch.rand(4)

np.random.seed(0)
np.random.rand(4)

#%%
from torch_geometric.datasets import KarateClub

# Import dataset from PyTorch Geometric
dataset = KarateClub()

# Print information
print(dataset)
print('------------')
print(f'Number of graphs: {len(dataset)}')
print(f'Number of features: {dataset.num_features}')
print(f'Number of classes: {dataset.num_classes}')

#%%
# Print first element
print(f'Graph: {dataset[0]}')

#%%
data = dataset[0]

print(f'x = {data.x.shape}')
print(data.x)

#%%
print(f'edge_index = {data.edge_index.shape}')
print(data.edge_index)

#%%
A = to_dense_adj(data.edge_index)[0].numpy().astype(int)
print(f'A = {A.shape}')
print(A)

#%%
print(f'y = {data.y.shape}')
print(data.y)

#%% Change class 3 to 1 and class 2 to 0
data.y[data.y==3] = 1
data.y[data.y==2] = 0

#%%
print(f'train_mask = {data.train_mask.shape}')
print(data.train_mask)

#%%
print(f'Edges are directed: {data.is_directed()}')
print(f'Graph has isolated nodes: {data.has_isolated_nodes()}')
print(f'Graph has loops: {data.has_self_loops()}')

#%%
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
        self.FCL1 = nn.Linear(dataset.num_features, self.num_hiddens_1, bias=False)
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
        h1 = self.relu(temp@W[0]) # y = xW'
        # 2
        temp2 = torch.from_numpy(A).float()@h1
        h2 = self.swish(temp2@W[1]) # y = xW'
        # 3
        temp3 = torch.from_numpy(A).float()@h2
        h3 = self.elu(temp3@W[2]) # y = xW'
        # 4
        temp4 = torch.from_numpy(A).float()@h3
        h4 = self.leakyrelu(temp4@W[3]) # y = xW'
        # 5
        temp5 = torch.from_numpy(A).float()@h4
        h5 = temp5@W[4]
        # 6
        z = self.sigmoid(h5)
        h = [h0,h1,h2,h3,h4,h5]
        return h, z
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
def dsigmoid():
    return lambda x : (np.exp(-x))/(1+np.exp(-x))**2

def didentity():
    return lambda x : np.ones(np.shape(x))

def dswish():
    return lambda x : (1+np.exp(-x)+x*np.exp(-x))/(np.exp(-x)+1)**2

def delu(alpha):
    return lambda x : (x<0)*alpha*np.exp(x) + (x>=0)*1

def dleakyrelu(alpha):
    return lambda x : (x<0)*alpha + (x>=0)*1

def drelu():
    return lambda x : (x>0)*1

#%%
def dyhatijdWs(i,j,s,d,A,Ws,W,H,U_bar):
    # dSigma = [dsigmoid(),didentity(),dswish(),delu(),dleakyrelu(),drelu()]
    dSigma = [dleakyrelu(LeakyReLU_negative_slope),delu(ELU_alpha),dswish(),drelu(),didentity(),dsigmoid()]
    dSigma = [drelu(),dswish(),delu(ELU_alpha),dleakyrelu(LeakyReLU_negative_slope),didentity(),dsigmoid()]
    dSigma_dplus = dSigma[d]
    dSigma_d = dSigma[d-1]
    Hd = H[d]
    Hd_1 = H[d-1]
    Ari = np.expand_dims(A[i,:],0)
    Wd = W[d-1]
    Wdcj = np.expand_dims(Wd[:,j],1)
    ns_1 = np.shape(Ws)[0]
    ns = np.shape(Ws)[1]
    Ins = np.identity(ns)
    Ins_1 = np.identity(ns_1)
    J = np.ones((ns_1, ns))
    # Sigma' d+1
    term1 = dSigma_dplus(Hd[i,j])
    term2 = myKron(J,dSigma_d(Ari@Hd_1@Wdcj))
    term3 = dAHWdWs(i,j,s,d,A,Ws,W,H,dSigma,U_bar)
    # print(np.shape(term1))
    # print(np.shape(term2))
    # print(np.shape(term3))
    return term1*(term2*term3)
    
def dAHWdWs(i,j,s,d,A,Ws,W,H,dSigma,U_bar):
    Hd_1 = H[d-1]
    Wd = W[d-1]
    Wdcj = np.expand_dims(Wd[:,j],1)
    Ari = np.expand_dims(A[i,:],0)
    ns_1 = np.shape(Ws)[0]
    ns = np.shape(Ws)[1]
    Ins = np.identity(ns)
    Ins_1 = np.identity(ns_1)
    nd = np.shape(Wd)[1]
    if s == d:
        term1 = myKron(Ins_1,(Ari@Hd_1))
        term2 = U_bar[s-1]@myKron(Ins,unitvector( nd, j+1 ))
        # U_bar = PermutationRelatedMatrix(ns_1,ns)
        return term1@term2
    else:
        k = d - 1
        return myKron(Ins_1,Ari)@dHdW(i,j,s,d,A,Ws,W,H,dSigma,k,U_bar)@myKron(Ins,Wdcj)
    
def dHdW(i,j,s,d,A,Ws,W,H,dSigma,k,U_bar):
    dSigma_k = dSigma[k-1]
    Hk = H[k]
    Hk_1 = H[k-1]
    Wk = W[k-1]
    ns_1 = np.shape(Ws)[0]
    ns = np.shape(Ws)[1]
    Ins = np.identity(ns)
    Ins_1 = np.identity(ns_1)
    J = np.ones((ns_1, ns))
    if s==k:
        # U_bar = PermutationRelatedMatrix(ns_1,ns)
        term1 = myKron(J,dSigma_k(A@Hk_1@Wk))
        term2 = myKron(Ins_1,(A@Hk_1))@U_bar[s-1]
        return term1*term2
    else:
        term1 = myKron(J,dSigma_k(A@Hk_1@Wk))
        term2 = myKron(Ins_1,A)@dHdW(i,j,s,d,A,Ws,W,H,dSigma,k-1,U_bar)@myKron(Ins,Wk)
        return term1*term2
        
#%%
def my_weight_update(output, target, H_in, W_kron):
    
  W = []  
  for s in range(len(W_kron)):
        W.append(W_kron[s].detach().clone().float().numpy().T)
      
  H = []
  for s in range(len(H_in)):
       H.append(H_in[s].numpy())
       
  U_bar = []
  for s in range(len(W_kron)):
       ns_1 = np.shape(W[s])[0]
       ns = np.shape(W[s])[1]
       U_bar.append(PermutationRelatedMatrix(ns_1,ns))

  # ct = 1e-15 # to deal with log(0)
  # loss_true = -sum(sum(A*torch.log(output+ct)))
  # loss_mis = -sum(sum(B*torch.log(1-output+ct)))
  # loss = (loss_true + loss_mis)/mask.sum()

  nr = np.shape(output)[0]
  nc = np.shape(output)[1]
  
  d = len(W)
  gW = []
  for s in range(len(W)):
      gW.append(np.zeros(np.shape(W[s]),dtype="float32"))
  count = 0
  for i in range(0,nr,1):
    for j in range(0,nc,1):
      # if count % 50 == 0:
      #     print(str(count)+'/'+str(nr*nc))
      # count = count + 1
      for s in range(len(W)):
          temp = (target[i,j]-output[i,j])/(output[i,j]*(1-output[i,j]))
          dLdWs = temp*dyhatijdWs(i,j,s+1,d,A,W[s],W,H,U_bar)
          gW[s] = gW[s] - dLdWs


  return gW

#%%
# def my_weight_update(output, target, A, H0):
#     n = np.shape(dummy_y)[0]
#     n1 = np.shape(dummy_y)[1]
#     first_flag = 1
#     for i in range(0,n,1):
#         for j in range(0,n1,1):
#             term1 = target[i,j] - output[i,j]
#             term2 = np.kron(np.identity(n),A[i,:]@H0.numpy())
#             term3 = PermutationRelatedMatrix(n,n1)@np.kron(np.identity(n1),unitvector(n1,j+1)) # position of the unit vector starts from 1
#             if first_flag:
#                 g = term1*(term2@term3)
#                 first_flag = 0
#             else:
#                 g = g+term1*(term2@term3)
#     return -g
    
#%%
all_log = []

itr = 0
while(1):
# for itr in range(0, 1060):# run 1060 times
    if itr == 1060:
        break
    
    device = torch.device('cpu')
    model = GCN().to(device)
    # print(model)
    
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
    
        # Updated weight
        W_temp = []
        for s in range(len(W_peak)):
            W_temp.append(W_peak[s].detach().clone())
        W_change.append(W_temp)
    
        # Store data for animations
        losses.append(loss.detach().numpy())
        accuracies.append(acc)
        outputs.append(1*(z>0.5))
    
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
        # Clear gradients
        # optimizer.zero_grad()
    
        # Forward pass
        # h, z = model(data.x, data.edge_index)
        H, z = mymodel_Kron(data.x, A, W_kron)
    
        # Calculate loss function
        loss = my_loss(z, data.y.numpy())
    
        # Calculate accuracy
        acc = accuracy(z, data.y)
    
        # Compute gradients
        # loss.backward()
        # g = my_weight_update(z, dummy_y, A, data.x)
        gW = my_weight_update(z.detach().numpy(), data.y.unsqueeze(1).numpy(), H, W_kron)
    
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
    axs.yaxis.grid(True)
    
    # plt.yscale("log")
    # plt.grid()
    plt.rcParams.update(plt.rcParamsDefault)
    plt.show()