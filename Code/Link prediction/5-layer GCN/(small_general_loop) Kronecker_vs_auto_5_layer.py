# -*- coding: utf-8 -*-
"""
Created on Sat Jan 20 23:27:11 2024

@author: xiaoyenche
"""

#%%
import pickle

name = 'D:/PhD thesis/ICML/GCN-Kronecker_product/Link prediction/DDI_100_nodes.pkl'

with open(name, 'rb') as f:
    loaded_dict = pickle.load(f)
    
#%%
import numpy as np
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
from torch_geometric.utils import to_dense_adj
from torch_geometric.utils import negative_sampling
import random
from torch_geometric.utils import to_networkx
import networkx as nx
import matplotlib.pyplot as plt
import itertools

#%%
random.seed(0)
random.random()

torch.manual_seed(0)
torch.rand(4)

np.random.seed(0)
np.random.rand(4)

#%%
graph_origin = loaded_dict
xx  = torch.stack(list(loaded_dict.edge_index), dim=0)
graph_origin.edge_index = xx

#%%
def convert_to_networkx(graph):

    g = to_networkx(graph, node_attrs=["x"])

    return g


def plot_graph(g):

    plt.figure(figsize=(9, 7))
    nx.draw_spring(g, with_labels=True, node_size=200, arrows=False)
    plt.show()


g_origin = convert_to_networkx(graph_origin)
plot_graph(g_origin)

#%%
subset = np.arange(0, 10, 1)
g = g_origin.subgraph(subset)
plot_graph(g)

#%%
from torch_geometric.utils.convert import from_networkx
graph_directed = from_networkx(
    G=g
)

#%%
g_sub = convert_to_networkx(graph_directed)
plot_graph(g_sub)

#%%
from torch_geometric.transforms.to_undirected import ToUndirected
graph = ToUndirected()(graph_directed)

#%%
# import torch_geometric.transforms as T

# split = T.RandomLinkSplit(
#     num_val=0.05,
#     num_test=0.1,
#     is_undirected=True,
#     add_negative_train_samples=False,
#     neg_sampling_ratio=1.0,
# )
# train_data, val_data, test_data = split(graph)

train_data = graph

n = len(subset)
temp_A=((to_dense_adj(train_data.edge_index, max_num_nodes=n)>0)*1)[0]
train_edge_index = temp_A.nonzero().t().contiguous()
train_edge_label = torch.ones(train_edge_index[0].size())

#%%
#%%
def my_loss(output, target, target_neg):

  # ct = 1e-15 # to deal with log(0)
  n = len(subset)
  A=((to_dense_adj(target, max_num_nodes=n)>0)*1)[0]
  B=((to_dense_adj(target_neg, max_num_nodes=n)>0)*1)[0]
  mask = A+B
  # loss_true = -sum(sum(A*torch.log(output+ct)))
  # loss_mis = -sum(sum(B*torch.log(1-output+ct)))
  loss_true = -sum(sum(A*torch.log(output)))
  loss_mis = -sum(sum(B*torch.log(1-output)))
  loss = (loss_true + loss_mis)/mask.sum()

  return loss

#%%
def Gaussian(x):
    return torch.exp(-(x**2))

#%%
LeakyReLU_negative_slope=0.01
ELU_alpha=1.0

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
        self.num_hiddens_5 = 2*20
        self.FCL1 = nn.Linear(self.num_features, self.num_hiddens_1, bias=False)
        self.FCL2 = nn.Linear(self.num_hiddens_1, self.num_hiddens_2, bias=False)
        self.FCL3 = nn.Linear(self.num_hiddens_2, self.num_hiddens_3, bias=False)
        self.FCL4 = nn.Linear(self.num_hiddens_3, self.num_hiddens_4, bias=False)
        self.FCL5 = nn.Linear(self.num_hiddens_4, self.num_hiddens_5, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.leakyrelu = nn.LeakyReLU(negative_slope=LeakyReLU_negative_slope)
        self.elu = nn.ELU(alpha=ELU_alpha)
        self.swish = nn.SiLU()

    def forward(self, x, edge_index):
        D = torch.diag(sum(temp_A)+1).float()
        D_inv = torch.inverse(torch.sqrt(D))
        n = len(subset)
        A=((to_dense_adj(edge_index, max_num_nodes=n)>0)*1)[0].numpy().astype(int)
        A_hat = torch.from_numpy(A).float()+torch.eye(n)
        h0 = x
        # 1
        temp = D_inv@A_hat@D_inv@x
        h1 = self.leakyrelu(self.FCL1(temp)) # y = xW'
        # 2
        temp2 = D_inv@A_hat@D_inv@h1
        h2 = self.elu(self.FCL2(temp2)) # y = xW'
        # 3
        temp3 = D_inv@A_hat@D_inv@h2
        h3 = self.swish(self.FCL3(temp3)) # y = xW'
        # 4
        temp4 = D_inv@A_hat@D_inv@h3
        h4 = self.relu(self.FCL4(temp4)) # y = xW'
        # 5
        temp5 = D_inv@A_hat@D_inv@h4
        h5 = self.FCL5(temp5)
        # 6
        h6 = h5@h5.t()
        z = self.sigmoid(h6)
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
        self.num_hiddens_1 = 2*5
        self.num_hiddens_2 = 3*5
        self.num_hiddens_3 = 2*5
        self.num_hiddens_4 = 3*5
        self.num_hiddens_5 = 2*5
        # self.FCL1 = nn.Linear(self.num_features, self.num_hiddens_1, bias=False)
        # self.FCL2 = nn.Linear(self.num_hiddens_1, self.num_hiddens_2, bias=False)
        # self.FCL3 = nn.Linear(self.num_hiddens_2, self.num_hiddens_3, bias=False)
        # self.FCL4 = nn.Linear(self.num_hiddens_3, self.num_hiddens_4, bias=False)
        # self.FCL5 = nn.Linear(self.num_hiddens_4, self.num_hiddens_5, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.leakyrelu = nn.LeakyReLU(negative_slope=LeakyReLU_negative_slope)
        self.elu = nn.ELU(alpha=ELU_alpha)
        self.swish = nn.SiLU()

    def forward(self, x, edge_index, W_kron):
        W = []
        for s in range(len(W_kron)):
            W.append(W_kron[s].detach().clone().float().numpy().T)
        D = torch.diag(sum(temp_A)+1).float()
        D_inv = torch.inverse(torch.sqrt(D))
        n = len(subset)
        A=((to_dense_adj(edge_index, max_num_nodes=n)>0)*1)[0].numpy().astype(int)
        A_hat = torch.from_numpy(A).float()+torch.eye(n)
        h0 = x
        # 1
        temp = D_inv@A_hat@D_inv@x
        h1 = self.leakyrelu(temp@W[0]) # y = xW'
        # 2
        temp2 = D_inv@A_hat@D_inv@h1
        h2 = self.elu(temp2@W[1]) # y = xW'
        # 3
        temp3 = D_inv@A_hat@D_inv@h2
        h3 = self.swish(temp3@W[2]) # y = xW'
        # 4
        temp4 = D_inv@A_hat@D_inv@h3
        h4 = self.relu(temp4@W[3]) # y = xW'
        # 5
        temp5 = D_inv@A_hat@D_inv@h4
        h5 = temp5@W[4]
        # 6
        h6 = h5@h5.t()
        z = self.sigmoid(h6)
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
   e = np.zeros( (dimension, 1) ,dtype="float32")
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
    U = np.zeros( (row*col, row*col) ,dtype="float32")
    for i in range(0, row):
        for j in range(0, col):
            U = U + myKron(ElementaryMatrix(row,col,i+1,j+1), ElementaryMatrix(col,row,j+1,i+1))
    return U

#test
#U2 = PermutationMatrix(2,3)

def PermutationRelatedMatrix( row, col ):
    U = np.zeros( (row*row, col*col) ,dtype="float32")
    for i in range(0, row):
        for j in range(0, col):
            U = U + myKron(ElementaryMatrix(row,col,i+1,j+1), ElementaryMatrix(row,col,i+1,j+1))
    return U

#test
#U3 = PermutationRelatedMatrix(2,3)

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
def dyhatijdWs(i,j,s,d,A,Ws,W,H,U,U_bar):
    # dSigma = [dsigmoid(),didentity(),dswish(),delu(),dleakyrelu(),drelu()]
    dSigma = [dleakyrelu(LeakyReLU_negative_slope),delu(ELU_alpha),dswish(),drelu(),didentity(),dsigmoid()]
    dSigma_dplus = dSigma[d]
    dSigma_d = dSigma[d-1]
    Hd = H[d]
    Hd_1 = H[d-1]
    Hdri = np.expand_dims(Hd[i,:],0)
    Hdrj = np.expand_dims(Hd[j,:],0)
    Ari = np.expand_dims(A[i,:],0)
    Arj = np.expand_dims(A[j,:],0)
    Wd = W[d-1]
    ns_1 = np.shape(Ws)[0]
    ns = np.shape(Ws)[1]
    Ins = np.identity(ns)
    Ins_1 = np.identity(ns_1)
    J = np.ones((ns_1, ns))
    # Sigma' d+1
    term1 = dSigma_dplus(Hdri@Hdrj.T)
    # print(np.shape(Ari))
    # print(np.shape(Hd_1))
    # print(np.shape(Wd))
    term2 = myKron(J,dSigma_d(Ari@Hd_1@Wd))*dAHWdWs(i,j,s,d,A,Ws,W,H,dSigma,U,U_bar)
    term3 = myKron(Ins,Hdrj.T)
    term4 = myKron(Ins_1,Hdri)
    term5 = myKron(J,dSigma_d(Arj@Hd_1@Wd).T)
    term6 = dAHWdWsT(i,j,s,d,A,Ws,W,H,dSigma,U,U_bar)
    return term1*(term2@term3+term4@(term5*term6))
    
def dAHWdWs(i,j,s,d,A,Ws,W,H,dSigma,U,U_bar):
    Hd_1 = H[d-1]
    Wd = W[d-1]
    Ari = np.expand_dims(A[i,:],0)
    ns_1 = np.shape(Ws)[0]
    ns = np.shape(Ws)[1]
    Ins = np.identity(ns)
    Ins_1 = np.identity(ns_1)
    if s == d:
        # U_bar = PermutationRelatedMatrix(ns_1,ns)
        return myKron(Ins_1,(Ari@Hd_1))@U_bar[s-1]
    else:
        k = d - 1
        return myKron(Ins_1,Ari)@dHdW(i,j,s,d,A,Ws,W,H,dSigma,k,U,U_bar)@myKron(Ins,Wd)
    
def dAHWdWsT(i,j,s,d,A,Ws,W,H,dSigma,U,U_bar):
    Hd_1 = H[d-1]
    Wd = W[d-1]
    Arj = np.expand_dims(A[j,:],0)
    ns_1 = np.shape(Ws)[0]
    ns = np.shape(Ws)[1]
    Ins = np.identity(ns)
    Ins_1 = np.identity(ns_1)
    if s == d:
        # U = PermutationMatrix(ns_1,ns)
        return U[s-1]@myKron(Ins,(Arj@Hd_1).T)
    else:
        k = d - 1
        return myKron(Ins_1,Wd.T)@dHTdW(i,j,s,d,A,Ws,W,H,dSigma,k,U,U_bar)@myKron(Ins,Arj.T)
    
def dHdW(i,j,s,d,A,Ws,W,H,dSigma,k,U,U_bar):
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
        term2 = myKron(Ins_1,A)@dHdW(i,j,s,d,A,Ws,W,H,dSigma,k-1,U,U_bar)@myKron(Ins,Wk)
        return term1*term2
    
def dHTdW(i,j,s,d,A,Ws,W,H,dSigma,k,U,U_bar):
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
        # U = PermutationMatrix(ns_1,ns)
        term1 = myKron(J,dSigma_k(A@Hk_1@Wk).T)
        term2 = U[s-1]@myKron(Ins,(A@Hk_1).T)
        return term1*term2
    else:
        term1 = myKron(J,dSigma_k(A@Hk_1@Wk).T)
        term2 = myKron(Ins_1,Wk.T)@dHTdW(i,j,s,d,A,Ws,W,H,dSigma,k-1,U,U_bar)@myKron(Ins,A.T)
        return term1*term2
        
#%%
def my_weight_update(output, target, target_neg, H_in, W_kron):
    
  W = []  
  for s in range(len(W_kron)):
        W.append(W_kron[s].detach().clone().float().numpy().T)
      
  H = []
  for s in range(len(H_in)):
       H.append(H_in[s].numpy())
  
  U = []
  for s in range(len(W_kron)):
       ns_1 = np.shape(W[s])[0]
       ns = np.shape(W[s])[1]
       U.append(PermutationMatrix(ns_1,ns))
       
  U_bar = []
  for s in range(len(W_kron)):
       ns_1 = np.shape(W[s])[0]
       ns = np.shape(W[s])[1]
       U_bar.append(PermutationRelatedMatrix(ns_1,ns))

  # ct = 1e-15 # to deal with log(0)
  n = len(subset)
  A_true=((to_dense_adj(target, max_num_nodes=n)>0)*1)[0]
  B=((to_dense_adj(target_neg, max_num_nodes=n)>0)*1)[0]
  mask = A_true+B
  N = mask.sum()
  A = A_true.numpy().astype(int)
  A_hat = torch.from_numpy(A).float()+torch.eye(n)
  D = torch.diag(sum(temp_A)+1).float()
  D_inv = torch.inverse(torch.sqrt(D))
  A = (D_inv@A_hat@D_inv).numpy()
  # loss_true = -sum(sum(A*torch.log(output+ct)))
  # loss_mis = -sum(sum(B*torch.log(1-output+ct)))
  # loss = (loss_true + loss_mis)/mask.sum()

  nr = np.shape(output)[0]
  nc = np.shape(output)[1]
  
  d = len(W)
  gW_pos = []
  gW_neg = []
  for s in range(len(W)):
      gW_pos.append(np.zeros(np.shape(W[s]),dtype="float32"))
      gW_neg.append(np.zeros(np.shape(W[s]),dtype="float32"))
  count = 0
  for i in range(0,nr,1):
    for j in range(0,nc,1):
      # if count % 50 == 0:
      #     print(str(count)+'/'+str(nr*nc))
      # count = count + 1
      if A_true[i,j] == 1:
          # positive edge
          for s in range(len(W)):
              dL1dWs = dyhatijdWs(i,j,s+1,d,A,W[s],W,H,U,U_bar)/output[i,j]
              gW_pos[s] = gW_pos[s] - dL1dWs
      elif B[i,j] == 1:
        # negative edge 
        for s in range(len(W)):
            dL2dWs = dyhatijdWs(i,j,s+1,d,A,W[s],W,H,U,U_bar)/(1-output[i,j])
            # dL2dWs = dyhatijdWs(i,j,s+1,d,A,W[s],W,H,U,U_bar)/(1-output[i,j])
            gW_neg[s] = gW_neg[s] + dL2dWs
      else:
        continue
  gW = []
  for s in range(len(W)):
      gW.append((gW_pos[s] + gW_neg[s])/N)

  return gW

#%%
all_log = []

itr = 0
while(1):
# for itr in range(0, 1060):# run 1060 times
    if itr == 1060:
        break
    #%%  
    device = torch.device('cpu')
    mymodel = GCN().to(device)
    count = 0
    W = []
    W_peak = []
    W_kron = []
    for name, param in mymodel.named_parameters():
        if param.requires_grad:
            W.append(param.data.detach().clone())
            W_kron.append(param.data.detach().clone())
            W_peak.append(param.data)
            # print(name, param.numel())
            count = count + 1
            
    #%%
    n_epochs = 10
    # criterion = torch.nn.BCEWithLogitsLoss()
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # mymodel = GCN().to(device)
    lr = 0.9
    optimizer = torch.optim.SGD(params=mymodel.parameters(), lr=lr)
    W_change = []
    W_temp = []
    for s in range(len(W)):
        W_temp.append(W_peak[s].detach().clone())
    W_change.append(W_temp)
    # W1_change = []
    # W1_change.append(W1_peak.detach().clone())
    # W2_change = []
    # W2_change.append(W2_peak.detach().clone())
    neg_samples_list = []
    outputs = []
    losses = []
    for epoch in range(1, n_epochs + 1):
    
        mymodel.train()
        optimizer.zero_grad()
    
        # sampling training negatives for every training epoch
        neg_edge_index = negative_sampling(
            edge_index=train_edge_index, num_nodes=train_data.num_nodes,
            num_neg_samples=train_data.edge_index.size(1), method='sparse')
    
        neg_samples_list.append(neg_edge_index)
    
        n = len(subset)
        temp_B=((to_dense_adj(neg_edge_index, max_num_nodes=n)>0)*1)[0]
        re_neg_edge_index = temp_B.nonzero().t().contiguous()
    
        edge_label_index = torch.cat(
            [train_edge_index, neg_edge_index],
            dim=-1,
        )
        edge_label = torch.cat([
            train_edge_label,
            train_edge_label.new_zeros(neg_edge_index.size(1))
        ], dim=0)
    
        h, out = mymodel(train_data.x, train_data.edge_index)
        loss = my_loss(out, train_data.edge_index, neg_edge_index)
    
        loss.backward()
        optimizer.step()
        
        W_temp = []
        for s in range(len(W)):
            W_temp.append(W_peak[s].detach().clone())
        W_change.append(W_temp)
    
        outputs.append(out.detach().numpy())
        losses.append(loss.detach().numpy())
    
        # val_auc = accuracy(mymodel, val_data)
        # val_auc = 0
    
        # if epoch % 1 == 0:
        #     print(f"Epoch: {epoch:03d}, Train Loss: {loss:.3f}")
    
    #%%
    n_epochs = 10
    # criterion = torch.nn.BCEWithLogitsLoss()
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # mymodel = GCN().to(device)
    # optimizer = torch.optim.SGD(params=mymodel.parameters(), lr=0.1)
    mymodel_Kron = GCN_Kron().to(device)
    W_change_2 = []
    W_temp = []
    for s in range(len(W)):
        W_temp.append(W_kron[s])
    W_change_2.append(W_temp)
    outputs_2 = []
    losses_2 = []
    for epoch in range(1, n_epochs + 1):
    
        # sampling training negatives for every training epoch
        # neg_edge_index = negative_sampling(
        #     edge_index=train_edge_index, num_nodes=train_data.num_nodes,
        #     num_neg_samples=train_data.edge_label_index.size(1), method='sparse')
    
        neg_edge_index = neg_samples_list[epoch-1]
    
        temp_B=((to_dense_adj(neg_edge_index, max_num_nodes=100)>0)*1)[0]
        re_neg_edge_index = temp_B.nonzero().t().contiguous()
    
        edge_label_index = torch.cat(
            [train_edge_index, neg_edge_index],
            dim=-1,
        )
        edge_label = torch.cat([
            train_edge_label,
            train_edge_label.new_zeros(neg_edge_index.size(1))
        ], dim=0)
    
        # h1, h2, h3, out = mymodel(train_data.x, train_data.edge_index)  
        H, out = mymodel_Kron(train_data.x, train_data.edge_index, W_kron)
        loss = my_loss(out, train_data.edge_index, neg_edge_index)
    
        gW = my_weight_update(out.numpy(), train_data.edge_index, neg_edge_index, H, W_kron)
    
        # Update weight
        for s in range(len(W)):
            W_kron[s] = W_kron[s] - (lr*gW[s]).T
            
        # loss.backward()
        # optimizer.step()
        
        W_temp = []
        for s in range(len(W)):
            W_temp.append(W_kron[s])
        W_change_2.append(W_temp)
    
        outputs_2.append(out.detach().numpy())
        losses_2.append(loss.detach().numpy())
    
        # val_auc = accuracy(mymodel, val_data)
        # val_auc = 0
    
        # if epoch % 1 == 0:
        #     print(f"Epoch: {epoch:03d}, Train Loss: {loss:.3f}")
    
    output_dir = 'D:/PhD thesis/ICML/GCN-Kronecker_product/Link prediction/5-layer general loop (small)'
    np.save(output_dir+'/outputs.npy', outputs)
    np.save(output_dir+'/outputs_2.npy', outputs_2)
    np.save(output_dir+'/losses.npy', losses)
    np.save(output_dir+'/losses_2.npy', losses_2)
    np.save(output_dir+'/W_change.npy', W_change)
    np.save(output_dir+'/W_change_2.npy', W_change_2)
            
    #%%
    W_dif = []
    nan_flag = False
    for i in range(n_epochs+1):
        W_temp = []
        for s in range(len(W)):
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
    plt.figure(figsize=(16,8))
    lines = plt.plot(list(range(n_epochs+1)), np.asarray(all_log).T[s], color=next(color),linewidth=3, marker=next(marker), markersize=20,label='SSE of W' + str(s+1), alpha=0.1)
    plt.setp(lines[1:], label="_")
    # Sum of Squared Errors
    plt.title('Sum of Squared Errors (SSE) of W' + str(s+1) + ' between \nautograd and our method', fontsize=30)
    plt.xlabel('Iterations', fontsize=30)
    plt.ylabel('SSE', fontsize=30)
    plt.legend(loc="best", fontsize=30)
    plt.grid()
    plt.rcParams.update(plt.rcParamsDefault)
    plt.show()

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

color = itertools.cycle(('#0173b2','#de8f05','#029e73','#d55e00','#cc78bc','#ca9161', '#fbafe4', '#949494', '#ece133', '#56b4e9'))
marker = itertools.cycle(('v', 'P', '^', 'o', '<', '*', '>',))
for s in range(len(W)):
    plt.rcParams["xtick.labelsize"] = 20
    plt.rcParams["ytick.labelsize"] = 20
    plt.figure(figsize=(16,8))
    lines = plt.plot(list(range(n_epochs+1)), np.asarray(all_log).T[s], color=next(color),linewidth=3, marker=next(marker), markersize=20,label='SSE of W' + str(s+1), alpha=0.1)
    plt.setp(lines[1:], label="_")
    # Sum of Squared Errors
    plt.title('Sum of Squared Errors (SSE) of W' + str(s+1) + ' between \nautograd and our method', fontsize=30)
    plt.xlabel('Iterations', fontsize=30)
    plt.ylabel('SSE', fontsize=30)
    plt.yscale("log")
    plt.legend(loc="best", fontsize=30)
    plt.grid()
    plt.rcParams.update(plt.rcParamsDefault)
    plt.show()

np.save('all_log.npy', np.asarray(all_log))

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
    axs.set_title('Sum of Squared Errors (SSE) of W' + str(s+1) + ' between \nautograd and our method', fontsize=30)
    
    # adding horizontal grid lines
    axs.yaxis.grid(True)
    axs.set_xticks([y + 1 for y in range(len(np.asarray(all_log).T[s]))],
                  labels=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10'])
    axs.set_xlabel('epochs', fontsize=30)
    axs.set_ylabel('SSE', fontsize=30)
    
    plt.yscale("log")
    plt.rcParams.update(plt.rcParamsDefault)
    plt.show()

#%%
plt.rcParams["xtick.labelsize"] = 20
plt.rcParams["ytick.labelsize"] = 20
fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(9, 4))

# data = []
# for box_i in range(len(np.asarray(all_log).T[0])):
#     data.append()

# plot box plot
# axs.boxplot(np.log10(np.asarray(all_log).T[0]).tolist())
axs.boxplot(np.asarray(all_log).T[0].tolist())
axs.set_title('Box plot')

# adding horizontal grid lines
axs.yaxis.grid(True)
axs.set_xticks([y + 1 for y in range(len(np.asarray(all_log).T[0]))],
              labels=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10'])
axs.set_xlabel('epochs', fontsize=30)
axs.set_ylabel('SSE', fontsize=30)

plt.yscale("log")
plt.rcParams.update(plt.rcParamsDefault)
plt.show()

# #%%
# # https://stackoverflow.com/questions/65116715/different-colours-and-alpha-values-based-on-array-values
# def get_color_negative(v,
#               true_v,
#              ):
#     pos_color=(0,0,0) # black
#     neg_color=(1,0,0) # red
#     scaled = np.array(v)
#     color_list = []
#     i = 0
#     for s in scaled:
#       if true_v[i] == 1:
#         color_list.append(pos_color+(s,))
#       else:
#         color_list.append(neg_color+(s,))
#       i = i + 1
#     return color_list

# #%%
# def make_prob_list(out):
#   prob_list = []
#   for i in range(0,100,1):
#     for j in range(i+1,100,1):
#       prob_list.append(out[i][j].item())
#   return prob_list

# #%%
# temp_A_2=((to_dense_adj(graph.edge_index, max_num_nodes=100)>0)*1)[0]
# temp_A_3 = torch.maximum( temp_A_2, temp_A_2.T )
# print(temp_A_3)
# true_A = temp_A_3

#%%
# from IPython.display import HTML
# from matplotlib import animation
# plt.rcParams["animation.bitrate"] = 3000
# my_G = nx.complete_graph(100)

# def animate(i):
#     # G = to_networkx(data, to_undirected=True)
#     G = my_G
#     nx.draw_networkx(G,
#                     pos=nx.spring_layout(G, seed=0),
#                     with_labels=True,
#                     node_size=800,
#                     node_color="#0173b2",
#                     cmap="hsv",
#                     vmin=-2,
#                     vmax=3,
#                     width=0.8,
#                     edge_color=get_color_negative(make_prob_list(outputs[i]),make_prob_list(true_A)),
#                     font_size=14
#                     )
#     plt.title(f'Epoch {i} | Training loss: {losses[i]:.2f}',
#               fontsize=18, pad=20)

# fig = plt.figure(figsize=(12, 12))
# plt.axis('off')

# anim = animation.FuncAnimation(fig, animate, \
#             np.arange(0, 10, 1), interval=500, repeat=True)

# html = HTML(anim.to_html5_video())

# #%%
# display(html)