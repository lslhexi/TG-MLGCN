import json
import numpy as np
import torch
from torch.nn import Parameter

def gen_A(num_classes, t=0.4, adj_file=r'D:\yolo\ML-GCN-master\data\TG1\anno\train_no_rpt.json'):
    '''
    计算共现标签矩阵
    :param num_classes:类别数量
    :param t: 二值化阈值
    :param adj_file:计算共现矩阵的概率的文件路径
    :return:
    '''

    num=np.zeros((num_classes,1))
    _adj=np.zeros((num_classes,num_classes),dtype=int)
    with open(adj_file,'r') as f :
        anno= json.load(f)
        category=len(anno["metainfo"]["categories"])
        data_list=anno["data_list"]
        for annotations in data_list:
            label=annotations["gt_label"]
            for id in label:
                num[id]+=1
            for idx in range(20):
                for idy in range(20):
                    if idx==idy:
                        continue
                    if idx in label and idy in label:
                        _adj[idx][idy] += 1
                        _adj[idy][idx] += 1
    _adj=_adj/2
    _adj = _adj / num
    _adj[_adj < t] = 0
    _adj[_adj >= t] = 1
    _adj = _adj * 0.25 / (_adj.sum(0, keepdims=True) + 1e-6)
    _adj = _adj + np.identity(num_classes, int)
    #print(_adj)
    return _adj

def gen_adj(A):
    '''
    聚合节点特征矩阵
    :param A: 共现标签矩阵
    :return:
    '''
    D = torch.pow(A.sum(1).float(), -0.5)
    D = torch.diag(D)
    adj = torch.matmul(torch.matmul(A, D).t(), D)
    return adj


if __name__ == '__main__':

    _adj = gen_A(20, 0.4)
    A = Parameter(torch.from_numpy(_adj).float())
    adj = gen_adj(A)
    print(adj)