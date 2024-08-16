import torch.utils.data as data
import json
import os
import cv2
import subprocess
from PIL import Image
import numpy as np
import torch
import pickle
from util import *

class TGdata(data.Dataset):
    def __init__(self,root,img_size=448,transform=None, phase='train', inp_name=None):
        self.root=root
        self.phase = phase
        self.img_list = []
        self.transform = transform
        self.get_anno()
        self.num_classes = 20
        self.inp = np.load(inp_name)
        self.inp_name = inp_name
        self.img_size=img_size

    def get_anno(self):
        list_path=os.path.join(self.root, r'data\TG1\anno\{}_no_rpt.json'.format(self.phase))
        anno = json.load(open(list_path, 'r'))
        self.img_list = anno["data_list"]


    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        #data_list
        annotations = self.img_list[index]
        filename = annotations["img_path"]
        labels = sorted(annotations["gt_label"])
        img = cv2.imread(os.path.join(self.root, 'data/TG1/img', filename))
        fx = self.img_size / img.shape[0]
        fy = self.img_size / img.shape[1]
        out = cv2.resize(img,dsize=None,fx=fx,fy=fy,interpolation=cv2.INTER_LINEAR)
        # print(out.size)
        if self.transform is not None:
            out = self.transform(img)
        # out=out.permute(2, 0, 1)
        target = np.zeros(self.num_classes, np.float32)
        target[labels] = 1
        # 返回图像、文件名、词嵌入矩阵
        # print(out.shape)
        return (out, self.inp), target

