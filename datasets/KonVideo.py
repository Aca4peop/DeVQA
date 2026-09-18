import os

import torch
from torch.utils.data import Dataset
import scipy.io as sio
import numpy as np
import skvideo.io
import cv2
import math
from random import randint

def normalize(image):
    mean = 0.45
    var = 0.225
    image = (image - mean) / var
    return image


def normalize2(image):
    maxx = np.max(image)
    mii = np.min(image)
    image = (image - mii) / (maxx-mii)
    return image



class KonSource():
    def __init__(self):
        self.__files = self.__gen_file_name()
        self.__dmos_normed,self.__dmos = self.__get_dmos()
        self.dmos=self.__dmos
        self.fiveFolds=self.train_test_5fold()

    def train_test_5fold(self):

        num_videos=len(self.__files)
        index = np.array(range(num_videos), dtype=np.int32)
        index = index.astype(int).flatten()
        folds=[]
        fold_indexs=[]

        fold_indexs.append(index[0::5])
        fold_indexs.append(index[1::5])
        fold_indexs.append(index[2::5])
        fold_indexs.append(index[3::5])
        fold_indexs.append(index[4::5])
        del index

        # fold 1
        fold={}
        train_index = np.concatenate((fold_indexs[0],fold_indexs[1],fold_indexs[2],fold_indexs[3],),axis=None)
        np.random.shuffle(train_index)
        test_index = fold_indexs[4]
        fold["train_images"] = self.__image_indexing(train_index)
        fold["train_dmos"] = self.__dmos[train_index]
        fold["train_dmos_normed"] = self.__dmos_normed[train_index]
        fold["test_images"] = self.__image_indexing(test_index)
        fold["test_dmos"] = self.__dmos[test_index]
        fold["test_dmos_normed"] = self.__dmos_normed[test_index]
        folds.append(fold)
        del fold
        # fold 2
        fold={}
        train_index = np.concatenate((fold_indexs[4],fold_indexs[1],fold_indexs[2],fold_indexs[3],),axis=None)
        np.random.shuffle(train_index)
        test_index = fold_indexs[0]
        fold["train_images"] = self.__image_indexing(train_index)
        fold["train_dmos"] = self.__dmos[train_index]
        fold["train_dmos_normed"] = self.__dmos_normed[train_index]
        fold["test_images"] = self.__image_indexing(test_index)
        fold["test_dmos"] = self.__dmos[test_index]
        fold["test_dmos_normed"] = self.__dmos_normed[test_index]
        folds.append(fold)
        del fold
        # fold 3
        fold={}
        train_index = np.concatenate((fold_indexs[4],fold_indexs[0],fold_indexs[2],fold_indexs[3],),axis=None)
        np.random.shuffle(train_index)
        test_index = fold_indexs[1]
        fold["train_images"] = self.__image_indexing(train_index)
        fold["train_dmos"] = self.__dmos[train_index]
        fold["train_dmos_normed"] = self.__dmos_normed[train_index]
        fold["test_images"] = self.__image_indexing(test_index)
        fold["test_dmos"] = self.__dmos[test_index]
        fold["test_dmos_normed"] = self.__dmos_normed[test_index]
        folds.append(fold)
        del fold
        # fold 4
        fold={}
        train_index = np.concatenate((fold_indexs[4],fold_indexs[0],fold_indexs[1],fold_indexs[3],),axis=None)
        np.random.shuffle(train_index)
        test_index = fold_indexs[2]
        fold["train_images"] = self.__image_indexing(train_index)
        fold["train_dmos"] = self.__dmos[train_index]
        fold["train_dmos_normed"] = self.__dmos_normed[train_index]
        fold["test_images"] = self.__image_indexing(test_index)
        fold["test_dmos"] = self.__dmos[test_index]
        fold["test_dmos_normed"] = self.__dmos_normed[test_index]
        folds.append(fold)
        del fold
        # fold 5
        fold={}
        train_index = np.concatenate((fold_indexs[4],fold_indexs[0],fold_indexs[1],fold_indexs[2],),axis=None)
        np.random.shuffle(train_index)
        test_index = fold_indexs[3]
        fold["train_images"] = self.__image_indexing(train_index)
        fold["train_dmos"] = self.__dmos[train_index]
        fold["train_dmos_normed"] = self.__dmos_normed[train_index]
        fold["test_images"] = self.__image_indexing(test_index)
        fold["test_dmos"] = self.__dmos[test_index]
        fold["test_dmos_normed"] = self.__dmos_normed[test_index]
        folds.append(fold)
        del fold
        return folds





    def __gen_file_name(self):
        nameset = []
        mat = sio.loadmat('./datasets/KoNViD-1kinfo.mat')
        name = mat['video_names']
        for i in range(1200):
            nameset.append(name[i][0][0])
        return nameset

    def __get_dmos(self):
        mat = sio.loadmat('./datasets/KoNViD-1kinfo.mat')
        dmos = mat['scores']
        dmos = np.array(dmos).flatten()
        return dmos,dmos

    def __image_indexing(self, index):
        image_new = []
        for i in index:
            image_new.append(self.__files[i])
        return image_new



class KonData(Dataset):  # 继承Dataset
    def __init__(self, root_dir, images, dmos):  # __init__是初始化该类的一些基础参数
        self.root_dir = root_dir  # 文件目录
        self.images = images  # 目录里的所有文件
        self.dmos = dmos

    def __len__(self):  # 返回整个数据集的大小
        return len(self.images)

    def __getitem__(self, index):  # 根据索引index返回dataset[index]
        image_index = self.images[index]  # 根据索引index获取该图片
        img_path = os.path.join(self.root_dir, image_index)  # 获取索引为index的图片的路径名
        v=skvideo.io.vread(img_path,as_grey=False)
        v=np.transpose(v,[0,3,1,2])

        label = self.dmos[index]

        sample = {'image': v, 'label': label,'vname':image_index}  # 根据图片和标签创建字典
        return sample  # 返回该样本


class KonFeat(Dataset):  # 继承Dataset
    def __init__(self, root_dir,images, dmos):  # __init__是初始化该类的一些基础参数
        self.images = images  # 目录里的所有文件
        self.dmos = dmos
        self.__imset = None
        self.root_dir=root_dir

    def __len__(self):  # 返回整个数据集的大小
        return len(self.images)

    def __getitem__(self, index):  # 根据索引index返回dataset[index]
        vname = self.images[index]  # 根据索引index获取该图片
        features=np.load('./konfeats/'+vname[:-4]+'.npy')
        img_path = os.path.join(self.root_dir, vname)  # 获取索引为index的图片的路径名
        v = skvideo.io.vread(img_path, as_grey=False)
        v = np.transpose(v, [ 3,0, 1, 2])
        if not v.shape[1]%2==0:
            features=features[1:]
            v=v[:,1:]
        dmos=self.dmos[index]
        sample = {'feat': features, 'label': dmos,'video':v}
        return sample  # 返回该样本
if __name__=='__main__':
    exit()

