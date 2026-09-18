# five-fold split for ODV JVQD and 360SVQD
import pickle
import numpy as np
import pandas as pd
import scipy.io as sio

class DataSource():
    def __init__(self):
        self.files,self.scores=self.read_data()
        self.trains,self.evals,self.tests = self.parse_data()
        self.fiveFolds=self.train_test_5fold()
    def read_data(self):
        return
    def parse_data(self):
        indexs=np.argsort(self.scores)
        train_index = np.concatenate((indexs[0::5],indexs[1::5],indexs[2::5],), axis=None)
        eval_index=indexs[3::5]
        test_index=indexs[4::5]
        trains={'files':self.files[train_index],'scores':self.scores[train_index]}
        evals = {'files': self.files[eval_index], 'scores': self.scores[eval_index]}
        tests = {'files': self.files[test_index], 'scores': self.scores[test_index]}
        return trains,evals,tests
    def train_test_5fold(self):

        index=np.argsort(self.scores)
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
        fold["train"] = {'files':self.files[train_index],'scores':self.scores[train_index]}
        fold["test"] = {'files': self.files[test_index], 'scores': self.scores[test_index]}
        folds.append(fold)
        del fold
        # fold 2
        fold={}
        train_index = np.concatenate((fold_indexs[4],fold_indexs[1],fold_indexs[2],fold_indexs[3],),axis=None)
        np.random.shuffle(train_index)
        test_index = fold_indexs[0]
        fold["train"] = {'files': self.files[train_index], 'scores': self.scores[train_index]}
        fold["test"] = {'files': self.files[test_index], 'scores': self.scores[test_index]}
        folds.append(fold)
        del fold
        # fold 3
        fold={}
        train_index = np.concatenate((fold_indexs[4],fold_indexs[0],fold_indexs[2],fold_indexs[3],),axis=None)
        np.random.shuffle(train_index)
        test_index = fold_indexs[1]
        fold["train"] = {'files': self.files[train_index], 'scores': self.scores[train_index]}
        fold["test"] = {'files': self.files[test_index], 'scores': self.scores[test_index]}
        folds.append(fold)
        del fold
        # fold 4
        fold={}
        train_index = np.concatenate((fold_indexs[4],fold_indexs[0],fold_indexs[1],fold_indexs[3],),axis=None)
        np.random.shuffle(train_index)
        test_index = fold_indexs[2]
        fold["train"] = {'files': self.files[train_index], 'scores': self.scores[train_index]}
        fold["test"] = {'files': self.files[test_index], 'scores': self.scores[test_index]}
        folds.append(fold)
        del fold
        # fold 5
        fold={}
        train_index = np.concatenate((fold_indexs[4],fold_indexs[0],fold_indexs[1],fold_indexs[2],),axis=None)
        np.random.shuffle(train_index)
        test_index = fold_indexs[3]
        fold["train"] = {'files': self.files[train_index], 'scores': self.scores[train_index]}
        fold["test"] = {'files': self.files[test_index], 'scores': self.scores[test_index]}
        folds.append(fold)
        del fold
        return folds

    def get_trains(self):
        return self.trains
    def get_evals(self):
        return self.evals
    def get_tests(self):
        return self.tests
    def get_five_folds(self,r):
        train=self.fiveFolds[r]["train"]
        eval=self.fiveFolds[r]["test"]
        return train,eval

    def get_all(self):
        return {'files':self.files,'scores':self.scores}

class KonVid(DataSource):
    def read_data(self):
        nameset = []
        mat = sio.loadmat('./datasets/KoNViD-1kinfo.mat')
        name = mat['video_names']
        for i in range(1200):
            nameset.append(str(name[i][0][0]))
        dmos = mat['scores']
        dmos = np.array(dmos).flatten()
        return np.array(nameset),dmos

class LiveVQC(DataSource):
    def read_data(self):
        nameset = []
        mat = sio.loadmat('./datasets/vqc_sorted.mat')
        name = mat['video_list']
        for i in range(585):
            nameset.append(str(name[i][0][0]))
        dmos = mat['mos']
        dmos = np.array(dmos).flatten()
        return np.array(nameset),dmos
class CVD2014(DataSource):
    def read_data(self):
        nameset = []
        mat = sio.loadmat('./datasets/CVD_sorted.mat')
        name = mat['video_names']
        for i in range(234):
            nameset.append(str(name[i][0][0]).split('/')[-1])
        dmos = mat['scores']
        dmos = np.array(dmos).flatten()
        return np.array(nameset),dmos

class LIVEVQA(DataSource):
    def read_data(self):
        nameset = []
        mat = sio.loadmat('./datasets/LIVEVIDEOData.mat')
        name = mat['file_name']
        for i in range(160):
            nameset.append(str(name[i][0][0]).replace('.yuv', '.mp4'))
        dmos = mat['dmos_all']
        dmos = np.array(dmos).flatten()
        nameset = np.array(nameset)

        return nameset, dmos



class ODVSource():
    def __init__(self):
        self.__files = self.__gen_file_name()
        self.__dmos = self.__get_dmos()
        self.fiveFolds=self.train_test_5fold()
        self.files=self.__files
        self.dmos=self.__dmos
    #     self.maxv=np.max(self.dmos)+0.0001
    #     self.minv=np.min(self.dmos)
    #
    # def v2l(self,value):
    #     value=(value-self.minv)/(self.maxv-self.minv)
    #     if isinstance(value,float):
    #         value=int(value*5)
    #     else:
    #         value=(value*5).astype(int)
    #     return value


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
        fold["test_images"] = self.__image_indexing(test_index)
        fold["test_dmos"] = self.__dmos[test_index]
        folds.append(fold)
        del fold
        # fold 2
        fold={}
        train_index = np.concatenate((fold_indexs[4],fold_indexs[1],fold_indexs[2],fold_indexs[3],),axis=None)
        np.random.shuffle(train_index)
        test_index = fold_indexs[0]
        fold["train_images"] = self.__image_indexing(train_index)
        fold["train_dmos"] = self.__dmos[train_index]
        fold["test_images"] = self.__image_indexing(test_index)
        fold["test_dmos"] = self.__dmos[test_index]
        folds.append(fold)
        del fold
        # fold 3
        fold={}
        train_index = np.concatenate((fold_indexs[4],fold_indexs[0],fold_indexs[2],fold_indexs[3],),axis=None)
        np.random.shuffle(train_index)
        test_index = fold_indexs[1]
        fold["train_images"] = self.__image_indexing(train_index)
        fold["train_dmos"] = self.__dmos[train_index]
        fold["test_images"] = self.__image_indexing(test_index)
        fold["test_dmos"] = self.__dmos[test_index]
        folds.append(fold)
        del fold
        # fold 4
        fold={}
        train_index = np.concatenate((fold_indexs[4],fold_indexs[0],fold_indexs[1],fold_indexs[3],),axis=None)
        np.random.shuffle(train_index)
        test_index = fold_indexs[2]
        fold["train_images"] = self.__image_indexing(train_index)
        fold["train_dmos"] = self.__dmos[train_index]
        fold["test_images"] = self.__image_indexing(test_index)
        fold["test_dmos"] = self.__dmos[test_index]
        folds.append(fold)
        del fold
        # fold 5
        fold={}
        train_index = np.concatenate((fold_indexs[4],fold_indexs[0],fold_indexs[1],fold_indexs[2],),axis=None)
        np.random.shuffle(train_index)
        test_index = fold_indexs[3]
        fold["train_images"] = self.__image_indexing(train_index)
        fold["train_dmos"] = self.__dmos[train_index]
        fold["test_images"] = self.__image_indexing(test_index)
        fold["test_dmos"] = self.__dmos[test_index]
        folds.append(fold)
        del fold
        return folds

    def __gen_file_name(self):
        nameset = []
        mat = sio.loadmat('./datasets/ODV_dmos_sorted.mat')
        name = mat['videos']
        for i in range(540):
            tmp=name[i]
            tmp = tmp.replace(' ', '')
            tmp=tmp.replace('.yuv', '.mp4')
            nameset.append(tmp)
        return nameset

    def __get_dmos(self):
        mat = sio.loadmat('./datasets/ODV_dmos_sorted.mat')
        dmos = mat['dmos']
        dmos = np.array(dmos).flatten()/100.0
        return dmos

    def __image_indexing(self, index):
        image_new = []
        for i in index:
            image_new.append(self.__files[i])
        return image_new


class JVQDSource():
    def __init__(self):
        self.files = self.__gen_file_name()
        self.__dmos = self.__get_dmos()
        self.fiveFolds = self.train_test_5fold()
        self.dmos = self.__dmos
        self.maxv = np.max(self.dmos)+0.0001
        self.minv = np.min(self.dmos)

    def v2l(self, value):
        value = (value - self.minv) / (self.maxv - self.minv)
        if isinstance(value, float):
            value = int(value * 5)
        else:
            value = (value * 5).astype(int)
        return value

    def train_test_5fold(self):

        num_videos=len(self.files)
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
        fold = {}
        train_index = np.concatenate((fold_indexs[0], fold_indexs[1], fold_indexs[2], fold_indexs[3],), axis=None)
        np.random.shuffle(train_index)
        test_index = fold_indexs[4]
        fold["train_images"] = self.__image_indexing(train_index)
        fold["train_dmos"] = self.__dmos[train_index]
        fold["test_images"] = self.__image_indexing(test_index)
        fold["test_dmos"] = self.__dmos[test_index]
        folds.append(fold)
        del fold
        # fold 2
        fold = {}
        train_index = np.concatenate((fold_indexs[4], fold_indexs[1], fold_indexs[2], fold_indexs[3],), axis=None)
        np.random.shuffle(train_index)
        test_index = fold_indexs[0]
        fold["train_images"] = self.__image_indexing(train_index)
        fold["train_dmos"] = self.__dmos[train_index]
        fold["test_images"] = self.__image_indexing(test_index)
        fold["test_dmos"] = self.__dmos[test_index]
        folds.append(fold)
        del fold
        # fold 3
        fold = {}
        train_index = np.concatenate((fold_indexs[4], fold_indexs[0], fold_indexs[2], fold_indexs[3],), axis=None)
        np.random.shuffle(train_index)
        test_index = fold_indexs[1]
        fold["train_images"] = self.__image_indexing(train_index)
        fold["train_dmos"] = self.__dmos[train_index]
        fold["test_images"] = self.__image_indexing(test_index)
        fold["test_dmos"] = self.__dmos[test_index]
        folds.append(fold)
        del fold
        # fold 4
        fold = {}
        train_index = np.concatenate((fold_indexs[4], fold_indexs[0], fold_indexs[1], fold_indexs[3],), axis=None)
        np.random.shuffle(train_index)
        test_index = fold_indexs[2]
        fold["train_images"] = self.__image_indexing(train_index)
        fold["train_dmos"] = self.__dmos[train_index]
        fold["test_images"] = self.__image_indexing(test_index)
        fold["test_dmos"] = self.__dmos[test_index]
        folds.append(fold)
        del fold
        # fold 5
        fold = {}
        train_index = np.concatenate((fold_indexs[4], fold_indexs[0], fold_indexs[1], fold_indexs[2],), axis=None)
        np.random.shuffle(train_index)
        test_index = fold_indexs[3]
        fold["train_images"] = self.__image_indexing(train_index)
        fold["train_dmos"] = self.__dmos[train_index]
        fold["test_images"] = self.__image_indexing(test_index)
        fold["test_dmos"] = self.__dmos[test_index]
        folds.append(fold)
        del fold
        return folds

    def __gen_file_name(self):
        nameset = []
        mat = sio.loadmat('./datasets/JVQD2.mat')
        name = mat['videos']
        for i in range(60):
            tmp=name[i]
            tmp = tmp.replace(' ', '').replace('\'', '')
            nameset.append(tmp)
        return nameset

    def __get_dmos(self):
        mat = sio.loadmat('./datasets/JVQD2.mat')
        dmos = mat['Jmos']
        dmos = np.array(dmos).flatten()/100.0
        return dmos

    def __image_indexing(self, index):
        image_new = []
        for i in index:
            image_new.append(self.files[i])
        return image_new
