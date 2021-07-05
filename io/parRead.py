from re import L
from torch.utils.data import Dataset, DataLoader

class Reader(Dataset):
    def __init__(self, eL,*args,**kwargs):
        #定义好 image 的路径
        self.eL =eL
        self.args = args
        self.kwargs = kwargs
    def __getitem__(self, index):
        e = self.eL[index]
        res = e.loadPSSacs(*self.args,**self.kwargs)
        return res
    def __len__(self):
        return len(self.eL)

def collate_function(data):
    return data

class StaReader(Reader):
    def __init__(self, eL,func,*args,**kwargs):
        #定义好 image 的路径
        self.eL =eL
        self.func= func
        self.args = args
        self.kwargs = kwargs
    def __getitem__(self, index):
        e = self.eL[index]
        return self.func(e, *self.args,**self.kwargs) 

class TomoCal(Reader):
    def __init__(self, eL,T3L,func,*args,**kwargs):
        #定义好 image 的路径
        self.eL =eL
        self.T3L =T3L
        self.func= func
        self.args = args
        self.kwargs = kwargs
    def __getitem__(self, i):
        eL = self.eL
        T3L = self.T3L
        resL= []
        print('cal ',i)
        for j in range(len(self.eL)):
            if j<=i:
                resL.append(None)
                continue
            resL.append(self.func(eL[i],eL[j],T3L[i],T3L[j], *self.args,**self.kwargs) )
        return  resL

class hander(Dataset):
    def __init__(self, h,L):
        #定义好 image 的路径
        self.h =h
        self.L = L
    def __getitem__(self, index):
        self.h.handleDay(*self.L[index])
        return 0
    def __len__(self):
        return len(self.L)