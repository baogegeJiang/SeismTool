import numpy as np

def compareST(st0,st1):
    keyXS = 73
    keyYS=  77
    keyXG = 81
    keyYG=  85
    bool0 = np.ones(len(st0.header))
    bool1 = np.ones(len(st1.header))
    j=0
    for i in range(st0.header):
        head0 = st0.header[i]
        if i%1000:
            print('comparing',i,j)
        for j in range(j,len(st1.header)):
            head1 = st0.header[j]
            if head1[keyXS]==head0[keyXS] and head1[keyYS]==head0[keyYS] and head1[keyXG]==head0[keyXG] and head1[keyYG]==head0[keyYG]:
                j=j+1
                break
            else:
                bool1[j]=0
    return bool0,bool1