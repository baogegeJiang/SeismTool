from obspy import read
from glob import glob
from matplotlib import pyplot as plt
import numpy as np
dirName = '/autofs/2022pai/2022_test_code/2022Pai30/02_SuFaceWave_Supression/2021HJQ3D/INPUT/Deonise_B4_Sgy/'
keyXG = 'group_coordinate_x'
keyYG=  'group_coordinate_y'
keyXS = 'source_coordinate_x'
keyYS=  'source_coordinate_y'
traceKey = 'trace_sequence_number_within_line'
sourceKey='energy_source_point_number'
sourceIdD = {}
print(glob(dirName+'*sgy'))
for file in glob(dirName+'*sgy')[:1]:
    st = read(file,unpack_trace_headers=True)
    print(st[0].stats['segy']['trace_header'],st[-1].stats['segy']['trace_header'])


timeL = np.arange(5001/10)*0.01
def handle(st,file):
    with open(file+'.loc','w+') as f:
        for ST in st:
            head = ST.stats['segy']['trace_header']
            sourceId = head[sourceKey]
            if sourceId not in sourceIdD:
                sourceIdD[sourceId]=[]
            ST = ST.copy()
            ST = ST.decimate(10)
            sourceIdD[sourceId].append(ST)
        for sourceId in sourceIdD:
            plt.close()
            st = sourceIdD[sourceId]
            st = st.copy()
            if False:
                head = st[0].stats['segy']['trace_header']
                plt.plot(head[keyXS],head[keyYS],'.r')
                for ST in st:
                    head = ST.stats['segy']['trace_header']
                    plt.plot(head[keyXG],head[keyYG],'.k')
                    f.write('%.1f %.1f %.1f %1.f %d %d\n'%(head[keyXG],head[keyYG],head[keyXS],head[keyYS],head[traceKey],head[sourceKey]))
                plt.savefig('%s.jpg'%sourceId,dpi=300)
            plt.close()
            xg  = np.array([ST.stats['segy']['trace_header'][keyXG]for ST in st])
            yg  = np.array([ST.stats['segy']['trace_header'][keyYG]for ST in st])
            data = np.array([ST.data for ST in st])
            data /= data.max(axis=1)
            Y = yg.copy()
            Y.sort()
            N = len(Y)
            yL = [Y[int(N/3)],Y[int(N/3*2)],Y[-1]]
            print('ploting')
            for i in range(3):
                ymin =yL[i]-50
                ymax =yL[i]+50
                plt.close()
                plt.pcolor(xg[(yg>ymin)*(yg<ymax)],timeL,data[(yg>ymin)*(yg<ymax)].transpose(),cmap='bwr',vmin=-1,vmax=1,rasterized=True)
                plt.xlabel('x/m')
                plt.ylabel('time/s')
                plt.savefig('%s_%d.jpg'%(sourceId,i),dpi=300)

handle(st,file)