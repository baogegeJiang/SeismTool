from matplotlib import pyplot as plt
import obspy
import os
import numpy as np

file = 'CEA/predicted_prob.txt'
showDir = os.path.dirname(file)
N=20
count=0
lines=[
    '/media/jiangyr/1TSSD/eventSac/1288543135.90000_-6.63000_150.33000/AH.ANQ._remove_resp_DISP.BHZ,     0.0562', 
    '/media/jiangyr/1TSSD/eventSac/1288543135.90000_-6.63000_150.33000/SC.PZH._remove_resp_DISP.BHZ,     0.2095', 
    '/media/jiangyr/1TSSD/eventSac/1288543135.90000_-6.63000_150.33000/HI.WET._remove_resp_DISP.BHZ,     0.5482',
    '/media/jiangyr/1TSSD/eventSac/1288543135.90000_-6.63000_150.33000/GS.JTA._remove_resp_DISP.BHZ,     0.8788',
    '/media/jiangyr/1TSSD/eventSac/1288543135.90000_-6.63000_150.33000/QH.DLH._remove_resp_DISP.BHZ,     0.9145',
    '/media/jiangyr/1TSSD/eventSac/1288543135.90000_-6.63000_150.33000/GS.LTT._remove_resp_DISP.BHZ,     1.0000',
]
with open(file) as f:
    probs =[ float(line.split()[-1])for line in f.readlines()]
    for line in lines:#f.readlines()[:N]:
        sacFile,prob = line.split()
        sacFile = sacFile[:-1]
        prob    = float(prob)
        name = os.path.basename(sacFile)
        net,sta,tmp,comp = name.split('.')
        name = '%s.%s.%s'%(net,sta,comp)
        trace  = obspy.read(sacFile)[0]
        plt.close()
        plt.figure(figsize=[5,2.5])
        plt.gca().set_position([0.2,0.15,0.6,0.7])
        timeL = np.arange(len(trace.data))/trace.stats.sampling_rate+trace.stats.sac['b']
        dist=trace.stats.sac['gcarc']*111.19
        time0=0
        time1=dist/1.8
        timeB=dist/5
        timeE=dist/2.5
        trace.filter('bandpass',freqmin=1/60, freqmax=1/30, corners=4, zerophase=False)
        data=trace.data[(timeL>time0)*(timeL<time1)]
        
        timeLNew= timeL[(timeL>time0)*(timeL<time1)]
        dataS=trace.data[(timeL>timeB)*(timeL<timeE)]
        timeLNewS= timeL[(timeL>timeB)*(timeL<timeE)]

        dataS/=data.std()
        data/=data.std()
        plt.plot(timeLNew,data,'k',linewidth=0.5)
        plt.plot(timeLNewS,dataS,'r',linewidth=0.5)
        plt.xlim([time0,time1])
        plt.xlabel('t/s')
        plt.ylabel('$D_r$')
        #plt.legend()
        plt.text(plt.xlim()[0]+plt.xlim()[1]*0.025,plt.ylim()[1]*0.95,'Prob. : %.3f'%(prob),ha='left',va='top')
        plt.savefig('%s/%d.pdf'%(showDir,count))
        plt.close()
        count+=1
    plt.close()
    plt.figure(figsize=[5,3])
    plt.hist(probs,color='gray',bins=np.arange(-0.000001,1.0001,0.01),cumulative=-1)
    plt.gca().set_position([0.2,0.15,0.6,0.7])
    plt.ylabel('count')
    plt.xlabel('probability')
    plt.savefig(showDir+'/probDis.pdf')
