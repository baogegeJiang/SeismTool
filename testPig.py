import numpy as np
from matplotlib import pyplot as plt
P0 = np.zeros(60)+1/60
P1 = np.zeros(90)+0.000000
P2 = np.zeros(60)
for i in range(15):
    P1[i:i+60] = P1[i:i+60]+P0*1/15

for i in range(60):
    P2[i]=P0[max(0,i-14):i+1].sum()

S0 = -(np.log2(P2[:60])*P1[:60]).sum()
S1 = -np.log2(P0[-15:].sum())*(P1[60:].sum())
plt.close()
plt.plot(P0,'b')
plt.plot(P1,'--r')
plt.plot([30,30],[0,1.5/60],'k')
plt.plot([15,15],[0,1.5/60],'--k')
plt.ylim([-0.01,1/60*2])
plt.xlim([0,75])
plt.xlabel('t/minute')
plt.legend(['0','1'])
plt.savefig('P0P1.jpg',dpi=300)
print(S0+S1,S0/np.log2(15)+S1)