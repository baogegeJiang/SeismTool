from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
fontsize0=7
styleD={}
styleD['HBDZKX']={'size': 8, 'family': 'sans-serif','sans-serif':'Times New Roman'}
styleD['ZGKX']={'size': 7, 'family': 'sans-serif','sans-serif':'Arial'}
def init(key='ZGKX'):
    plt.switch_backend('pgf')
    plt.rcParams['pgf.texsystem'] ='pdflatex'
    font = styleD[key]
    plt.rc('font', **font)

def setABC(ABC,pos=[0.05,0.95],c='k',m=None,key='ZGKX'):
    #fontsize=styleD['ZGKX']['size']
    xlim = plt.xlim()
    ylim = plt.ylim()
    xpos = (1-pos[0])*xlim[0]+pos[0]*xlim[1]
    ypos = (1-pos[1])*ylim[0]+pos[1]*ylim[1]
    if not isinstance(m,type(None)):
        xpos,ypos=m(xpos,ypos)
        print(xpos,ypos)
    plt.text(xpos,ypos,ABC,verticalalignment='top',horizontalalignment='left',c=c)
def setColorbar(pc,label='',key='ZGKX',pos='bottom'):
    ax=plt.gca()
    ax_divider = make_axes_locatable(ax)
    if pos in ['bottom','top']:
        cax = ax_divider.append_axes(pos, size="7%", pad="10%")
        cbar=plt.colorbar(pc, cax=cax, orientation="horizontal")
    elif pos in ['right','left']:
        cax = ax_divider.append_axes(pos, size="7%", pad="10%")
        cbar=plt.colorbar(pc, cax=cax, orientation="vertical")
    if len(label)>0:
        cbar.set_label(label)