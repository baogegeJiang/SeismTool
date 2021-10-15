from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
fontsize0=7
styleD={}
styleD['HBDZKX']={'font.size': 8, 'font.family': 'sans-serif','font.sans-serif':'Times New Roman',"mathtext.fontset" : "custom"}
styleD['ZGKX']={'font.size': 7, 'font.family': 'sans-serif','font.sans-serif':'Arial',"mathtext.fontset" : "custom"}
def init(key='ZGKX'):
    plt.switch_backend('pgf')
    plt.rcParams['pgf.texsystem'] ='pdflatex'
    plt.rc('text', usetex=False)
    font = styleD[key]
    #plt.rc(font)
    plt.rcParams.update(font)
    plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]

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
    elif pos in ['HBDZKX']:
        cax = ax_divider.append_axes('bottom', size="3%", pad="2%")
        cbar=plt.colorbar(pc, cax=cax, orientation="horizontal")
    elif pos in ['HBDZKXPer']:
        cax = ax_divider.append_axes('bottom', size="3%", pad="2%")
        cbar=plt.colorbar(pc, cax=cax, orientation="horizontal",ticks=[-0.03,0,0.03])
    if pos in ['ZGKX']:
        cax = ax_divider.append_axes('bottom', size="7%", pad="3%")
        cbar=plt.colorbar(pc, cax=cax, orientation="horizontal")
    if pos in ['Surf']:
        cax = ax_divider.append_axes('bottom', size="20%", pad="-60%")
        cbar=plt.colorbar(pc, cax=cax, orientation="horizontal")
        #plt.xticks([])
    if len(label)>0:
        cbar.set_label(label)

def getCAX(pos='bottom',size="7%", pad="10%"):
    ax=plt.gca()
    ax_divider = make_axes_locatable(ax)
    if pos in ['bottom','top']:
        cax = ax_divider.append_axes(pos, size=size, pad=pad)
    elif pos in ['right','left']:
        cax = ax_divider.append_axes(pos, size=size, pad=pad)
    return cax