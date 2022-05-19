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
def setColorbar(pc,label='',key='ZGKX',pos='bottom',isAppend=True,is3D=False,isShow=True,ax=None):
    if ax==None:
        ax=plt.gca()
    ax_divider = make_axes_locatable(ax)
    cax = ax
    if pos in ['bottom','top']:
        if isAppend:
            cax = ax_divider.append_axes(pos, size="7%", pad="12%",)
        if pc!=None:
            cbar=plt.colorbar(pc, cax=cax, orientation="horizontal")
    elif pos in ['right','left']:
        if isAppend:
            cax = ax_divider.append_axes(pos, size="7%", pad="10%")
        if pc!=None:
            cbar=plt.colorbar(pc, cax=cax, orientation="vertical")
    elif pos in ['HBDZKX']:
        if isAppend:
            cax = ax_divider.append_axes('bottom', size="3%", pad="2%")
        if pc!=None:
            cbar=plt.colorbar(pc, cax=cax, orientation="horizontal")
    elif pos in ['HBDZKXPer']:
        if isAppend:
            cax = ax_divider.append_axes('bottom', size="3%", pad="2%")
        if pc!=None:
            cbar=plt.colorbar(pc, cax=cax, orientation="horizontal",ticks=[-0.03,0,0.03])
    if pos in ['ZGKX']:
        if isAppend:
            cax = ax_divider.append_axes('bottom', size="7%", pad="3%")
        if pc!=None:
            cbar=plt.colorbar(pc, cax=cax, orientation="horizontal")
    if pos in ['Surf']:
        if isAppend:
            cax = ax_divider.append_axes('bottom', size="20%", pad="-60%")
        if pc!=None:
            cbar=plt.colorbar(pc, cax=cax, orientation="horizontal")
        #plt.xticks([])
    if len(label)>0:
        if pc!=None :
            cbar.set_label(label)
            if is3D:
                surf=cax.collections[0]
                surf._facecolors2d=surf._facecolors3d
                surf._edgecolors2d=surf._edgecolors3d

def getCAX(pos='bottom',size="7%", pad="10%"):
    ax=plt.gca()
    ax_divider = make_axes_locatable(ax)
    if pos in ['bottom','top']:
        cax = ax_divider.append_axes(pos, size=size, pad=pad)
    elif pos in ['right','left']:
        cax = ax_divider.append_axes(pos, size=size, pad=pad)
    return cax