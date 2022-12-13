
import matplotlib.pyplot as plt
### MatplotlibRCs
plt.rc('font',family='STIXGeneral',size=20) # STIX looks like latex
plt.rc('mathtext',fontset='stix')
plt.rc('figure', figsize=(1.41421356237*6.,6.) )
plt.rc('figure.subplot', right=0.96464466094,top=0.95 )
plt.rc('lines', linewidth=1.8,marker=None,markersize=8 )
plt.rc('axes', linewidth=1.5,labelsize=24,prop_cycle=plt.cycler(color=('k','r','c','darkorange','steelblue','hotpink','gold','b','maroon','darkgreen')) )
plt.rc(('xtick.major','ytick.major'), size=5.2,width=1.5)
plt.rc(('xtick.minor','ytick.minor'), size=3.2,width=1.5,visible=True)
plt.rc(('xtick','ytick'), labelsize=20, direction='in' )
plt.rc(('xtick'), top=True,bottom=True ) # For some stupid reason you have to do these separately
plt.rc(('ytick'), left=True,right=True )
plt.rc('legend',numpoints=1,scatterpoints=1,labelspacing=0.2,fontsize=18,fancybox=True,handlelength=1.5,handletextpad=0.5)
plt.rc('savefig', dpi=150,format='pdf',bbox='tight' )
plt.rc('errorbar',capsize=3.)

plt.rc('image',cmap='gist_rainbow')
