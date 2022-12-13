
from matplotlib.pyplot import plot,show


	### --- Modules --- ###
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import plot,show,savefig,gca,legend
from matplotlib.pyplot import cm
from traceback import print_exc,format_exc

from .mpl_rcs import *

def merge_pdfs(filenames,outfilename,delete=False):
	''' () -> float
	
	'''
	from PyPDF2 import PdfFileMerger, PdfFileReader
	
	merger = PdfFileMerger()
	for filename in filenames:
		with open(filename,'rb') as fd:
			merger.append(PdfFileReader(fd))
	
	while True:
		try:
			merger.write(outfilename)
			break
		except IOError as msg:
			print("Could not write file:",msg)
			raw_input( "Fix it and press Enter to try again" )
	
	if delete:
		from os import remove
		for filename in filenames: remove(filename)
	
	return

def get_log_minorticks(norm_instance):
	'''
	Generate ticks in linspace ranging over orders of magnitude
	and normalize to log for minorticks
	for colorbar.
	'''
	from numpy import where,log10,array,floor,ceil,logical_and
	vmin,vmax = norm_instance.vmin,norm_instance.vmax # Get min and max value of colorbar (in linear space)
	minorticks = array([ suborder*pow(10.,float(order)) for order in range(int(floor(log10(vmin))) , int(ceil(log10(vmax)))) for suborder in range(2,10) ]) # Generate array ranging over orders of magnitude, exclude 10^order, those are majorticks
	minorticks = norm_instance( minorticks[ where( logical_and( minorticks > vmin , minorticks < vmax ) ) ] ) # Take those ticks larger,smaller than min,max resp. and normalise to colorbar	
	
	return minorticks


def make_colormap(seq):
	"""Return a LinearSegmentedColormap
	seq: a sequence of floats and RGB-tuples. The floats should be increasing
	and in the interval (0,1).
	"""
	from matplotlib import colors
	seq = [(None,) * 3, 0.0] + list(seq) + [1.0, (None,) * 3]
	cdict = {'red': [], 'green': [], 'blue': []}
	for i, item in enumerate(seq):
		if isinstance(item, float):
			r1, g1, b1 = seq[i - 1]
			r2, g2, b2 = seq[i + 1]
			cdict['red'].append([item, r1, r2])
			cdict['green'].append([item, g1, g2])
			cdict['blue'].append([item, b1, b2])
	return colors.LinearSegmentedColormap('CustomMap', cdict)

# Register changed colormap
def register_spitler_cmaps():
	import matplotlib.colors as colors
	def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
		new_cmap = colors.LinearSegmentedColormap.from_list('trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),cmap(np.linspace(minval, maxval, n)))
		return new_cmap
	
	cmap = plt.get_cmap('inferno')
	new_cmap = truncate_colormap(cmap, 0.0, 0.92)
	
	plt.register_cmap(name='inferno_spitler', cmap=new_cmap)
register_spitler_cmaps()

def handle_colorbar( cb_lolim,cb_uplim, cmap='default', lognorm=False, prepare=True,plot=False, label='', labelsize=24., ticklabelsize=14., cut_min=None,cut_max=None):
	''' Handle the colorbar
	
	No autorange supported. Think for yourself.
	'''
	from matplotlib.pyplot import get_cmap
	if cmap == 'default':
		cmap = 'gist_rainbow'
	
		
	
	if cmap[-4:] == '_cut':
		if cut_min is None or cut_max is None:
			msg = "if you ask for a _cut map, you have to set cut_min and cut_max (as float in range 0-1)"
			raise ValueError(msg)
			
		cb_cmap = get_cmap( cmap[:-4] )
		
		ind_min = int(cut_min*len(cb_cmap.colors))
		ind_max = int(cut_max*len(cb_cmap.colors))
		
		from matplotlib.colors import ListedColormap
		cb_cmap = ListedColormap(cb_cmap.colors[ind_min:ind_max])
		
		
	else:
		
		cb_cmap = get_cmap( cmap )
	
	
	if cb_lolim > cb_uplim: # colorbar does not like inverted range
		cb_lolim,cb_uplim = cb_uplim,cb_lolim
	
	
	# Define normalize log or not
	if not lognorm:
		from matplotlib.colors import Normalize
		cb_norm = Normalize(vmin=cb_lolim,vmax=cb_uplim)
	else:
		from matplotlib.colors import LogNorm
		from warnings import warn
		if cb_lolim <= 0.:
			warn( "cb_lolim <= 0 with lognorm=True, impossible. Changing cb_lolim to 1e-2." )
			cb_lolim = 1e-2
		cb_norm = LogNorm(vmin=cb_lolim,vmax=cb_uplim)
	
	def cb_color(val):
		return cb_cmap(cb_norm(val))
	
	
	if plot:
		from matplotlib.pyplot import cm,colorbar
		mappable,mappable._A = cm.ScalarMappable(cmap=cb_cmap, norm=cb_norm),[]
		cb = colorbar( mappable )
		# label
		if label != '':
			cb.ax.text(-0.58,0.5,label,rotation=90,va='center',ha='center',fontsize=labelsize)
		# ticks
		cb.ax.tick_params(left='on',labelleft='off',right='on',labelright='on',which='both',width=1.,labelsize=ticklabelsize)
		if lognorm:
			cb.ax.yaxis.set_ticks(get_log_minorticks( cb_norm ), minor=True)
	else:
		cb = None
	
	return cb_color,cb

def dict_set_default(thedict,key,default):
	thedict[key] = thedict.get(key,default)
	return


def set_kwargdict_defaults(indict,defaults,name="indict"):
	"""
	Set values from indict, else defaults, and check if any are left.
	
	Usually: indict = set_kwargdict_defaults( indict, defaults, name="indict" )
	"""
	out = {}
	
	for key in defaults:
		out[key] = indict.pop( key , defaults[key] )
	
	if indict:
		raise KeyError("kwarg key '%s' in dictionary '%s' not recognized"%(next(iter(indict.keys())),name))
	
	return out

def get_filenames(folder,start="",end="",folders=False):
	''' (string,string="",string="") -> list
	Get filenames from folder that start with start AND start with start
	
	if Folders: add folders
	'''
	from os import listdir
	
	#~ filenames = []
	fnames = listdir(folder)
	#~ print fnames
	#~ for fname in fnames:
		#~ if fname[:len(start)] == start and fname[-len(end):] == end: filenames.append(fname)
	return fnames
def get_fnames(*args,**kwargs):
	return get_filenames(*args,**kwargs)




	### --- Manipulate files --- ###
class manipulate_files(object):
	def __init__(self,filestring,fdir=""):
		''' (str,str="")
		By Nathan Grin
		
		Use for manipulating files.
		filename = fdir + filestring
		Functions:
		- read_alllines()
		- write_appline(, writeline)
		'''
		self.filename = fdir+filestring
	
	# Read file and put content into nested list filecontent
	def read_alllines(self,skipto=0):
		''' (int=0) -> list
		Read file 'self.filename' from line 'skipto' on and put content into nested list of strings 'filecontent'
		'''
		my_file = open(self.filename,'r')
		
		raw_filecontent = my_file.readlines()
		filecontent = list(range(len(raw_filecontent)))
		
		for i in range(skipto,len(raw_filecontent)):
			filecontent[i] = raw_filecontent[i].strip().split()
		
		my_file.close()		# Close file
		return filecontent
	
	# Read columns
	def read_cols(self,cols=[0,1]):
		''' (int=[0,1]) -> list
		Read from file, columns cols (default first 2)
		'''
		data,datalist = self.read_alllines(),[[] for i in range(len(cols))]
		for i in cols:
			for row in data: 
				if len(row)>=i: datalist[cols.index(i)].append(row[i])
		return datalist
	
	# Read columns from DAT file of calculated lines generated by FASTWINDs inicalc
	def read_FWdata(self):
		''' () -> list
		Read from file, columns cols (default first 2)
		from DAT file of calculated lines generated by FASTWINDs inicalc
		also read VFTS spectra data
		'''
		data,datalist = self.read_alllines(),[[],[]]
		
		if len(data[2]) == 2: # Convolved data contains 2 cols
			cols=[0,1]
			ran = 1
			rowtotal = len(data)-1 # and is prepended by line #$$$ stating length
		elif len(data[2]) == 6: # FASTWIND data contains 6 cols
			cols=[2,4]
			ran = 0
			rowtotal = 161
		elif len(data[2]) == 3: # VFTS data contains 3 cols
			cols=[0,1]
			ran = 0
			rowtotal = len(data)
		
		from numpy import array
		data = data[ran:]
		data = array(data,dtype='float64').transpose()
		datalist = [ data[cols[0]].tolist(),data[cols[1]].tolist() ]
		return datalist
		
		## OLD sequence
		# start from row ran, run for rowtotal rows
		# FASTWIND + convolved data have 161 rows, spectra depends, resampled data depends
		for i in cols:
			for j in range(ran,rowtotal+ran):
				if len(data[j])>=i: datalist[cols.index(i)].append(float(data[j][i]))
		return datalist
		
	
	
	
	# Append one line ('writeline') to file
	def write_appline(self, writeline):
		''' (str) -> NULL
		Append one line 'writeline' to file 'filename'
		'''
		my_file = open(self.filename,'a')
		my_file.write(writeline)
		my_file.write('\n')
		my_file.close()		# Close file
		#return
	
	def clearfile(self):
		''' (NULL) -> NULL
		Clear file contents
		'''
		open(self.filename,'w').close()
	
		#return
	
	def read_csv_all(self):
		''' (NULL) -> array
		Read file contents
		'''
		from numpy import array
		from csv import reader
		read,datalist = reader(open(self.filename,'rb')),[]
		for data in read: datalist.append([i for i in data])
		datalist = datalist
		return datalist
	
	def read_datafile(self):
		''' (NULL) -> array
		Read file contents,
		return dictionary with keys from first row, per key the column as list
		'''
		datadict = {}
		data = self.read_alllines()
		header = data[0][1:] # Skip the hashtag, which was first element of the header
		data = data[1:]
		for i in range(len(header)):
			datadict[header[i]] = [ data[j][i] for j in range(len(data)) ]
		return datadict
		
	def read_datafile_float(self):
		''' (NULL) -> array
		Read file contents,
		return dictionary with keys from first row, per key the column as list
		'''
		datadict = {}
		data = self.read_alllines()
		header = data[0][1:] # Skip the hashtag, which was first element of the header
		data = data[1:]
		for i in range(len(header)):
			datadict[header[i]] = [ float(data[j][i]) for j in range(len(data)) ]
		return datadict
		
		
	def read_datafile_rows(self):
		''' (NULL) -> array
		Read file contents,
		return dictionary, with keys from first column,
		with as values dictionary with keys from first row
		'''
		datadict = {}
		data = self.read_alllines()
		header = data[0][1:] # Skip the hashtag, which was first element of the header
		#~ print header
		data = data[1:]
		for i in range(len(data)):
			datadict[data[i][0]] = {}
			for j in range(len(header)-1):
				datadict[data[i][0]][header[j+1]] = data[i][j+1]
		return datadict
		
		
	def read_datafile_rows_float(self):
		''' (NULL) -> array
		Read file contents,
		return dictionary, with keys from first column,
		with as values dictionary with keys from first row
		'''
		datadict = {}
		data = self.read_alllines()
		header = data[0][1:] # Skip the hashtag, which was first element of the header
		#~ print header
		data = data[1:]
		for i in range(len(data)):
			datadict[data[i][0]] = {}
			#~ print data[i]
			for j in range(len(header)-1):
				datadict[data[i][0]][header[j+1]] = float(data[i][j+1])
		return datadict
		
	def load_file_bin(self,nhead=1,delimiter=None,confirm_use_binary=False,verbose=False,nonverbose=False,delete=False):
		""" Will load ascii, make binary file if not exists
		    or load binary data file, if it exists
		    
		    if confirm_use_binary: Ask if you want to use the binary else refresh it
		    nhead: # Should one skip the first few lines, e.g. because they contain a header?
		    note: better not save header in binary file, because of load() function
		    
		credit to Selma
		"""
		from os.path import isfile
		from numpy import genfromtxt,save,load
		from os.path import getmtime
		
		
		# Insert filename of your asscii file here
		myfile = self.filename
		
		# The new binary file will be stored with the same name but with the extention.npy iso .any
		mybinfile = myfile.rsplit('.',1)[0]+".npy"
		
		binary_exists = isfile(mybinfile) # Check if .npy exists already...
		
		if binary_exists:
			if getmtime(myfile) > getmtime(mybinfile): # Check if binary is outdated
				if not nonverbose:
					print( " >>> Warning!" )
					print( "     Verbose or not, I will warn you that:" )
					print( "     %s is newer than\n     %s"%(myfile,mybinfile) )
					print( "     This probably means you have updated the ASCII and" )
					print( "     I propose you delete the binary such that I will refresh it" )
					
			
			if confirm_use_binary:
				print( " Refresh binary file %s? "%mybinfile )
				answer = input(""" Type 'yes' or 'y' to refresh, anything else to use: """ )
				if answer == 'yes' or answer == 'y': binary_exists = False
			elif delete:
				print( " >>> Uh Oh!!! I am refreshing the binary file, but did not bother to ask you !! " )
				binary_exists = False
			
			
		if not binary_exists: 
			
			if verbose:
				print( "... Binary file does not yet exist" )
				print( "...    reading ascci file ", myfile )
				print( "...    and storing it for you in binary format" )
				print( "...    patience please, next time will be way faster " )
				print( "...    I promise" )
			
			# If binary does not exist, read the original ascii file:
			if (not isfile(myfile)):  raise IOError( "File %s does not exist "%myfile )
		
			# Read after skipping "nhead" lines, store in a numpy.array
			data = genfromtxt(myfile, delimiter=delimiter,skip_header=nhead)
		
			# Save the numpy array to file in binary format
			save(mybinfile, data)
		
		else:
			
			if verbose: print( "... Great. Binary file exists, reading data directly from: ", mybinfile )
			
			# If binary file exists load it directly.
			data = load(mybinfile)
		
		# You can now access all data with this array
		return data




