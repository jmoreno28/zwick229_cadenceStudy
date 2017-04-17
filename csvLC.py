import math as math
import numpy as np
import urllib
import urllib2
import os as os
import sys as sys
import warnings
import fitsio
from fitsio import FITS, FITSHDR
import subprocess
import argparse
import pdb
import pandas

from astropy import units
from astropy.coordinates import SkyCoord

try:
    os.environ['DISPLAY']
except KeyError as Err:
    warnings.warn('No display environment! Using matplotlib backend "Agg"')
    import matplotlib
    matplotlib.use('Agg')
import matplotlib.pyplot as plt

try:
    import kali.lc
except ImportError:
    print 'kali is not setup. Setup kali by sourcing bin/setup.sh'
    sys.exit(1)


#pass a data frame, assume an oscillator period and select sampling frequency [N observations per cycle]    
def reg_sampler(df, ocillatorP, N):
	zwickdata = df
	oscillatorP = 60 #units = days
	time = zwickdata['t']
	length = time.iloc[-1] - time.iloc[0]
	cycles = length/oscillatorP
	freq1 = N * cycles
	sampRate1= np.floor(len(zwickdata)/freq1)
	ones = np.arange(0,len(zwickdata),sampRate1,dtype=int)
	newdf = zwickdata.iloc[ones]
	print("Number of points in the lightcurve")
	print(len(newdf))
	return np.require(newdf.as_matrix(newdf.columns), requirements=['F', 'A', 'W', 'O', 'E'])    
	
def irreg_sampler(df, ocillatorP, N):
	zwickdata = df
	oscillatorP = 60 #units = days
	time = zwickdata['t']
	length = time.iloc[-1] - time.iloc[0]
	cycles = np.floor(length/oscillatorP)
	freq1 = N * cycles
	newdf = zwickdata.sample(freq1)
	newdf = newdf.sort_values(['t'])
	#newdf = zwickdata.iloc[ones]
	print("Number of points in the lightcurve")
	print(len(newdf))
	return np.require(newdf.as_matrix(newdf.columns), requirements=['F', 'A', 'W', 'O', 'E'])  	
	
def smart_irreg_sampler(df, ocillatorP, N):
	zwickdata = df
	oscillatorP = 60 #units = days
	time = zwickdata['t']
	length = time.iloc[-1] - time.iloc[0]
	cycles = np.floor(length/oscillatorP)
	sampRate1= np.floor(len(zwickdata)/cycles)
	ones = np.arange(0,len(zwickdata),sampRate1,dtype=int)
	sample = np.array([])
	constraint = np.random.randint(N-2, N+2)
	for i in range(0,len(ones)-1):
		dummy = np.arange(ones[i],ones[i+1],1,dtype=int)  
		np.random.shuffle(dummy)
		dummy2 = dummy[:constraint]
		sample = np.append(sample,dummy2)
	sample = sample.astype(int)	
	newdf = zwickdata.iloc[sample]
	newdf = newdf.sort_values(['t'])

	print("Number of points in the lightcurve")
	print(len(newdf))
	return np.require(newdf.as_matrix(newdf.columns), requirements=['F', 'A', 'W', 'O', 'E']) 	


class csvLC(kali.lc.lc):

    def _read(self, name, path , skipNrows,downsample,N):
        fileName = name
        fileNameCSV = ''.join([fileName[0:-3], 'csv'])
        filePathCSV = os.path.join(path, fileNameCSV)
        
        dataInFile  = np.loadtxt(filePathCSV,delimiter=',',skiprows=skipNrows)
	zcols = [ 't', 'cadence', 'y', 'yerr']
	df = pandas.DataFrame(data=dataInFile,columns=zcols, dtype=float)
        
        if downsample == True:
        	dataInFile0 = smart_irreg_sampler(df, 60, N)
        	dataInFile = np.empty(dataInFile0.shape)
        	dataInFile = dataInFile0
        irr_downsample =1	
        if irr_downsample == 0:
        	dataInFile0 = irreg_sampler(df, 60, N)
        	dataInFile = np.empty(dataInFile0.shape)
        	dataInFile = dataInFile0	
        	
        #IF STATEMENT DOWNSAMPLE OR NOT 
        self._numCadences = dataInFile.shape[0]

        startT = -1.0
        lineNum = 0
        while startT == -1.0:
            startTCand = dataInFile[lineNum][0]
            startTCandNext = dataInFile[lineNum + 1][0]
            if not np.isnan(startTCand) and not np.isnan(startTCandNext):
                startT = float(startTCand)
                dt = float(startTCandNext) - float(startTCand)
            else:
                lineNum += 1
        self.startT = startT
        self._dt = dt  # Increment between epochs.
        self.cadence = np.require(np.zeros(self.numCadences), requirements=['F', 'A', 'W', 'O', 'E'])
        self.t = np.require(np.zeros(self.numCadences), requirements=['F', 'A', 'W', 'O', 'E'])
        self.terr = np.require(np.zeros(self.numCadences), requirements=['F', 'A', 'W', 'O', 'E'])
        self.x = np.require(np.zeros(self.numCadences), requirements=['F', 'A', 'W', 'O', 'E'])
        self.y = np.require(np.zeros(self.numCadences), requirements=['F', 'A', 'W', 'O', 'E'])
        self.yerr = np.require(np.zeros(self.numCadences), requirements=['F', 'A', 'W', 'O', 'E'])
        self.mask = np.require(np.zeros(self.numCadences), requirements=[
                               'F', 'A', 'W', 'O', 'E'])  # Numpy array of mask values.
         
    	for i in xrange(self.numCadences):
            dataLine = dataInFile[i]
            self.cadence[i] = int(dataLine[1])
            self.t[i] = float(dataLine[0]) - self.startT
            self.terr[i] = float(0)
            self.y[i] = float(dataLine[2])
            self.yerr[i] = float(dataLine[3])
            self.mask[i] = 1.0 
                    
        #self.cadence = np.arange(0, len(dataInFile), dtype=int)
        
       

    def read(self, name, path=None,skipNrows=0, downsample=False, N=50 ,**kwargs):
        self.z = kwargs.get('z', 0.0)
	fileName = name
    	if path is None:
            try:
                self.path = os.environ['DATADIR']
            except KeyError:
                raise KeyError('Environment variable "DATADIR" not set! Please set "DATADIR" to point \
                where csv data should live first...')
        else:
            self.path = path
        filePath = os.path.join(self.path, fileName)
	if downsample is True:
            try:
                self.N = N
            except KeyError:
                raise KeyError('N = total number of points in the lightcurve to read')
        else:
            self.path = path
            self.N = N
        self._name = str(name)  # The name of the light curve (usually the object's name).
        self._band = str(r'Kep')  # The name of the photometric band (eg. HSC-I or SDSS-g etc..).
        self._xunit = r'$t$~(MJD)'  # Unit in which time is measured (eg. s, sec, seconds etc...).
        self._yunit = r'$F$~($\mathrm{e^{-}}$)'  # Unit in which the flux is measured (eg Wm^{-2} etc...).

    	Ret = True

        if Ret == True:
             self._read(name, self.path,skipNrows,downsample, self.N )
        else:
            raise ValueError('csv light curve not found!')
            
        for i in xrange(self._numCadences):
            self.t[i] = self.t[i]/(1.0 + self.z)
        


    def write(self, name, path=None, **kwrags):
        pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-id', '--ID', type=str, default='iizw229015_kepler_q04_q17', help=r'pass filename as -id arg')
    parser.add_argument('-z', '--z', type=float, default='0.3056', help=r'object redshift')
    parser.add_argument('-skiprows', '--skiprows',  type=int, default='0', help=r'how many rows to skip in csv file')
    args = parser.parse_args()

    LC = csvLC(name=args.ID,  z=args.z, skipNrows=args.skiprows)

    LC.plot()
    LC.plotacf()
    LC.plotsf()
    plt.show()
