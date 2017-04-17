import numpy as np
import math as math
import cmath as cmath
import psutil as psutil
import matplotlib.pyplot as plt
from matplotlib import cm as cm
from matplotlib import gridspec as gridspec
import argparse as argparse
import operator as operator
import warnings as warnings
import copy as copy
import time as time
import pdb
import os as os
import random

import kali.csvLC
#import kali.s82
import kali.carma
import kali.util.mcmcviz as mcmcviz
from kali.util.mpl_settings import set_plot_params
import kali.util.triangle as triangle
import pickle
import pandas



import seaborn as sns
sns.set_palette("colorblind")
sns.set_style("white")
def plotlc(t,y,mask,name):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(t,y*mask, marker = '+',label=name)
    ax.set_xlabel('Time [days]')
    ax.set_ylabel('Flux')
    ax.legend(loc="upper right", fontsize=18,markerscale=0) 


def doubleMADsfromMedian(y,thresh=3.5):
    # warning: this function does not check for NAs
    # nor does it address issues when 
    # more than 50% of your data have identical values
    m = np.median(y)
    abs_dev = np.abs(y - m)
    left_mad = np.median(abs_dev[y <= m])
    right_mad = np.median(abs_dev[y >= m])
    y_mad = left_mad * np.ones(len(y))
    y_mad[y > m] = right_mad
    modified_z_score = 0.6745 * abs_dev / y_mad
    modified_z_score[y == m] = 0
    return np.where(modified_z_score < thresh)[0]
    
# make new kepler flux
def kepFlux(sdsslc_r,sdsslc_g):
    sdss_g = sdsslc_g.y
    sdss_r = sdsslc_r.y
    sdss_t = sdsslc_g.t
    sdss_gerr = sdsslc_g.yerr
    
    if len(sdss_r) != len(sdss_g):
        print sdsslc_r.t[-1], sdsslc_g.t[-1]
        #catch missing g band measurements in the middle of the array 
        for i in range(0, len(sdss_t)):
            tolerance = 0.0002
            missed = np.isclose(sdsslc_r.t, sdss_t[i],tolerance)
            m = np.where(missed == True)[0]
            #print m, i
            if m != i:
                print m, i, len(sdss_t)
                sdss_g = np.insert(sdss_g, m ,0.)
                sdss_gerr = np.insert(sdss_gerr, m ,0.)
                sdss_t = np.insert(sdss_t, m ,sdsslc_r.t[m])
                print sdss_g[m], sdss_t[i], len(sdss_t),len(sdsslc_r.t)
            if len(sdss_r) == len(sdss_g):
                break
    c = pow(sdss_r, 0.2) + pow(sdss_g,0.8)
    fullr = np.where(sdss_g == 0.)[0]
    c[fullr] = sdss_r[fullr]
    print(c[fullr],sdss_r[fullr-1])
    c_err = pow(sdsslc_r.yerr, 0.2) + pow(sdss_gerr,0.8)
    c_t = sdss_t
    #return 0,0,0                   
    return c, c_err, c_t
    
def valOrBlank( i, x, size=2 ):
	if i >= len(x):
		return " "*size 
	else:
		value = x[i] 
		remainder = size - len( str(x[i] ) ) 
		outStr = str(value)
		if remainder > 0: 
			outStr += ( remainder*" ") 
		return outStr
def maxSize( listOfLists ):
	maxSizeOut = 0 
	for li in listOfLists :
		if len( li ) > maxSizeOut : 
			maxSizeOut = len(li ) 
	return maxSizeOut
def maxWidth( listOfLists ):
	maxWidthOut = 0 
	for li in listOfLists:
		for q in li : 
			if len( str(q ) ) > maxWidthOut:
				maxWidthOut = len( str(q) ) 
	return maxWidthOut 
#---------------------------------------------------------------------------------
parser = argparse.ArgumentParser()
#info for data file to be read
parser.add_argument('-id', '--id', type=str, default='iizw229015_kepler_q04_q17', help=r'pass filename as -id arg')
parser.add_argument('-z', '--z', type=float, default='0.3056', help=r'object redshift')
parser.add_argument('-skiprows', '--skiprows',  type=int, default='0', help=r'how many rows to skip in csv file')

# params for CARMA fit and plotting
parser.add_argument('-libcarmaChain', '--lC', type = str, default = 'libcarmaChain', help = r'libcarma Chain Filename')
parser.add_argument('-nsteps', '--nsteps', type = int, default =500, help = r'Number of steps per walker')
parser.add_argument('-nwalkers', '--nwalkers', type = int, default = 25*psutil.cpu_count(logical = True), help = r'Number of walkers')
parser.add_argument('-pMax', '--pMax', type = int, default = 2, help = r'Maximum C-AR order')
parser.add_argument('-pMin', '--pMin', type = int, default = 1, help = r'Minimum C-AR order')
parser.add_argument('-qMax', '--qMax', type = int, default = -1, help = r'Maximum C-MA order')
parser.add_argument('-qMin', '--qMin', type = int, default = -1, help = r'Minimum C-MA order')
parser.add_argument('--plot', dest = 'plot', action = 'store_true', help = r'Show plot?')
parser.add_argument('--no-plot', dest = 'plot', action = 'store_false', help = r'Do not show plot?')
parser.set_defaults(plot = True)
parser.add_argument('-minT', '--minTimescale', type = float, default = 2.0, help = r'Minimum allowed timescale = minTimescale*lc.dt')
parser.add_argument('-maxT', '--maxTimescale', type = float, default = 0.5, help = r'Maximum allowed timescale = maxTimescale*lc.T')
parser.add_argument('-maxS', '--maxSigma', type = float, default = 5.0, help = r'Maximum allowed sigma = maxSigma*var(lc)')
parser.add_argument('--stop', dest = 'stop', action = 'store_true', help = r'Stop at end?')
parser.add_argument('--no-stop', dest = 'stop', action = 'store_false', help = r'Do not stop at end?')
parser.set_defaults(stop = False)
parser.add_argument('--save', dest = 'save', action = 'store_true', help = r'Save files?')
parser.add_argument('--no-save', dest = 'save', action = 'store_false', help = r'Do not save files?')
parser.set_defaults(save = False)
parser.add_argument('--log10', dest = 'log10', action = 'store_true', help = r'Compute distances in log space?')
parser.add_argument('--no-log10', dest = 'log10', action = 'store_false', help = r'Do not compute distances in log space?')
parser.set_defaults(log10 = False)
parser.add_argument('--viewer', dest = 'viewer', action = 'store_true', help = r'Visualize MCMC walkers')
parser.add_argument('--no-viewer', dest = 'viewer', action = 'store_false', help = r'Do not visualize MCMC walkers')
parser.set_defaults(viewer = True)
args = parser.parse_args()

if (args.qMax >= args.pMax):
	raise ValueError('pMax must be greater than qMax')
if (args.qMax == -1):
	args.qMax = args.pMax - 1
if (args.qMin == -1):
	args.qMin = 0
if (args.pMin < 1):
	raise ValueError('pMin must be greater than or equal to 1')
if (args.qMin < 0):
	raise ValueError('qMin must be greater than or equal to 0')
	

#------------------------------------------------------------------------
#Load target list ids


zwick = np.loadtxt('iizw229015_kepler_q04_q17.csv',delimiter=',')
zcols = [ 't', 'cadence', 'y', 'yerr']
zwickdata = pandas.DataFrame(data=zwick,columns=zcols)
zwickplot = plotlc(zwickdata['t'],zwickdata['y'],mask=1, name = 'Zwicky 229.15')
plt.savefig('zwick.png')

Number = 10
downsampled = False
k2lc = kali.csvLC.csvLC(name='iizw229015_kepler_q04_q17.csv', path='/Users/Jackster/Research/code/k2/python/zwicky', downsample=downsampled,N=Number)

def chop(data,mask):
    mask0 = np.zeros(len(mask))
    days = 100 
    start = np.random.randint(0, high=len(data) - days*30)
    stop = int(days*30)
    print stop
    print len(data)
    sliceofpi = data.loc[start:start+stop]
    length = len(sliceofpi)
    mask0[:length] = 1.0 
    return sliceofpi, mask0
pi, mask = chop(zwickdata, k2lc.mask)
piplot = plotlc(pi['t'],pi['y'],mask=1, name = 'Zwicky 229.15')
plt.savefig('zwick_chunkworksss.png')


plt.clf()

plt.plot(k2lc.t, k2lc.y, "k")
plt.savefig('poop_downsampled.png')

#reset K2LC
#k2lc.mask = mask
#k2lc.y[:len(pi)] = pi['y']

#k2lc.yerr[:len(pi)] = pi['yerr']

#k2lc.t[:len(pi)] = pi['t']-pi['t'].iloc[0]

#w = np.where(k2lc.mask == 1.0)[0]
#plt.clf()
#k2plot = plotlc(k2lc.t[w],k2lc.y[w],mask=1, name = 'Zwicky 229.15')
#plt.savefig('k2zwick_chunk.png')
#plt.clf()
#correct other timescale params 
#sigma = maxSigma*var(lc)
#k2lc.startT = 0.
#k2lc._dt = 100.0## Increment between epochs.
#k2lc._mindt = 0.02
#k2lc._maxdt = 3010.
#k2lc._T = pi['t'].iloc[-1] - pi['t'].iloc[0] ## Total duration of the light curve.
#k2lc._numCadences = len(k2lc.y)
#self.cadence = 0.02

#k2lc.maxT = args.maxT

taskDict = dict()
DICDict= dict()

dataForResultsFile = [] 
for p in xrange(args.pMin, args.pMax + 1):
#	for q in xrange(args.qMin, p):
	q = p-1
	nt = kali.carma.CARMATask(p, q, nwalkers = args.nwalkers, nsteps = args.nsteps )

	print 'Starting libcarma fitting for p = %d and q = %d...'%(p, q)
	startLCARMA = time.time()
	nt.fit(k2lc)
	stopLCARMA = time.time()
	timeLCARMA = stopLCARMA - startLCARMA
	print 'libcarma took %4.3f s = %4.3f min = %4.3f hrs'%(timeLCARMA, timeLCARMA/60.0, timeLCARMA/3600.0)

	Deviances = copy.copy(nt.LnPosterior[:,args.nsteps/2:]).reshape((-1))
	DIC = 0.5*math.pow(np.std(-2.0*Deviances),2.0) + np.mean(-2.0*Deviances)
	print 'C-ARMA(%d,%d) DIC: %+4.3e'%(p, q, DIC)
	DICDict['%d %d'%(p, q)] = DIC
	taskDict['%d %d'%(p, q)] = nt

		#print mid range value and confidence interval for timescales and arma coeffs
		#print r'Rho: ' + str(nt.rootChain[:,10, 200:])
	irand = random.randint(0 , args.nwalkers -1)
	print r'Tau: ' + str(nt.timescaleChain[:,irand, -1])
	labelList = []
	for k in xrange(1, p + 1):
		labelList.append('a$_{%d}$'%(k))
	for u in xrange(0, p):
		labelList.append('b$_{%d}$'%(u))	
		#print labelList

	figTitle = str("zw229")
	if downsampled == True :
		figTitle = str(figTitle +"smartirregular"+str(Number))
	fname = str(figTitle+"-"+'%i-%i'%(p, q)+'Timescales')
	fname2 = str(figTitle+"-"+'%i-%i'%(p, q)+'Chains')
	output = open(fname+'.pkl', 'wb')
	pickle.dump(np.array(taskDict['%i %i'%(p, q)].timescaleChain),output)
	output.close()
	output2 = open(fname2+'.pkl', 'wb')
	pickle.dump(np.array(taskDict['%i %i'%(p, q)].Chain),output2)
	output2.close()

sortedDICVals = sorted(DICDict.items(), key = operator.itemgetter(1))
pBest = int(sortedDICVals[0][0].split()[0])
qBest = int(sortedDICVals[0][0].split()[1])

print(sortedDICVals[0][:])
print(sortedDICVals[1][:])

pNext = int(sortedDICVals[1][0].split()[0])
qNext = int(sortedDICVals[1][0].split()[1])
print(pNext,qNext)

print 'Best model is C-ARMA(%d,%d)'%(pBest, qBest)
lnPrior = nt.logPrior(k2lc)
print 'The log prior is %e'%(lnPrior)
lnLikelihood = nt.logLikelihood(k2lc)
print 'The log likelihood is %e'%(lnLikelihood)
lnPosterior = nt.logPosterior(k2lc)

bestTask = taskDict['%d %d'%(pBest, qBest)]
Theta = bestTask.Chain[:,np.where(bestTask.LnPosterior == np.max(bestTask.LnPosterior))[0][0],np.where(bestTask.LnPosterior == np.max(bestTask.LnPosterior))[1][0]]
nt = kali.carma.CARMATask(pBest, qBest)

#newTask.set(newLC.dt, newTask.bestTheta)
#BEFORE calling newTask.smooth(newTask)
nt.set(k2lc.dt, Theta)
nt.smooth(k2lc)

#-------------------PLOTTING STATS--------------------------
fwid, fhgt = 5 , 5
plt.clf()
#plt.figure(1, figsize=(fwid, fhgt))
#lagsEst, sfEst, sferrEst = k2lc.sf()
#lagsModel, sfModel = bestTask.sf(start=lagsEst[1], stop=lagsEst[-1], num=5000, spacing='log')
#plt.loglog(lagsModel, sfModel, label=r'$SF(\delta t)$ (model)', color='#000000', zorder=5)
#plt.errorbar(lagsEst, sfEst, sferrEst, label=r'$SF(\delta t)$ (est)',
#             fmt='o', capsize=0, color='#ff7f00', markeredgecolor='none', zorder=0)
#plt.xlabel(r'$\log_{10}\delta t$')
#plt.ylabel(r'$\log_{10} SF$')
#plt.legend(loc=2)
#plt.show(True)
#plt.savefig(figTitle + '_SF.png')
plt.clf()

PSDplot = nt.plotpsd(LC = k2lc, doShow = False)
PSDplot.savefig(figTitle+'_PSD.png')
plt.clf()


#FUNCTIONS IN LIBCARMA.PY
#a = nt.plotacvf(LC = k2lc)
#a.savefig(figTitle + '_acvf.png')
#plt.clf()

#b = nt.plotacf(LC = k2lc)
#b.savefig(figTitle + '_acf.png')
#plt.clf()

#-------------------PLOTTING triangles--------------------

labelList = []
labelT = []
labelRoots = []
for k in xrange(1, pBest + 1):
	labelList.append(r"$a_{%d} $" %(k))
	labelT.append(r"$t_{%d}  $ " %(k))
	labelRoots.append(r"$_{%d}  $" %(k))
for u in xrange(0, pBest):
	labelList.append(r"$b _{%d} $" %(u))	
	labelT.append(r"$t_{MA%d}   $" %(u))
	labelRoots.append(r"$_{MA%d} $" %(u))


res_coeffs = mcmcviz.vizTriangle(pBest, qBest, taskDict['%d %d'%(pBest, qBest)].Chain, labelList , str(figTitle+'BestChain'+'%d-%d'%(pBest, qBest)))
plt.clf()
res_times = mcmcviz.vizTriangle(pBest, qBest, taskDict['%d %d'%(pBest, qBest)].timescaleChain, labelT, str(figTitle+'BestTimescales'+'%d-%d'%(pBest, qBest)))
plt.clf()

#reset labels list
pNext = int(sortedDICVals[1][0].split()[0])
qNext = int(sortedDICVals[1][0].split()[1])

labelList2 = []
labelT2 = []
labelRoots2 = []
for k in xrange(1, pNext + 1):
	labelList2.append('a$_{%d}$'%(k))
	labelT2.append('$tau_{%d}$'%(k))
	labelRoots2.append('r$_{%d}$'%(k))
for u in xrange(0, pNext):
	labelList2.append('b$_{%d}$'%(u))	
	labelT2.append('$tau_{MA%d}$'%(u))
	labelRoots2.append('r$_{MA%d}$'%(u))	
#vizTriangle(p, q, Chain, labelList, figTitle, doShow=False)
res_coeffsN = mcmcviz.vizTriangle(pBest, qBest, taskDict['%d %d'%(pNext, qNext)].Chain, labelList2 , str(figTitle+'Chain'+'%d-%d'%(pNext, qNext)),doShow=False)
plt.clf()
res_timesN = mcmcviz.vizTriangle(pBest, qBest, taskDict['%d %d'%(pNext, qNext)].timescaleChain, labelT2, str(figTitle+'Timescales'+'%d-%d'%(pNext, qNext)),doShow=False)
plt.clf()




if args.stop:
    pdb.set_trace()










