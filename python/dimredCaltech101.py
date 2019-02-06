'''
Created on 4 Jul 2011
Dimensionality Reduction of visual categories using PCA to various
lower dimensions and measuring classification performance at various
lower dimensions
@author: ag00087
'''

import os
import numpy as np
import mdp

rootDir = '/vol/vssp/diplecs/ash/Data/'
dataset = 'Caltech101'
dataDir = '/FeatureMatrix/'
outDir = '/PCA/'    
nSamples = 10000
dataPath = rootDir+dataset+dataDir
outPath = rootDir+dataset+outDir
catFileList = os.listdir(dataPath)
dataExt = catFileList[0].split('.')[1]
    
def dimred(iCategory,catname):
    print iCategory,catname
    catfilename = dataPath+catname+'.'+dataExt
    dataOri = np.loadtxt(catfilename,dtype=np.float)       
    dataOri = dataOri[:,:-2]
    nInputDim = dataOri.shape[1]
    nOutputDim = nInputDim
    pcanode = mdp.nodes.PCANode()
    pcanode.set_input_dim(nInputDim)
    pcanode.set_output_dim(nOutputDim)
    nInputVec = dataOri.shape[0]
    if(nInputVec > nSamples):
        iSamples = nInputVec
    else:
        iSamples = nSamples
    pcanode.train(dataOri[np.random.randint(0, high=nInputVec, size=iSamples),:])
    dataOuttemp = pcanode.execute(dataOri)
    dataOut = np.array(np.round(dataOuttemp).astype(np.int16),dtype=np.int16)
    print dataOut
    outFilename = outPath+catname+'.'+dataExt
    np.savetxt(outFilename, dataOut, delimiter=' ', fmt='%d')
    print '%s written.' %(outFilename)

def main():   
    catList = [catfilename.split('.')[0] for catfilename in catFileList]
    for iCategory,catname in enumerate(catList):
        dimred(iCategory,catname)

if __name__ == '__main__':
    main()