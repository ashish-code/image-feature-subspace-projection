'''
Created on 16 Jan 2012
Project the ImgWrdMat data using PCA
The dictionary sizes are {32,64,128,256,512,1024,2048,4096,8192,16384}

@author: ag00087
'''

higherDims = [512,1024,2048,4096]
lowerDims = [32,64,128,256,512]

#imports
import numpy as np
from optparse import OptionParser
import sys
from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA
from sklearn.decomposition import ProbabilisticPCA
from sklearn.decomposition import MiniBatchSparsePCA
from sklearn.decomposition import RandomizedPCA

#acquire program arguments
parser = OptionParser()
parser.add_option('-d','--dataset',action='store',type='string',dest='dataset',default='VOC2006',metavar='dataset',help='visual dataset')
parser.add_option('-m','--method',action='store',type='string',dest='method',default='pca',metavar='method',help='the decomposition method to be used {pca, ppca, rpca, kpca, spca')
parser.add_option('-q','--quiet',action='store_false',dest='verbose',default=True)

#global paths
rootDir = '/vol/vssp/diplecs/ash/Data/'
iwmDir = '/ImgWrdMat/'
outputDir = '/PCA/'
universalcb = '/UniversalCB/'
imgidDir = '/ImgIds/'
universalworddictionary = '/UniversalWordDictionary/'


#global variables
catidfname = 'catidlist.txt' # list of categories in the dataset
cbext = '.ucb' # universal codebook
iwmext = '.iwm' # image word matrix
uwdext = '.uwd' # universal word dictionary
iidext = '.iid' # image id

def getCatMap(dataset):
    catidfpath = rootDir+dataset+'/'+catidfname
    catnames = np.genfromtxt(catidfpath,delimiter=',',dtype='|S32',usecols=[0])
    catnum = np.genfromtxt(catidfpath,delimiter=',',dtype=np.int,usecols=[1])
    catmap = dict(zip(catnames,catnum))
    return catmap

def pcaImgWrdMat(highDim,lowDim):
    (options, args) = parser.parse_args(sys.argv[1:]) #@UnusedVariable
    dataset = options.dataset
#    method = options.method
    
    #acquire the category list
    catmap = getCatMap(dataset)
    catList = catmap.keys()
    # the number of categories in category list
#    nCategory = len(catList)
    
    for catName in catList:
        print '%s : %d : %d\n'%(catName,highDim,lowDim)
        catPosFileName = rootDir+dataset+iwmDir+catName+str(highDim)+iwmext
        catPosData = np.loadtxt(catPosFileName, dtype=np.int, delimiter=' ')
        nPosImages = catPosData.shape[0]
        catNegFileName = rootDir+dataset+iwmDir+'NEG'+catName+str(highDim)+iwmext
        catNegData = np.loadtxt(catNegFileName,dtype=np.int,delimiter=' ')
        nNegImages = catNegData.shape[0]
        catData = np.vstack((catPosData,catNegData))
        labels = np.vstack((np.ones((nPosImages,1),np.int),np.zeros((nNegImages,1),np.int)))
        print 'pca...'
        pcaData = PCA(n_components=lowDim).fit(catData).transform(catData)
        pcaData = np.hstack((pcaData,labels))
        pcaDataFileName = rootDir+dataset+outputDir+catName+str(highDim)+str(lowDim)+'.pca'
        np.savetxt(pcaDataFileName, pcaData, fmt='%f', delimiter=' ')
        print 'ppca...'
        ppcaData = ProbabilisticPCA(n_components=lowDim).fit(catData).transform(catData)
        ppcaData = np.hstack((ppcaData,labels))
        ppcaDataFileName = rootDir+dataset+outputDir+catName+str(highDim)+str(lowDim)+'.ppca'
        np.savetxt(ppcaDataFileName, ppcaData, fmt='%f', delimiter=' ')
        print 'rpca...'
        rpcaData = RandomizedPCA(n_components=lowDim).fit(catData).transform(catData)
        rpcaData = np.hstack((rpcaData,labels))
        rpcaDataFileName = rootDir+dataset+outputDir+catName+str(highDim)+str(lowDim)+'.rpca'
        np.savetxt(rpcaDataFileName,rpcaData,fmt='%f',delimiter=' ')
        print 'kpca...'
        kpcaData = KernelPCA(n_components=lowDim).fit(catData).transform(catData)
        kpcaData = np.hstack((kpcaData,labels))
        kpcaDataFileName = rootDir+dataset+outputDir+catName+str(highDim)+str(lowDim)+'.kpca'
        np.savetxt(kpcaDataFileName, kpcaData, fmt='%f', delimiter=' ')      
        print 'spca...'
        spcaData = MiniBatchSparsePCA(n_components=lowDim,n_iter=100).fit(catData).transform(catData)
        spcaData = np.hstack((spcaData,labels))
        spcaDataFileName = rootDir+dataset+outputDir+catName+str(highDim)+str(lowDim)+'.spca'
        np.savetxt(spcaDataFileName, spcaData, fmt='%f', delimiter=' ')
    pass

if __name__=='__main__':
    for highDim in higherDims:
        for lowDim in lowerDims:
            pcaImgWrdMat(highDim,lowDim)
    pass