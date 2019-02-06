'''
Created on 5 Jul 2011

@author: ag00087
'''
# imports
import os
import numpy as np
import mdp
from scipy.cluster.vq import kmeans2,vq
from scikits.learn import svm
from scikits.learn.metrics import roc_curve,precision_recall_curve,auc
from scikits.learn.cross_val import StratifiedKFold
import matplotlib.pyplot as plt
import sys

# global variables
rootDir = '/vol/vssp/diplecs/ash/Data/'
dataset = 'VOC2006'
dataDir = '/FeatureMatrix/'
tempDir = '/Temp/'
resultDir = '/results/'    
nDimSamples = 10000
dataPath = rootDir+dataset+dataDir
tempPath = rootDir+dataset+tempDir
resultPath = rootDir+dataset+resultDir
catFileList = os.listdir(dataPath)
dataExt = catFileList[0].split('.')[1]
catList = [catfilename.split('.')[0] for catfilename in catFileList]
ldims=[2,5,9,16,27,47,81,128]
nDim = len(ldims)
nClusterSamples = 100000 # approximate number of samples from dataset
nIterKmeans = 20   # number of iterations of kmeans2
nCategory = len(catFileList)
codebookext = '.cb'
bofext = '.bof'
metrics = ['auc-roc','map']
nMetrics = len(metrics)
nFold = 10
nCodewords = 1000
kernelType='linear'
catidfname = 'catidlist.txt'
outDir = '/results/'
evalext = '.eval'
# methods
def dimred(iCategory,catname,nOutputDim):
    print iCategory,catname
    catfilename = dataPath+catname+'.'+dataExt
    dataOri = np.loadtxt(catfilename,dtype=np.float64)       
    dataOri = dataOri[:,:-2]
    nInputDim = dataOri.shape[1]
    if(nOutputDim == None): nOutputDim = nInputDim
    pcanode = mdp.nodes.PCANode()
    pcanode.set_input_dim(nInputDim)
    pcanode.set_output_dim(nOutputDim)
    nInputVec = dataOri.shape[0]
    if(nInputVec > nDimSamples):
        iSamples = nInputVec
    else:
        iSamples = nDimSamples
    pcanode.train(dataOri[np.random.randint(0, high=nInputVec, size=iSamples),:])
    dataOut = pcanode.execute(dataOri)
    print '%s projected...' % (catname)
    return dataOut

def getCatMap(dataset):
    catidfpath = rootDir+dataset+'/'+catidfname
    catnames = np.genfromtxt(catidfpath,delimiter=',',dtype='|S32',usecols=[0])
    catnum = np.genfromtxt(catidfpath,delimiter=',',dtype=np.int,usecols=[1])
    catmap = dict(zip(catnames,catnum))
    return catmap

def main():
    try:
        kernelType = sys.argv[1]
    except(IndexError):
        kernelType='linear'
    
    #catmap = getCatMap(dataset)
    #initialise output matrices
    rocauc = np.zeros((nDim,nCategory),dtype=np.float32)
    mapauc = np.zeros((nDim,nCategory),dtype=np.float32)
    
    nSamplesPerCat = int(np.round(nClusterSamples/nCategory))
    for iLDim,ldim in enumerate(ldims):
        #write the lower dimensional projections for each category
        for iCategory,catname in enumerate(catList):
            dataOuttemp = dimred(iCategory,catname,ldim)
            dataOut = np.array(np.round(dataOuttemp).astype(np.int16),dtype=np.int16)
            outFilename = tempPath+catname+'.'+dataExt
            np.savetxt(outFilename, dataOut, delimiter=' ', fmt='%d')
            if(dataOut.shape[0] <= nSamplesPerCat):
                catSample = dataOut
            else:
                rndsample = np.random.randint(0,dataOut.shape[0],nSamplesPerCat)
                catSample = dataOut[rndsample,:]
            if(iCategory==0):
                dataLower = catSample
            else:
                dataLower = np.concatenate((dataLower,catSample),axis=0)
        #cluster random sampled lower dimensional data
        # compute the code-book for the data-set
        [CodeBook,label] = kmeans2(dataLower,nCodewords,iter=nIterKmeans,minit='points',missing='warn') #@UnusedVariable
        # write code-book to file
        cbfilepath = tempPath+dataset+codebookext
        cbfile = open(cbfilepath,'w')
        np.savetxt(cbfile,CodeBook,fmt='%f', delimiter=' ',)
        cbfile.close()
        
        for iCategory,catname in enumerate(catList):
            tempFilename = tempPath+catname+'.'+dataExt
            catData = np.loadtxt(tempFilename, dtype=np.int16, delimiter=' ')
            [catLabel,catDist] = vq(catData,CodeBook) #@UnusedVariable
            catfilePath = dataPath+catname+'.'+dataExt
            catImgId = np.genfromtxt(catfilePath,dtype=np.int,usecols=[-2])
            catId = np.genfromtxt(catfilePath,dtype=np.int,usecols=[-1])[0]
            ImgId = np.unique(catImgId)
            catboffilepath = tempPath+catname+bofext
            imgcount=0
            for imgid in ImgId:
                imgLabel = catLabel[catImgId==imgid]
                [hist,edges] = np.histogram(imgLabel,nCodewords) #@UnusedVariable
                if imgcount==0:
                    dataout = np.hstack((hist.T,imgid,catId))
                else:
                    dataout = np.vstack((dataout,np.hstack((hist.T,imgid,catId))))
                imgcount+=1
            np.savetxt(catboffilepath, dataout, fmt='%d', delimiter=' ', )
        
        select = np.concatenate((np.arange(nCodewords),[nCodewords+1]),axis=1)
        for iCategory,catname in enumerate(catList):
            #posLabel = catmap.get(catname)
            #negLabel = 0
            #read the category data which will positive
            catboffilepath = tempPath+catname+bofext
            catpos = np.genfromtxt(catboffilepath,dtype=np.int)   
            catpos = catpos.take(select,axis=1)
            catpos[:,-1] = 1
            #posLabel = catpos[0][-1]
            catset = set(catList)
            catset.remove(catname)
            firstvisit = True
            for cat in catset: #@UnusedVariable
                catboffilepath = tempPath+catname+bofext
                if(firstvisit):
                    catneg = np.genfromtxt(catboffilepath,dtype=np.int)
                    firstvisit = False
                else : 
                    catneg = np.concatenate((catneg,np.genfromtxt(catboffilepath,dtype=np.int)),axis=0)
                
            #sample the negative data to have equal size as the positive
            nPos = catpos.shape[0]
            nNeg = catneg.shape[0]
            catneg = catneg[np.random.randint(0,nNeg,nPos),:]
            catneg = catneg.take(select,axis=1)
            catneg[:,-1] = -1
            #combine positive and negative data
            data = np.concatenate((catpos,catneg),axis=0)
            
            #shuffle the rows to aid in random selection of train and test
            #np.random.shuffle(data)
            
            X = data[:,:nCodewords]
            y = data[:,nCodewords]
            #labels for cross validation
            
            #y2 = np.where(y!=posLabel,0,y)
            #y2 = np.where(y2==posLabel,1,y2)
            
            #cross-validation
            cv = StratifiedKFold(y, k=nFold)
            #select classifier
            classifier = svm.SVC(kernel=kernelType, probability=True)
            metricstemp = np.zeros((nFold,nMetrics),np.float)
            
            for i, (train, test) in enumerate(cv):
                probas_ = classifier.fit(X[train], y[train]).predict_proba(X[test])
                print y[test]
                print probas_[:,1]
                try:
                    fpr, tpr, thresholds = roc_curve(y[test], probas_[:,1]) #@UnusedVariable
                    roc_auc = auc(fpr, tpr)
                except:
                    roc_auc = 0.
                try:
                    precision, recall, thresholds = precision_recall_curve(y[test], probas_[:,1]) #@UnusedVariable
                    pr_auc = auc(recall,precision)
                except:
                    pr_auc = 0.
                metricstemp[i] = [roc_auc,pr_auc]
                
            rocauc[iLDim,iCategory] = np.mean(metricstemp[0],axis=0)
            mapauc[iLDim,iCategory] = np.mean(metricstemp[1],axis=0)
            print '%s classified...' % (catname)
     
    outPath = rootDir + dataset + outDir + '%s%s%s%s'%('dimensionality',dataset,kernelType,'.svg')
    outPath1 = rootDir + dataset + outDir + '%s%s%s%s' % ('dimensionality',dataset,kernelType,'.npz') 
    plt.figure(0)
    #ax = plt.subplot(111)
    plt.errorbar(np.arange(1,nDim+1), np.mean(rocauc,axis=1), np.std(rocauc,axis=1), fmt = '-', elinewidth=1, marker = 'x', label = 'AUC-ROC')
    plt.errorbar(np.arange(1,nDim+1), np.mean(mapauc,axis=1), np.std(mapauc,axis=1), fmt = '--', elinewidth=1, marker = 'o', label = 'MAP')
    plt.xlabel('Visual Categories')
    plt.ylabel('Performance Metric')
    plt.title('BOF Performance: %s : %s' % (dataset,kernelType))
    plt.legend(loc="lower right")
    #ax.set_xticks()
    #ax.set_xticklabels(ldim,size='small',ha='center')
    plt.savefig(outPath,format='svg')
    try:
        np.savez(outPath1,rocauc,mapauc)
    except:
        print 'unable to write file %s' % (outPath1)

    plt.show()
    plt.close()
if __name__ == '__main__':
    main()