% Comparative analysis of entropy in sub-manifolds projected using
% different dimensionality reduction techniques; on different datasets

function calcSubmanifoldEntropy(alpha)
dataSets = {'VOC2006','VOC2007','VOC2010','Scene15','Caltech101','Caltech256'};
nDataSet = max(size(dataSets));
rootDir = '/vol/vssp/diplecs/ash/Data/';
% initialize matlab
run('~/startup.m');
methods = {
 'PCA';
 'ProbPCA';
 'FactorAnalysis';
 'Isomap';         
 'LandmarkIsomap';
 'LLE';  
 'KernelPCA';
 'NPE';
 'LPP'; 
};
nMethods = size(methods,1);

estmethods = {
 'MDS';   
 'DiffusionMaps';
 'SymSNE';
 'tSNE';
 'SPE';
};
nEstMethods = size(estmethods,1);

dictType = 'categorical';
mappingDir = '/Mapping/';
intDimMethod = 'MLE';
sampleDir = '/collated/';
sampleSize = 100000;
rndSampleSize = 1000;
resultDir = 'Result/';
resultFileName = strcat(rootDir,resultDir,'renyiEntropy.csv');
resultFile = fopen(resultFileName,'w');
fprintf(resultFile,'%s,%s,%s,%s\n','DataSet','Category','SubspaceMethod','RenyiEntropy');

for iDataSet = 1 : nDataSet
    dataSet = dataSets{iDataSet};
    categoryListFileName = 'categoryList.txt';
    categoryListPath = strcat(rootDir,dataSet,'/',categoryListFileName);
    fid = fopen(categoryListPath);
    categoryList = textscan(fid,'%s');
    categoryList = categoryList{1};
    fclose(fid);
    nCategory = size(categoryList,1);
    for iCategory = 1 : nCategory
        category = categoryList{iCategory};
        for iMethod = 1 : nMethods
            method = methods{iMethod};
            mappingFileName = strcat(rootDir,dataSet,mappingDir,category,dictType,intDimMethod,method,'.mat');
            mapping = load(mappingFileName);
            sampleDataFileName = strcat(rootDir,dataSet,sampleDir,category,num2str(sampleSize),'.cat');
            sampleData = load(sampleDataFileName);
            rndIdx = randsample(max(size(sampleData)),rndSampleSize);
            sampleData = sampleData(:,rndIdx);
            sampleData = sampleData';
            subspaceData = out_of_sample(sampleData,mapping);
            intDim = size(subspaceData,2);
            pDist = pdist(subspaceData,'minkowski',intDim);
            try
                Hrenyi = renyi_entro(pDist',alpha);
            catch err
                Hrenyi = NaN;
            end
            fprintf(resultFile,'%s,%s,%s,%f\n',dataSet,category,method,Hrenyi);
            fprintf('%s,%s,%s,%f\n',dataSet,category,method,Hrenyi);
        end
        for iMethod = 1 : nEstMethods
            method = estmethods{iMethod};
%             mappingFileName = strcat(rootDir,dataSet,mappingDir,dataSet,dictType,intDimMethod,method,'.mat');
%             mapping = load(mappingFileName);
            sampleDataFileName = strcat(rootDir,dataSet,sampleDir,category,num2str(sampleSize),'.cat');
            sampleData = load(sampleDataFileName);
            rndIdx = randsample(max(size(sampleData)),rndSampleSize);
            sampleData = sampleData(:,rndIdx);
            sampleData = sampleData';
            rndFileName = strcat(rootDir,dataSet,mappingDir,category,dictType,intDimMethod,method,'.rnd');
            rndData = load(rndFileName);
            mappedFileName = strcat(rootDir,dataSet,mappingDir,category,dictType,intDimMethod,method,'.ms');
            mappedData = load(mappedFileName);
            subspaceData = out_of_sample_est(sampleData,rndData,mappedData);
            intDim = size(subspaceData,2);
            pDist = pdist(subspaceData,'minkowski',intDim);
            try
                Hrenyi = renyi_entro(pDist',alpha);
            catch err
                Hrenyi = NaN;
            end
            fprintf(resultFile,'%s,%s,%s,%f\n',dataSet,category,method,Hrenyi);
            fprintf('%s,%s,%s,%f\n',dataSet,category,method,Hrenyi);
        end
    end
end
fclose(resultFile);

end