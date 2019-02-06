% calculate sub-space dictionary
function calcSubSpaceKmeans(dataSet,dictType,dictSize,sampleSize,method)
% function calcSubSpaceDLDict(dataSet,dictType,dictSize,sampleSize,method)
rootDir = '/vol/vssp/diplecs/ash/Data/';
categoryListFileName = 'categoryList.txt';
sampleDir = '/collated/';
dictDir = '/Dictionary/';
mappingDir = '/Mapping/';
intDimMethod = 'MLE';

% initialize matlab
% cdir = pwd;
% cd ~
% startup;
% cd (cdir)


% read the category list in the dataset
categoryListPath = [(rootDir),(dataSet),'/',(categoryListFileName)];
fid = fopen(categoryListPath);
categoryList = textscan(fid,'%s');
categoryList = categoryList{1};
fclose(fid);
%
nCategory = size(categoryList,1);

if strcmp(dictType,'universal')
    sampleDataFile = [(rootDir),(dataSet),(sampleDir),(dataSet),num2str(sampleSize),'.uni'];
    dictDataFile = [(rootDir),(dataSet),(dictDir),(dataSet),num2str(dictSize),(dictType),num2str(sampleSize),'kmeans',method,'.dict'];
    mappingFile = [(rootDir),(dataSet),(mappingDir),(dataSet),(dictType),intDimMethod,method,'.mat'];
    rndSampleFile = [(rootDir),(dataSet),(mappingDir),(dataSet),(dictType),intDimMethod,method,'.rnd'];
    mappedSampleFile = [(rootDir),(dataSet),(mappingDir),(dataSet),(dictType),intDimMethod,method,'.ms'];
    fprintf('%s\n',sampleDataFile);
    % load the sample data file
    callSubSpaceKmeans(sampleDataFile,rndSampleFile,mappedSampleFile,dictDataFile,dictSize,mappingFile);
    fprintf('%s\n',sampleDataFile)
    
elseif strcmp(dictType,'categorical')
    for iCategory = 1 : nCategory
        sampleDataFile = [(rootDir),(dataSet),(sampleDir),(categoryList{iCategory}),num2str(sampleSize),'.cat'];
        dictDataFile = [(rootDir),(dataSet),(dictDir),(categoryList{iCategory}),num2str(dictSize),(dictType),num2str(sampleSize),'kmeans',method,'.dict'];
        mappingFile = [(rootDir),(dataSet),(mappingDir),(categoryList{iCategory}),(dictType),intDimMethod,method,'.mat'];
        rndSampleFile = [(rootDir),(dataSet),(mappingDir),(categoryList{iCategory}),(dictType),intDimMethod,method,'.rnd'];
        mappedSampleFile = [(rootDir),(dataSet),(mappingDir),(categoryList{iCategory}),(dictType),intDimMethod,method,'.ms'];
        fprintf('%s\n',sampleDataFile);
        callSubSpaceKmeans(sampleDataFile,rndSampleFile,mappedSampleFile,dictDataFile,dictSize,mappingFile);
        fprintf('%s\n',dictDataFile)
    end
elseif strcmp(dictType,'balanced')
    for iCategory = 1 : nCategory
        sampleDataFile = [(rootDir),(dataSet),(sampleDir),(categoryList{iCategory}),num2str(sampleSize),'.bal'];
        dictDataFile = [(rootDir),(dataSet),(dictDir),(categoryList{iCategory}),num2str(dictSize),(dictType),num2str(sampleSize),'kmeans',method,'.dict'];
        mappingFile = [(rootDir),(dataSet),(mappingDir),(categoryList{iCategory}),(dictType),intDimMethod,method,'.mat'];
        rndSampleFile = [(rootDir),(dataSet),(mappingDir),(categoryList{iCategory}),(dictType),intDimMethod,method,'.rnd'];
        fprintf('%s\n',sampleDataFile);
        callSubSpaceKmeans(sampleDataFile,rndSampleFile,dictDataFile,dictSize,mappingFile);
        fprintf('%s\n',dictDataFile)
    end
end

end

function callSubSpaceKmeans(sampleDataFile,rndSampleFile,mappedSampleFile,dictDataFile,dictSize,mappingFile)
% load the sample data file
    if exist(dictDataFile,'file')
        return;
    end
    sampleData = load(sampleDataFile);
    addpath(genpath('/vol/vssp/diplecs/ash/code/drtoolbox/'));
    if exist(mappedSampleFile,'file')    
        rndsample = dlmread(rndSampleFile,' ');
        mappedSample = dlmread(mappedSampleFile,' ');        
        sampleSubSpace = out_of_sample_est(sampleData',rndsample,mappedSample);
    else        
        mapping = load(mappingFile);
        sampleSubSpace = out_of_sample(sampleData',mapping);
    end
    
    nVec = size(sampleSubSpace,1);
    rndIdx = randsample(nVec,20000);
    sampleSubSpace = sampleSubSpace(rndIdx,:);    
        
    opts = statset('MaxIter',20);
    tic
    [~, C] = kmeans(sampleSubSpace,dictSize,'Start','cluster','EmptyAction','singleton','Options',opts);
    t = toc; fprintf('clustering time: %d',t);
    % write the dictionary to file
    dlmwrite(dictDataFile,C','delimiter',',');
        
end