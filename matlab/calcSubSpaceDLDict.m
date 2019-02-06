% calculate sub-space dictionary
function calcSubSpaceDLDict(dataSet,dictType,dictSize,sampleSize,method)
% function calcSubSpaceDLDict(dataSet,dictType,dictSize,sampleSize,method)
rootDir = '/vol/vssp/diplecs/ash/Data/';
categoryListFileName = 'categoryList.txt';
sampleDir = '/collated/';
dictDir = '/Dictionary/';
mappingDir = '/Mapping/';
intDimMethod = 'MLE';

% initialize matlab
cdir = pwd;
cd ~
startup;
cd (cdir)

SpamsMatlabPath = '/vol/vssp/diplecs/ash/Code/spams-matlab/';
cd (SpamsMatlabPath);
start_spams

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
    dictDataFile = [(rootDir),(dataSet),(dictDir),(dataSet),num2str(dictSize),(dictType),num2str(sampleSize),'dl','neg',method,'.dict'];
    mappingFile = [(rootDir),(dataSet),(mappingDir),(dataSet),(dictType),intDimMethod,method,'.mat'];
    fprintf('%s\n',sampleDataFile);
    % load the sample data file
    callSubSpaceDL(sampleDataFile,dictDataFile,dictSize,dictType,mappingFile);
    fprintf('%s\n',sampleDataFile)
    
elseif strcmp(dictType,'categorical')
    for iCategory = 1 : nCategory
        sampleDataFile = [(rootDir),(dataSet),(sampleDir),(categoryList{iCategory}),num2str(sampleSize),'.cat'];
        dictDataFile = [(rootDir),(dataSet),(dictDir),(categoryList{iCategory}),num2str(dictSize),(dictType),num2str(sampleSize),'dl','neg',method,'.dict'];
        mappingFile = [(rootDir),(dataSet),(mappingDir),(categoryList{iCategory}),(dictType),intDimMethod,method,'.mat'];
        fprintf('%s\n',sampleDataFile);
        callSubSpaceDL(sampleDataFile,dictDataFile,dictSize,dictType,mappingFile);
        fprintf('%s\n',dictDataFile)
    end
elseif strcmp(dictType,'balanced')
    for iCategory = 1 : nCategory
        sampleDataFile = [(rootDir),(dataSet),(sampleDir),(categoryList{iCategory}),num2str(sampleSize),'.bal'];
        dictDataFile = [(rootDir),(dataSet),(dictDir),(categoryList{iCategory}),num2str(dictSize),(dictType),num2str(sampleSize),'dl','neg',method,'.dict'];
        mappingFile = [(rootDir),(dataSet),(mappingDir),(categoryList{iCategory}),(dictType),intDimMethod,method,'.mat'];
        fprintf('%s\n',sampleDataFile);
        callSubSpaceDL(sampleDataFile,dictDataFile,dictSize,dictType,mappingFile);
        fprintf('%s\n',dictDataFile)
    end
end

end

function callSubSpaceDL(sampleDataFile,dictDataFile,dictSize,dictType,mappingFile)
     if exist(dictDataFile,'file')
         return;
     end
    sampleData = load(sampleDataFile);
%     if exist(mappedSampleFile,'file')    
%         rndsample = dlmread(rndSampleFile,' ');
%         mappedSample = dlmread(mappedSampleFile,' ');        
%         sampleSubSpace = out_of_sample_est(sampleData',rndsample,mappedSample);
%     else        
%         mapping = load(mappingFile);
%         sampleSubSpace = out_of_sample(sampleData',mapping);
%     end
    try
        mapping = load(mappingFile);
        sampleSubSpace = out_of_sample(sampleData',mapping);
    catch err1
        fprintf('%s\n',err1.identifier);
        
    end
    
    sampleSubSpace = sampleSubSpace';
    
    param.K = dictSize;
    if strcmp(dictType,'universal')
        param.iter = -36000;
    else
        param.iter = -10000;
    end
    param.batchsize = 1000;
    param.modeParam = 0;
    param.lambda = 10;
    param.mode = 0;
    param.posAlpha = false;
    param.posD = false;
    param.modeD = 0;
    param.iter_updateD = 1;
    param.verbose = true;
    param.numThreads = -1;
    [D] = mexTrainDL(sampleSubSpace,param);  
    % write the dictionary to file
    fprintf('computed dictionary\n');
    dlmwrite(dictDataFile,D,'delimiter',',');
end