function calcSubspaceKmeansCoeff(dataSet,dictType,dictSize)
% function calcKmeansCoeff(dataSet,dictType,dictSize,sampleSize)

% initialize matlab
cdir = pwd;
cd ~;
startup;
cd (cdir);

algo = 'kmeans';
param = '';
sampleSize = 100000;
method = 'PCA';

rootDir = '/vol/vssp/diplecs/ash/Data/';
coeffDir = '/Coeff/';
categoryListFileName = 'categoryList.txt';
dictDir = '/Dictionary/';
imageListDir = '/ImageLists/';
mappingDir = '/Mapping/';

% read the category list in the dataset
categoryListPath = [(rootDir),(dataSet),'/',(categoryListFileName)];
fid = fopen(categoryListPath);
categoryList = textscan(fid,'%s');
categoryList = categoryList{1};
fclose(fid);
%
nCategory = size(categoryList,1);
listSizes = 30;
nListSizes = max(size(listSizes));
intDimMethod = 'MLE';

% loop over each category to compute its decomposition coefficient
% according the dictionary specified

for iCategory = 1 : nCategory
    % load the images from imagelist
    if strcmp(dictType,'universal')
        dictDataFile = [(rootDir),(dataSet),(dictDir),(dataSet),num2str(dictSize),(dictType),num2str(sampleSize),(algo),num2str(param),'.dict'];
        mappingFile = [(rootDir),(dataSet),(mappingDir),(dataSet),(dictType),intDimMethod,method,'.mat'];
        mapping = load(mappingFile);
    elseif ismember(dictType,['categorical','balanced']);
        dictDataFile = [(rootDir),(dataSet),(dictDir),(categoryList{iCategory}),num2str(dictSize),(dictType),num2str(sampleSize),(algo),num2str(param),'.dict'];
    end
    dict = load(dictDataFile);
    if ismember(dataSet,['Scene15','Caltech101','Caltech256'])
        coeffCatDir = [(rootDir),(dataSet),(coeffDir),categoryList{iCategory}];
        if exist(coeffCatDir,'dir') ~= 7
            mkdir(coeffCatDir)
        end
    end
    for iListSize = 1 : nListSizes
        listTrainPosFile = [(rootDir),(dataSet),(imageListDir),categoryList{iCategory},'Train',num2str(listSizes(iListSize)),'.pos'];
        listValPosFile = [(rootDir),(dataSet),(imageListDir),categoryList{iCategory},'Val',num2str(listSizes(iListSize)),'.pos'];
        listTrainNegFile = [(rootDir),(dataSet),(imageListDir),categoryList{iCategory},'Train',num2str(listSizes(iListSize)),'.neg'];
        listValNegFile = [(rootDir),(dataSet),(imageListDir),categoryList{iCategory},'Val',num2str(listSizes(iListSize)),'.neg'];
        
        
        fid = fopen(listTrainPosFile,'r');
        listTrainPos = textscan(fid,'%s');
        fclose(fid);
        listTrainPos = listTrainPos{1};
        
        fid = fopen(listValPosFile,'r');
        listValPos = textscan(fid,'%s');
        fclose(fid);
        listValPos = listValPos{1};
        
        fid = fopen(listTrainNegFile,'r');
        listTrainNeg = textscan(fid,'%s');
        fclose(fid);
        listTrainNeg = listTrainNeg{1};
        
        fid = fopen(listValNegFile,'r');
        listValNeg = textscan(fid,'%s');
        fclose(fid);
        listValNeg = listValNeg{1};
        
        nListTrainPos = size(listTrainPos,1);
        nListValPos = size(listValPos,1);
        nListTrainNeg = size(listTrainNeg,1);
        nListValNeg = size(listValNeg,1);
        
        % DEBUG
    
        % Train ; Pos
        for iter = 1 : nListTrainPos
            imageName = listTrainPos{iter};
            callVQEnc(imageName,dict,dataSet,dictType,dictSize,sampleSize,algo,param,mapping);            
        end
        
        % Val ; Pos
        for iter = 1 : nListValPos
            imageName = listValPos{iter};
            callVQEnc(imageName,dict,dataSet,dictType,dictSize,sampleSize,algo,param,mapping);           
        end
        
        % Train ; Neg
        for iter = 1 : nListTrainNeg
            imageName = listTrainNeg{iter};
            callVQEnc(imageName,dict,dataSet,dictType,dictSize,sampleSize,algo,param,mapping);           
        end
        
        % Val ; Neg
        for iter = 1 : nListValNeg
            imageName = listValNeg{iter};
            callVQEnc(imageName,dict,dataSet,dictType,dictSize,sampleSize,algo,param,mapping);         
        end
    end
end

end

function callVQEnc(imageName,dict,dataSet,dictType,dictSize,sampleSize,algo,param,mapping)
rootDir = '/vol/vssp/diplecs/ash/Data/';
coeffDir = '/Coeff/';
dsiftDir = '/DSIFT/';

% to conform with the nomenclature of OMP and Lasso
method = 'PCA';

coeffFilePathAvg = [(rootDir),(dataSet),(coeffDir),imageName,num2str(dictSize),(dictType),num2str(sampleSize),(algo),num2str(param),(method),num2str(dictSize),'.avg'];
if exist(coeffFilePathAvg,'file')
    return;
end

imageFilePath = [(rootDir),(dataSet),(dsiftDir),(imageName),'.dsift'];
imageData = load(imageFilePath);
imageData = imageData(3:130,:);

imageSubspace = out_of_sample(imageData',mapping);
imageSubspace = imageSubspace';

hvqenc = dsp.VectorQuantizerEncoder('Codebook',dict,'CodewordOutputPort', false,'QuantizationErrorOutputPort', false, ...
            'OutputIndexDataType', 'uint16');
idx = step(hvqenc,imageData);
idx = idx+1;
coeff = zeros(dictSize,size(imageSubspace,2));
for i = 1 : size(imageSubspace,2)
    coeff(idx(i),i) = 1;
end
Favg = mean(coeff,2);
dlmwrite(coeffFilePathAvg,Favg,'delimiter',',');
fprintf('%s\n',coeffFilePathAvg);

end