% compute entropy of image feature vector distribution
function calcSubspaceEntropy(dataSet,dictType,dictSize,sampleSize,method)
% initialize matlab
run('~/startup.m');

rootDir = '/vol/vssp/diplecs/ash/Data/';
coeffDir = '/Coeff/';
categoryListFileName = 'categoryList.txt';
imageListDir = '/ImageLists/';
mappingDir = '/Mapping/';
entropyDir = '/Entropy/';

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
%
intDimMethod = 'MLE';

% loop over each category to compute its decomposition coefficient
% according the dictionary specified

for iCategory = 1 : nCategory
    % load the images from imagelist
    if strcmp(dictType,'universal')
        mappingFile = [(rootDir),(dataSet),(mappingDir),(dataSet),(dictType),intDimMethod,method,'.mat'];
    elseif ismember(dictType,['categorical','balanced'])
        mappingFile = [(rootDir),(dataSet),(mappingDir),(dataSet),(dictType),intDimMethod,method,'.mat'];
    end
    
    if ~ismember(dataSet,['VOC2006','VOC2007','VOC2010'])
        coeffCatDir = [(rootDir),(dataSet),(coeffDir),categoryList{iCategory},'/'];
        if ~exist(coeffCatDir,'dir')
            mkdir(coeffCatDir)
        end
    end
    mapping = load(mappingFile);
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
        
        H = [];
               
        % Train ; Pos
        for iter = 1 : nListTrainPos
            imageName = listTrainPos{iter};
            Hr = callEntropy(imageName,dataSet,dictType,dictSize,sampleSize,method,mapping);
            H = [H ; Hr];
        end
        
%         % Val ; Pos
%         for iter = 1 : nListValPos
%             imageName = listValPos{iter};
%             [Hs,Hr,Ht,Favg] = callEntropy(imageName,dataSet,dictType,dictSize,sampleSize,method,mapping);           
%             H = [H ; Hs,Hr,Ht];
%             Coeffs = [Coeffs ; Favg'];
%         end
%         
%         % Train ; Neg
%         for iter = 1 : nListTrainNeg
%             imageName = listTrainNeg{iter};
%             [Hs,Hr,Ht,Favg] = callEntropy(imageName,dataSet,dictType,dictSize,sampleSize,method,mapping);           
%             H = [H ; Hs,Hr,Ht];
%             Coeffs = [Coeffs ; Favg'];
%         end
%         
%         % Val ; Neg
%         for iter = 1 : nListValNeg
%             imageName = listValNeg{iter};
%             [Hs,Hr,Ht,Favg] = callEntropy(imageName,dataSet,dictType,dictSize,sampleSize,method,mapping);         
%             H = [H ; Hs,Hr,Ht];
%             Coeffs = [Coeffs ; Favg'];
%         end
    end
    disp(H);
    entDescFileName = strcat(rootDir,dataSet,entropyDir,categoryList{iCategory},method,'.entD');
    dlmwrite(entDescFileName,H);    
    
end

end

function Hr = callEntropy(imageName,dataSet,dictType,dictSize,sampleSize,method,mapping)
    rootDir = '/vol/vssp/diplecs/ash/Data/';
    
    dsiftDir = '/DSIFT/';
    imageFilePath = [(rootDir),(dataSet),(dsiftDir),(imageName),'.dsift'];
    imageData = load(imageFilePath);
    imageData = imageData(3:130,:);
    nVec = size(imageData,2);
    nSample = 200;
    if nSample < nVec
        rndidx = randsample(nVec,nSample);
        imageData = imageData(:,rndidx);
    end
    
    try
        imageSubspace = out_of_sample(imageData',mapping);
    catch err
        
        imageSubspace = out_of_sample_est(imageData',mapping);
    end
    dim = size(imageSubspace,2);
    D = pdist(imageSubspace,'minkowski',dim);
%     Hs = shannon_entro(D');
     Hr = renyi_entro(D',2);
%      Hr = Tsallis_entro(D',0);

    
% --------------------------------------------------------------------
%     imageSubspace = imageSubspace';
%     mult = 1;
%     shannon = HShannon_kNN_k_initialization(mult);
%     renyi = HRenyi_kNN_k_initialization(mult);
%     tsallis = HTsallis_kNN_k_initialization(mult);
%     
%     shannon.k = 10;
%     renyi.k = 10;
%     renyi.alpha = 2;
%     tsallis.k = 10;
%     tsallis.alpha = 2;
%     
%     Hs = HShannon_kNN_k_estimation(imageSubspace,shannon);
%     Hr = HRenyi_kNN_k_estimation(imageSubspace,renyi);
%     Ht = HTsallis_kNN_k_estimation(imageSubspace,tsallis);    
end
