function calcSSProjClassPerf(dataSet,dictSize,colClust,ccType)
cdir = pwd;
cd ~;
startup;
cd (cdir);
% ----------------------------------------------------------------
dictType = 'universal';
sampleSize = 100000;
rowClust = colClust;

rootDir = '/vol/vssp/diplecs/ash/Data/';
coeffDir = '/Coeff/';
categoryListFileName = 'categoryList.txt';
imageListDir = '/ImageLists/';
coeffPerfDir = '/CoeffPerf/';

% read the category list in the dataset
categoryListPath = [(rootDir),(dataSet),'/',(categoryListFileName)];
fid = fopen(categoryListPath,'r');
categoryList = textscan(fid,'%s');
categoryList = categoryList{1};
fclose(fid);
%
%
nCategory = size(categoryList,1);
listSize = 30;
%
coeffPerfFileAvg  = [(rootDir),(dataSet),(coeffPerfDir),num2str(dictSize),(dictType),num2str(colClust),ccType,'.ssproj'];
coeffPerfAvg = zeros(nCategory,2);

for iCategory = 1 : nCategory
    category = categoryList{iCategory};
    fprintf('%s\n',categoryList{iCategory});
    listTrainPosFile = [(rootDir),(dataSet),(imageListDir),(category),'Train',num2str(listSize),'.pos'];
    listValPosFile = [(rootDir),(dataSet),(imageListDir),(category),'Val',num2str(listSize),'.pos'];
    listTrainNegFile = [(rootDir),(dataSet),(imageListDir),(category),'Train',num2str(listSize),'.neg'];
    listValNegFile = [(rootDir),(dataSet),(imageListDir),(category),'Val',num2str(listSize),'.neg'];
            
    fid = fopen(listTrainPosFile,'r');listTrainPos = textscan(fid,'%s');fclose(fid);
    listTrainPos = listTrainPos{1};

    fid = fopen(listValPosFile,'r');listValPos = textscan(fid,'%s');fclose(fid);
    listValPos = listValPos{1};

    fid = fopen(listTrainNegFile,'r');listTrainNeg = textscan(fid,'%s');fclose(fid);
    listTrainNeg = listTrainNeg{1};

    fid = fopen(listValNegFile,'r');listValNeg = textscan(fid,'%s');fclose(fid);
    listValNeg = listValNeg{1};

    nListTrainPos = size(listTrainPos,1);
    nListValPos = size(listValPos,1);
    nListTrainNeg = size(listTrainNeg,1);
    nListValNeg = size(listValNeg,1);

    FTrainPosAvg = ones(nListTrainPos,dictSize+1);
    FTrainNegAvg = zeros(nListTrainNeg,dictSize+1);
    FValPosAvg = ones(nListValPos,dictSize+1);
    FValNegAvg = zeros(nListValNeg,dictSize+1);        

    % Train ; Pos
    for iter = 1 : nListTrainPos
        imageName = listTrainPos{iter};
        coeffFilePathAvg = [(rootDir),(dataSet),(coeffDir),imageName,num2str(dictSize),(dictType),num2str(colClust),ccType,'.ssproj'];
        try
            Favg = dlmread(coeffFilePathAvg,',');
        catch err
            disp(err.identifier);
            Favg = zeros(1,dictSize);
        end
        if size(Favg,1) > size(Favg,2)
            Favg = Favg';
        end                 
        FTrainPosAvg(iter,1:dictSize) = Favg;            
    end

    % Val ; Pos
    for iter = 1 : nListValPos
        imageName = listValPos{iter};
        coeffFilePathAvg = [(rootDir),(dataSet),(coeffDir),imageName,num2str(dictSize),(dictType),num2str(colClust),ccType,'.ssproj'];
        try
            Favg = dlmread(coeffFilePathAvg,',');
        catch err
            disp(err.identifier)
            Favg = zeros(1,dictSize);
        end
        if size(Favg,1) > size(Favg,2)
            Favg = Favg';
        end            
        FValPosAvg(iter,1:dictSize) = Favg;           
    end

    % Train ; Neg
    for iter = 1 : nListTrainNeg
        imageName = listTrainNeg{iter};
        coeffFilePathAvg = [(rootDir),(dataSet),(coeffDir),imageName,num2str(dictSize),(dictType),num2str(colClust),ccType,'.ssproj'];
        try
            Favg = dlmread(coeffFilePathAvg,',');
        catch err
            disp(err.identifier);
            Favg = zeros(1,dictSize);
        end
        if size(Favg,1) > size(Favg,2)
            Favg = Favg';
        end           
        FTrainNegAvg(iter,1:dictSize) = Favg;            
    end

    % Val ; Neg
    for iter = 1 : nListValNeg
        imageName = listValNeg{iter};
        coeffFilePathAvg = [(rootDir),(dataSet),(coeffDir),imageName,num2str(dictSize),(dictType),num2str(colClust),ccType,'.ssproj'];
        try
            Favg = dlmread(coeffFilePathAvg,',');
        catch err
            disp(err.identifier);
            Favg = zeros(1,dictSize);
        end
        if size(Favg,1) > size(Favg,2)
            Favg = Favg';
        end            
        FValNegAvg(iter,1:dictSize) = Favg;          
    end

FTrainAvg = [FTrainPosAvg;FTrainNegAvg];
FValAvg = [FValPosAvg;FValNegAvg];

%---------------------------------------------------------------------
% CLASSIFICATION
%---------------------------------------------------------------------
% echo pipeline stage: classification
fprintf('%s\n','classification');


mdl = ClassificationKNN.fit(FTrainAvg(:,1:dictSize),FTrainAvg(:,dictSize+1),'NumNeighbors',10);
predLabel = predict(mdl,FValAvg(:,1:dictSize));

tp = sum(FValAvg(:,dictSize+1) == 1 & predLabel == 1);
fp = sum(FValAvg(:,dictSize+1) == 0 & predLabel == 1);
tn = sum(FValAvg(:,dictSize+1) == 0 & predLabel == 0);
fn = sum(FValAvg(:,dictSize+1) == 1 & predLabel == 0);

prec = tp/(tp+fp);
recall = tp/(tp+fn);

f1 = 2*prec*recall/(prec+recall);

% cdir = pwd;
% cd '/vol/vssp/diplecs/ash/code/libsvm-3.11/matlab/';
% % train the classifier
% svmStruct = svmtrain(FTrainAvg(:,dictSize+1),FTrainAvg(:,1:dictSize), '-h 0');
% % predict label of data using the trained classifier
% [predLabel, ~, predValues] = svmpredict(FValAvg(:,dictSize+1), FValAvg(:,1:dictSize), svmStruct);
% cd (cdir);
% 
% [~, ~, ~, averagePrecision] = precisionRecall(predValues, FValAvg(:,dictSize+1), 'r');
% [~, ~, AUC] = ROCcurve(predValues, FValAvg(:,dictSize+1));
cp = classperf(FValAvg(:,dictSize+1),predLabel);
fprintf('%s\t%f\t%f\n',categoryList{iCategory},f1,cp.CorrectRate);

coeffPerfAvg(iCategory,:) = [f1,cp.CorrectRate];
end

disp(coeffPerfAvg);
dlmwrite(coeffPerfFileAvg,coeffPerfAvg,'delimiter',',');
fprintf('%s\n',coeffPerfFileAvg);
end

