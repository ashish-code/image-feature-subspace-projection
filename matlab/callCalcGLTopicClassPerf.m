% function to call clacGLTopicClassPerf
function callCalcGLTopicClassPerf()
dataSets = {'VOC2006', 'VOC2007', 'VOC2010', 'Scene15', 'Caltech101',};
ccTypes = {'i','e','r'};

dictType = 'universal';
dictSize=  1000;
algo = 'kmeans';
param = '';
method = 'VQ';
colClust = 100;

nDataSets = max(size(dataSets));
nccTypes = max(size(ccTypes));


for iDataSet = 1 : nDataSets
    dataSet = dataSets{iDataSet};
    for iccType = 1 : nccTypes
        ccType = ccTypes{iccType};
        try
            calcGLTopicClassPerf(dataSet,dictType,dictSize,algo,param,method,colClust,ccType);
        catch err
            fprintf('%s\n',err.identifier);
        end
    end
end

end