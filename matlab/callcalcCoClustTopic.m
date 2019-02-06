% call coclustertopic for different coclustering methods
function callcalcCoClustTopic(dictType,dictSize,algo,algoParam,method,rowClust,colClust)
% initialize matlab
cdir = pwd;
cd ~
startup;
cd (cdir)
%
dataSets = {'VOC2006', 'VOC2007', 'VOC2010', 'Scene15', 'Caltech101', 'Caltech256'};
ccTypes = {'i','e','r'};
sampleSize = 100000;
nDataSets = max(size(dataSets));
nccTypes = max(size(ccTypes));

for iDataSet = 1 : nDataSets
    dataSet = dataSets{iDataSet};
    for iccType = 1 : nccTypes
        ccType = ccTypes{iccType};
        calcCoClustTopic(dataSet,dictType,dictSize,sampleSize,algo,algoParam,method,rowClust,colClust,ccType);
    end
end

end