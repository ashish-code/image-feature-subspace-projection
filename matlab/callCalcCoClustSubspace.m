function callCalcCoClustSubspace(rowClust,colClust)
% initialize matlab
cdir = pwd;
cd ~
startup;
cd (cdir)
%
dataSets = {'VOC2006', 'VOC2007', 'VOC2010', 'Scene15'};
ccTypes = {'i','e','r'};
sampleSize = 100000;
dictSize = 1000;
dictType = 'universal';
nDataSets = max(size(dataSets));
nccTypes = max(size(ccTypes));

for iDataSet = 1 : nDataSets
    dataSet = dataSets{iDataSet};
    for iccType = 1 : nccTypes
        ccType = ccTypes{iccType};
        calcCoClustSubspace(dataSet,dictType,dictSize,sampleSize,rowClust,colClust,ccType)
    end
end

end