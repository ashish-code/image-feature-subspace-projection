% call subspace classification performance function for various subspace
% methods

function callCalcSubspaceClassPerf(dataSet,dictType,dictSize)
methods = {
 'PCA';
 'MDS';
 'ProbPCA';
 'FactorAnalysis';
 'LLE';                   
 'DiffusionMaps';
 'KernelPCA';
 'SymSNE';
 'tSNE';
 'NPE';
 'LPP';
 'SPE';
 'Isomap';         
 'LandmarkIsomap';
};

nMethods = max(size(methods));

missedmethods = [];

for iMethod = 1 : nMethods
    method = methods{iMethod};
    fprintf('%s\n',method);
    try
        calcSubspaceClassPerf(dataSet,dictType,dictSize,method);
    catch err
        fprintf('%s\n',err.identifier);
        missedmethods = [missedmethods ; method];
    end
end
disp(dataSet);
disp(missedmethods);
end