% call calcSubSpaceDlDict for different subspace methods
function callcalcSubSpaceDlDict(dataSet,dictType,dictSize,sampleSize)
% initialize matlab
cdir = pwd;
cd ~
startup;
cd (cdir)

progDir = '/vol/vssp/diplecs/ash/Thesis/';

methods = {
 'PCA';
 'LDA';
 'MDS';
 'ProbPCA';
 'FactorAnalysis';
%'Sammon';
 'Isomap';         
 'LandmarkIsomap';
 'LLE';            
%'Laplacian';       
%'HessianLLE'; 
%'LTSA';                   
 'DiffusionMaps';
 'KernelPCA';
%'KernelLDA';
%'SNE';
 'SymSNE';
 'tSNE';
 'NPE';
 'LPP';
 'SPE';
% 'LLTSA';
%'CCA';
%'MVU';
%'LandmarkMVU';
%'FastMVU';
%'LLC';
%'ManifoldChart';
%'CFA';
%'GPLVM';
%'AutoEncoderRBM';
%'AutoEncoderEA';
%'NCA';
%'MCML';
};
nMethods = size(methods,1);
addpath (progDir);
for iMethod = 1 : nMethods
    try
        method = methods{iMethod};
        disp(method);
        calcSubSpaceDLDict(dataSet,dictType,dictSize,sampleSize,method);
        
    catch err
        fprintf('%s\n',err.identifier);
    end
end

end

