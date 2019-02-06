% callsubspacedictionary
function callcolclustSubspaceDictionary(dataSet)

dictSizes = [100,500,1000];
colClusts = [5,10,20,30,40,50];
ccTypes = {'i','r'};

for i = 1 : max(size(dictSizes))
    dictSize = dictSizes(i);
    for j = 1 : max(size(colClusts))
        colClust = colClusts(j);
        for k = 1 : max(size(ccTypes))
            ccType = ccTypes{k};
            rowClust = colClust;
            try
                coclustSubspaceDictionary(dataSet,dictSize,rowClust,colClust,ccType);
            catch err
                fprintf('%s\n',err.identifier);
                fprintf('%d\t%d\t%s\n',dictSize,colClust,ccType);
            end
        end
    end
end

