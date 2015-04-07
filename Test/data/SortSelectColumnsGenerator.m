
N = 512;
M = 512;
matrix = 7*randn(N, M);
sums = [];
sizes = [];
goldColumnSet = [];
for i = 1:10
    numRows = floor(rand(1)^2*N/2)+1;
    numCols = floor(rand(1)^2*M/2)+1;
    sizes = [sizes; numCols, numRows];
    
    rowSet = (randperm(N)<=numRows);
    colSums = rowSet * matrix;
    sums = [sums; colSums];

    [sortedColSums orderColSums] = sort(colSums,'descend');
    columnSet = zeros(M, 1);
    columnSet(orderColSums(1:numCols)) = true;
    goldColumnSet = [goldColumnSet, columnSet];
end;
goldChanges = sum(goldColumnSet);

save 'sortSelectColumns.sums.csv' sums
save 'sortSelectColumns.sizes.csv' sizes
save 'sortSelectColumns.goldColumnSet.csv' goldColumnSet
save 'sortSelectColumns.goldChanges.csv' goldChanges
