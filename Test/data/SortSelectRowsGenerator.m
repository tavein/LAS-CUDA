
N = 512;
M = 512;
matrix = 7*randn(N, M);
sums = [];
sizes = [];
goldRowSet = [];
for i = 1:10
    numRows = floor(rand(1)^2*N/2)+1;
    numCols = floor(rand(1)^2*M/2)+1;
    sizes = [sizes; numCols, numRows];
    
    columnSet = (randperm(M)<=numCols);
    rowSums = matrix * columnSet';
    sums = [sums, rowSums];

    [sortedRowSums orderRowSums] = sort(rowSums,'descend');
    rowSet = zeros(1, N);
    rowSet(orderRowSums(1:numRows)) = true;
    goldRowSet = [goldRowSet; rowSet];
end;
goldChanges = sum(goldRowSet, 2);

save 'sortSelectRows.sums.csv' sums
save 'sortSelectRows.sizes.csv' sizes
save 'sortSelectRows.goldRowSet.csv' goldRowSet
save 'sortSelectRows.goldChanges.csv' goldChanges
