
N = 512;
M = 512;
matrix = 7*randn(N, M);
sums = [];
sizes = [];
goldNewSizes = [];

goldRowSet = [];
for i = 1:10
    numRows = floor(rand(1)^2*N/2)+1;
    numCols = floor(rand(1)^2*M/2)+1;
    sizes = [sizes; numCols, numRows];
    
    columnSet = (randperm(M)<=numCols);
    rowSums = matrix * columnSet';
    sums = [sums, rowSums];

    [sortedRowSums orderRowSums] = sort(rowSums,'descend');

    potBcSumsR = cumsum(sortedRowSums);
    potScoresR = LAS_score(potBcSumsR, (1:N)', numCols, N, M);
    [maxPotScoreR numRows] = max(potScoresR);
        
    rowSet = zeros(1, N);
    rowSet(orderRowSums(1:numRows)) = true;
    goldRowSet = [goldRowSet; rowSet];
    goldNewSizes = [goldNewSizes; numCols, numRows];
end;
goldChanges = sum(goldRowSet, 2);

save 'sortSelectHeightAndRows.sums.csv' sums
save 'sortSelectHeightAndRows.sizes.csv' sizes
save 'sortSelectHeightAndRows.goldRowSet.csv' goldRowSet
save 'sortSelectHeightAndRows.goldChanges.csv' goldChanges
save 'sortSelectHeightAndRows.goldNewSizes.csv' goldNewSizes
save 'sortSelectHeightAndRows.matrixSize.csv' M N
