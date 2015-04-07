
N = 512;
M = 512;
matrix = 7*randn(N, M);
sums = [];
sizes = [];
goldNewSizes = [];
goldMaxScores = [];
goldColumnSet = [];
for i = 1:10
    numRows = floor(rand(1)^2*N/2)+1;
    numCols = floor(rand(1)^2*M/2)+1;
    sizes = [sizes; numCols, numRows];
    
    rowSet = (randperm(N)<=numRows);
    colSums = rowSet * matrix;
    sums = [sums; colSums];

    [sortedColSums orderColSums] = sort(colSums,'descend');

    potBcSumsC = cumsum(sortedColSums);
    potScoresC = LAS_score(potBcSumsC, numRows, (1:M), N, M);
    [maxPotScoreC numCols] = max(potScoresC);
    
    columnSet = zeros(M, 1);
    columnSet(orderColSums(1:numCols)) = true;
    goldColumnSet = [goldColumnSet, columnSet];
    goldNewSizes = [goldNewSizes; numCols, numRows];
    goldMaxScores = [goldMaxScores, maxPotScoreC];
end;
goldChanges = sum(goldColumnSet);

save 'sortSelectWidthAndColumns.sums.csv' sums
save 'sortSelectWidthAndColumns.sizes.csv' sizes
save 'sortSelectWidthAndColumns.goldColumnSet.csv' goldColumnSet
save 'sortSelectWidthAndColumns.goldChanges.csv' goldChanges
save 'sortSelectWidthAndColumns.goldNewSizes.csv' goldNewSizes
save 'sortSelectWidthAndColumns.goldMaxScores.csv' goldMaxScores
save 'sortSelectWidthAndColumns.matrixSize.csv' M N
