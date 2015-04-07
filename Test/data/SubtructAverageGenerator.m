
N = 512;
M = 512;
matrix = 7*randn(N, M);
columnSets = [];
rowSets = [];
sizes = [];

for j = 1:10
    bcNumRows = floor(rand(1)^2*N/2)+1;
    bcNumCols = floor(rand(1)^2*M/2)+1;
    sizes = [sizes; bcNumCols, bcNumRows];

    columnSet = (randperm(M)<=bcNumCols)';
    rowSet = (randperm(N)<=bcNumRows);

    columnSets = [columnSets, columnSet];
    rowSets = [rowSets; rowSet];
end;

i = randi([1, 10]);

colSums = rowSets(i, :) * matrix;
average = mean(colSums(logical(columnSets(:, i)))/sizes(i, 2))
tmp1 = matrix * columnSets(:, i);
tmp1'
average2 = dot(tmp1, rowSets(i, :));
average2 /= (sizes(i, 1) * sizes(i, 2))
assert(abs(average - average2) < 0.00000001);

originalMatrix = matrix;

matrix(logical(rowSets(i, :)), logical(columnSets(:, i))) = matrix(logical(rowSets(i, :)), logical(columnSets(:, i))) - average;

matrix2 = originalMatrix;
matrix2 = -average * (rowSets(i, :)' * columnSets(:, i)') + matrix2;
assert(matrix = matrix2);

i -= 1;

save 'subtractAverage.matrix.csv' originalMatrix
save 'subtractAverage.sizes.csv' sizes
save 'subtractAverage.columnSet.csv' columnSets
save 'subtractAverage.rowSet.csv' rowSets
save 'subtractAverage.i.csv' i

save 'subtractAverage.goldUpdatedMatrix.csv' matrix
