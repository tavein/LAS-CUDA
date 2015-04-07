
N = 260;
M = 270;
data = 7*randn(N, M) + 0.1*rand(N, M);

numberOfBiclusters = 10;
invocationsPerBicluster = 262;
scoreThreshold = 3.14;

res = LAS_SearchForRedBCs(data, numberOfBiclusters, invocationsPerBicluster, scoreThreshold);
numberOfFoundBiclusters = length(res);

rowSet = [];
columnSet = [];
scores = [];
for i = 1:length(res)
    rowSet = [rowSet; res(i).rows'];
    columnSet = [columnSet, res(i).cols'];
    scores = [scores, res(i).score];
end

save 'wholeAlgo.params.csv' numberOfFoundBiclusters numberOfBiclusters invocationsPerBicluster scoreThreshold
save 'wholeAlgo.matrix.csv' data
save 'wholeAlgo.goldScores.csv' scores
save 'wholeAlgo.goldRowSet.csv' rowSet
save 'wholeAlgo.goldColumnSet.csv' columnSet


