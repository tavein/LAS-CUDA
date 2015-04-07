
N = 512;
M = 512;
matrix = 7*randn(N, M);

sizes = [];
initialColumnSets = [];
goldColumnSets = [];
goldRowSets = [];

numIterations = 0;

for i = 1:10
    
    bcNumRows = floor(rand(1)^2*N/2)+1;
    bcNumCols = floor(rand(1)^2*M/2)+1;
    sizes = [sizes; bcNumCols, bcNumRows];

    columnSet = (randperm(M)<=bcNumCols);
    initialColumnSets = [initialColumnSets, columnSet'];

    prevAvg = -Inf;
    currAvg = 0;
    j = 0;
    while(prevAvg~=currAvg)
        prevAvg = currAvg;
        
        % calculate the row sums over the selected columns
        % rowSums = sum(data(:,columnSet),2);
        
        rowSums = matrix*columnSet';

        % sort sums, saving the info about order
        [sortedRowSums orderRowSums] = sort(rowSums,'descend');

        % select bcNumRows rows with larges averages
        rowSet = zeros(N, 1);
        rowSet(orderRowSums(1:bcNumRows)) = true;

        % debug info
        %      disp(mean(mean(data(logical(rowSet),logical(columnSet)))));
        %     disp(mean(sortedRowSums(1:bcNumRows))/bcNumCols);

        % calculate the column sums over the selected rows
        % colSums = sum(data(logical(rowSet),:),1);
        colSums = rowSet'*matrix;

        % sort sums, saving the information about permutation]
        [sortedColSums orderColSums] = sort(colSums,'descend');

        % select bcNumRows rows with larges averages
        columnSet = zeros(1, M);
        columnSet(orderColSums(1:bcNumCols)) = true;

        % mean(mean(data(logical(rowSet),logical(columnSet))));
        currAvg = mean(sortedColSums(1:bcNumCols)/bcNumRows);
        %disp([currAvg prevAvg currAvg==prevAvg]);

    %     disp(mean(sortedColSums(1:bcNumCols))/bcNumRows);
        j += 1;
    end;

    j

    numIterations = max(numIterations, j);

    goldColumnSets = [goldColumnSets, columnSet'];
    goldRowSets = [goldRowSets; rowSet'];

end;

numIterations

save 'firstStage.matrix.csv' matrix
save 'firstStage.sizes.csv' sizes
save 'firstStage.columnSet.csv' initialColumnSets
save 'firstStage.goldColumnSet.csv' goldColumnSets
save 'firstStage.goldRowSet.csv' goldRowSets
