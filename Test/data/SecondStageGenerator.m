
N = 512;
M = 512;
matrix = 7*randn(N, M) + 0.3;

sizes = [];
initialColumnSets = [];
goldColumnSets = [];
goldRowSets = [];
goldFinalScore = [];

numIterations = 0;

for i = 1:10
    
    bcNumRows = floor(rand(1)^2*N/2)+1;
    bcNumCols = floor(rand(1)^2*M/2)+1;
    sizes = [sizes; bcNumCols, bcNumRows];

    columnSet = (randperm(M)<=bcNumCols);
    initialColumnSets = [initialColumnSets, columnSet'];

    j = 0;
    prevScore = -Inf;
    currScore = 0;
    while (prevScore~=currScore)
        prevScore = currScore;
        
        % calculate the row sums over the selected columns
        % rowSums = sum(data(:,columnSet),2);
        
        rowSums = matrix*columnSet';

        % sort sums, saving the info about order
        [sortedRowSums orderRowSums] = sort(rowSums,'descend');

        % Calculate the sums of potential biclusters
        potBcSumsR = cumsum(sortedRowSums);
        % the Scores of the potential biclusters
        potScoresR = LAS_score(potBcSumsR, (1:N)', bcNumCols, N, M);
        [maxPotScoreR bcNumRows] = max(potScoresR);
            
        %disp(['bcNumRows = ' num2str(bcNumRows)]);
        
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

        % Calculate the sums of potential biclusters
        potBcSumsC = cumsum(sortedColSums);
        % the Scores of the potential biclusters
        potScoresC = LAS_score(potBcSumsC, bcNumRows, (1:M), N, M);
        [maxPotScoreC bcNumCols] = max(potScoresC);
        
        %disp(['bcNumCols = ' num2str(bcNumCols)]);
        
        % select bcNumRows rows with larges averages
        columnSet = zeros(1, M);
        columnSet(orderColSums(1:bcNumCols)) = true;

        % mean(mean(data(logical(rowSet),logical(columnSet))));
        currScore = maxPotScoreC;
    %     disp([currScore prevScore currScore==prevScore]);
    %     disp(['currScore = ' num2str(currScore)]);

    %     disp(mean(sortedColSums(1:bcNumCols))/bcNumRows);
        j += 1;
    end;

    j
    numIterations = max(numIterations, j);

    goldColumnSets = [goldColumnSets, columnSet'];
    goldRowSets = [goldRowSets; rowSet'];
    goldFinalScore = [goldFinalScore, currScore];

end;

numIterations

save 'secondStage.matrix.csv' matrix
save 'secondStage.sizes.csv' sizes
save 'secondStage.columnSet.csv' initialColumnSets
save 'secondStage.goldColumnSet.csv' goldColumnSets
save 'secondStage.goldRowSet.csv' goldRowSets
save 'secondStage.goldFinalScore.csv' goldFinalScore
