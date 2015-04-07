function bc = SingleBiclusterSearch(data, bcNum, i)
%% Search for a bicluster

[m n] = size(data);
bcNumRows = floor(rand(1)^2*m/2)+1;
bcNumCols = floor(rand(1)^2*n/2)+1;

% start with a random columnset
% 
randomColPermutation = randperm(n);
columnSet = (randomColPermutation<=bcNumCols);
clear fraction2select randomColPermutation;

if i == 1
    sizes = [bcNumCols; bcNumRows];
    fullColumnSet = columnSet';
else
    load(strcat(['wholeAlgo.' num2str(bcNum) '.init.csv']), 'sizes', 'fullColumnSet');
    sizes = [sizes, [bcNumCols; bcNumRows]];
    fullColumnSet = [fullColumnSet, columnSet'];
end;
save(strcat(['wholeAlgo.' num2str(bcNum) '.init.csv']), 'sizes', 'fullColumnSet');
clear sizes fullColumnSet



%% First we search with bcNumRows and bcNumCols fixed
prevAvg = -Inf;
currAvg = 0;
j = 0;
while(prevAvg~=currAvg)
    prevAvg = currAvg;
    
    % calculate the row sums over the selected columns
    % rowSums = sum(data(:,columnSet),2);
    
    rowSums = data*columnSet';

    % sort sums, saving the info about order
    [sortedRowSums orderRowSums] = sort(rowSums,'descend');

    % select bcNumRows rows with larges averages
    rowSet = zeros(m,1);
    rowSet(orderRowSums(1:bcNumRows)) = true;

    %{
    if j == 0
        if i == 1
            firstRowSums = rowSums;
            firstRowSet = rowSet';
        else
            load(strcat([num2str(bcNum) '.first-sortSelectRows.csv']), 'firstRowSums', 'firstRowSet');
            firstRowSums = [firstRowSums, rowSums];
            firstRowSet = [firstRowSet; rowSet'];
        end;
        save(strcat([num2str(bcNum) '.first-sortSelectRows.csv']), 'firstRowSums', 'firstRowSet');
        clear firstRowSums firstRowSet
    end;
    %}


    % debug info
    %      disp(mean(mean(data(logical(rowSet),logical(columnSet)))));
    %     disp(mean(sortedRowSums(1:bcNumRows))/bcNumCols);

    % calculate the column sums over the selected rows
    % colSums = sum(data(logical(rowSet),:),1);
    colSums = rowSet'*data;

    % sort sums, saving the information about permutation]
    [sortedColSums orderColSums] = sort(colSums,'descend');

    % select bcNumRows rows with larges averages
    columnSet = zeros(1,n);
    columnSet(orderColSums(1:bcNumCols)) = true;

    %{
    if j == 0
        if i == 1
            firstColSums = colSums;
            firstColumnSet = columnSet';
        else
            load(strcat([num2str(bcNum) '.first-sortSelectCols.csv']), 'firstColSums', 'firstColumnSet');
            firstColSums = [firstColSums; colSums];
            firstColumnSet = [firstColumnSet, columnSet'];
        end;
        save(strcat([num2str(bcNum) '.first-sortSelectCols.csv']), 'firstColSums', 'firstColumnSet');
        clear firstColSums firstColumnSet
    end;
    %}

    % mean(mean(data(logical(rowSet),logical(columnSet))));
    currAvg = mean(sortedColSums(1:bcNumCols)/bcNumRows);
%     disp([currAvg prevAvg currAvg==prevAvg]);

%     disp(mean(sortedColSums(1:bcNumCols))/bcNumRows);
    j += 1;
end;

%{
if i == 1
    fullColumnSet = columnSet';
    fullRowSet = rowSet';
else
    load(strcat([num2str(bcNum) '.firstStage.csv']), 'fullRowSet', 'fullColumnSet');
    fullColumnSet = [fullColumnSet, columnSet'];
    fullRowSet = [fullRowSet; rowSet'];
end;
save(strcat([num2str(bcNum) '.firstStage.csv']), 'fullRowSet', 'fullColumnSet');
clear fullRowSet fullColumnSet
%}

clear prevAvg currAvg rowSums colSums sortedRowSums sortedColSums;
clear orderRowSums orderColSums j;

%% now let bcNumRows and bcNumCols be flexible
prevScore = -Inf;
currScore = 0;
j = 0;
while(prevScore~=currScore)
    prevScore = currScore;
    
    % calculate the row sums over the selected columns
    % rowSums = sum(data(:,columnSet),2);
    
    rowSums = data*columnSet';

    % sort sums, saving the info about order
    [sortedRowSums orderRowSums] = sort(rowSums,'descend');

    % Calculate the sums of potential biclusters
    potBcSumsR = cumsum(sortedRowSums);
    % the Scores of the potential biclusters
    potScoresR = LAS_score(potBcSumsR,(1:m)',bcNumCols,m,n);
    [maxPotScoreR bcNumRows] = max(potScoresR);
        
%     disp(['bcNumRows = ' num2str(bcNumRows)]);
    
    rowSet = zeros(m,1);
    rowSet(orderRowSums(1:bcNumRows)) = true;

    %{
    if j == 0
        if i == 1
            firstRowSums = rowSums;
            firstRowSet = rowSet';
            sizes = [bcNumCols; bcNumRows];
        else
            load(strcat([num2str(bcNum) '.first-sortSelectHeightAndRows.csv']), 'sizes', 'firstRowSums', 'firstRowSet');
            firstRowSums = [firstRowSums, rowSums];
            firstRowSet = [firstRowSet; rowSet'];
            sizes = [sizes, [bcNumCols; bcNumRows]];
        end;
        save(strcat([num2str(bcNum) '.first-sortSelectHeightAndRows.csv']), 'sizes', 'firstRowSums', 'firstRowSet');
        clear sizes firstRowSums firstRowSet
    end;
    %}


    % debug info
    %      disp(mean(mean(data(logical(rowSet),logical(columnSet)))));
    %     disp(mean(sortedRowSums(1:bcNumRows))/bcNumCols);

    % calculate the column sums over the selected rows
    % colSums = sum(data(logical(rowSet),:),1);
    colSums = rowSet'*data;

    % sort sums, saving the information about permutation]
    [sortedColSums orderColSums] = sort(colSums,'descend');

    % Calculate the sums of potential biclusters
    potBcSumsC = cumsum(sortedColSums);
    % the Scores of the potential biclusters
    potScoresC = LAS_score(potBcSumsC,bcNumRows,(1:n),m,n);
    [maxPotScoreC bcNumCols] = max(potScoresC);
    
%     disp(['bcNumCols = ' num2str(bcNumCols)]);
    
    % select bcNumRows rows with larges averages
    columnSet = zeros(1,n);
    columnSet(orderColSums(1:bcNumCols)) = true;

    %{
    if j == 0
        if i == 1
            firstColSums = colSums;
            firstColumnSet = columnSet';
            sizes = [bcNumCols; bcNumRows];
        else
            load(strcat([num2str(bcNum) '.first-sortSelectWidthAndCols.csv']), 'sizes', 'firstColSums', 'firstColumnSet');
            firstColSums = [firstColSums; colSums];
            firstColumnSet = [firstColumnSet, columnSet'];
            sizes = [sizes, [bcNumCols; bcNumRows]];
        end;
        save(strcat([num2str(bcNum) '.first-sortSelectWidthAndCols.csv']), 'sizes', 'firstColSums', 'firstColumnSet');
        clear sizes firstColSums firstColumnSet
    end;
    %}

    % mean(mean(data(logical(rowSet),logical(columnSet))));
    currScore = maxPotScoreC;
%     disp([currScore prevScore currScore==prevScore]);
%     disp(['currScore = ' num2str(currScore)]);

%     disp(mean(sortedColSums(1:bcNumCols))/bcNumRows);
    j += 1;
end;

%{
if i == 1
    fullColumnSet = columnSet';
    fullRowSet = rowSet';
    sizes = [bcNumCols; bcNumRows];
    scores = [currScore];
else
    load(strcat([num2str(bcNum) '.secondStage.csv']), 'fullRowSet', 'fullColumnSet', 'sizes', 'scores');
    fullColumnSet = [fullColumnSet, columnSet'];
    fullRowSet = [fullRowSet; rowSet'];
    sizes = [sizes, [bcNumCols; bcNumRows]];
    scores = [scores, currScore];
end;
save(strcat([num2str(bcNum) '.secondStage.csv']), 'fullRowSet', 'fullColumnSet', 'sizes', 'scores');
clear fullRowSet fullColumnSet sizes scores
%}

% clear prevScore rowSums colSums sortedRowSums sortedColSums;
% clear orderRowSums orderColSums;
% clear potBcSumsR potScoresR potBcSumsC potScoresC

bc = struct( ...
    'score', currScore, ...
    'rows', logical(rowSet), ...
    'cols', logical(columnSet), ...
    'avg', mean(sortedColSums(1:bcNumCols))/bcNumRows);
return;
