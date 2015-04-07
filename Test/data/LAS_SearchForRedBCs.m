function TheBiclusters = LAS_SearchForRedBCs(data, numBCfind, iterationsPerBC,scoreThreshold)
    % iterationsPerBC = 100;
    % scoreThreshold = 10;
    % numBCfind = 30;

    TheBiclusters = struct('score', {}, ...
                'rows', {}, ...
                'cols', {}, ...
                'avg', {});

    for bcNum = 1:numBCfind
        bestBC.score = -Inf;
        
        parfor i=1:iterationsPerBC
            bc = SingleBiclusterSearch(data, bcNum, i);
            bestBC = BetterBicluster(bestBC,bc);
%             disp(['Iter=' num2str(i) ',    ' num2str(sum(bestBC.rows)) 'x' num2str(sum(bestBC.cols)) ',    score=' num2str(bestBC.score)]);
        end;
        disp(['BC#=' num2str(bcNum) ' found,    ' num2str(sum(bestBC.rows)) 'x' num2str(sum(bestBC.cols)) ',    score=' num2str(bestBC.score)]);
        TheBiclusters(bcNum) = bestBC;

        data(bestBC.rows,bestBC.cols) = ...
            data(bestBC.rows,bestBC.cols) - bestBC.avg;

        if(bestBC.score < scoreThreshold)
            break;
        end;
    end;
end

function bc = BetterBicluster(bc1, bc2)
    if (bc1.score>bc2.score)
        bc = bc1;
    else
        bc = bc2;
    end;
end
