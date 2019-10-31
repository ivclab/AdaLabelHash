function [map, precision_at_k, agg_prec] = precision_singlelabel (trn_label, trn_binary, tst_label, tst_binary, mode)
K = size(trn_binary,1);
QueryTimes = size(tst_binary,1);

AP = zeros(QueryTimes,1);

Ns = 1:1:K;
sum_tp = zeros(1, length(Ns));

agg_prec = zeros(1, 11);

for i = 1:QueryTimes

    %img_path = tst_list(i,1);
    query_label = tst_label(i);
    fprintf('query %d\n',i);
    query_binary = tst_binary(i,:);
    if mode==1
        tic
        similarity = pdist2(double(trn_binary),double(query_binary),'hamming');
        toc
        fprintf('Complete Query [Hamming] %.2f seconds\n',toc);
    elseif mode ==2
        tic
        similarity = pdist2(trn_binary,query_binary,'euclidean');
        toc
        fprintf('Complete Query [Euclidean] %.2f seconds\n',toc);
    end

    [x2,y2] = sort(similarity);

    buffer_yes = trn_label(y2(1:K)) == query_label;
    % compute precision
    P = cumsum(buffer_yes) ./ Ns';
    if (sum(buffer_yes) == 0)
        AP(i) = 0;
    else
        AP(i) = sum(P .* buffer_yes) / sum(buffer_yes);
    end
    sum_tp = sum_tp + cumsum(buffer_yes)';

    if (sum(buffer_yes) > 0)
        % recall
        R = cumsum(buffer_yes) ./ sum(buffer_yes);

        % aggregated precision
        for mm = 1: 11
            rr = (mm - 1)/10;
            idx = find(R >= rr);
            agg_prec(mm) = agg_prec(mm) + max(P(idx));
        end
    end
end
    precision_at_k = sum_tp ./ (Ns * QueryTimes);
    map = mean(AP);

    agg_prec = agg_prec / QueryTimes;
end

