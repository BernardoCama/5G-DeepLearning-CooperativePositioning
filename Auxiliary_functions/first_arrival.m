function [peak, delayEst] = first_arrival(corr, strategy)
    % first_arrival(corr) return the peak corresponding to the first
    % arrival and its corresponding delay

    switch strategy
        case 0
            % MAXIMUM
            % Delay estimate is at point of maximum correlation
            peak = max(corr);
            delayEst= find(corr == peak,1)-1;    
        case 1
            [pks, locs] = findpeaks(corr, ...
                'MinPeakHeight',0.5 * 10^-5, ...
                'MinPeakDistance',10,...
                'MinPeakProminence',0.5 * 10^-5);
            try
                peak = pks(1);
                delayEst = locs(1) - 1; 
            catch
                peak = max(corr);
                delayEst= find(corr == peak,1)-1;  
            end
        case 2
            num_pks = 3;
            percentage_high_pks = 0.2;
            distance_pks = 10;

            [pks, locs] = findpeaks(corr, ...
                'SortStr','descend');
            pks = reshape(pks(1:num_pks), [1, num_pks]);
            locs = reshape(locs(1:num_pks), [1, num_pks]);

            percentages = pks'*(pks.^-1);
            percentages = percentages(:,1)>percentage_high_pks;

            distance_matrix = pair_distances(locs);
            distance_matrix = distance_matrix(:,1)>distance_pks;
            distance_matrix(1) = 1;

            peak_idx = percentages .* distance_matrix;
            [~, peak_idx] = min(locs(logical(peak_idx)), [], 'all');
            peak = pks(peak_idx);
            delayEst= find(corr == peak,1)-1;  
    end
 
end
