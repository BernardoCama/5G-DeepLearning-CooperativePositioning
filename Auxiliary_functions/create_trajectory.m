function [lat_, lon_] = create_trajectory(lat_, lon_)

% Method 2 shows a solution using proximity. 

% This method relies on the following assumptions:
% The perimeter line does not cross over itself. 
% The distance between coordinate A and it's neightboring coordinate B is 
% the minimum distance between coordinate A and any other coordinate in 
% the perimeter execept, perhaps the coordinate that come sequentially 
% before A.  In other words, if parts of the parimeter come very close 
% to eachother such that their distance is closer than the interval of 
% coordinates, this will fail. 

    % Set up loop variables.  The first coordinate
    % will be the first coordinate in the new, sorted order.
    yIdx = [1,nan(size(lat_)-1)]; 
    xyTemp = [lon_(:), lat_(:)]; 
    xyTemp(1,:) = NaN; 
    idx = [1, nan(1, numel(lon_)-1)]; 
    counter = 0; 
    % Loop through each coordinate and find its nearest neightbor,
    % Then eliminate that coordinate from being chosen again, and
    % then start the loop again finding the nearest neighbor to 
    % the pointe we just located.
    while any(isnan(idx))
        counter = counter+1; 
        % find closest coordinate to (x(i),y(i)) that isn't already taken
        [~, idx(counter+1)] = min(pdist2(xyTemp,[lon_(idx(counter)), lat_(idx(counter))])); 
        % Eliminate that from the pool
        xyTemp(idx(counter+1),:) = NaN; 
    end
    
    lat_ = lat_(idx);
    lon_ = lon_(idx);

end