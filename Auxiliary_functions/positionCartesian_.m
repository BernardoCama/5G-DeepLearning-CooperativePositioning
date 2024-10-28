% from xyz to lat_lon
function [lat,lon,h] = positionCartesian_(coords, referenceCoords)
    %positionCartesian   Return position from Cartesian reference
    
%     srcpos = referenceCoords.AntennaPosition;
%     tgtpos = coords.AntennaPosition;
    
%     % Initialize outputs
%     numSourceSites = size(srcpos,1);
%     X = zeros(size(tgtpos,1), numSourceSites);
%     Y = X;
%     Z = X;
    
%     for srcInd = 1:numSourceSites
%         relPos = tgtpos - srcpos(srcInd,:);
%         X(:,srcInd) = relPos(:,1);
%         Y(:,srcInd) = relPos(:,2);
%         Z(:,srcInd) = relPos(:,3);
%     end
    
    pos = coords';
    [lat0, lon0, h0] = referenceCoords.geodetic;

    [lat,lon,h] = rfprop.internal.MapUtils.enu2geodetic(lat0, lon0, h0, [pos, pos]);
    lat = lat(1);
    lon = lon(1);
    h = h(1);

%     [lat,lon,h] = enu2geodetic(lat0, lon0, h0, pos);


    function [lat,lon,h] = enu2geodetic(lat0, lon0, h0, pos, spheroid)
            % Return geodetic coordinates from relative ENU coordinates
            
            if nargin < 5
                spheroid = map.geodesy.internal.wgs84Spheroid;
            end
            [lat,lon,h] = enu2geodetic_(pos(1),pos(2),pos(3), ...
                lat0, lon0, h0, spheroid);
    end
end
