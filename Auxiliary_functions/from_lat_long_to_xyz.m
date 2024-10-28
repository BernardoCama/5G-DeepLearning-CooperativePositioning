function gNBPos_xyz = from_lat_long_to_xyz(bsSite, refsystemSite, varargin)
% Convert site objects coordinate in bsSite (lat, lon, high) according to
% refsystemSite site objects (lat, lon, high) in an xyz space.
    
    gNBPos_xyz = zeros(1,3);
    
    %%  SLOW
%     % Validate sites
%     validateattributes(bsSite,{'txsite','rxsite'},{'nonempty'}, ...
%         'angle','',1);
%     validateattributes(refsystemSite,{'txsite','rxsite'},{'nonempty'}, ...
%         'angle','',2);
%     % Process optional name/value pairs
%     p = inputParser;
%     addOptionalPath = nargin > 2 && mod(numel(varargin),2);
%     travelPath = 'euclidean';
%     if addOptionalPath
%         % Validator function is necessary for inputParser to allow string
%         % option instead of treating it like parameter name
%         p.addOptional('Path',travelPath,@(x)ischar(x)||isstring(x));
%     end
%     p.addParameter('Map', []);
%     p.addParameter('SourceAntennaSiteCoordinates', []);
%     p.addParameter('TargetAntennaSiteCoordinates', []);
%     p.parse(varargin{:});
%     
%     % Get usingCartesian from CoordinateSystem validation or from pre-validated 
%     % AntennaSiteCoordinates
%     usingCartesian = rfprop.internal.Validators.validateCoordinateSystem(bsSite, refsystemSite);
%     
%     if addOptionalPath
%         try
%             travelPath = validatestring(p.Results.Path, {'euclidean','greatcircle'}, ...
%                 'angle','',3);
%         catch me
%             % Check if option is a match for deprecated "geodesic" option
%             try
%                 travelPath = validatestring(p.Results.Path, {'geodesic'}, ...
%                     'angle','',3);
%             catch
%                 rethrow(me)
%             end
%         end
%     end
% 
%     % Get antenna site coordinates
%     map = rfprop.internal.Validators.validateMapTerrainSource(p, 'angle');
%         coordSys = 'geographic';
% 
%     rfprop.internal.Validators.validateMapCoordinateSystemMatch(map, coordSys);
%     bsCoords = rfprop.internal.Validators.validateAntennaSiteCoordinates(...
%         p.Results.SourceAntennaSiteCoordinates, bsSite, map, 'angle');
%     refCoords = rfprop.internal.Validators.validateAntennaSiteCoordinates(...
%         p.Results.TargetAntennaSiteCoordinates, refsystemSite, map, 'angle');
% 
%     [x,y,z] = positionGeographic(bsCoords, refCoords);

    %% FAST
    [x,y,z] = customGeodetic2Enu(bsSite.Latitude,bsSite.Longitude,bsSite.AntennaHeight,refsystemSite.Latitude,refsystemSite.Longitude,refsystemSite.AntennaHeight);

    gNBPos_xyz(1) = x;
    gNBPos_xyz(2) = y;
    gNBPos_xyz(3) = z;

    function [xEast, yNorth, zUp] = customGeodetic2Enu(latObs, lonObs, hObs, latRef, lonRef, hRef)
        % Define Earth radius (mean radius)
        R = 6371e3; % in meters
    
        % Convert latitude and longitude from degrees to radians
        latObs = deg2rad(latObs);
        lonObs = deg2rad(lonObs);
        latRef = deg2rad(latRef);
        lonRef = deg2rad(lonRef);
    
        % Calculate differences in coordinates
        dLat = latObs - latRef;
        dLon = lonObs - lonRef;
        dH = hObs - hRef;
    
        % Calculate ENU coordinates
        xEast = (dLon) * R * cos(latRef);
        yNorth = (dLat) * R;
        zUp = dH;
    end

end