function [lat,lon,h] = from_xyz_to_site(bsxyz, refsystemSite, varargin)
% function bsSite = from_xyz_to_site(bsName, bsxyz, bsArrayOrientation, bsSite_AntennaHeight, fc, ...
%     refsystemName, refsystemxyz, refsystemArrayOrientation, refsystemSite_AntennaHeight, varargin)
% Convert coordinate in bsSite (x, y, z) according to
% refsystemxyz site objects (lat, lon, high) in an xyz space.

        
%% SLOW

%     [lat_ref,lon_ref] = location(ref_system);
%     high_ref = elevation(ref_system);
%     refsystemxyz = [0,0,0];

%     d = norm(bs_xyz);
%     d_xy = norm(bs_xyz(1:2));
%     az = atan( bs_xyz(1) / bs_xyz(2) ); 
%     el = asin( bs_xyz(3) / d );
    % [az,el] = cart2sph(bsSite_xyz(1),bsSite_xyz(2),bsSite_xyz(3));
    % az = wrapToPi(az);

%     [bsSite_lat, bsSite_lon] = location(ref_system,d_xy,az);

%     bsSite = txsite("Name",Name, ...
%         "Latitude",bsSite_lat,"Longitude",bsSite_lon,...
%         "AntennaAngle",bsArrayOrientation,...
%         "AntennaHeight",bsSite_AntennaHeight,...  % in m
%         "TransmitterFrequency",fc);



%     bsSite = txsite("Name",bsName, ...
%      'CoordinateSystem','cartesian', ...
%     'AntennaPosition',reshape(bsxyz,[3,1]), ...
%     "AntennaAngle",bsArrayOrientation,...
%     'AntennaHeight',bsSite_AntennaHeight, ...
%     'TransmitterFrequency',fc);
%     refsystemSite = txsite("Name",refsystemName, ...
%      'CoordinateSystem','cartesian', ...
%     'AntennaPosition', reshape(refsystemxyz,[3,1]), ...
%     "AntennaAngle",refsystemArrayOrientation,...
%     'AntennaHeight',refsystemSite_AntennaHeight, ...
%     'TransmitterFrequency',fc);

    % Validate sites
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
%     usingCartesian = 1;
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
%     if usingCartesian
%         map = rfprop.internal.Validators.validateCartesianMap(p);
%         coordSys = 'cartesian';
%     else
%         map = rfprop.internal.Validators.validateMapTerrainSource(p, 'angle');
%         coordSys = 'geographic';
%     end
%     rfprop.internal.Validators.validateMapCoordinateSystemMatch(map, coordSys);
% %     bsCoords = rfprop.internal.Validators.validateAntennaSiteCoordinates(...
% %         p.Results.SourceAntennaSiteCoordinates, bsSite, map, 'angle');
%     refCoords = rfprop.internal.Validators.validateAntennaSiteCoordinates(...
%         p.Results.TargetAntennaSiteCoordinates, refsystemSite, map, 'angle');
% 
%     [lat,lon,h] = positionCartesian_(bsxyz, refCoords);


    %% FAST

    % Retrieve the reference site coordinates
    latRef = refsystemSite.Latitude;
    lonRef = refsystemSite.Longitude;
    hRef = refsystemSite.AntennaHeight;

    % Define Earth radius (mean radius)
    R = 6371e3; % in meters

    % Convert reference latitude and longitude from degrees to radians
    latRef = deg2rad(latRef);
    lonRef = deg2rad(lonRef);

    % Retrieve the ENU coordinates from the input xyz
    xEast = bsxyz(1);
    yNorth = bsxyz(2);
    zUp = bsxyz(3);

    % Convert ENU to lat, lon, and h
    dLat = yNorth / R;
    dLon = xEast / (R * cos(latRef));
    h = hRef + zUp;

    % Convert lat and lon differences to absolute lat and lon values
    lat = rad2deg(latRef + dLat);
    lon = rad2deg(lonRef + dLon);


end