% credit to: https://gist.github.com/govert/1b373696c9a27ff4c72a
% (WGS-84) Geodetic point (lat0, lon0, h0)
% h0: ellipsoidal height. h0 (ellipsoidal) = H (orthometric) + N (geoid height)

function r = coord2xyz(varargin)

if length(varargin) == 1
    lat_lon_h = deal(varargin{1});
    [lat, lon, h] = deal(lat_lon_h(1),lat_lon_h(2), lat_lon_h(3));
elseif length(varargin) == 3
    lat = varargin{1};
    lon = varargin{2};
    h = varargin{3};
end

a = 6378137.0; % meters
b = 6356752.314245; % meters

f = (a - b) / a;     % flattening
lambda = deg2rad(lat);
e2 = f*(2-f);        % Square of eccentricity
phi = deg2rad(lon);
sin_lambda = sin(lambda);
cos_lambda = cos(lambda);
cos_phi = cos(phi);
sin_phi = sin(phi);

N = a / sqrt(1 - e2 * sin_lambda * sin_lambda);

x = (h + N) * cos_lambda * cos_phi;
y = (h + N) * cos_lambda * sin_phi;
z = (h + (1 - e2) * N) * sin_lambda;
r = [x, y, z];