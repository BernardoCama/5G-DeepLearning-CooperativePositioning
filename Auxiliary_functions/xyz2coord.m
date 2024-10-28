% credit to: https://gist.github.com/govert/1b373696c9a27ff4c72a

function pos = xyz2coord(varargin)

if length(varargin) == 1
    xyz = deal(varargin{1});
    [x, y, z] = deal(xyz(1),xyz(2), xyz(3));
elseif length(varargin) == 3
    x = varargin{1};
    y = varargin{2};
    z = varargin{3};
end



a = 6378137.0;      % meters
b = 6356752.314245; % meters

f = (a - b) / a;     % Flattening
%f_inv = 1.0 / f;

e2 = f*(2-f);        % Square of eccentricity

eps = e2 / (1.0 - e2);

p = sqrt(x * x + y * y);
q = atan2((z * a), (p * b));

sin_q = sin(q);
cos_q = cos(q);

sin_q_3 = sin_q * sin_q * sin_q;
cos_q_3 = cos_q * cos_q * cos_q;

phi = atan2((z + eps * b * sin_q_3), (p - e2 * a * cos_q_3));
lam = atan2(y, x);

v = a / sqrt(1.0 - e2 * sin(phi) * sin(phi));
h = (p / cos(phi)) - v;

lat = rad2deg(phi);
lon = rad2deg(lam);

pos = [lat, lon, h];
