function angle = convert_angle_in_range(angle, range)

% input
% angle (degree)
% range (vector) within which the angle must be

diff = range(2)-range(1);
angle = mod(angle, diff);

if angle > range(2)
    angle = angle - diff;
elseif angle < range(1)
    angle = angle + diff;
end


end