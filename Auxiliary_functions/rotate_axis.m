
function rotated = rotate_axis(xyz, az, el, roll, backtransformation)

    az = deg2rad(az);
    el = deg2rad(el); 
    roll = deg2rad(roll);

    if ~exist('backtransformation','var')
        % third parameter does not exist, so default it to something
        backtransformation = 0;
    end

    % Ensure the input vector is a column vector
    if size(xyz, 2) > size(xyz, 1)
        xyz = xyz';
    end
    
    % Define the rotation matrices
    Rz = [cos(az), -sin(az), 0; sin(az), cos(az), 0; 0, 0, 1];
    Ry = [cos(el), 0, sin(el); 0, 1, 0; -sin(el), 0, cos(el)];
    Rx = [1, 0, 0; 0, cos(roll), -sin(roll); 0, sin(roll), cos(roll)];
    
    % Compute the overall rotation matrix
    R = Rz * Ry * Rx;
    R = R';
    
    if backtransformation
        R = inv(R);
    end
    
    % Rotate the axes (not the vector) by using the transpose of the rotation matrix
    rotated = R * xyz;
end
