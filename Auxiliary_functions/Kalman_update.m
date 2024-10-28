function [estimatedPos_xyz, R, Q, P] = Kalman_update(estimatedPos_xyz, s, rho_TOA, R, Q, P)
    
    num_measurements = size(rho_TOA, 1);
    num_variables_to_estimate = size(P, 1);

    % Kalman
    H = createMatrixH(estimatedPos_xyz(1:3),s,'TOA');
    temp = zeros(num_measurements,num_variables_to_estimate - 3); % 3 are ux, uy, uz -> 0 in vx, vy, vz
    H = [H temp];

    G = P * H' * inv(H * P * H' + R);
    
    % estimatedPos_xyz = [estimatedPos_xyz zeros(1, num_variables_to_estimate - 3)];
    estimatedPos_xyz = estimatedPos_xyz + ( rho_TOA - sqrt(sum((s - estimatedPos_xyz(1:3)).^2, 2)))' * G';
    P = P - G * H * P;

end 