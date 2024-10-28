function [estimatedPos_xyz, P] = Kalman_prediction(estimatedPos_xyz, F, P, Q)
    
    estimatedPos_xyz = reshape(estimatedPos_xyz, [length(estimatedPos_xyz), 1]);
    estimatedPos_xyz = F * estimatedPos_xyz;
    P = F * P * F' + Q;
    
    estimatedPos_xyz = reshape(estimatedPos_xyz, [1, length(estimatedPos_xyz)]);
end 