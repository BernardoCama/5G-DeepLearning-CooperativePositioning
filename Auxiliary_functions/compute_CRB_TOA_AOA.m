function lb = compute_CRB_TOA_AOA(sigma_AOA, sigma_TOA, H)

    % sigma_AOA
    % sigma_TOA
    % H: num_meas x dimensions (3 or 2)
    c = physconst('LightSpeed');

    R_a = diag(sigma_AOA.^2);
    num_angles = length(sigma_AOA); 
    num_t = size(H,1)-num_angles;
    R_t = diag(sigma_TOA.^2*ones(1,num_t));
    R = [R_a zeros(num_angles,num_t); zeros(num_t,num_angles) R_t];
    lb=trace(inv(H'*(R)^(-1)*H));
   
end