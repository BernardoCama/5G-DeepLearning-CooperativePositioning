function sigma_AOA = compute_CRB_sigma_AOA(SNR, beta, Na, da, phi)

    % SNR: linear
    % beta: effective bandwidth = N_RB * ∆f * N_SC_per_RB,
    % Na: number of array elements of the antenna array (in transmission for AoD and in reception for AoA)
    % da: antenna elements distance
    % phi: angle in rad
    c = physconst('LightSpeed');
    sigma_AOA = (sqrt(3)*c)./(sqrt(2) * pi * sqrt(SNR) * beta * sqrt(Na*(Na^2-1)) * da * cos(phi));
   
end