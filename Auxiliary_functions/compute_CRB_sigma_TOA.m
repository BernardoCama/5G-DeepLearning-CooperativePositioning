function sigma_TOA = compute_CRB_sigma_TOA(SNR, beta)

    % SNR: linear
    % beta: effective bandwidth = N_RB * ∆f * N_SC_per_RB,
    c = physconst('LightSpeed');
    sigma_TOA = c./(2*sqrt(2)*pi*sqrt(SNR)*beta);
    
end