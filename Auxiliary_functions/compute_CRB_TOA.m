function lb = compute_CRB_TOA(sigma_TOA, H)

    % sigma_TOA
    % H: num_meas x dimensions (3 or 2)
    c = physconst('LightSpeed');
    lb = trace(sigma_TOA^2./(H'*H));
   
end