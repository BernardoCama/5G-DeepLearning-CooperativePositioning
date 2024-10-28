function u_k = Non_linear_LS_TDOA(rho, u_0, s, K, dim)
% Perform NLS solution with Iterative search (Gauss-Newton)

    u_k = u_0;
    mu_k = 0.01;
    
    for k=1:K
       delta_rho = compute_delta_rho (s, rho, u_k);
       H = compute_H (s, u_k, dim);
       u_k = u_k + (mu_k * (H'*H)^(-1) * (H') * delta_rho)';
    end


    function H = compute_H (s, u, dim)
        N = size(s,1);
        a = zeros(N, dim);
        H = zeros(N-1, dim);
        for i=1:N
            a(i,:) = compute_a(s(i,:), u);
        end
        for i=2:N
            H(i-1,:) = a(1,:) - a(i,:); 
        end
    end

    function a = compute_a(s, u)
         d = norm(s-u);
         a = (s-u)./d;
    end
    
    function delta_rho = compute_delta_rho (s, rho, u)
        N = size(s,1);
        d = zeros(1, N);
        for i=1:N
            d(i) = norm(s(i,:)-u);
        end
        delta_rho = zeros(N-1, 1);
        for i=2:N
            delta_rho(i-1) = rho(i-1) - (d(i) - d(1));
        end
    end

end 