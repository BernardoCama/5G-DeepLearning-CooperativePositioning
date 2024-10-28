function u_k = Non_linear_LS_TOA(rho, u_0, s, K, dim, diag_R)
% Perform NLS solution with Iterative search (Gauss-Newton)

    u_k = u_0(1:dim);
    mu_0 = 0.01;
    mu_k = mu_0;

    num_meas = length(rho);

    if length(diag_R) ~= num_meas
        diag_R = ones(1, num_meas);
    else
        diag_R = normalize(diag_R,'range', [0,1]);
    end
    R = diag(diag_R);
    
    for k=1:K
       delta_rho = compute_delta_rho (s, rho, u_k);
       H = compute_H (s, u_k, dim);
       delta_u = (H'*(R^(-1))*H)^(-1) * (H') * (R^(-1)) * delta_rho;
       u_k = u_k + (mu_k * delta_u)';

%         if sum(delta_u.^2 < 1e-6)
%             k
%             return
%         end
    end


    function H = compute_H (s, u, dim)
        N = size(s,1);
        a = zeros(N, dim);
        H = zeros(N, dim);
        for i=1:N
            a(i,:) = compute_a(s(i,:), u);
        end
        for i=1:N
            H(i,:) = - a(i,:); 
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
        delta_rho = zeros(N, 1);
        for i=1:N
            delta_rho(i) = rho(i) - d(i);
        end
    end

end 