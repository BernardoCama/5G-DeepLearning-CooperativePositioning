function u = LS_TOA(rho, s, dim)
% Perform LS solution

    H = compute_H (s, dim);
    rho = compute_rho(rho, s);
    x = ((H'*H)^(-1) * (H') * rho)';
    u = x + s(1,:);

    function H = compute_H (s, dim)
        N = size(s,1);
        H = zeros(N-1, dim);
        for i=2:N
            H(i-1,:) = s(i,:) - s(1,:); 
        end
    end
    
    function rho_new = compute_rho (rho, s)
        N = size(s,1);
        d = zeros(1, N);
        for i=1:N
            d(i) = norm(s(i,:)-s(1,:));
        end
        rho_new = zeros(N-1, 1);
        for i=2:N
            rho_new(i-1) = ((rho(1)^2) - (rho(i)^2) + (d(i)^2))/2;
        end
    end

end 