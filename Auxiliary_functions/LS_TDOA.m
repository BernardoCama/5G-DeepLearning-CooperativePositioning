function u = LS_TDOA(rho, s, dim)
% Perform LS solution

    H = compute_H (rho, s, dim);
    w = compute_w(rho, s);
    u = (H' * H)^(-1) * H' * w;
    u = reshape(u(1:dim), [1, dim]);

    function H = compute_H (rho, s, dim)
        N = size(s,1);
        H = zeros(N-1, dim + 1);
        for i=2:N
            H(i-1,:) = [s(1,:) - s(i,:), rho(i-1)]; 
        end
    end
    
    function w = compute_w (rho, s)
        N = size(s,1);
        w = zeros(N-1, 1);
        for i=2:N
            w(i-1) = ((rho(i-1)^2) ...
                + sum(s(1,:).^2) - sum(s(i,:).^2))/2;
        end
    end

end 