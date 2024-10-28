function [C, G] = CRB_TDOA(u, s, sigma, dim)
% Perform CRB for TDOA measurements and
% same uncorrelaed noise across dimensions simga [m]

    H = compute_H (s, u, dim);
    G = (H' * H)^(-1);
    C = sigma.*G;

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

end 