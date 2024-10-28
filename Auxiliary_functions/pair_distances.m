function distance_matrix = pair_distances(locs)
    N = length(locs);
    distance_matrix = zeros(N, N);
    for i = 1:N
        for j = 1:N
           distance_matrix(i,j) = norm(locs(i)-locs(j)); 
        end
    end
end