function mdifference = distance_2latlon(lat1, lon1, lat2, lon2)

    distgc = distance(lat1,lon1,lat2,lon2);  
    distrh = distance('rh',lat1,lon1,lat2,lon2);  
    mdifference = deg2m(distrh-distgc);
end