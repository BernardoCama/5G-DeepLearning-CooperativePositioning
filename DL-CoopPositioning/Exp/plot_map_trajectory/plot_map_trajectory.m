% clear all; clc; close all;

% Consider only these cells
cells_to_plot = [12]; % [1:57]; [1]
% Consider only test trajectory

GT_height = 10;
estimated_points_height = 1;

%% Plot
% Import and Visualize 3-D Environment with Buildings for Ray Tracing
if exist('viewer','var') && isvalid(viewer) % viewer handle exists and viewer window is open
    viewer.clearMap();
else
    viewer = siteviewer("Basemap","openstreetmap","Buildings","Politecnico.osm");    
end
% close(viewer)


%% Params
channel = 'ADCRM'; % 'ADCRM' 'SFCRM'
channel_element = 'abs_ADCRM'; 
FR = 1;  % 1 or 2
type = 'BOTH'; % 'LOS' 'NLOS' 'BOTH'
    
back_dir = '../../../../';
name_dataset = 'Positioning1';
DB_destination = sprintf('%s../../DB/FR%d_singleBS_raw_%s', back_dir, FR, name_dataset);


%% BS Parameters
% Define center location site (cells 1-3)
centerSite = rxsite('Name','DEIB', ...
    'Latitude',45.4787,...
    'Longitude',9.2330,...
    'AntennaHeight',1);

antHeight = 25; % m

% Initialize arrays for distance and angle from center location to each cell site, where
% each site has 3 cells
numCellSites = 19;
siteDistances = zeros(1,numCellSites);
siteAngles = zeros(1,numCellSites);

% Define distance and angle for inner ring of 6 sites (cells 4-21)
isd = 200; % Inter-site distance
siteDistances(2:7) = isd;
siteAngles(2:7) = 30:60:360;

% Define distance and angle for middle ring of 6 sites (cells 22-39)
siteDistances(8:13) = 2*isd*cosd(30);
siteAngles(8:13) = 0:60:300;

% Define distance and angle for outer ring of 6 sites (cells 40-57)
siteDistances(14:19) = 2*isd;
siteAngles(14:19) = 30:60:360;

% Each cell site has three transmitters corresponding to each cell. 
% Create arrays to define the names, latitudes, longitudes, 
% and antenna angles of each cell transmitter.

% Initialize arrays for cell transmitter parameters
numCells = numCellSites*3;
cellLats = zeros(1,numCells);
cellLons = zeros(1,numCells);
cellNames = strings(1,numCells);
cellAngles = zeros(1,numCells);

% Define cell sector angles
cellSectorAngles = [30 150 270];

% For each cell site location, populate data for each cell transmitter
cellInd = 1;
for siteInd = 1:numCellSites
    % Compute site location using distance and angle from center site
    [cellLat,cellLon] = location(centerSite, siteDistances(siteInd), siteAngles(siteInd));
    
    % Assign values for each cell
    for cellSectorAngle = cellSectorAngles
        cellNames(cellInd) = "Cell " + cellInd;
        cellSiteInd(cellInd) = siteInd;
        cellLats(cellInd) = cellLat;
        cellLons(cellInd) = cellLon;
        cellAngles(cellInd) = cellSectorAngle;
        cellInd = cellInd + 1;
    end
end

downtilt = 15;

% Create cell transmitter sites
bsSite = rxsite('Name',cellNames, ...
    'Latitude',cellLats, ...
    'Longitude',cellLons, ...
    'AntennaAngle',cellAngles, ...
    'AntennaHeight',antHeight);
j = 1;
for gNBIdx = 1:numCells
    show(bsSite(gNBIdx), "Icon",sprintf("placeholder_rx%d.png", j), "IconSize", [100, 100]);
    % pattern(bsSite(gNBIdx),fc,"Size",4);
    if mod(gNBIdx,3) == 0
        j = j + 1;
    end
end


%% LOAD
load(sprintf('%s/TDOA_estimatedPos_latlon.mat', DB_destination))
load(sprintf('%s/TDOA_LOS_index.mat', DB_destination))
load(sprintf('%s/UE_ML_estimatedPos_latlon.mat', DB_destination))
load(sprintf('%s/UE_Pos_latlon.mat', DB_destination))
% load(sprintf('ris_%s_%s_epoch_30.mat', name_dataset, type))

% remove_pos_TDOA = all(TDOA_estimatedPos_latlon == 0,2);
% TDOA_estimatedPos_latlon = TDOA_estimatedPos_latlon(remove_pos_TDOA, :);
% for gNBIdx = 1:cells_to_plot%numCells
%         try
%             UE_ML_estimatedPos_latlon{gNBIdx} = UE_ML_estimatedPos_latlon{gNBIdx}(remove_pos_TDOA, :);
%         end
% end
% UE_Pos_latlon = UE_Pos_latlon(remove_pos_TDOA, :);

%% REAL POS
UE_Site = rxsite('Latitude',UE_Pos_latlon(:,1), ...
    'Longitude',UE_Pos_latlon(:,2),...
    'AntennaHeight',GT_height);
% ORIGINAL TEST
% Whole LOS testing trajectory
% idx_test = UE_Pos_latlon(:,1) > 45.477036 & UE_Pos_latlon(:,2) <= 9.231598;
% For picture paper
idx_test = UE_Pos_latlon(:,1) > 45.477036 & UE_Pos_latlon(:,2) <= 9.231598 & UE_Pos_latlon(:,2) > 9.229998;
idx_train = ~(idx_test);
idx_test_LOS = idx_test & TDOA_LOS_index(1:end-2);
idx_test_NLOS = idx_test & (~TDOA_LOS_index(1:end-2));

show(UE_Site(idx_test), "Icon",sprintf("pink_filled.png", j), "IconSize", [20, 20]); % [10, 10]);
% show(UE_Site(idx_train), "Icon",sprintf("red.png", j), "IconSize", [10, 10]);
% show(UE_Site(idx_test_LOS), "Icon",sprintf("pink_filled.png", j), "IconSize", [20, 20]); % [10, 10]);
% show(UE_Site(idx_test_NLOS), "Icon",sprintf("pink_filled_dark.png", j), "IconSize", [20, 20]); % [10, 10]);

latlon_toplot_UE_Site = UE_Pos_latlon(idx_test,:);

%% SINGLE ML POSITIONING
latlon_toplot_UE_ML_estimated = [];
indexes_positions = zeros(length(cells_to_plot), length(UE_ML_estimatedPos_latlon{cells_to_plot(1)}(:,1)));
j = 1;
for gNBIdx = cells_to_plot % 1:numCells
    UE_ML_estimated = [];
    try
        UE_ML_estimated = cat(1, UE_ML_estimated, txsite('Latitude',UE_ML_estimatedPos_latlon{gNBIdx}(:,1), ...
                                            'Longitude',UE_ML_estimatedPos_latlon{gNBIdx}(:,2),...
                                            'AntennaHeight', estimated_points_height));
    end
    % indexes_positions = cat(2, indexes_positions, zeros(length(UE_ML_estimated), 1));
    for pos = find(idx_test==1)' % 1:length(UE_ML_estimated)
        if ~isequal(UE_ML_estimatedPos_latlon{gNBIdx}(pos,:), [0,0,0])
            if strcmp(type, 'LOS')
                if TDOA_LOS_index(pos)
                    show(UE_ML_estimated(1, pos), "Icon",sprintf("blue.png", j), "IconSize", [10, 10]);
                    link(UE_Site(1, pos), UE_ML_estimated(1, pos), 'SuccessColor', 'blue', 'FailColor', 'black')
                    indexes_positions(j, pos) = 1;

                    % to plot
                    latlon_toplot_UE_ML_estimated = cat(1, latlon_toplot_UE_ML_estimated, [UE_ML_estimated(1,pos).Latitude; UE_ML_estimated(1,pos).Longitude]');
                end
            elseif strcmp(type, 'NLOS')
                if ~TDOA_LOS_index(pos)
                    show(UE_ML_estimated(1, pos), "Icon",sprintf("blue.png", j), "IconSize", [10, 10]);
                    link(UE_Site(1, pos), UE_ML_estimated(1, pos), 'SuccessColor', 'blue', 'FailColor', 'black')
                    indexes_positions(j, pos) = 1;

                    % to plot
                    latlon_toplot_UE_ML_estimated = cat(1, latlon_toplot_UE_ML_estimated, [UE_ML_estimated(1,pos).Latitude; UE_ML_estimated(1,pos).Longitude]');
                end
            elseif strcmp(type, 'BOTH')
                show(UE_ML_estimated(1, pos), "Icon",sprintf("blue.png", j), "IconSize", [10, 10]);
                link(UE_Site(1, pos), UE_ML_estimated(1, pos), 'SuccessColor', 'blue', 'FailColor', 'black')
                indexes_positions(j, pos) = 1;

                % to plot
                latlon_toplot_UE_ML_estimated = cat(1, latlon_toplot_UE_ML_estimated, [UE_ML_estimated(1,pos).Latitude; UE_ML_estimated(1,pos).Longitude]');
            end
        end
    end
    j = j + 1;
end

%% COOPERATION TDOA
latlon_toplot_TDOA_estimated = [];
TDOA_estimatedSite = txsite('Latitude',TDOA_estimatedPos_latlon(:,1), ...
    'Longitude',TDOA_estimatedPos_latlon(:,2),...
    'AntennaHeight', estimated_points_height);
for pos = 1:length(TDOA_estimatedSite)
    try
        if ~isequal(TDOA_estimatedPos_latlon(pos,:),[0,0,0])
            if strcmp(type, 'LOS')
                if TDOA_LOS_index(pos)
                    status = link(UE_Site(pos), TDOA_estimatedSite(1, pos));
                    if status & all(indexes_positions(:, pos)==1) 
                        show(TDOA_estimatedSite(1, pos), "Icon",sprintf("red.png", j), "IconSize", [10, 10]);
                        link(UE_Site(pos), TDOA_estimatedSite(1, pos), 'SuccessColor', [1 0 0], 'FailColor', 'black')

                        % to plot
                        latlon_toplot_TDOA_estimated = cat(1, latlon_toplot_TDOA_estimated, [TDOA_estimatedSite(1,pos).Latitude; TDOA_estimatedSite(1,pos).Longitude]');
                    end
                end
            elseif strcmp(type, 'NLOS')
                if ~TDOA_LOS_index(pos)
                    status = link(UE_Site(pos), TDOA_estimatedSite(1, pos));
                    if status & all(indexes_positions(:, pos)==1) 
                        show(TDOA_estimatedSite(1, pos), "Icon",sprintf("red.png", j), "IconSize", [10, 10]);
                        link(UE_Site(pos), TDOA_estimatedSite(1, pos), 'SuccessColor', [1 0 0], 'FailColor', 'black')

                        % to plot
                        latlon_toplot_TDOA_estimated = cat(1, latlon_toplot_TDOA_estimated, [TDOA_estimatedSite(1,pos).Latitude; TDOA_estimatedSite(1,pos).Longitude]');
                    end
                end
            elseif strcmp(type, 'BOTH')
                status = link(UE_Site(pos), TDOA_estimatedSite(1, pos));
                if status & all(indexes_positions(:, pos)==1) 
                    show(TDOA_estimatedSite(1, pos), "Icon",sprintf("red.png", j), "IconSize", [10, 10]);
                    link(UE_Site(pos), TDOA_estimatedSite(1, pos), 'SuccessColor', [1 0 0], 'FailColor', 'black')
                
                    % to plot
                    latlon_toplot_TDOA_estimated = cat(1, latlon_toplot_TDOA_estimated, [TDOA_estimatedSite(1,pos).Latitude; TDOA_estimatedSite(1,pos).Longitude]');
                end
            end

        end
    end
end
% show(TDOA_estimatedSite, "Icon",sprintf("orange.png", j), "IconSize", [10, 10]);


%% MODEL PREDICTION
latlon_toplot_UE_Model_estimated = [];
load(sprintf('ris_%s_%s_epoch_30.mat', name_dataset, 'LOS'))
for gNBIdx = cells_to_plot %1:numCells

    indexes_cell = cell_index == gNBIdx;
    UE_Model_estimated = [];

    for pos = 1:length(UE_ML_estimated)
        indexes_pos = pos_index == pos;

        if (sum(indexes_cell & indexes_pos) ~= 0) & (all(indexes_positions(:, pos)==1))

            gNBPos_xyz = from_lat_long_to_xyz(bsSite(gNBIdx), UE_Site(pos));
            
            xyz_true_cell = test_realpos(indexes_cell & indexes_pos, :);
            xyz_true_cell = xyz_true_cell(1, :);
            xyz_true = rotate_axis(xyz_true_cell, (cellAngles(gNBIdx) + 90),  (-downtilt) , 0, 1);
            % in UE coord system
            xyz_true = xyz_true + gNBPos_xyz;
            [lat, lon, h] = from_xyz_to_site(double(xyz_true), UE_Site(pos));
            lat_lon_true = [lat, lon, h];


            % TEST RIGHT CONVERSION
            % in BS coord system
            % UEpos_xyz = [0,0,0] - gNBPos_xyz;
            % in cell coord system
            % UEpos_xyz_cell = rotate_axis(UEpos_xyz, (cellAngles(gNBIdx) + 90),  (-downtilt) , 0);

        
            xyz_predicted_cell = pos_predicted(indexes_cell & indexes_pos, :);
            xyz_predicted_cell = xyz_predicted_cell(1, :); % PLOT FIRST ESTIMATE POSITION
            xyz_predicted = rotate_axis(xyz_predicted_cell, (cellAngles(gNBIdx) + 90),  (-downtilt) , 0, 1);
            % in UE coord system
            xyz_predicted = xyz_predicted + gNBPos_xyz;
            [lat, lon, h] = from_xyz_to_site(double(xyz_predicted), UE_Site(pos));
            lat_lon_predicted = [lat, lon, h];
    
            % error = positioning_error(indexes_pos & indexes_pos);

            UE_Model_estimated = cat(1, UE_Model_estimated, txsite('Latitude',lat_lon_predicted(1), ...
                                    'Longitude',lat_lon_predicted(2),...
                                    'AntennaHeight', estimated_points_height));
            show(UE_Model_estimated(pos), "Icon",sprintf("orange.png", j), "IconSize", [10, 10]);
            link(UE_Site(pos), UE_Model_estimated(pos), 'SuccessColor', [0.9290 0.6940 0.1250], 'FailColor', 'black')

            % to plot
            latlon_toplot_UE_Model_estimated = cat(1, latlon_toplot_UE_Model_estimated, [UE_Model_estimated(pos).Latitude; UE_Model_estimated(pos).Longitude]');
        else
            UE_Model_estimated = cat(1, UE_Model_estimated, txsite('Latitude',0, ...
                                                'Longitude',0));
        end

    end
    
end

%% REMOVE USELESS UESITES
hide(UE_Site(~indexes_positions(1:length(UE_Site))))


%% COOPERATIVE MODEL PREDICTION
load(sprintf('ris_%s_%s_epoch_120.mat', name_dataset, 'BOTH'))

latlon_toplot_UE_Model_cooperative_estimated= []; % lat lon to plot trajectory
UE_Model_cooperative_estimated = []; % Sites

if strcmp(type, 'LOS') 
    indexes_los = test_LOS_identification == 1;
elseif strcmp(type, 'NLOS')
    indexes_los = test_LOS_identification == 0;
elseif strcmp(type, 'BOTH')
    indexes_los = (test_LOS_identification == 1) | (test_LOS_identification == 0);
end

for pos = 1:length(UE_ML_estimated)

    % check the correct position
    indexes_pos = pos_index == pos;

    xyz_predicted_pos = [];
    xyz_true_pos = [];
     
    for gNBIdx = cells_to_plot%numCells

        % check the correct cell
        indexes_cell = cell_index == gNBIdx;

        if (sum(indexes_cell & indexes_pos & indexes_los) ~= 0)

            gNBPos_xyz = from_lat_long_to_xyz(bsSite(gNBIdx), UE_Site(pos));

            xyz_true_cell = test_realpos(indexes_cell & indexes_pos & indexes_los, :);
            xyz_true_cell = xyz_true_cell(1, :);
            xyz_true = rotate_axis(xyz_true_cell, (cellAngles(gNBIdx) + 90),  (-downtilt) , 0, 1);
            % in UE coord system
            xyz_true = xyz_true + gNBPos_xyz;
            [lat, lon, h] = from_xyz_to_site(double(xyz_true), UE_Site(pos));
            lat_lon_true = [lat, lon, h];
            % global xyz
            xyz_true = coord2xyz(lat_lon_true(1), lat_lon_true(2), lat_lon_true(3));

            xyz_true_pos = cat(1, xyz_true_pos, xyz_true);

            xyz_predicted_cell = pos_predicted(indexes_cell & indexes_pos & indexes_los, :);        
            xyz_predicted_cell = xyz_predicted_cell(1, :);
            xyz_predicted = rotate_axis(xyz_predicted_cell, (cellAngles(gNBIdx) + 90),  (-downtilt) , 0, 1);
            % in UE coord system
            xyz_predicted = xyz_predicted + gNBPos_xyz;
            [lat, lon, h] = from_xyz_to_site(double(xyz_predicted), UE_Site(pos));
            lat_lon_predicted = [lat, lon, h];
            % global xyz
            xyz_predicted = coord2xyz(lat_lon_predicted(1), lat_lon_predicted(2), lat_lon_predicted(3));

            xyz_predicted_pos = cat(1, xyz_predicted_pos, xyz_predicted);
        end
    
    end

    if (size(xyz_predicted_pos, 1) >=1) & (all(indexes_positions(:, pos)==1))
        xyz_predicted = mean(xyz_predicted_pos, 1);
        xyz_true = mean(xyz_true_pos, 1);
        lat_lon_predicted = xyz2coord(xyz_predicted(1), xyz_predicted(2), xyz_predicted(3));

        % errors_MODEL_cooperative_estimated = cat(1, errors_MODEL_cooperative_estimated, sqrt(sum((xyz_true(1:dim) - xyz_predicted(:, 1:dim)).^2,2)));

        UE_Model_cooperative_estimated = cat(1, UE_Model_cooperative_estimated, txsite('Latitude',lat_lon_predicted(1), ...
                            'Longitude',lat_lon_predicted(2),...
                            'AntennaHeight', estimated_points_height));
        show(UE_Model_cooperative_estimated(pos), "Icon",sprintf("green.png", j), "IconSize", [10, 10]);
        link(UE_Site(pos), UE_Model_cooperative_estimated(pos), 'SuccessColor', [0 1 0], 'FailColor', 'black')

        % to plot
        latlon_toplot_UE_Model_cooperative_estimated = cat(1, latlon_toplot_UE_Model_cooperative_estimated, [UE_Model_cooperative_estimated(pos).Latitude; UE_Model_cooperative_estimated(pos).Longitude]');
    else
        UE_Model_cooperative_estimated = cat(1, UE_Model_cooperative_estimated, txsite('Latitude',0, ...
                                            'Longitude',0));
    end
end


%% Plot trajectories
fig1 = figure(1);
h = zeros(5, 1);
alpha_predictions = 0.7;

% Test Trajectory
a = latlon_toplot_UE_Site;
lat_ = a(:,1);
lon_ = a(:,2);
[lat_, lon_] = create_trajectory(lat_, lon_);
a(:,1) = lat_;
a(:,2) = lon_;
a = a(60:end-10, :); % LOS
geoscatter(a(:,1), a(:,2), 64, 'magenta' ,'x','filled', 'MarkerFaceAlpha',1);
hold on 
h(1) = geoplot(a(:,1), a(:,2), 'LineWidth', 4, 'Color', [1 0 1 1]); % 'magenta'

geobasemap streets-light % satellite openstreetmap

% Single-BS ToF-AoA
a = latlon_toplot_UE_ML_estimated;
lat_ = a(:,1);
lon_ = a(:,2);
[lat_, lon_] = create_trajectory(lat_, lon_);
a(:,1) = lat_;
a(:,2) = lon_;
a = a(40:end-56, :); % LOS (end is upper-right part of trajectory)  a(40:end-56, :)
geoscatter(a(:,1), a(:,2),64,'blue','s', 'filled', 'MarkerFaceAlpha',alpha_predictions);
h(2) = geoplot(a(:,1), a(:,2), 'LineWidth', 2, 'Color', [0 0 1 alpha_predictions]); % 'blue'

% Multi-BS TDoF
a = latlon_toplot_TDOA_estimated;
lat_ = a(:,1);
lon_ = a(:,2);
[lat_, lon_] = create_trajectory(lat_, lon_);
a(:,1) = lat_;
a(:,2) = lon_;
a = a(1:end-62, :); % LOS ( is upper-right part of trajectory) a(1:end-62, :)
geoscatter(a(:,1), a(:,2),64,'red','s', 'filled', 'MarkerFaceAlpha',alpha_predictions);
h(3) = geoplot(a(:,1), a(:,2), 'LineWidth', 2, 'Color', [1 0 0 alpha_predictions]); % 'red'

% Ego DL model
a = latlon_toplot_UE_Model_estimated;
lat_ = a(:,1);
lon_ = a(:,2);
[lat_, lon_] = create_trajectory(lat_, lon_);
a(:,1) = lat_;
a(:,2) = lon_;
a = a(21:end-1, :); % LOS (end is upper-right part of trajectory) a(21:end-1, :)
geoscatter(a(:,1), a(:,2),64,[0.9290 0.6940 0.1250],'o', 'filled', 'MarkerFaceAlpha',alpha_predictions);
h(4) = geoplot(a(:,1), a(:,2), 'LineWidth', 2, 'Color', [0.9290 0.6940 0.1250 alpha_predictions]);

% Cooperative DL model
a = latlon_toplot_UE_Model_cooperative_estimated;
lat_ = a(:,1);
lon_ = a(:,2);
[lat_, lon_] = create_trajectory(lat_, lon_);
a(:,1) = lat_;
a(:,2) = lon_;
a = a(25:end-46, :); % LOS (end is upper-right part of trajectory) a(25:end-46, :)
geoscatter(a(:,1), a(:,2),64,'green','o', 'filled', 'MarkerFaceAlpha',alpha_predictions);
h(5) = geoplot(a(:,1), a(:,2), 'LineWidth', 2, 'Color', [0 1 0 alpha_predictions]); % 'green'




[leg,icons] = legend(h, 'Test Trajectory','Single-BS ToF-AoA','Multi-BS TDoF', 'Ego DL model', 'Cooperative DL model', 'FontSize',24, 'Location', 'southeast');

fig = gcf;
set(gca,'FontSize',20)

fig.WindowState = 'maximized';

% a = get(gca,'LabelFontSizeMultiplier');
% set(gca,'LabelFontSizeMultiplier',a,'fontsize',14)
% Change only xaxis size and not xlabel
xl = get(gca,'LatitudeLabel');
xlFontSize = get(xl,'FontSize');
xAX = get(gca,'LatitudeAxis');
set(xAX,'FontSize', 14)
set(xl, 'FontSize', xlFontSize);
set(xAX, 'TickLabelRotation', 90);

xl = get(gca,'LongitudeLabel');
xlFontSize = get(xl,'FontSize');
xAX = get(gca,'LongitudeAxis');
set(xAX,'FontSize', 14)
set(xl, 'FontSize', xlFontSize);

pause(1);
% [latlim, lonlim] = geolimits;
% ORIGINAL
% geolimits([45.4784, 45.4794],[9.2296, 9.2320])
% NEW
geolimits([45.4786, 45.4793],[ 9.2297, 9.2315])
pause(1);
exportgraphics(fig,sprintf('Trajectory1.pdf', 1),'BackgroundColor','none')
% savefig(fig,sprintf('Trajectory1.fig', 1))


%% Angle upper-left
geolimits([45.4791, 45.4793],[9.2299, 9.2303])
exportgraphics(fig,sprintf('Zoom_trajectory.pdf', 1),'BackgroundColor','none')
