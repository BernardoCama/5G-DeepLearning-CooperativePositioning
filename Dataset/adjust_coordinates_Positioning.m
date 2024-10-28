clear all; clc; close all;

%% Params
channel = 'ADCRM'; % 'ADCRM' 'SFCRM'
channel_element = 'abs_ADCRM'; 
FR = 1;  % 1 or 2

%% UE positions
% load vehicle positions 
T = readtable('Traiettorie/DeibTraces.csv');
T = rmmissing(T);
% 1 timestep_time, 3 vehicle_id, 9 vehicle_lon, 10 vehicle_lat.
T = T(:,[1,3,9,10]); 
num_vehicles = length(unique(T.vehicle_id));
num_instants = length(unique(T.timestep_time));
uePositions = zeros(1,2);
for v = 0:num_vehicles-1
    uePositions = cat(1,uePositions,fliplr(table2array(T(T.vehicle_id == v,3:4))));
end
uePositions = uePositions(2:end,:);
numPos = size(uePositions, 1);% size(uePositions, 1); 2

% Create cell transmitter sites
ueSite = rxsite('Latitude',uePositions(:,1), ...
    'Longitude',uePositions(:,2));
UE_Pos_latlon = uePositions;

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

% Create cell transmitter sites
bsSite = rxsite('Name',cellNames, ...
    'Latitude',cellLats, ...
    'Longitude',cellLons, ...
    'AntennaAngle',cellAngles, ...
    'AntennaHeight',antHeight);


%% Load estimated positions
name_dataset = 'Positioning1';
DB_LOS = sprintf('../../DB/LOS_%s', name_dataset);
DB_NLOS = sprintf('../../DB/NLOS_%s', name_dataset);
DB_destination = sprintf('../../DB/FR%d_singleBS_raw_%s', FR, name_dataset);

load(sprintf('%s/TDOA_estimatedPos_xyz_relativeUE.mat', DB_destination))

TDOA_estimatedPos_latlon = [];
for pos = 1:numPos
    if ~any(TDOA_estimatedPos_xyz_relativeUE(pos,:))
        TDOA_estimatedPos_latlon(pos,:) = [0,0,0];
    else
        [lat,lon,h] = from_xyz_to_site(double(TDOA_estimatedPos_xyz_relativeUE(pos,:)), ueSite(pos));
        TDOA_estimatedPos_latlon(pos,:) = [lat,lon,h];
    end
end



% convert UE_ML_real_est_pos (BS coord system) to UE_ML_estimatedPos_latlon
filesAndFolders = dir(DB_LOS);  
Folders_pos = filesAndFolders(([filesAndFolders.isdir]));
numOfFolders_pos = length(Folders_pos);
UE_ML_estimatedPos_latlon = cell(1, 57);
BS_pos_LOS_index = nan(57, numPos);
TOA_AOA_CRB = cell(1, 57);

for specific_cell = 1:57

    %% Folders    
    DB_LOS = sprintf('../../DB/LOS_%s', name_dataset);
    DB_NLOS = sprintf('../../DB/NLOS_%s', name_dataset);
    
    DB_destination = sprintf('../../DB/FR%d_singleBS_raw_%s/cell%d', FR, name_dataset, specific_cell);
    
    filesAndFolders = dir(DB_LOS);  
    Folders_pos_LOS = filesAndFolders(([filesAndFolders.isdir]));
    filesAndFolders = dir(DB_NLOS);  
    Folders_pos_NLOS = filesAndFolders(([filesAndFolders.isdir]));

    numOfFolders_pos = length(Folders_pos_LOS);
    
    pos=1;

    while(pos<=numOfFolders_pos)

        pos_folder_LOS = sprintf('%s/pos_%d', DB_LOS, pos);
        filesAndFolders_LOS = dir(pos_folder_LOS);  
        Folders_cells_LOS = filesAndFolders_LOS(([filesAndFolders_LOS.isdir]));

        pos_folder_NLOS = sprintf('%s/pos_%d', DB_NLOS, pos);
        filesAndFolders_NLOS = dir(pos_folder_NLOS);  
        Folders_cells_NLOS = filesAndFolders_NLOS(([filesAndFolders_NLOS.isdir]));
      
        cell = specific_cell;

        cell_detected_in_pos = -1;
        
        if any(strcmp({Folders_cells_LOS.name}, sprintf('cell_%d', cell)))
            Folders_cells = Folders_cells_LOS;
            pos_folder = pos_folder_LOS;
            cell_detected_in_pos = 1;

        elseif any(strcmp({Folders_cells_NLOS.name}, sprintf('cell_%d', cell)))
            Folders_cells = Folders_cells_NLOS;
            pos_folder = pos_folder_NLOS;
            cell_detected_in_pos = 0;
        end

        % if it was detected, either in LOS (1) or NLOS (0)
        if (cell_detected_in_pos == 1) || (cell_detected_in_pos == 0)
            
            file_dataset = sprintf('%s/cell_%d/UE_ML_real_est_pos.mat', pos_folder, cell);
            load(file_dataset)

            gNBPos_xyz = from_lat_long_to_xyz(bsSite(specific_cell), ueSite(pos));
            [lat,lon,h] = from_xyz_to_site(double(UE_ML_estimatedPos_xyz + gNBPos_xyz), ueSite(pos));
            UE_ML_estimatedPos_latlon{specific_cell} = cat(1,UE_ML_estimatedPos_latlon{specific_cell}, [lat,lon,h]);

            try
                file_dataset = sprintf('%s/cell_%d/CRB_TOA_AOA.mat', pos_folder, cell);
                load(file_dataset)
                TOA_AOA_CRB{specific_cell} = cat(1,TOA_AOA_CRB{specific_cell}, CRB_TOA_AOA);
            end

            BS_pos_LOS_index(specific_cell, pos) = cell_detected_in_pos;
        else
            UE_ML_estimatedPos_latlon{specific_cell} = cat(1,UE_ML_estimatedPos_latlon{specific_cell}, [0, 0, 0]);
            TOA_AOA_CRB{specific_cell} = cat(1,TOA_AOA_CRB{specific_cell}, nan);
        end
    
        pos = pos+1;
        pos;
    end

    specific_cell


end

DB_destination = sprintf('../../DB/FR%d_singleBS_raw_%s', FR, name_dataset);
save(sprintf('%s/UE_Pos_latlon.mat', DB_destination), 'UE_Pos_latlon');
save(sprintf('%s/TDOA_estimatedPos_latlon.mat', DB_destination), 'TDOA_estimatedPos_latlon');
save(sprintf('%s/UE_ML_estimatedPos_latlon.mat', DB_destination), 'UE_ML_estimatedPos_latlon');
save(sprintf('%s/TOA_AOA_CRB.mat', DB_destination), 'TOA_AOA_CRB');
save(sprintf('%s/BS_pos_LOS_index.mat', DB_destination), 'BS_pos_LOS_index');

% alert_body = 'Check code on Server Nicoli';
% alert_subject = 'MATLAB Terminated';
% alert_api_key = 'TAKyOjO6VipPPf2CV7f';
% alert_url= "https://api.thingspeak.com/alerts/send";
% jsonmessage = sprintf(['{"subject": "%s", "body": "%s"}'], alert_subject,alert_body);
% options = weboptions("HeaderFields", {'Thingspeak-Alerts-API-Key', alert_api_key; 'Content-Type','application/json'});
% result = webwrite(alert_url, jsonmessage, options);
