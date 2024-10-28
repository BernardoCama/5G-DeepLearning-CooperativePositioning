clear all; clc; close all;

%% Params
channel = 'ADCRM'; % 'ADCRM' 'SFCRM'
channel_element = 'abs_ADCRM'; 
FR = 1;  % 1 or 2

%% UE
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


%% FOLDERS
name_dataset = 'Positioning2';
DB_LOS = sprintf('../../DB/LOS_%s', name_dataset);
DB_NLOS = sprintf('../../DB/NLOS_%s', name_dataset);
DB_destination = sprintf('../../DB/FR%d_singleBS_raw_%s', FR, name_dataset);
mkdir(DB_destination)

filesAndFolders = dir(DB_LOS);  
Folders_pos = filesAndFolders(([filesAndFolders.isdir]));
numOfFolders_pos = length(Folders_pos);

TDOA_estimatedPos_xyz_relativeUE = [];
TDOA_LOS_index = [];
TDOA_CRB = [];

pos=1; % 3 2725
while(pos<=numOfFolders_pos)
    try
        pos_folder = sprintf('%s/pos_%d', DB_LOS, pos);
        filesAndFolders_LOS = dir(pos_folder);  
        Folders_cells_LOS = filesAndFolders_LOS(([filesAndFolders_LOS.isdir]));        
        file_dataset = sprintf('%s/estimatedPos_xyz.mat', pos_folder);
        load(file_dataset)
        TDOA_estimatedPos_xyz_relativeUE = cat(1, TDOA_estimatedPos_xyz_relativeUE, estimatedPos_xyz);

        file_dataset = sprintf('%s/CRB_TDOA.mat', pos_folder);
        load(file_dataset)
        TDOA_CRB = cat(1, TDOA_CRB, CRB_TDOA);

        if length({Folders_cells_LOS.name}) >= 5 % if we see 4 or more BS in LOS
            TDOA_LOS_index = cat(1, TDOA_LOS_index, 1);
        else
            TDOA_LOS_index = cat(1, TDOA_LOS_index, 0);
        end
    catch
        try
            pos_folder = sprintf('%s/pos_%d', DB_NLOS, pos);
            file_dataset = sprintf('%s/estimatedPos_xyz.mat', pos_folder);
            load(file_dataset)
            TDOA_estimatedPos_xyz_relativeUE = cat(1, TDOA_estimatedPos_xyz_relativeUE, estimatedPos_xyz);

            file_dataset = sprintf('%s/CRB_TDOA.mat', pos_folder);
            load(file_dataset)
            TDOA_CRB = cat(1, TDOA_CRB, CRB_TDOA);

            TDOA_LOS_index = cat(1, TDOA_LOS_index, 0);
        catch
            TDOA_estimatedPos_xyz_relativeUE = cat(1, TDOA_estimatedPos_xyz_relativeUE, [0,0,0]);

            TDOA_CRB = cat(1, TDOA_CRB, nan);

            TDOA_LOS_index = cat(1, TDOA_LOS_index, 0);
        end
    end
    pos = pos+1;
    pos
end
save(sprintf('%s/TDOA_estimatedPos_xyz_relativeUE.mat', DB_destination), 'TDOA_estimatedPos_xyz_relativeUE');
save(sprintf('%s/TDOA_LOS_index.mat', DB_destination), 'TDOA_LOS_index');
save(sprintf('%s/TDOA_CRB.mat', DB_destination), 'TDOA_CRB');

%% BS
% Define center location site (cells 1-3)
centerSite = rxsite('Name','DEIB', ...
    'Latitude',45.4787,...
    'Longitude',9.2330,...
    'AntennaHeight',1);

antHeight = 25; % m
downtilt = 15; % degrees

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


%% Create dataset
for specific_cell = 1:57
    clear dataset

    %% Folders    
    DB_LOS = sprintf('../../DB/LOS_%s', name_dataset);
    DB_NLOS = sprintf('../../DB/NLOS_%s', name_dataset);
    
    DB_destination = sprintf('../../DB/FR%d_singleBS_raw_%s/cell%d', FR, name_dataset, specific_cell);

    mkdir(DB_destination)
    
    filesAndFolders = dir(DB_LOS);  
    Folders_pos_LOS = filesAndFolders(([filesAndFolders.isdir]));
    filesAndFolders = dir(DB_NLOS);  
    Folders_pos_NLOS = filesAndFolders(([filesAndFolders.isdir]));

    numOfFolders_pos = length(Folders_pos_LOS);
    
    pos=1; % 3 2725

    dataset = struct();
    dataset_realPos_latlon = [];
    dataset_realPos_xyz = [];
    dataset_realPos_xyz_cell = [];
    dataset_estimatedPos_xyz = [];
    dataset_estimatedPos_xyz_cell = [];
    dataset_CRB_TDOA = [];

    dataset_pos = [];
    dataset_LOS = [];
    dataset_est_ToA = [];
    dataset_est_AoA = [];
    dataset_est_AoA_cell = [];
    dataset_true_ToA = [];
    dataset_true_AoA = [];
    dataset_true_AoA_cell = [];
    dataset_ML_estPos_xyz = [];
    dataset_ML_estPos_xyz_cell = [];
    dataset_CRB_TOA_AOA = [];
    
    dataset_input = [];
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

        if (cell_detected_in_pos == 1) || (cell_detected_in_pos == 0)

            % BS position wrt to UE coord
            gNBPos_xyz = from_lat_long_to_xyz(bsSite(specific_cell), ueSite(pos));
            
            file_dataset = sprintf('%s/cell_%d/%s/%s.mat', pos_folder, cell, channel, channel_element);
            load(file_dataset)
            dataset_input = cat (2, dataset_input, eval(channel)); % 2 OFDM temporal dimension

            num_samples = size(ADCRM,2);

            dataset_pos = cat(1, dataset_pos, repmat(pos, num_samples, 1));
            dataset_LOS = cat(1, dataset_LOS, repmat(cell_detected_in_pos, num_samples, 1));

            file_dataset = sprintf('%s/cell_%d/est_measurements.mat', pos_folder, cell);
            load(file_dataset)
            dataset_est_ToA = cat(1, dataset_est_ToA, repmat(est_ToA, num_samples, 1));
            dataset_est_AoA = cat(1, dataset_est_AoA, repmat(est_AoA', num_samples, 1));
            dataset_est_AoA_cell = cat(1, dataset_est_AoA_cell, ...
                                    repmat((mod(est_AoA - [cellAngles(specific_cell); -downtilt], 360))', ...
                                    num_samples, 1));

            file_dataset = sprintf('%s/cell_%d/measurements.mat', pos_folder, cell);
            load(file_dataset)
            dataset_true_ToA = cat(1, dataset_true_ToA, repmat(ToA, num_samples, 1));
            dataset_true_AoA = cat(1, dataset_true_AoA, repmat(AoA', num_samples, 1));
            dataset_true_AoA_cell = cat(1, dataset_true_AoA_cell, ...
                                    repmat((mod(AoA - [cellAngles(specific_cell); -downtilt], 360))', ...
                                    num_samples, 1));

            file_dataset = sprintf('%s/cell_%d/UE_ML_real_est_pos.mat', pos_folder, cell);
            load(file_dataset)
            dataset_ML_estPos_xyz = cat(1, dataset_ML_estPos_xyz, repmat(UE_ML_estimatedPos_xyz, num_samples, 1));
            dataset_ML_estPos_xyz_cell = cat(1, dataset_ML_estPos_xyz_cell, ...
                                    repmat(rotate_axis(UE_ML_estimatedPos_xyz, cellAngles(specific_cell) + 90, -downtilt, 0), ...
                                    num_samples, 1));

            file_dataset = sprintf('%s/cell_%d/CRB_TOA_AOA.mat', pos_folder, cell);
            load(file_dataset)
            dataset_CRB_TOA_AOA = cat(1, dataset_CRB_TOA_AOA, repmat(CRB_TOA_AOA, num_samples, 1));

            % estimatedPos_xyz is with respect to the UE coord system
            % (placed in [0,0,0])
            % In coord system of BS: 
            % estimatedPos_xyz = estimatedPos_xyz - gNBPos_xyz 
            % estimatedPos_xyz = estimatedPos_xyz + UE_ML_realPos_xyz 
            file_dataset = sprintf('%s/estimatedPos_xyz.mat', pos_folder);
            load(file_dataset)
            estimatedPos_xyz = estimatedPos_xyz + UE_ML_realPos_xyz;
            dataset_estimatedPos_xyz = cat(1, dataset_estimatedPos_xyz, repmat(estimatedPos_xyz, num_samples, 1));
            dataset_estimatedPos_xyz_cell = cat(1, dataset_estimatedPos_xyz_cell, ...
                                    repmat(rotate_axis(estimatedPos_xyz, cellAngles(specific_cell) + 90, -downtilt, 0), ...
                                    num_samples, 1));
            dataset_realPos_latlon = cat(1, dataset_realPos_latlon, repmat(uePositions(pos,:), num_samples, 1));
            dataset_realPos_xyz = cat(1, dataset_realPos_xyz, repmat(UE_ML_realPos_xyz, num_samples, 1));
            dataset_realPos_xyz_cell = cat(1, dataset_realPos_xyz_cell, ...
                                    repmat(rotate_axis(UE_ML_realPos_xyz, cellAngles(specific_cell) + 90, -downtilt, 0), ...
                                    num_samples, 1));

            file_dataset = sprintf('%s/CRB_TDOA.mat', pos_folder);
            load(file_dataset)
            dataset_CRB_TDOA = cat(1, dataset_CRB_TDOA, repmat(CRB_TDOA, num_samples, 1));
            

        end
    
        pos = pos+1;
        pos;
    end

    dataset.(sprintf('cell_%d', specific_cell)) = struct();
    dataset.(sprintf('cell_%d', specific_cell)).realPos_latlon = dataset_realPos_latlon;
    dataset.(sprintf('cell_%d', specific_cell)).realPos_xyz = dataset_realPos_xyz;
    dataset.(sprintf('cell_%d', specific_cell)).realPos_xyz_cell = dataset_realPos_xyz_cell;
    dataset.(sprintf('cell_%d', specific_cell)).estimatedPos_xyz = dataset_estimatedPos_xyz;
    dataset.(sprintf('cell_%d', specific_cell)).estimatedPos_xyz_cell = dataset_estimatedPos_xyz_cell;
    dataset.(sprintf('cell_%d', specific_cell)).CRB_TDOA = dataset_CRB_TDOA;

    dataset.(sprintf('cell_%d', specific_cell)).pos = dataset_pos;
    dataset.(sprintf('cell_%d', specific_cell)).LOS = dataset_LOS;
    dataset.(sprintf('cell_%d', specific_cell)).est_ToA = dataset_est_ToA;
    dataset.(sprintf('cell_%d', specific_cell)).est_AoA = dataset_est_AoA;
    dataset.(sprintf('cell_%d', specific_cell)).est_AoA_cell = dataset_est_AoA_cell;
    dataset.(sprintf('cell_%d', specific_cell)).true_ToA = dataset_true_ToA;
    dataset.(sprintf('cell_%d', specific_cell)).true_AoA = dataset_true_AoA;
    dataset.(sprintf('cell_%d', specific_cell)).true_AoA_cell = dataset_true_AoA_cell;
    dataset.(sprintf('cell_%d', specific_cell)).ML_estPos_xyz = dataset_ML_estPos_xyz;
    dataset.(sprintf('cell_%d', specific_cell)).ML_estPos_xyz_cell = dataset_ML_estPos_xyz_cell;
    dataset.(sprintf('cell_%d', specific_cell)).CRB_TOA_AOA = dataset_CRB_TOA_AOA;

    dataset.(sprintf('cell_%d', specific_cell)).input = dataset_input;
    
    if ~ isempty(dataset_input)
        specific_cell
        save(sprintf('%s/dataset.mat', DB_destination), 'dataset', '-v7.3');
    end

end

alert_body = 'Check code on Server Nicoli';
alert_subject = 'MATLAB Terminated';
alert_api_key = 'TAKyOjO6VipPPf2CV7f';
alert_url= "https://api.thingspeak.com/alerts/send";
jsonmessage = sprintf(['{"subject": "%s", "body": "%s"}'], alert_subject,alert_body);
options = weboptions("HeaderFields", {'Thingspeak-Alerts-API-Key', alert_api_key; 'Content-Type','application/json'});
result = webwrite(alert_url, jsonmessage, options);
