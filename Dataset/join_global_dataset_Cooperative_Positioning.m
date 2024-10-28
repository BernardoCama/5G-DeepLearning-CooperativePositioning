clear all; clc; close all;

%% Params
channel = 'ADCRM'; % 'ADCRM' 'SFCRM'
channel_element = 'abs_ADCRM'; 
FR = 1;  % 1 or 2
type = 'BOTH'; % 'LOS' 'NLOS' 'BOTH'

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

%% FOLDERS
positioning = 'Positioning1';
name_dataset = sprintf('Cooperative_%s', positioning);
DB_destination = sprintf('../../DB/FR%d_allBS_raw_%s/%s', FR, name_dataset, type);
mkdir(DB_destination)
      
% load input dataset
DB_input = sprintf('../../DB/FR%d_allBS_raw_%s/%s/', FR, positioning, type);
load(sprintf('%s/dataset.mat', DB_input))

% load BS-position matrix
DB_input = sprintf('../../DB/FR%d_singleBS_raw_%s/BS_pos_LOS_index.mat', FR, positioning);
load(DB_input)

%% 
dataset_whole = struct();

for pos = 1:numPos
    
    if strcmp(type, 'LOS')
        cell_detected = find(BS_pos_LOS_index(:, pos) == 1);
    elseif strcmp(type, 'NLOS')
        cell_detected = find(BS_pos_LOS_index(:, pos) == 0);
    elseif strcmp(type, 'BOTH')
        cell_detected = find((BS_pos_LOS_index(:, pos) == 1) | (BS_pos_LOS_index(:, pos) == 0));
    end

    dataset_temp = struct();

    dataset_temp.realPos_latlon = [];
    dataset_temp.realPos_xyz = [];
    dataset_temp.realPos_xyz_cell = [];
    dataset_temp.estimatedPos_xyz = [];
    dataset_temp.estimatedPos_xyz_cell = [];
    dataset_temp.CRB_TDOA = [];

    dataset_temp.pos = [];
    dataset_temp.LOS = [];
    dataset_temp.cell_index = [];
    dataset_temp.est_ToA = [];
    dataset_temp.est_AoA = [];
    dataset_temp.est_AoA_cell = [];
    dataset_temp.true_ToA = [];
    dataset_temp.true_AoA = [];
    dataset_temp.true_AoA_cell = [];
    dataset_temp.ML_estPos_xyz = [];
    dataset_temp.ML_estPos_xyz_cell = [];
    dataset_temp.CRB_TOA_AOA = [];
    dataset_temp.input = [];

    for specific_cell = cell_detected'

        if strcmp(type, 'LOS')
            type_index = dataset.(sprintf('cell_%d', specific_cell)).LOS == 1;
        elseif strcmp(type, 'NLOS')
            type_index = dataset.(sprintf('cell_%d', specific_cell)).LOS == 0;
        elseif strcmp(type, 'BOTH')
            type_index = (dataset.(sprintf('cell_%d', specific_cell)).LOS == 1) | (dataset.(sprintf('cell_%d', specific_cell)).LOS == 0);
        end
        pos_index = dataset.(sprintf('cell_%d', specific_cell)).pos == pos;
        indexes = type_index & pos_index;
   
        dataset_temp.realPos_latlon = cat(3, dataset_temp.realPos_latlon, dataset.(sprintf('cell_%d', specific_cell)).realPos_latlon(indexes, :));
        dataset_temp.realPos_xyz = cat(3, dataset_temp.realPos_xyz, dataset.(sprintf('cell_%d', specific_cell)).realPos_xyz(indexes, :));
        dataset_temp.realPos_xyz_cell = cat(3, dataset_temp.realPos_xyz_cell, dataset.(sprintf('cell_%d', specific_cell)).realPos_xyz_cell(indexes, :));
        dataset_temp.estimatedPos_xyz = cat(3, dataset_temp.estimatedPos_xyz, dataset.(sprintf('cell_%d', specific_cell)).estimatedPos_xyz(indexes, :));
        dataset_temp.estimatedPos_xyz_cell = cat(3, dataset_temp.estimatedPos_xyz_cell, dataset.(sprintf('cell_%d', specific_cell)).estimatedPos_xyz_cell(indexes, :));
        try
        dataset_temp.CRB_TDOA = cat(3, dataset_temp.CRB_TDOA, dataset.(sprintf('cell_%d', specific_cell)).CRB_TDOA(indexes, :));
        end

        dataset_temp.pos = cat(3, dataset_temp.pos, dataset.(sprintf('cell_%d', specific_cell)).pos(indexes, :));
        dataset_temp.LOS = cat(3, dataset_temp.LOS, dataset.(sprintf('cell_%d', specific_cell)).LOS(indexes, :));
        num_samples = size(dataset.(sprintf('cell_%d', specific_cell)).input(:,indexes, :), 2);
        dataset_temp.cell_index = cat(3, dataset_temp.cell_index, repmat(specific_cell, num_samples, 1));
        dataset_temp.est_ToA = cat(3, dataset_temp.est_ToA, dataset.(sprintf('cell_%d', specific_cell)).est_ToA(indexes, :));
        dataset_temp.est_AoA = cat(3, dataset_temp.est_AoA, dataset.(sprintf('cell_%d', specific_cell)).est_AoA(indexes, :));
        dataset_temp.est_AoA_cell = cat(3, dataset_temp.est_AoA_cell, dataset.(sprintf('cell_%d', specific_cell)).est_AoA_cell(indexes, :));
        dataset_temp.true_ToA = cat(3, dataset_temp.true_ToA, dataset.(sprintf('cell_%d', specific_cell)).true_ToA(indexes, :));
        dataset_temp.true_AoA = cat(3, dataset_temp.true_AoA, dataset.(sprintf('cell_%d', specific_cell)).true_AoA(indexes, :));
        dataset_temp.true_AoA_cell = cat(3, dataset_temp.true_AoA_cell, dataset.(sprintf('cell_%d', specific_cell)).true_AoA_cell(indexes, :));
        dataset_temp.ML_estPos_xyz = cat(3, dataset_temp.ML_estPos_xyz, dataset.(sprintf('cell_%d', specific_cell)).ML_estPos_xyz(indexes, :));
        dataset_temp.ML_estPos_xyz_cell = cat(3, dataset_temp.ML_estPos_xyz_cell, dataset.(sprintf('cell_%d', specific_cell)).ML_estPos_xyz_cell(indexes, :));
        try
        dataset_temp.CRB_TOA_AOA = cat(3, dataset_temp.CRB_TOA_AOA, dataset.(sprintf('cell_%d', specific_cell)).CRB_TOA_AOA(indexes, :));
        end

        dataset_temp.input = cat(4, dataset_temp.input, dataset.(sprintf('cell_%d', specific_cell)).input(:,indexes, :));
        
    end
    dataset_whole.(sprintf('pos_%d', pos)) = dataset_temp;
   
    pos
end


if ~ isempty(dataset_whole)
    dataset = dataset_whole;
    save(sprintf('%s/dataset.mat', DB_destination), 'dataset', '-v7.3');
end



alert_body = 'Check code on Server Nicoli';
alert_subject = 'MATLAB Terminated';
alert_api_key = 'TAKyOjO6VipPPf2CV7f';
alert_url= "https://api.thingspeak.com/alerts/send";
jsonmessage = sprintf(['{"subject": "%s", "body": "%s"}'], alert_subject,alert_body);
options = weboptions("HeaderFields", {'Thingspeak-Alerts-API-Key', alert_api_key; 'Content-Type','application/json'});
result = webwrite(alert_url, jsonmessage, options);
