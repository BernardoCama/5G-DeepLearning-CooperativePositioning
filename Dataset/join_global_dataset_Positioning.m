clear all; clc; close all;

%% Params
channel = 'ADCRM'; % 'ADCRM' 'SFCRM'
channel_element = 'abs_ADCRM'; 
FR = 1;  % 1 or 2
type = 'LOS'; % 'LOS' 'NLOS' 'BOTH'


%% FOLDERS
name_dataset = 'Positioning2';
DB_destination = sprintf('../../DB/FR%d_allBS_raw_%s/%s', FR, name_dataset, type);
mkdir(DB_destination)

dataset_whole = struct();

%% Create dataset
for specific_cell = 1:57

    %% Folders        
    DB_input = sprintf('../../DB/FR%d_singleBS_raw_%s/cell%d', FR, name_dataset, specific_cell);
    
    filesAndFolders = dir(DB_input);  
    Folders_cell = filesAndFolders(([filesAndFolders.isdir]));

    numOfFolders_cell = length(Folders_cell);

    try
        load(sprintf('%s/dataset.mat', DB_input))
        if strcmp(type, 'LOS') || strcmp(type, 'NLOS')
            if strcmp(type, 'LOS')
                type_index = dataset.(sprintf('cell_%d', specific_cell)).LOS == 1;
            else
                type_index = dataset.(sprintf('cell_%d', specific_cell)).LOS == 0;
            end
            if sum(type_index) ~= 0
                dataset.(sprintf('cell_%d', specific_cell)).realPos_latlon = dataset.(sprintf('cell_%d', specific_cell)).realPos_latlon(type_index,:);
                dataset.(sprintf('cell_%d', specific_cell)).realPos_xyz = dataset.(sprintf('cell_%d', specific_cell)).realPos_xyz(type_index,:);
                dataset.(sprintf('cell_%d', specific_cell)).realPos_xyz_cell = dataset.(sprintf('cell_%d', specific_cell)).realPos_xyz_cell(type_index,:);
                dataset.(sprintf('cell_%d', specific_cell)).estimatedPos_xyz = dataset.(sprintf('cell_%d', specific_cell)).estimatedPos_xyz(type_index,:);
                dataset.(sprintf('cell_%d', specific_cell)).estimatedPos_xyz_cell = dataset.(sprintf('cell_%d', specific_cell)).estimatedPos_xyz_cell(type_index,:);
                dataset.(sprintf('cell_%d', specific_cell)).CRB_TDOA = dataset.(sprintf('cell_%d', specific_cell)).CRB_TDOA(type_index,:);


                dataset.(sprintf('cell_%d', specific_cell)).pos = dataset.(sprintf('cell_%d', specific_cell)).pos(type_index,:);
                dataset.(sprintf('cell_%d', specific_cell)).LOS = dataset.(sprintf('cell_%d', specific_cell)).LOS(type_index,:);
                dataset.(sprintf('cell_%d', specific_cell)).est_ToA = dataset.(sprintf('cell_%d', specific_cell)).est_ToA(type_index,:);
                dataset.(sprintf('cell_%d', specific_cell)).est_AoA = dataset.(sprintf('cell_%d', specific_cell)).est_AoA(type_index,:);
                dataset.(sprintf('cell_%d', specific_cell)).est_AoA_cell = dataset.(sprintf('cell_%d', specific_cell)).est_AoA_cell(type_index,:);
                dataset.(sprintf('cell_%d', specific_cell)).true_ToA = dataset.(sprintf('cell_%d', specific_cell)).true_ToA(type_index,:);
                dataset.(sprintf('cell_%d', specific_cell)).true_AoA = dataset.(sprintf('cell_%d', specific_cell)).true_AoA(type_index,:);
                dataset.(sprintf('cell_%d', specific_cell)).true_AoA_cell = dataset.(sprintf('cell_%d', specific_cell)).true_AoA_cell(type_index,:);
                dataset.(sprintf('cell_%d', specific_cell)).ML_estPos_xyz = dataset.(sprintf('cell_%d', specific_cell)).ML_estPos_xyz(type_index,:);
                dataset.(sprintf('cell_%d', specific_cell)).ML_estPos_xyz_cell = dataset.(sprintf('cell_%d', specific_cell)).ML_estPos_xyz_cell(type_index,:);
                dataset.(sprintf('cell_%d', specific_cell)).CRB_TOA_AOA = dataset.(sprintf('cell_%d', specific_cell)).CRB_TOA_AOA(type_index,:);
                dataset.(sprintf('cell_%d', specific_cell)).input = dataset.(sprintf('cell_%d', specific_cell)).input(:,type_index, :);

            end
        end
        dataset_whole.(sprintf('cell_%d', specific_cell)) = dataset.(sprintf('cell_%d', specific_cell));
    end
    specific_cell

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
