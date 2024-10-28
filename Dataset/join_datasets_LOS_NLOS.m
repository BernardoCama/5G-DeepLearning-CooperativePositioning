clear all; clc; close all;



for specific_cell = 1:57
    clear dataset

    %% Folders
    type = 'NLOS'; % 'LOS' 'NLOS'
    channel = 'ADCRM'; % 'ADCRM' 'SFCRM'
    channel_element = 'abs_ADCRM'; 
    FR = 1;  % 1 or 2
    
    switch type
        case 'LOS'
            boolean_los = 1;
        case 'NLOS'
            boolean_los = 0;
    end
    
    % If we join dataset of all cells or specific cell
    % specific_cell = 1;  % 0 all cells
    
    DB = sprintf('../../DB/%s', type);
    
    if (specific_cell == 0)
        DB_destination = sprintf('../../DB/FR%d_allBS_raw/%s', FR, type);
    else
        DB_destination = sprintf('../../DB/FR%d_singleBS_raw/cell%d/%s', FR, specific_cell, type);
    end
    mkdir(DB_destination)
    
    filesAndFolders = dir(DB);  
    Folders_pos = filesAndFolders(([filesAndFolders.isdir]));
    
    numOfFolders_pos = length(Folders_pos);
    
    pos=2725; % 3 2725
    dataset = [];
    while(pos<=numOfFolders_pos)
        pos_folder = sprintf('%s/%s/%s', DB, Folders_pos(pos).name, channel);
    
        if ~ specific_cell
            filesAndFolders = dir(pos_folder);  
            Folders_cells = filesAndFolders(([filesAndFolders.isdir]));
            numOfFolders_cells = length(Folders_cells);
    
            if numOfFolders_cells>2
                cell = 3;
                file_dataset = sprintf('%s/%s/%s.mat', pos_folder, Folders_cells(cell).name, channel_element);
                load(file_dataset)
                % save(sprintf('%s/%s.mat', DB, channel_element), channel);
    
                dataset = cat (2, dataset, eval(channel)); % 2 OFDM temporal dimension
                
                while (cell<numOfFolders_cells)
                    
    
                    cell = cell + 1;
    
                    file_dataset = sprintf('%s/%s/%s.mat', pos_folder, Folders_cells(cell).name, channel_element);
                    load(file_dataset)
                    dataset = cat (2, dataset, eval(channel)); % 2 OFDM temporal dimension
                    % save(sprintf('%s/%s.mat', DB, channel_element), channel, '-append');
                end
    
    
            end
    
        else
            filesAndFolders = dir(pos_folder);  
            Folders_cells = filesAndFolders(([filesAndFolders.isdir]));
            cell = specific_cell;
            
            if any(strcmp({Folders_cells.name}, sprintf('cell_%d', cell)))
    
                file_dataset = sprintf('%s/cell_%d/%s.mat', pos_folder, cell, channel_element);
                load(file_dataset)
                dataset = cat (2, dataset, eval(channel)); % 2 OFDM temporal dimension
    
            end
            
    
    
        end
        
    
    
        pos = pos+1;
        pos-3
    end
    
    if ~ isempty(dataset)
        save(sprintf('%s/%s.mat', DB_destination, channel_element), 'dataset');
    end

end

% alert_body = 'Check code on Server Nicoli';
% alert_subject = 'MATLAB Terminated';
% alert_api_key = 'TAKyOjO6VipPPf2CV7f';
% alert_url= "https://api.thingspeak.com/alerts/send";
% jsonmessage = sprintf(['{"subject": "%s", "body": "%s"}'], alert_subject,alert_body);
% options = weboptions("HeaderFields", {'Thingspeak-Alerts-API-Key', alert_api_key; 'Content-Type','application/json'});
% result = webwrite(alert_url, jsonmessage, options);
