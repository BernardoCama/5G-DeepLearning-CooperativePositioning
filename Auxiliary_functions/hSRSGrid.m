function [Grid,dispGrid] = hSRSGrid(carrier,srs,Duration,displayGrid,chplevels)
% [GRID,DISPGRID] = hSRSGrid(CARRIER,SRS,DURATION,DISPLAYGRID,CHPLEVELS)
% returns a multi-slot OFDM resource grid GRID containing a set of sounding
% reference signals in a carrier, as specified by the configuration objects
% CARRIER and SRS. This function also returns a scaled version of the grid
% used for display purposes. The optional input DURATION (Default 1)
% specifies the number of slots of the generated grid. The resource grid
% can be displayed using the optional input DISPLAYGRID (Default false).
% CHPLEVELS specifies the channel power levels for display purposes only
% and it must be of the same size as SRS.

    numSRS = length(srs);
    if nargin < 5
        chplevels = 1:-1/numSRS:1/numSRS;
        if nargin < 4
            displayGrid = false;
            if nargin < 3
                Duration = 1;
            end
        end
    end
    
    SymbolsPerSlot = carrier.SymbolsPerSlot;
    emptySlotGrid = nrResourceGrid(carrier,max([srs(:).NumSRSPorts])); % Initialize slot grid
    
    % Create the SRS symbols and indices and populate the grid with the SRS symbols
    Grid = repmat(emptySlotGrid,1,Duration);
    dispGrid = repmat(emptySlotGrid,1,Duration); % Frame-size grid for display
    for ns = 0:Duration-1
        slotGrid = emptySlotGrid;
        dispSlotGrid = emptySlotGrid; % Slot-size grid for display
        for ich = 1:numSRS
            srsIndices = nrSRSIndices(carrier,srs(ich));
            srsSymbols = nrSRS(carrier,srs(ich));
            slotGrid(srsIndices) = srsSymbols;
            dispSlotGrid(srsIndices) = chplevels(ich)*srsSymbols; % Scale the SRS for display only
        end
        OFDMSymIdx = ns*SymbolsPerSlot + (1:SymbolsPerSlot);
        Grid(:,OFDMSymIdx,:) = slotGrid;
        dispGrid(:,OFDMSymIdx,:) = dispSlotGrid;
        carrier.NSlot = carrier.NSlot+1;
    end
    
    if displayGrid
        plotGrid(dispGrid(:,:,1),chplevels,"SRS " + (1:numSRS)'); 
    end

    function varargout = plotGrid(Grid,chplevels,leg)
    % plotGrid(GRID, CHPLEVEL,LEG) displays a resource grid GRID containing
    % channels or signals at different power levels CHPLEVEL and create a
    % legend for these using a cell array of character vector LEG
    
        if nargin < 3
            leg = {'SRS'};
            if nargin < 2
                chplevels = 1;
            end
        end
        
        cmap = colormap(gcf);
        chpscale = length(cmap); % Scaling factor
        
        h = figure;
        image(0:size(Grid,2)-1,(0:size(Grid,1)-1)/12,chpscale*abs(Grid(:,:,1))); % Multiplied with scaling factor for better visualization
        axis xy;
        
        title('Carrier Grid Containing SRS')
        xlabel('OFDM Symbol'); ylabel('RB');
        
        clevels = chpscale*chplevels(:);
        N = length(clevels);
        L = line(ones(N),ones(N),'LineWidth',8); % Generate lines
        
        % Index the color map and associate the selected colors with the lines
        set(L,{'color'},mat2cell(cmap( min(1+fix(clevels),length(cmap) ),:),ones(1,N),3)); % Set the colors according to cmap
        
        % Create legend
        legend(leg(:));
        
        if nargout > 0 
            varargout = {h};
        end
    end


end
    

