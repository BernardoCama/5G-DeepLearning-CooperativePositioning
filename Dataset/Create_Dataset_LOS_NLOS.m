clearvars -except viewer; 
clc; close all
rng('default');                    % Set RNG state for repeatability


%% Folders
type = 'LOS'; % 'LOS' 'NLOS'
switch type
    case 'LOS'
        boolean_los = 1;
    case 'NLOS'
        boolean_los = 0;
end
DB = sprintf('../../DB/%s', type);
[status, msg, msgID] = mkdir(sprintf('%s', DB));

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
for pos = 1:numPos
    [status, msg, msgID] = mkdir(sprintf('%s/pos_%d/ADCRM', DB, pos));
    [status, msg, msgID] = mkdir(sprintf('%s/pos_%d/SFCRM', DB, pos));
end


%% Simulation Paramters
PerfectChannelEstimator = 1;
dimensions = 2; % 2D or 3D
plotting = 0; 
add_awgn = 0;
FR = 1; % 1 or 2 % Frequency ranges

nFrames = 1; % Number of 10 ms frames per position
switch FR
    case 1
        SCS = 30;
        NSizeGrid = 273;
        fc = 4e9;    % Carrier frequency (Hz)
    case 2
        SCS = 120;
        NSizeGrid = 273;
        fc = 30e9;    % Carrier frequency (Hz)
end
% (52 RBs at 15 kHz SCS for 10 MHz BW)
% (106 RBs at 15 kHz SCS for 20 MHz BW)
% (216 RBs at 15 kHz SCS for 40 MHz BW)
% (270 RBs at 15 kHz SCS for 50 MHz BW)

% (51 RBs at 30 kHz SCS for 20 MHz BW)
% (78 RBs at 30 kHz SCS for 30 MHz BW)
% (106 RBs at 30 kHz SCS for 40 MHz BW)
% (217 RBs at 30 kHz SCS for 80 MHz BW)
% (273 RBs at 30 kHz SCS for 100 MHz BW)

% (24 RBs at 60 kHz SCS for 20 MHz BW)
% (51 RBs at 60 kHz SCS for 40 MHz BW)
% (135 RBs at 60 kHz SCS for 100 MHz BW)
% (273 RBs at 60 kHz SCS for 200 MHz BW)


%% Physical Parameters
c = physconst('LightSpeed');
lambda = c/fc;
reflectionsOrder = 3;                 % number of reflections for ray tracing analysis (0 for LOS)

% Noise AWGN
EbNo = 40;              % in dB
bitsPerCarrier = 2;     % pdsch(1, 1).Modulation = 'QPSK'
codeRate = 193/1024;
SNR = convertSNR(EbNo,"ebno", ...
  "BitsPerSymbol",bitsPerCarrier, ...
  "CodingRate",codeRate);
SNRLin = 10^(SNR/10);      % Linear
% bitsPerSymbol = bitsPerCarrier*codeRate;
% SNRLin = 10^(EbNo/10) * bitsPerSymbol;      % Linear


%% UE Parameters
% Configure UE position
% uePositions = [45.4785, 9.2345;
%                45.4785, 9.2347];
numPos = size(uePositions, 1);
ueAntSize = [1 1];                    % number of rows and columns in rectangular array (UE).
NTxAnts = prod(ueAntSize);

ueArray = phased.NRRectangularPanelArray( ...
    'Size',[ueAntSize(1:2) 1 1], ...
    'Spacing', [0.5*lambda*[1 1] 1 1]);
ueArray.ElementSet = {phased.IsotropicAntennaElement};   % isotropic antenna element


%% BS Parameters
% Define center location site (cells 1-3)
centerSite = rxsite('Name','DEIB', ...
    'Latitude',45.4787,...
    'Longitude',9.2330);

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


%% Create BS Sites
antHeight = 25; % m
txPowerDBm = 44; % Total transmit power in dBm
txPower = 10.^((txPowerDBm-30)/10); % Convert dBm to W

% Create cell transmitter sites
bsSite = rxsite('Name',cellNames, ...
    'Latitude',cellLats, ...
    'Longitude',cellLons, ...
    'AntennaAngle',cellAngles, ...
    'AntennaHeight',antHeight);

bsAntSize = [8 8];       % number of rows and columns in rectangular array (base station)
number_polarizations = 1;
NRxAnts = prod(bsAntSize)*number_polarizations; 

% % Too complex from computational POV
% % Design half-wavelength rectangular microstrip patch antenna
% patchElement = design(patchMicrostrip,fc);
% patchElement.Width = patchElement.Length;
% patchElement.Tilt = 90;
% patchElement.TiltAxis = [0 1 0];

% Define pattern parameters
azvec = -180:180;
elvec = -90:90;
Am = 30; % Maximum attenuation (dB)
tilt = 0; % Tilt angle
az3dB = 65; % 3 dB bandwidth in azimuth
el3dB = 65; % 3 dB bandwidth in elevation
% Define antenna pattern
[az,el] = meshgrid(azvec,elvec);
azMagPattern = -12*(az/az3dB).^2;
elMagPattern = -12*((el-tilt)/el3dB).^2;
combinedMagPattern = azMagPattern + elMagPattern;
combinedMagPattern(combinedMagPattern<-Am) = -Am; % Saturate at max attenuation
phasepattern = zeros(size(combinedMagPattern));
% Create antenna element
antennaElement = phased.CustomAntennaElement(...
    'AzimuthAngles',azvec, ...
    'ElevationAngles',elvec, ...
    'MagnitudePattern',combinedMagPattern, ...
    'PhasePattern',phasepattern);

% Base station array (single panel)
cellAntenna = phased.URA('Size', [bsAntSize(1:2)], ...
                         'Element',antennaElement, ...  antennaElement ,patchElement phased.IsotropicAntennaElement
                         'ElementSpacing', [0.5*lambda*[1 1]],...
                          'ArrayNormal', 'x');
% pattern(cellAntenna,fc);
viewArray(cellAntenna, 'ShowTaper', true)

% Assign the antenna element for each cell transmitter
downtilt = 15;
for rx = bsSite
    rx.Antenna = cellAntenna;
    rx.AntennaAngle = [rx.AntennaAngle; -downtilt]; % -downtilt
end

% Import and Visualize 3-D Environment with Buildings for Ray Tracing
if exist('viewer','var') && isvalid(viewer) % viewer handle exists and viewer window is open
    viewer.clearMap();
else
    viewer = siteviewer("Basemap","openstreetmap","Buildings","Politecnico.osm");    
end
% close(viewer)
if plotting == 1
    estimator = phased.MUSICEstimator2D(...
    'SensorArray',cellAntenna,...
    'OperatingFrequency',fc,...
    'NumSignalsSource','Property',...
    'DOAOutputPort',true, ...
    'ForwardBackwardAveraging', true, ...
    'NumSignals',1,...
    'AzimuthScanAngles',-180:.1:180,...
    'ElevationScanAngles',0:.5:90);
end

% Show sites on a map
j = 1;
for gNBIdx = 1:numCells
    show(bsSite(gNBIdx), "Icon",sprintf("placeholder_rx%d.png", j), "IconSize", [150, 150]);
    % pattern(bsSite(gNBIdx),fc,"Size",4);
    if mod(gNBIdx,3) == 0
        j = j + 1;
    end
end
% pause()


%% Carrier Properties
nLayers = min(NTxAnts,NRxAnts);

% Configure carrier properties
carrier = repmat(nrCarrierConfig,1,numCells);
for gNBIdx = 1:numCells
    carrier(gNBIdx).NCellID = gNBIdx;
    carrier(gNBIdx).SubcarrierSpacing = SCS;
    carrier(gNBIdx).NSizeGrid = NSizeGrid;
end
validateCarriers(carrier);

% Compute OFDM information using first carrier, assuming all carriers are
% at same sampling rate
ofdmInfo = nrOFDMInfo(carrier(1));
sr = ofdmInfo.SampleRate; % sampling rate 
cellRange = min(ofdmInfo.CyclicPrefixLengths)/sr * c / 2;


%% SRS Configuration
% Configure a periodic multi-port SRS and enable frequency hopping
srs_single = nrSRSConfig;
srs_single.NumSRSSymbols = 4;          % Number of OFDM symbols allocated per slot (1,2,4)
srs_single.SymbolStart = 0;            % Starting OFDM symbol within a slot
srs_single.NumSRSPorts = NTxAnts;      % Number of SRS antenna ports (1,2,4).
srs_single.FrequencyStart = 0;         % Frequency position of the SRS in BWP in RBs
srs_single.NRRC = 0;                   % Additional offset from FreqStart specified in blocks of 4 PRBs (0...67)
srs_single.CSRS = 63;                  % Bandwidth configuration C_SRS (0...63). Increases with C_SRS. It controls the allocated bandwidth to the SRS
srs_single.BSRS = 0;                   % Bandwidth configuration B_SRS (0...3). Decreases with B_SRS. It controls the allocated bandwidth to the SRS
srs_single.BHop = 0;                   % Frequency hopping configuration (0...3). Set BHop < BSRS to enable frequency hopping
srs_single.KTC = 2;                    % Comb number (2,4). Frequency density in subcarriers
srs_single.Repetition = 4;             % Repetition (1,2,4). It disables frequency hopping in blocks of |Repetition| symbols
srs_single.SRSPeriod = [2 0];          % Periodicity and offset in slots. SRSPeriod(2) must be < SRSPeriod(1)
srs_single.ResourceType = 'periodic';%'periodic';  % Resource type ('periodic', 'semi-persistent','aperiodic'). Use 'aperiodic' to disable inter-slot frequency hopping


%% PUSCH Configuration
pusch_single = nrPUSCHConfig;
pusch_single.PRBSet = 0:NSizeGrid-1;

pusch_single.MappingType = 'A';
pusch_single.RNTI = 1;

% Define the transform precoding enabling, layering and transmission scheme
pusch_single.TransformPrecoding = false; % Enable/disable transform precoding. 0: (CP-OFDM), 1: (DFT-s-OFDM)
pusch_single.NumLayers = 1;              % Number of PUSCH transmission layers
pusch_single.TransmissionScheme = 'nonCodebook'; % Transmission scheme ('nonCodebook','codebook')
pusch_single.NumAntennaPorts = 1;        % Number of antenna ports for codebook based precoding
pusch_single.TPMI = 0;                   % Precoding matrix indicator for codebook based precoding

% Define codeword modulation
pusch_single.Modulation = 'QPSK'; % 'pi/2-BPSK', 'QPSK', '16QAM', '64QAM', '256QAM'

% PUSCH DM-RS configuration
pusch_single.DMRS.DMRSTypeAPosition = 2;       % Mapping type A only. First DM-RS symbol position (2,3)
pusch_single.DMRS.DMRSLength = 1;              % Number of front-loaded DM-RS symbols (1(single symbol),2(double symbol))
pusch_single.DMRS.DMRSAdditionalPosition = 1;  % Additional DM-RS symbol positions (max range 0...3)
pusch_single.DMRS.DMRSConfigurationType = 1;   % DM-RS configuration type (1,2)
pusch_single.DMRS.NumCDMGroupsWithoutData = 2; % Number of CDM groups without data
pusch_single.DMRS.NIDNSCID = 0;                % Scrambling identity (0...65535)
pusch_single.DMRS.NSCID = 0;                   % Scrambling initialization (0,1)
pusch_single.DMRS.NRSID = 0;                   % Scrambling ID for low-PAPR sequences (0...1007)
pusch_single.DMRS.GroupHopping = 0;            % Group hopping (0,1)
pusch_single.DMRS.SequenceHopping = 0;         % Sequence hopping (0,1)


%% Start Simulation
for posIdx = 1:numPos
    
    % Create ueSite
    ueSite = txsite("Name","UE", ...
        "Latitude",uePositions(posIdx, 1), ...
        "Longitude",uePositions(posIdx, 2),...
        "AntennaHeight",1,... % in m
        "Antenna", ueArray,...
        "TransmitterFrequency",fc); %     "AntennaAngle",ueArrayOrientation(1:2),...
    if plotting == 1
        ueSite.show();
    end


    %% Compute cells in visibility
    % Get xyx coordinates with origin placed in ueSite
    uePosition_xyz = from_lat_long_to_xyz(ueSite, ueSite);
    gNBPos_xyz = cell(1,numCells);
    angles_bs = zeros(2, numCells);
    angles_ue = zeros(2, numCells);
    for gNBIdx = 1:numCells
        gNBPos_xyz{gNBIdx} = from_lat_long_to_xyz(bsSite(gNBIdx), ueSite);
        [angles_bs(1,gNBIdx),angles_bs(2,gNBIdx)] = angle(bsSite(gNBIdx),ueSite);
        [angles_ue(1,gNBIdx),angles_ue(2,gNBIdx)] = angle(ueSite, bsSite(gNBIdx));
        angles_bs(1,gNBIdx) = mod(angles_bs(1,gNBIdx), 360);
    end
    % [lat,lon,h] = from_xyz_to_site(gNBPos_xyz{1}, ueSite);

    s = zeros(numCells, dimensions);
    for bs=1:numCells
        s(bs,:) = reshape(gNBPos_xyz{bs}(1:dimensions), [dimensions,1]);
    end

    true_dist = zeros(1, numCells);
    for bs=1:numCells
        true_dist(bs) = norm(s(bs,:));
    end

    % Cells in visibility within a range and angle
    detectedgNBs_ind = (true_dist < cellRange) & ...
                   (abs(rad2deg(angdiff(deg2rad(cellAngles), deg2rad(angles_bs(1,:)))))< 60);
    detectedgNBs = find(detectedgNBs_ind~=0);
    not_detectedgNBs = find(detectedgNBs_ind~=1);
    numdetectedgNBs = length(detectedgNBs);

    % Hide all sites on map
    for gNBIdx = 1:numCells
        hide(bsSite(gNBIdx));
    end
    % Show sites on map
    for gNBIdx = detectedgNBs
        show(bsSite(gNBIdx), "Icon",sprintf("placeholder_rx%d.png", ceil(gNBIdx/3)), "IconSize", [150, 150]);
    end
    
    % If at least one cell in visibility
    if numdetectedgNBs ~= 0

        %% Ray Tracing Channel configuration
        pm = cell(1,numdetectedgNBs);
        channel = cell(1,numdetectedgNBs);
        tempdetectedgNBs = detectedgNBs;
        tempgNBIdx = 1;
        for gNBIdx = 1:numdetectedgNBs    
            cellID = tempdetectedgNBs(gNBIdx);
    
            % Ray Tracing Analysis
            pm{tempgNBIdx} = propagationModel("raytracing",...
                                        "Method","sbr", ...
                                        "MaxNumReflections",reflectionsOrder);
            rays = raytrace(ueSite,bsSite(cellID),pm{tempgNBIdx},"Type","pathloss");   
    
            % Do not consider cells that are not reached with multipath of MaxNumReflections = reflectionsOrder
            if isempty(rays{1}) || ~xor(~any([rays{1}.LineOfSight]), boolean_los)
                numdetectedgNBs = numdetectedgNBs - 1;
                detectedgNBs = detectedgNBs(detectedgNBs~=cellID);
                hide(bsSite(cellID));
            else
        
                pathToAs = [rays{1}.PropagationDelay];                                  % Time of arrival of each ray
                avgPathGains  = -[rays{1}.PathLoss];                                    % Average path gains of each ray
                pathAoDs = [rays{1}.AngleOfDeparture];                                  % AoD of each ray
                pathAoAs = [rays{1}.AngleOfArrival];                                    % AoA of each ray
                isLOS = any([rays{1}.LineOfSight]);                                     % Line of sight flag
                
                if plotting == 1
                    plot(rays{1})
                end
                
                % Set Up CDL Channel Model
                channel{tempgNBIdx} = nrCDLChannel;
                channel{tempgNBIdx}.DelayProfile = 'Custom';
                channel{tempgNBIdx}.PathDelays = pathToAs;           % Time of arrival of each ray (normalized to 0 sec)
                channel{tempgNBIdx}.AveragePathGains = avgPathGains;
                channel{tempgNBIdx}.AnglesAoD = pathAoDs(1,:);       % azimuth of departure
                channel{tempgNBIdx}.AnglesZoD = 90-pathAoDs(2,:);    % channel uses zenith angle, rays use elevation
                channel{tempgNBIdx}.AnglesAoA = pathAoAs(1,:);       % azimuth of arrival
                channel{tempgNBIdx}.AnglesZoA = 90-pathAoAs(2,:);    % channel uses zenith angle, rays use elevation
                channel{tempgNBIdx}.HasLOSCluster = isLOS;           % Line of sight flag
                channel{tempgNBIdx}.CarrierFrequency = fc;
                channel{tempgNBIdx}.NormalizeChannelOutputs = false; % do not normalize by the number of receive antennas, this would change the receive power
                channel{tempgNBIdx}.NormalizePathGains = false;      % set to false to retain the path gains
            
                channel{tempgNBIdx}.TransmitAntennaArray = ueArray;
                channel{tempgNBIdx}.TransmitArrayOrientation = [angles_ue(1, cellID); (-1)*(0); 0];  % the (-1) converts elevation to downtilt
                channel{tempgNBIdx}.ReceiveAntennaArray = bsSite(cellID).Antenna;  
                channel{tempgNBIdx}.ReceiveArrayOrientation = [bsSite(cellID).AntennaAngle; 0];   % the (-1) converts elevation to downtilt 
            
                channel{tempgNBIdx}.SampleRate = ofdmInfo.SampleRate;
    
                tempgNBIdx = tempgNBIdx + 1;
            end
        end
    
        % Specify the flag to configure the existence of the line of sight (LOS) 
        % path between each gNB and UE pair.
        los = zeros(1, numdetectedgNBs);
        for gNBIdx = 1:numdetectedgNBs
            los(gNBIdx) = channel{1,gNBIdx}.HasLOSCluster;
        end
    
        los = logical(los);
        if numel(los) ~= numdetectedgNBs
            error('nr5g:InvalidLOSLength',['Length of line of sight flag (' num2str(numel(los)) ...
                ') must be equal to the number of configured gNBs (' num2str(numdetectedgNBs) ').']);
        end
    

        % If at least one cell in visibility
        if numdetectedgNBs ~= 0
        
            %% SRS Configuration
            % Slot offsets of different SRS signals
            srsSlotOffsets = 0:2:(2*numdetectedgNBs - 1);
            srsIDs = randperm(4096,numdetectedgNBs) - 1;
        
            srs = repmat(srs_single,1,numdetectedgNBs);
            for gNBIdx = 1:numdetectedgNBs
                srs(gNBIdx).SRSPeriod(1) = 2; % 10;
                srs(gNBIdx).SRSPeriod(2) = 0; %  srsSlotOffsets(gNBIdx);  UE TRANSMITS TO ALL BS AT DIFFERENT INSTANTS
                srs(gNBIdx).NSRSID = srsIDs(gNBIdx);
            end
        
            if plotting==1
                % Visualize SRS
                hSRSGrid(carrier(1),srs(1),nFrames*carrier(1).SlotsPerFrame, true); % Generate a multi-slot resource grid containing SRS
                % hSRSGrid(carrier(1),srs,nFrames*carrier(1).SlotsPerFrame, true); % Generate a multi-slot resource grid containing SRS
                title('SRS with Different Slot Offsets (SRSPeriod)')
                ylim([0 14]);
                % ylim([0 srs(1).NRBPerTransmission]);
            end
        
        
            %% PUSCH Configuration
            pusch = repmat(pusch_single,1,numdetectedgNBs);
            for gNBIdx = 1:numdetectedgNBs
                pusch(gNBIdx).SymbolAllocation = [0 carrier(detectedgNBs(gNBIdx)).SymbolsPerSlot];
                pusch(gNBIdx).NID = carrier(detectedgNBs(gNBIdx)).NCellID;
            end
            validateNumLayers(pusch,NTxAnts,NRxAnts);
        
        
            %% Generate SRS and PUSCH Resources
            totSlots = nFrames*carrier(1).SlotsPerFrame;
            srsGrid = cell(1,numdetectedgNBs);
            dataGrid = cell(1,numdetectedgNBs);
            wtx = cell(1,numdetectedgNBs);
            for slotIdx = 0:totSlots-1 
                [carrier(:).NSlot] = deal(slotIdx);
                [srsSym,srsInd] = deal(cell(1,numdetectedgNBs));
                [srsAntSym,srsAntInd] = deal(cell(1,numdetectedgNBs));
                for gNBIdx = 1:numdetectedgNBs
            
                    % MIMO precoding
                    % wtx{gNBIdx} = newWtx{gNBIdx}; 
            
                    % Create an empty resource grid spanning one slot in time domain
                    slotGrid = nrResourceGrid(carrier(detectedgNBs(gNBIdx)),NTxAnts);
            
                    % Generate SRS symbols and indices
                    srsSym{gNBIdx} = nrSRS(carrier(detectedgNBs(gNBIdx)),srs(gNBIdx));
                    srsInd{gNBIdx} = nrSRSIndices(carrier(detectedgNBs(gNBIdx)),srs(gNBIdx));
        
                    if (strcmpi(pusch(gNBIdx).TransmissionScheme,'codebook'))
                        % Codebook based MIMO precoding, F precodes between PUSCH
                        % transmit antenna ports and transmit antennas
                        F = eye(pusch(gNBIdx).NumAntennaPorts,NTxAnts);
                    else
                        % Non-codebook based MIMO precoding, F precodes between PUSCH
                        % layers and transmit antennas
                        F = eye(pusch(gNBIdx).NumLayers,NTxAnts);
                    end
            
                    [srsAntSym{gNBIdx},srsAntInd{gNBIdx}] = nrExtractResources(srsInd{gNBIdx},slotGrid);
                    
                    % Map SRS resources to slot grid
                    slotGrid(srsAntInd{gNBIdx}) = srsSym{gNBIdx} * F;
                    srsGrid{gNBIdx} = [srsGrid{gNBIdx} slotGrid];
        
                end
                % Transmit data in slots in which the SRS is not transmitted by any of
                % the gNBs (to control the hearability problem)
                for gNBIdx = 1:numdetectedgNBs
                    dataSlotGrid = nrResourceGrid(carrier(detectedgNBs(gNBIdx)),NTxAnts);
                    if all(cellfun(@isempty,srsAntInd))
            
                        % MIMO precoding
                        % wtx{gNBIdx} = newWtx{gNBIdx};
            
                        % Generate PUSCH indices
                        [puschInd,puschInfo] = nrPUSCHIndices(carrier(detectedgNBs(gNBIdx)),pusch(gNBIdx));
            
                        % Generate random data bits for transmission
                        data = randi([0 1],puschInfo.G,1);
                        % Generate PUSCH symbols
                        puschSym = nrPUSCH(carrier(detectedgNBs(gNBIdx)),pusch(gNBIdx),data);
        
                        [puschAntSym,puschAntInd] = nrExtractResources(puschInd,dataSlotGrid);         
                        dataSlotGrid(puschAntInd) = puschSym * F;
            
                        % Generate demodulation reference signal (DM-RS) indices and symbols
                        dmrsInd = nrPUSCHDMRSIndices(carrier(detectedgNBs(gNBIdx)),pusch(gNBIdx));
                        dmrsSym = nrPUSCHDMRS(carrier(detectedgNBs(gNBIdx)),pusch(gNBIdx));
        
                        % Map DM-RS to slot grid
                        for p = 1:size(dmrsSym,2)
                            [~,dmrsAntInd] = nrExtractResources(dmrsInd(:,p),dataSlotGrid);
                            dataSlotGrid(dmrsAntInd) = dataSlotGrid(dmrsAntInd) + dmrsSym(:,p) * F(p,:);
                        end
                        % Map PUSCH to slot grid
                        dataSlotGrid(puschAntInd) = puschSym * F;
        
                    end
                    dataGrid{gNBIdx} = [dataGrid{gNBIdx} dataSlotGrid];
                end
            end
        
        
            %% Perform OFDM Modulation
            % Perform OFDM modulation of SRS and data signal at each gNB
            txWaveform = cell(1,numdetectedgNBs);
            for gNBIdx = 1:numdetectedgNBs
                carrier(detectedgNBs(gNBIdx)).NSlot = 0;
                txWaveform{gNBIdx} = nrOFDMModulate(carrier(detectedgNBs(gNBIdx)),srsGrid{gNBIdx} + dataGrid{gNBIdx});
            end
            ofdmInfo = nrOFDMInfo(carrier(1));
        
        
            %% Add Signal Delays
            maxChDelay = 0;
            sampleDelay = zeros(1,numdetectedgNBs);
            for gNBIdx = 1:numdetectedgNBs
               chInfo = info(channel{gNBIdx});
               sampleDelay(gNBIdx) = ceil(max(chInfo.PathDelays*channel{gNBIdx}.SampleRate)) + chInfo.ChannelFilterDelay;
               maxChDelay = max(maxChDelay, sampleDelay(gNBIdx));
            end
            maxChDelay = max(sampleDelay);
        
        
            %% Signal through Channel
            for gNBIdx = 1:numdetectedgNBs
                txWaveform{gNBIdx} = [txWaveform{gNBIdx}; zeros(sampleDelay(gNBIdx),size(txWaveform{gNBIdx},2))]; 
            end
            
            rxWaveform = cell(1,numdetectedgNBs);
            pathGains = cell(1,numdetectedgNBs);
            sampleTimes = cell(1,numdetectedgNBs);
            for gNBIdx = 1:numdetectedgNBs
                [rxWaveform{gNBIdx},pathGains{gNBIdx},sampleTimes{gNBIdx}] = channel{gNBIdx}(txWaveform{gNBIdx});
                % Add AWGN
                if add_awgn == 1
                    rxWaveform{gNBIdx} = awgn(rxWaveform{gNBIdx},SNR, 'measured');
                end
            end
        
        
            %% TOA Estimation
            corr = cell(1,numdetectedgNBs);
            delayEst = zeros(1,numdetectedgNBs);
            maxCorr = zeros(1,numdetectedgNBs);
            for gNBIdx = 1:numdetectedgNBs
                % REFGRID
                % K-by-N-by-P, (ci sono P grid)
                % K: number of subcarriers
                % N: number of OFDM symbols
                % P: number of reference signal ports
                [~,mag] = nrTimingEstimate(carrier(detectedgNBs(gNBIdx)),rxWaveform{gNBIdx},srsGrid{gNBIdx});
                % Extract correlation data samples spanning about 1/14 ms for normal
                % cyclic prefix and about 1/12 ms for extended cyclic prefix (this
                % truncation is to ignore noisy side lobe peaks in correlation outcome)
                corr{gNBIdx} = mag(1:(ofdmInfo.Nfft*carrier(detectedgNBs(gNBIdx)).SubcarrierSpacing/15));
            
                [maxCorr(gNBIdx), delayEst(gNBIdx)] = first_arrival(corr{gNBIdx}, 2);
            end
            
            % Plot PRS correlation results
            if plotting == 1
                plotPRSCorr(carrier(detectedgNBs),corr,ofdmInfo.SampleRate, maxCorr);
            end
            
            
            %% AOA Estimation with MUSIC
            if plotting == 1
                doas = zeros(2, numdetectedgNBs);
                for gNBIdx = 1:numdetectedgNBs
                    [~,doas(:,gNBIdx)] = estimator(rxWaveform{gNBIdx});
                    figure(10000)
                    plotSpectrum(estimator);
                    pause()
                end
            end
            
        
            %% OFDM Demodulation
            % Perform OFDM demodulation on the received data to recreate the
            % resource grid, including padding in the event that practical
            % synchronization results in an incomplete slot being demodulated
            rxGrid = cell(1,numdetectedgNBs);
            for gNBIdx = 1:numdetectedgNBs
                % RIMUOVERE OFFSET
                % rxGrid{gNBIdx} = nrOFDMDemodulate(carrier(detectedgNBs(gNBIdx))),rxWaveform(1+offset(gNBIdx):end,:));
                rxGrid{gNBIdx} = nrOFDMDemodulate(carrier(detectedgNBs(gNBIdx)),rxWaveform{gNBIdx});
                [K,L,R] = size(rxGrid{gNBIdx});
                if (L < carrier(detectedgNBs(gNBIdx)).SymbolsPerSlot)
                    rxGrid{gNBIdx} = cat(2,rxGrid{gNBIdx},zeros(K,carrier(detectedgNBs(gNBIdx)).SymbolsPerSlot-L,R));
                end
            end
        
        
            %% Channel Estimation
            estChannelGrid = cell(1,numdetectedgNBs);
            noiseEst = cell(1,numdetectedgNBs);
            if (PerfectChannelEstimator == 1)
                pathFilters = cell(1,numdetectedgNBs);
                for gNBIdx = 1:numdetectedgNBs
                    % Perfect channel estimation, using the value of the path gains
                    % provided by the channel. This channel estimate does not
                    % include the effect of transmitter precoding
                    % size(estChannelGrid) =  K(Subcar)-N(OFDM sym)-R(Rx ant)-128(Tx ant)
                    pathFilters{gNBIdx} = getPathFilters(channel{gNBIdx});
                    % estChannelGrid{gNBIdx} = nrPerfectChannelEstimate(carrier(detectedgNBs(gNBIdx)),pathGains{gNBIdx},pathFilters{gNBIdx},offset(gNBIdx),sampleTimes{gNBIdx});
                    estChannelGrid{gNBIdx} = nrPerfectChannelEstimate(carrier(detectedgNBs(gNBIdx)),pathGains{gNBIdx},pathFilters{gNBIdx},0,sampleTimes{gNBIdx});
                end
            else
                for gNBIdx = 1:numdetectedgNBs
                    gNBIdx
                    % Practical channel estimation between the received grid and
                    % each transmission layer, using the PDSCH DM-RS for each
                    % layer. This channel estimate includes the effect of
                    % transmitter precoding
                    % size(estChannelGrid) =  K(Subcar)-N(OFDM sym)-R(Rx ant)-P(Tx port)
                    % [estChannelGrid{gNBIdx},noiseEst{gNBIdx}] = nrChannelEstimate(carrier(detectedgNBs(gNBIdx)),rxGrid{gNBIdx},srsGrid{gNBIdx} + dataGrid{gNBIdx});
                    [estChannelGrid{gNBIdx},noiseEst{gNBIdx}] = nrChannelEstimate(carrier(detectedgNBs(gNBIdx)),rxGrid{gNBIdx},srsGrid{gNBIdx}, 'CDMLengths',hSRSCDMLengths(srs(1)));
                    % [estChannelGrid{gNBIdx},noiseEst{gNBIdx}] = nrChannelEstimate(carrier(detectedgNBs(gNBIdx)),rxGrid{gNBIdx},srsGrid{gNBIdx}, 'CDMLengths',pusch.DMRS.CDMLengths);
                end
            end
        
        
            %% Save Amplitude ADCRM
            % From Space-Frequency domain Channel Response Matrix (SFCRM) to 
            % Angle-Delay domain Channel Response Matrix (ADCRM)
            % Save ADCRM
            sr = ofdmInfo.SampleRate; % sampling rate 
            for gNBIdx = 1: numdetectedgNBs
                [status, msg, msgID] = mkdir(sprintf('%s/pos_%d/ADCRM/cell_%d', DB, posIdx, detectedgNBs(gNBIdx)));
                ADCRM = zeros(max(ofdmInfo.CyclicPrefixLengths), size(estChannelGrid{gNBIdx}, 2), R);
                for sample = 1: size(estChannelGrid{gNBIdx}, 2)
                    slotIdx = mod(sample, 14) + 1;
                    tx_port = 1;
            
                    SFCRM = squeeze(estChannelGrid{gNBIdx}(:, sample, :, tx_port));
                    % tic
                    temp_ADCRM = ifft2(SFCRM, ofdmInfo.Nfft, R);
                    % toc
                    ADCRM(1:ofdmInfo.CyclicPrefixLengths(slotIdx),sample,:) = abs(temp_ADCRM(1:ofdmInfo.CyclicPrefixLengths(slotIdx),:));
        
                end
                ADCRM = ADCRM(:,1:7:252,:);
                save(sprintf('%s/pos_%d/ADCRM/cell_%d/abs_ADCRM.mat', DB, posIdx, detectedgNBs(gNBIdx)), 'ADCRM'); 
            end
        
        
            %% Amplitude ADCRM
        %     % From Space-Frequency domain Channel Response Matrix (SFCRM) to 
        %     % Angle-Delay domain Channel Response Matrix (ADCRM)
        %     % Plot ADCRM
        %     sr = ofdmInfo.SampleRate; % sampling rate 
        %     for gNBIdx = 1: numdetectedgNBs
        %         for sample = 1: size(estChannelGrid{gNBIdx}, 2)
        %             set(gca,'view',[45 45])
        %             slotIdx = mod(sample, 14) + 1;
        %             tx_port = 1;
        %     
        %             SFCRM = squeeze(estChannelGrid{gNBIdx}(:, sample, :, tx_port));
        %     
        %             ADCRM = ifft2(SFCRM, ofdmInfo.Nfft, R);
        %             ADCRM = ADCRM(1:ofdmInfo.CyclicPrefixLengths(slotIdx),:);
        %     
        %             % [angle,delay] = meshgrid(1:R,0 : 1/sr : (ofdmInfo.Nfft-1)*1/sr);
        %             [angle,delay] = meshgrid(1:R,0 : 1/sr : (ofdmInfo.CyclicPrefixLengths(slotIdx)-1)/sr);
        %             figure(sample)
        %             h = surf(angle, delay, squeeze(abs(ADCRM)));
        %             xlabel('Angle')
        %             ylabel('Delay (s)')
        %     
        %             % saveas(figure(sample),sprintf('../Foto/DL-AoA/1/%d/image_%d_%d.png', gNBIdx, gNBIdx, sample))
        %             pause()
        %             close(figure(sample))
        %         end
        %     end
            % pause()
            
        %     % Plot Azimuth - Elevation
        %     sr = ofdmInfo.SampleRate; % sampling rate 
        %     for gNBIdx = 1: numdetectedgNBs
        %         for sample = 1: 1 % size(estChannelGrid{gNBIdx}, 2)
        %             set(gca,'view',[45 45])
        %             slotIdx = mod(sample, 14) + 1;
        %             tx_port = 1;
        %     
        %             SFCRM = squeeze(estChannelGrid{gNBIdx}(:, sample, :, tx_port));
        %             ADCRM = ifft2(SFCRM, ofdmInfo.Nfft, R);
        %             ADCRM = ADCRM(1:ofdmInfo.CyclicPrefixLengths(slotIdx),:);
        %             ADCRM = reshape(ADCRM,[ofdmInfo.CyclicPrefixLengths(slotIdx),sqrt(R), sqrt(R)]);
        %     
        %             % plot azimuth and elavation at correct delay
        %             [azimut,elavation] = meshgrid(flip(-180:360/(sqrt(R)-1):180),0:90/(sqrt(R)-1):90);
        %             figure(sample)
        %             surf(azimut,elavation,squeeze(abs(ADCRM(delayEst(gNBIdx),:, :))))
        %             xlabel('Azimuth')
        %             ylabel('Elevation')
        %         
        %             % saveas(figure(sample),sprintf('../Foto/DL-AoA/1/%d/image_%d_%d.png', gNBIdx, gNBIdx, sample))
        %             pause()
        %             close(figure(sample))
        %         end
        %     end
        %     % pause()
        
            
            %% UE Position Estimation Using OTDOA Technique
            % Compute RSTD values for multilateration or trilateration
            rstdVals = getRSTDValues(delayEst,ofdmInfo.SampleRate);
            TOA = delayEst;
            
            % Plot gNB and UE positions
            txCellIDs = [carrier(:).NCellID];
            cellIdx = 1;
            rho = [];
            s = [];
            curveX = {};
            curveY = {};
            for jj = detectedgNBs(1) % Assuming first detected gNB as reference gNB
                txj = find(txCellIDs == carrier(jj).NCellID);
                s(1,:) = gNBPos_xyz{txj};
                for ii = detectedgNBs(2:end)
                    rstd = rstdVals(find(detectedgNBs == ii),find(detectedgNBs == jj))*c; % Delay distance
                    % Establish gNBs for which delay distance is applicable by
                    % examining detected cell identities
                    txi = find(txCellIDs == carrier(ii).NCellID);
                    if (~isempty(txi) && ~isempty(txj))
            
                        if dimensions == 2
                            [x,y] = getRSTDCurve(gNBPos_xyz{txi},gNBPos_xyz{txj},rstd);
                            curveX{1,cellIdx} = x; 
                            curveY{1,cellIdx} = y;
                            % Get the gNB numbers corresponding to the current hyperbola
                            % curve
                            gNBNums{cellIdx} = [find(detectedgNBs == jj) find(detectedgNBs == ii)];
                        end
            
                        rho(cellIdx) = rstd;
                        cellIdx = cellIdx + 1;
                    end
                end
            end
            
            
            s = zeros(numdetectedgNBs, dimensions);
            for gNBIdx=1:numdetectedgNBs
                s(gNBIdx,:) = reshape(gNBPos_xyz{detectedgNBs(gNBIdx)}(1:dimensions), [dimensions,1]);
            end
            
            % % Estimate UE position with TDOA.
            u = zeros(1,dimensions);
            u_0 = randi([-100,100],1,dimensions);
            K = 1000;
            estimatedPos_xyz = Non_linear_LS_TDOA(rho, u_0, s, K, dimensions)
            % estimatedPos_xyz = LS_TDOA(rho, s, dimensions)
            
            % Estimate UE position with TOA.
            % rho = TOA./ofdmInfo.SampleRate*c;
            % estimatedPos_xyz = LS_TOA(rho, s, dimensions);
            
            % Check if measurement are good:
            % compute distance from each BS to u = [0,0,0] and compare with
            % measurements.
            true_dist = zeros(1, numdetectedgNBs);
            for gNBIdx=1:numdetectedgNBs
                true_dist(gNBIdx) = norm(s(gNBIdx,:));
            end
            true_dist;
            rho = TOA./ofdmInfo.SampleRate*c;
            EstimationErr = norm(uePosition_xyz(1:dimensions)-estimatedPos_xyz)
            if dimensions == 2
                estimatedPos_xyz(3) = uePosition_xyz(3);
            end
            
            % Plot Estimated UE in Siteviewer
            [lat,lon,h] = from_xyz_to_site(estimatedPos_xyz, ueSite);
            if (~isnan(lat)) && (~isnan(lon))
                ueSite = txsite("Name","UE", ...
                    "Latitude",lat,"Longitude",lon,...
                    "AntennaHeight",1);
                if plotting == 1
                    ueSite.show();
                end
            end
                    
            % Plot UEtrue, UEest, BS, Hyperbolas, GDOP
            if plotting == 1
                tempgNBPos_xyz = cell(1, numdetectedgNBs);
                for gNBIdx=1:numdetectedgNBs
                    tempgNBPos_xyz{gNBIdx} = gNBPos_xyz{detectedgNBs(gNBIdx)};
                end
                gNBsToPlot = unique([gNBNums{:}],'stable');
                plotGDOP(gNBPos_xyz,...
                    detectedgNBs,...
                    uePosition_xyz, ...
                    gNBsToPlot, ...
                    curveX,curveY, ...
                    gNBNums, ...
                    estimatedPos_xyz, ...
                    dimensions);
            end
        
            % Save results of Estimate UE position with TDOA.
            save(sprintf('%s/pos_%d/estimatedPos_xyz.mat', DB, posIdx),'estimatedPos_xyz');

        % If at least one cell in visibility
        end 

    % If at least one cell in visibility
    end

end



% celldetectedNames = strcat('cell_', string(detectedgNBs));
% cs = cell2struct(estChannelGrid, celldetectedNames, 2);
% save('prova.mat', '-struct', 'cs');
% 
% ADCRM = ndSparse(ADCRM);
% save('prova2.mat', 'ADCRM');


%% Local functions
% Displays perfect and practical channel estimates and the channel
% estimation error for the first transmit and receive ports. The channel
% estimation error is defined as the absolute value of the difference
% between the perfect and practical channel estimates.
function hChannelEstimationPlot(symbols,subcarriers,allHestPerfect,allHestInterp)

    figure
    subplot(311)
    imagesc(symbols, subcarriers, abs(allHestPerfect(:,:,1,1)));
    axis xy; xlabel('OFDM symbol'); ylabel('Subcarrier');
    colorbar;
    title('Perfect Channel Estimate (TxAnt=1, RxAnt=1)');

    subplot(312)
    imagesc(symbols, subcarriers, abs(allHestInterp(:,:,1,1)), ...
            'AlphaData',~isnan(allHestInterp(:,:,1,1)))
    axis xy; xlabel('OFDM symbol'); ylabel('Subcarrier');
    colorbar;
    title('SRS-based Practical Channel Estimate (TxAnt=1, RxAnt=1) ');

    % Display channel estimation error, defined as the difference between the
    % SRS-based and perfect channel estimates
    subplot(313)
    hestErr = abs(allHestInterp - allHestPerfect);
    imagesc(symbols, subcarriers, hestErr(:,:,1,1),...
            'AlphaData',~isnan(hestErr(:,:,1,1)));
    axis xy; xlabel('OFDM symbol'); ylabel('Subcarrier');
    colorbar;
    title('Channel Estimation Error (TxAnt=1, RxAnt=1)');

end

% Displays the PMI evolution and PMI estimation SINR loss over time and
% frequency. The SINR loss is defined as a ratio of the SINR after
% precoding with estimated and perfect PMIs. Estimated PMIs are
% obtained using a practical channel estimate and perfect PMIs are
% selected using a perfect channel estimate.
function hPMIPlot(slots,resourceBlocks,pmiRB,pmiPerfectRB,lossRB)

    figure
    subplot(311)
    imagesc(slots,resourceBlocks,pmiPerfectRB,'AlphaData',~isnan(pmiPerfectRB)); axis xy;
    c = caxis;
    cm = colormap;
    colormap( cm(1:floor(size(cm,1)/(c(2)-c(1)) -1):end,:) ); % Adjust colormap to PMI discrete values
    colorbar
    xlabel('Slot'); ylabel('Resource block'), title('PMI Selected using Perfect Channel Estimates')

    subplot(312)
    imagesc(slots,resourceBlocks,pmiRB,'AlphaData',~isnan(pmiRB)); axis xy;
    colorbar,
    xlabel('Slot'); ylabel('Resource block'), title('PMI Selected using SRS')

    subplot(313)
    imagesc(slots,resourceBlocks,lossRB,'AlphaData',~isnan(lossRB));
    colormap(gca,cm)
    xlabel('Slot'); ylabel('Resource block'); axis xy; colorbar;
    title('PMI Estimation SINR Loss (dB)')

end

% Displays the SINR per resource block obtained after precoding with the
% PMI that maximizes the SINR per subband.
function hBestSINRPlot(slots,resourceBlocks,sinrSubband,pmi,csiBandSize)

    % Display SINR after precoding with best PMI
    bestSINRPerSubband = nan(size(sinrSubband,[1 2]));

    % Get SINR per subband and slot using best PMI
    [sb,nslot] = find(~isnan(pmi));
    for i = 1:length(sb)
        bestSINRPerSubband(sb(i),nslot(i)) = sinrSubband(sb(i),nslot(i),pmi(sb(i),nslot(i))+1);
    end

    % First expand SINR from subbands into RBs for display purposes
    bestSINRPerRB = hExpandSubbandToRB(bestSINRPerSubband, csiBandSize, length(resourceBlocks));

    figure
    sinrdb = 10*log10(abs(bestSINRPerRB));
    imagesc(slots,resourceBlocks,sinrdb,'AlphaData',~isnan(sinrdb));
    axis xy; colorbar;
    xlabel('Slot');
    ylabel('Resource block')
    title('Average SINR Per Subband and Slot After Precoding with Best PMI (dB)')

end

% Expands a 2D matrix of values per subband in the first dimension into a
% matrix of values per resource block.
function rbValues = hExpandSubbandToRB(subbandValues, bandSize, NRB)

    lastBandSize = mod(NRB,bandSize);
    lastBandSize = lastBandSize + bandSize*(lastBandSize==0);

    rbValues = [kron(subbandValues(1:end-1,:),ones(bandSize,1));...
                subbandValues(end,:).*ones(lastBandSize,1)];
end



%% Local Functions
function validateCarriers(carrier)
%   validateCarriers(CARRIER) validates the carrier properties of all
%   gNBs.

    % Validate physical layer cell identities
    cellIDs = [carrier(:).NCellID];
    if numel(cellIDs) ~= numel(unique(cellIDs))
        error('nr5g:invalidNCellID','Physical layer cell identities of all of the carriers must be unique.');
    end

    % Validate subcarrier spacings
    scsVals = [carrier(:).SubcarrierSpacing];
    if ~isscalar(unique(scsVals))
        error('nr5g:invalidSCS','Subcarrier spacing values of all of the carriers must be same.');
    end

    % Validate cyclic prefix lengths
    cpVals = {carrier(:).CyclicPrefix};
    if ~all(strcmpi(cpVals{1},cpVals))
        error('nr5g:invalidCP','Cyclic prefix lengths of all of the carriers must be same.');
    end

    % Validate NSizeGrid values
    nSizeGridVals = [carrier(:).NSizeGrid];
    if ~isscalar(unique(nSizeGridVals))
        error('nr5g:invalidNSizeGrid','NSizeGrid of all of the carriers must be same.');
    end

    % Validate NStartGrid values
    nStartGridVals = [carrier(:).NStartGrid];
    if ~isscalar(unique(nStartGridVals))
        error('nr5g:invalidNStartGrid','NStartGrid of all of the carriers must be same.');
    end

end

function validateNumLayers(pusch,NTxAnts,NRxAnts)
% Validate the number of layers, relative to the antenna geometry

    numlayers = pusch.NumLayers;
    ntxants = NTxAnts;
    nrxants = NRxAnts;
    antennaDescription = sprintf('min(NTxAnts,NRxAnts) = min(%d,%d) = %d',ntxants,nrxants,min(ntxants,nrxants));
    if numlayers > min(ntxants,nrxants)
        error('The number of layers (%d) must satisfy NumLayers <= %s', ...
            numlayers,antennaDescription);
    end

    % Display a warning if the maximum possible rank of the channel equals
    % the number of layers
    if (numlayers > 2) && (numlayers == min(ntxants,nrxants))
        warning(['The maximum possible rank of the channel, given by %s, is equal to NumLayers (%d).' ...
            ' This may result in a decoding failure under some channel conditions.' ...
            ' Try decreasing the number of layers or increasing the channel rank' ...
            ' (use more transmit or receive antennas).'],antennaDescription,numlayers); %#ok<SPWRN>
    end

end



function plotGrid(prsGrid,dataGrid)
% PERCHE SENZA COLORI
%   plotGrid(PRSGRID,DATAGRID) plots the carrier grid from all gNBs.

    numdetectedgNBs = numel(prsGrid);
    figure()
    mymap = [1 1 1; ...       % White color for background
             0.8 0.8 0.8; ... % Gray color for PDSCH data from all gNBs
             getColors(numdetectedgNBs)];
    chpval = 3:numdetectedgNBs+2;

    gridWithPRS = zeros(size(prsGrid{1}));
    gridWithData = zeros(size(dataGrid{1}));
    names = cell(1,numdetectedgNBs);
    for gNBIdx = 1:numdetectedgNBs
        gridWithPRS = gridWithPRS + abs(prsGrid{gNBIdx})*chpval(gNBIdx);
        gridWithData = gridWithData + abs(dataGrid{gNBIdx});
        names{gNBIdx} = ['PRS from gNB' num2str(gNBIdx)];
    end
    names = [{'Data from all gNBs'} names];
    % Replace all zero values of gridWithPRS with proper scaling (value of 1)
    % for white background
    gridWithPRS(gridWithPRS == 0) = 1;
    gridWithData(gridWithData ~=0) = 1;

    % Plot grid
    image(gridWithPRS+gridWithData); % In the resultant grid, 1 represents
                                     % the white background, 2 (constitute
                                     % of 1s from gridWithPRS and
                                     % gridWithData) represents PDSCH, and
                                     % so on
    % Apply colormap
    colormap(mymap);
    axis xy;

    % Generate lines
    L = line(ones(numdetectedgNBs+1),ones(numdetectedgNBs+1),'LineWidth',8);
    % Index colormap and associate selected colors with lines
    set(L,{'color'},mat2cell(mymap(2:numdetectedgNBs+2,:),ones(1,numdetectedgNBs+1),3)); % Set the colors

    % Create legend
    legend(names{:});

    % Add title
    title('Carrier Grid Containing PRS and PDSCH from Multiple gNBs');

    % Add labels to x-axis and y-axis
    xlabel('OFDM Symbols');
    ylabel('Subcarriers');
end



function rstd = getRSTDValues(toa,sr)
%   RSTD = getRSTDValues(TOA,SR) computes RSTD values, given the time of
%   arrivals TOA and the sampling rate SR.

    % Compute number of samples delay between arrivals from different gNBs
    rstd = zeros(length(toa));
    for jj = 1:length(toa)
        for ii = 1:length(toa)
            rstd(ii,jj) = toa(ii) - toa(jj);
        end
    end

    % Get RSTD values in time
    rstd = rstd./sr;

end

function [x,y] = getRSTDCurve(gNB1,gNB2,rstd)
%   [X,Y] = getRSTDCurve(GNB1,GNB2,RSTD) returns the x- and y-coordinates
%   of a hyperbola equation corresponding to an RSTD value, given the pair
%   of gNB positions gNB1 and gNB2.

    % Calculate vector between two gNBs
    delta = gNB1 - gNB2;

    % Express distance vector in polar form
    [phi,r] = cart2pol(delta(1),delta(2));
    rd = (r+rstd)/2;

    % Compute the hyperbola parameters
    a = (r/2)-rd;         % Vertex
    c = r/2;              % Focus
    b = sqrt(c^2-a^2);    % Co-vertex
    hk = (gNB1 + gNB2)/2;
    mu = -2:1e-3:2;

    % Get x- and y- coordinates of hyperbola equation
    x = (a*cosh(mu)*cos(phi)-b*sinh(mu)*sin(phi)) + hk(1);
    y = (a*cosh(mu)*sin(phi)+b*sinh(mu)*cos(phi)) + hk(2);

end




