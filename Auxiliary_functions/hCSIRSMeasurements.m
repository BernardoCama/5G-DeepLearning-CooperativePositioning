function meas = hCSIRSMeasurements(carrier,csirs,grid)
%hCSIRSMeasurements CSI-RS based reference signal measurements
%   MEAS = hCSIRSMeasurements(CARRIER,CSIRS,GRID) returns the CSI-RS based
%   reference signal measurements (reference signal received power (RSRP),
%   received signal strength indicator (RSSI) and reference signal received
%   quality (RSRQ)) in a structure MEAS, as described in
%   TS 38.215 Sections 5.1.2 and 5.1.4, for the given carrier specific
%   configuration object CARRIER, CSI-RS specific configuration object
%   CSIRS, and received grid spanning one slot GRID.
%
%   CARRIER is a carrier specific configuration object as described in
%   <a href="matlab:help('nrCarrierConfig')">nrCarrierConfig</a> with the following properties:
%
%   NCellID           - Physical layer cell identity
%   SubcarrierSpacing - Subcarrier spacing in kHz
%   CyclicPrefix      - Cyclic prefix
%   NSizeGrid         - Number of resource blocks (RBs) in carrier resource
%                       grid
%   NStartGrid        - Start of carrier resource grid relative to common
%                       resource block 0 (CRB 0)
%   NSlot             - Slot number
%   NFrame            - System frame number
%
%   CSIRS is a CSI-RS specific configuration object as described in
%   <a href="matlab:help('nrCSIRSConfig')">nrCSIRSConfig</a> with the following properties specified for one or more
%   CSI-RS resources:
%
%   CSIRSType             - Type of a CSI-RS resource, whether it is ZP or
%                           NZP
%   CSIRSPeriod           - CSI-RS slot periodicity and offset
%   RowNumber             - Row number corresponding to a CSI-RS
%                           resource, as defined in TS 38.211
%                           Table 7.4.1.5.3-1
%   Density               - CSI-RS resource frequency density
%   SymbolLocations       - Time-domain locations of a CSI-RS resource
%   SubcarrierLocations   - Frequency-domain locations of a CSI-RS
%                           resource
%   NumRB                 - Number of RBs allocated for a CSI-RS resource
%   RBOffset              - Starting RB index of CSI-RS allocation
%                           relative to carrier resource grid
%   NID                   - Scrambling identity
%
%   GRID is a 3-dimensional array of the resource elements, for one slot
%   across all receive antennas. GRID is of size K-by-L-by-R array, where K
%   represents the number of subcarriers, L represents the number of OFDM
%   symbols, and R represents the number of receive antennas.
%
%   MEAS is a structure including the fields:
%   RSRPPerAntennaPerResource - Matrix of linear RSRP values with rows
%                               corresponding to receive antennas and
%                               columns corresponding to CSI-RS resources
%   RSSIPerAntennaPerResource - Matrix of linear RSSI values with rows
%                               corresponding to receive antennas and
%                               columns corresponding to CSI-RS resources
%   RSRQPerAntennaPerResource - Matrix of linear RSRQ values with rows
%                               corresponding to receive antennas and
%                               columns corresponding to CSI-RS resources
%   RSRP                      - Row vector of reported RSRP values for all
%                               CSI-RS resources, where each element
%                               represents the maximum value of linear
%                               RSRPs of all receive antennas
%   RSSI                      - Row vector of reported RSSI values for all
%                               CSI-RS resources, where each element
%                               represents the RSSI value of the receive
%                               antenna with the maximum RSRQ
%   RSRQ                      - Row vector of reported RSRQ values for all
%                               CSI-RS resources, where each element
%                               represents the maximum value of linear
%                               RSRQs of all receive antennas
%   RSRPdBm                   - The RSRP expressed in deciBels relative to
%                               1 milliwatt (in dBm)
%   RSSIdBm                   - The RSSI expressed in deciBels relative to
%                               1 milliwatt (in dBm)
%   RSRQdB                    - The RSRQ expressed in deciBels (in dB)
%
%   Note that the columns of above fields are in the same order, as the
%   resources are specified in the input configuration object CSIRS.
%
%   % Example:
%   % The following example demonstrates how to calculate CSI-RS based
%   % reference signal measurements
%
%   % Create carrier specific configuration object
%   carrier = nrCarrierConfig;
%
%   % Create CSI-RS specific configuration object
%   csirs = nrCSIRSConfig;
%   csirs.RowNumber = 1;
%   csirs.Density = 'three';
%   csirs.SymbolLocations = 6;
%   csirs.SubcarrierLocations = 0;
%
%   % Generate CSI-RS symbols and indices for the specified configurations
%   % and map the symbols to the slot grid
%   [ind,~] = nrCSIRSIndices(carrier,csirs);
%   [sym,~] = nrCSIRS(carrier,csirs);
%
%   % Initialize the carrier resource grid for one slot
%   ports = max(csirs.NumCSIRSPorts);
%   txGrid = nrResourceGrid(carrier,ports);
%   txGrid(ind) = sym;
%
%   % Perform OFDM modulation
%   txWaveform = nrOFDMModulate(carrier,txGrid);
%
%   % Apply power scaling to the transmitted waveform
%   EsdBm = -50;
%   rxWaveform = txWaveform * sqrt(10^((EsdBm-30)/10));
%
%   % Perform OFDM demodulation
%   rxGrid = nrOFDMDemodulate(carrier,rxWaveform);
%
%   % Perform CSI-RS based reference signal measurements
%   meas = hCSIRSMeasurements(carrier,csirs,rxGrid)

%   Copyright 2019-2020 The MathWorks, Inc.

    % Get the number of CSI-RS resources
    if iscell(csirs.CSIRSType)
        numCSIRSRes = numel(csirs.CSIRSType);
    else
        numCSIRSRes = 1;
    end

    % Number of receive antennas
    nRx = size(grid,3);

    % Initialize the variables for measurements
    rsrpPerAntPerRes = zeros(nRx,numCSIRSRes); % Row indicates the receive antenna index and column indicates the CSI-RS resource index
    rssiPerAntPerRes = zeros(nRx,numCSIRSRes);
    rsrqPerAntPerRes = zeros(nRx,numCSIRSRes);

    % Reference indices and symbols generation
    [refIndLin,info] = nrCSIRSIndices(carrier,csirs,'OutputResourceFormat','cell');
    tmpRefSym = nrCSIRS(carrier,csirs,'OutputResourceFormat','cell');

    for rxAntIdx = 1:nRx % Loop over all receive antennas
        gridRxAnt = grid(:,:,rxAntIdx);
        for resIdx = 1:numCSIRSRes % Loop over all CSI-RS resources for one receive antenna

            if isscalar(csirs.NumRB)
                N = csirs.NumRB;
            else
                N = csirs.NumRB(resIdx);
            end

            if isscalar(csirs.RBOffset)
                rbOffset = csirs.RBOffset;
            else
                rbOffset = csirs.RBOffset(resIdx);
            end
            if iscell(csirs.SymbolLocations)
                l0 = csirs.SymbolLocations{resIdx}(1); % l0 value
            else
                l0 = csirs.SymbolLocations(1);
            end
            % For RSSI measurement, generate the indices of all resource elements in OFDM symbols containing CSI-RS resource for single port (port 3000)
            csirsSymIndices = l0 + info.LPrime{resIdx} + 1; % 1-based symbol indices
            numCSIRSSym = numel(csirsSymIndices);
            rssiIndices = repmat((1:N*12).' + rbOffset*12,1,numCSIRSSym) + repmat((csirsSymIndices - 1)*12*carrier.NSizeGrid,12*N,1); % 1-based, linear carrier-oriented indices
            % Extract the modulation symbols using the indices, which
            % corresponds to RSSI measurement
            rssiSym = gridRxAnt(rssiIndices);

            numPorts = csirs.NumCSIRSPorts(resIdx);
            if numPorts > 2
                ports = 2; % Consider only ports 3000 and 3001
                tmp2 = reshape(tmpRefSym{resIdx == info.ResourceOrder},[],numPorts);
                refSym = reshape(tmp2(:,1:2),[],1);
            else
                ports = numPorts; % Port 3000 or Ports 3000 and 3001
                refSym = tmpRefSym{resIdx == info.ResourceOrder};
            end
            indTmp = reshape(refIndLin{resIdx == info.ResourceOrder},[],numPorts);
            indTmp = double(indTmp(:,1:ports)) - (0:ports-1)*carrier.NSizeGrid*12*carrier.SymbolsPerSlot; % Port 3000 or Ports 3000 and 3001
            % Extract the received CSI-RS symbols using locally generated
            % CSI-RS indices
            rxSym = reshape(gridRxAnt(indTmp),[],1);

            if ~isempty(refSym)
                % Perform the following calculations for NZP-CSI-RS
                % resources only
                rsrpPerAntPerRes(rxAntIdx,resIdx) = abs(mean(rxSym.*conj(refSym))*ports)^2;
                rssiPerAntPerRes(rxAntIdx,resIdx) = sum(rssiSym.*conj(rssiSym))/numCSIRSSym;
                rsrqPerAntPerRes(rxAntIdx,resIdx) = N*rsrpPerAntPerRes(rxAntIdx,resIdx)/rssiPerAntPerRes(rxAntIdx,resIdx);
            end
        end
    end

    rsrpPerAntPerRes(isnan(rsrpPerAntPerRes)) = 0;
    rssiPerAntPerRes(isnan(rssiPerAntPerRes)) = 0;
    rsrqPerAntPerRes(isnan(rsrqPerAntPerRes)) = 0;

    % Extract the maximum value of per antenna measurements (values
    % correspond to individual receiver branches) for all CSI-RS resources
    % to get the RSRP, RSRQ reporting values, as described in TS 38.215
    % Sections 5.1.2 and 5.1.4
    rsrp = max(rsrpPerAntPerRes,[],1);
    [rsrq,rsrqIdx] = max(rsrqPerAntPerRes,[],1);

    % Extract RSSI value of the receive antenna with the maximum RSRQ for
    % all CSI-RS resources
    rssi = rssiPerAntPerRes(rsrqIdx + (0:numCSIRSRes-1)*nRx);

    % Convert the RSRP, RSSI and RSRQ reporting values from linear scale to
    % logarithmic scale
    RSRPdBm = 10*log10(rsrp) + 30; % dBm
    RSSIdBm = 10*log10(rssi) + 30; % dBm
    RSRQdB = 10*log10(rsrq);       % dB

    % Create an output structure with the fields representing the
    % measurements
    meas.RSRPPerAntennaPerResource = rsrpPerAntPerRes;
    meas.RSSIPerAntennaPerResource = rssiPerAntPerRes;
    meas.RSRQPerAntennaPerResource = rsrqPerAntPerRes;
    meas.RSRP = rsrp;
    meas.RSSI = rssi;
    meas.RSRQ = rsrq;
    meas.RSRPdBm = RSRPdBm;
    meas.RSSIdBm = RSSIdBm;
    meas.RSRQdB = RSRQdB;

end