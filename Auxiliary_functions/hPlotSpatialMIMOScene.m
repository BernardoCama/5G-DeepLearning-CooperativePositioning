function hPlotSpatialMIMOScene(sceneParams,wT,wR)
%hPlotSpatialMIMOScene Plot spatial MIMO scenario
%   hPlotSpatialMIMOScene(SCENEPARAMS,WT,WR) plots the spatial MIMO
%   scenario by considering these inputs:
%
%   SCENEPARAMS is a structure with these fields:
%      TxArray         - Transmit antenna array System object
%      RxArray         - Receive antenna array System object
%      TxArrayPos      - Center of the transmitting antenna array,
%                        specified as a three-element vector in Cartesian
%                        form, [x;y;z], with respect to global coordinate
%                        system. Units are in meters
%      RxArrayPos      - Center of the receiving antenna array, specified
%                        as a three-element vector in Cartesian form,
%                        [x;y;z], with respect to global coordinate system.
%                        Units are in meters
%      ScatterersPos   - Scatterer locations as a 3-by-K matrix. K is the
%                        number of scatterers and each column of SCATPOS is
%                        a different scatterer and has the Cartesian form
%                        [x;y;z] with respect to global coordinate system.
%                        Units are in meters
%      Lambda          - Carrier wavelength, in meters
%      ArrayScaling    - Scaling factor to plot the transmit and receive
%                        antenna arrays. This is an optional field and the
%                        default value is calculated based on txArrayPos,
%                        rxArrayPos and lambda
%      MaxTxBeamLength - Maximum length of transmit beams in the plot. This
%                        is an optional field and the default value is
%                        considered based on txArrayPos and rxArrayPos
%      MaxRxBeamLength - Maximum length of receive beams in the plot. This
%                        is an optional field and the default value is
%                        considered based on txArrayPos and rxArrayPos
%
%   WT         - Steering weights for transmit beamforming. WT is specified
%                as a matrix, where each column represents the weights
%                corresponds to a separate direction
%   WR         - Steering weights for receive beamforming. WR is specified
%                as a matrix, where each column represents the weights
%                corresponds to a separate direction

%   Copyright 2020-2021 The MathWorks, Inc.

    % Extract the inputs
    txArray    = sceneParams.TxArray;
    rxArray    = sceneParams.RxArray;
    txArrayPos = sceneParams.TxArrayPos;
    rxArrayPos = sceneParams.RxArrayPos;
    scatPos    = sceneParams.ScatterersPos;
    lambda     = sceneParams.Lambda;
    fc = physconst('LightSpeed')/lambda;
    txRxDist = norm(txArrayPos-rxArrayPos);
    if isfield(sceneParams,'ArrayScaling')
        arrayScaling = sceneParams.ArrayScaling;
    else
        % Scaling factor for transmit and receive antenna arrays
        arrayScaling = txRxDist;
    end
    if isfield(sceneParams,'MaxTxBeamLength')
        txBeamScaling = sceneParams.MaxTxBeamLength;
    else
        % Calculate the scaling factor for normalized transmit beams
        txBeamScaling = txRxDist/3;
    end
    if isfield(sceneParams,'MaxRxBeamLength')
        rxBeamScaling = sceneParams.MaxRxBeamLength;
    else
        % Calculate the scaling factor for normalized receive beams
        rxBeamScaling = txRxDist/3;
    end

    % Get the positions of transmit antenna elements (in meters)
    txElemPos = getElementPosition(txArray);
    % Get the positions of receive antenna elements (in meters)
    rxElemPos = getElementPosition(rxArray);

    % Scale the antenna elements positions and shift them to centers of
    % antenna arrays
    txarraypos_plot = txElemPos*arrayScaling+txArrayPos;
    rxarraypos_plot = rxElemPos*arrayScaling+rxArrayPos;

    % Plot the transmit antenna array
    figure;
    h1 = plot3(txarraypos_plot(1,:),txarraypos_plot(2,:), ...
        txarraypos_plot(3,:),'ro','MarkerSize',2,'MarkerFaceColor','r');
    hold on;
    if isprop(txArray,'Size')
        txArraySize = txArray.Size;
    else
        txArraySize = txArray.NumElements;
    end
    txPanelCorners = txarraypos_plot(:,[1 txArraySize(1) prod(txArraySize) prod(txArraySize)-txArraySize(1)+1]);
    txPanel = patch(txPanelCorners(1,:),txPanelCorners(2,:),...
        txPanelCorners(3,:),[0.5725 0.6588 0.8196],'LineStyle',':','FaceAlpha',0.5);

    % Plot the receive antenna array
    h2 = plot3(rxarraypos_plot(1,:),rxarraypos_plot(2,:), ...
        rxarraypos_plot(3,:),'bo','MarkerSize',2,'MarkerFaceColor','b');
    if isprop(rxArray,'Size')
        rxArraySize = rxArray.Size;
    else
        rxArraySize = rxArray.NumElements;
    end
    rxPanelCorners = rxarraypos_plot(:,[1 rxArraySize(1) prod(rxArraySize) prod(rxArraySize)-rxArraySize(1)+1]);
    rxPanel = patch(rxPanelCorners(1,:),rxPanelCorners(2,:),...
        rxPanelCorners(3,:),[0.3571 0.3571 0.3571],'LineStyle',':','FaceAlpha',0.5);

    % Plot the scatterers and the paths connecting txArrayPos, scatPos and
    % rxArrayPos
    h3 = plot3(scatPos(1,:),scatPos(2,:),scatPos(3,:),'ro');
    numScatterers = size(scatPos,2);
    for m = 1:numScatterers
        h4(m) = plot3([txArrayPos(1) scatPos(1,m) rxArrayPos(1) ...
            scatPos(1,m)],[txArrayPos(2) scatPos(2,m) rxArrayPos(2) ...
            scatPos(2,m)],[txArrayPos(3) scatPos(3,m) rxArrayPos(3) ...
            scatPos(3,m)],'k'); %#ok<*AGROW>
    end

    % Plot the transmit beam patterns
    numTxBeams = size(wT,2);
    rng(42);
    txBeamColorMap = rand(numTxBeams,3);
    h5 = [];
    txBeamLegend = cell(1,numTxBeams);
    for txBeamIdx = 1:numTxBeams
        [txPat,txAZRange,txELRange] = pattern(txArray,fc,'Weights',wT(:,txBeamIdx));
        txbeam = db2mag(txPat);
        txbeam = (txbeam/max(txbeam(:)))*txBeamScaling;
        [x1,y1,z1] = sph2cart(deg2rad(txAZRange),deg2rad(txELRange'),txbeam);
        txhandle = surf(x1+txArrayPos(1),...
                        y1+txArrayPos(2), ...
                        z1+txArrayPos(3));
        txhandle.EdgeColor = 'none';
        txhandle.FaceColor = txBeamColorMap(txBeamIdx,:);
        txhandle.FaceAlpha = 0.9;
        h5(txBeamIdx) = txhandle;
        if numTxBeams == 1
            txBeamLegend = {'Transmit beam'};
        else
            txBeamLegend{txBeamIdx} = ['Transmit beam ' num2str(txBeamIdx)];
        end
    end

    % Plot the receive beam patterns
    numRxBeams = size(wR,2);
    rng(141);
    rxBeamColorMap = rand(numRxBeams,3);
    h6 = [];
    rxBeamLegend = cell(1,numRxBeams);
    for rxBeamIdx = 1:numRxBeams
        [rxPat,rxAZRange,rxELRange] = pattern(rxArray,fc,'Weights',wR(:,rxBeamIdx));
        rxbeam = db2mag(rxPat);
        rxbeam = (rxbeam/max(rxbeam(:)))*rxBeamScaling;
        [x2,y2,z2] = sph2cart(deg2rad(rxAZRange),deg2rad(rxELRange'),rxbeam);
        rxhandle = surf(x2+rxArrayPos(1),...
                        y2+rxArrayPos(2), ...
                        z2+rxArrayPos(3));
        rxhandle.EdgeColor = 'none';
        rxhandle.FaceColor = rxBeamColorMap(rxBeamIdx,:);
        rxhandle.FaceAlpha = 0.7;
        h6(rxBeamIdx) = rxhandle;
        if numRxBeams == 1
            rxBeamLegend = {'Receive beam'};
        else
            rxBeamLegend{rxBeamIdx} = ['Receive beam ' num2str(rxBeamIdx)];
        end
    end

    xlabel('X axis (m)')
    ylabel('Y axis (m)')
    zlabel('Z axis (m)')
    legend([h1 txPanel h2 rxPanel h3 h4(1) h5 h6], ...
        [{'Transmit antenna elements','Transmit antenna panel',...
        'Receive antenna elements','Receive antenna panel','Scatterer(s)', ...
        'Scatterer path(s)'} txBeamLegend rxBeamLegend],'Location', ...
        'bestoutside');
    hold off;
    axis equal;
    grid on;
end