function plotGDOP(gNBPos,detectedgNBs,UEPos,detgNBNums,curveX,curveY,gNBNums,estPos, dim)
%   plotPositionsAndHyperbolaCurves(GNBPOS,UEPOS,DETGNBNUMS,CURVEX,CURVEY,GNBNUMS,ESTPOS)
%   plots gNB, UE positions, and hyperbola curves by considering these
%   inputs:
%   GNBPOS     - Positions of gNBs
%   UEPOS      - Position of UE
%   DETGNBNUMS - Detected gNB numbers
%   CURVEX     - Cell array having the x-coordinates of hyperbola curves
%   CURVEY     - Cell array having the y-coordinates of hyperbola curves
%   GNBNUMS    - Cell array of gNB numbers corresponding to all hyperbolas
%   ESTPOS     - Estimated UE position

    numgNBs = length(gNBPos);
    s = zeros(numgNBs, dim);
    for bs=1:numgNBs
        s(bs,:) = reshape(gNBPos{bs}(1:dim), [dim,1]);
    end
    s = s(detectedgNBs,:);

    for gNBIdx = 1:numgNBs
        gNBPos{gNBIdx} = gNBPos{gNBIdx}(1:dim);
    end
    UEPos = UEPos(1:dim);
    estPos = estPos(1:dim);



    plotgNBAndUEPositions(gNBPos,UEPos,detgNBNums);
    for curveIdx = 1:numel(curveX)
        curves(curveIdx) = plot(curveX{1,curveIdx},curveY{1,curveIdx}, ...
               '--','LineWidth',1,'Color','k');
    end
    % Add estimated UE position to figure
    plot(estPos(1),estPos(2),'+','MarkerSize',10, ...
        'MarkerFaceColor','#D95319','MarkerEdgeColor','#D95319','LineWidth',2);

    for curveIdx = 1:numel(curves)
        % Create a data tip and add a row to the data tip to display
        % gNBs information for each hyperbola curve
        dt = datatip(curves(curveIdx));
        row = dataTipTextRow("Hyperbola of gNB" + gNBNums{curveIdx}(1) + ...
            " and gNB" + gNBNums{curveIdx}(2),'');
        curves(curveIdx).DataTipTemplate.DataTipRows = [row; curves(curveIdx).DataTipTemplate.DataTipRows];
        % Delete the data tip that is added above
        delete(dt);
    end
    grid on;
    hold on

    % Compute G
    [~, G] = CRB_TDOA(UEPos, s, 10^-3, dim);
    % Compute ellipse of GDOP
    [ellipsePoints] = calculateEllipse(G, UEPos, 1);
    
    % Plot ellipse of GDOP
    x_ell = ellipsePoints(1,:);
    y_ell = ellipsePoints(2,:);
    % fig = figure(1); 
    %fig.WindowState = 'maximized';
    % plot( AP(:,1) , AP(:,2) , '^','MarkerSize',10,'MarkerEdgeColor',[147,0,0]./255,'MarkerFaceColor',[147,0,0]./255)
    fill(x_ell,y_ell,[.9 .95 1],'edgecolor',[0, 0.4470, 0.7410],'linewidth',2);alpha(.5)
    % plot(UE(1),UE(2),'pk','LineWidth',1,'MarkerSize',9,'MarkerFaceColor',[.4 .4 1]);
    plot([s(:,1),repmat(UEPos(1),size(s,1),1)]',[s(:,2),repmat(UEPos(2),size(s,1),1)]','--','LineWidth',1,'color',[220,220,220]./255)
    % legend('Ellipse')
    xlabel('[m]'), ylabel('[m]');
    grid on
    xlim([min(s(:,1), [], 'all') max(s(:,1), [], 'all')])
    ylim([min(s(:,2), [], 'all') max(s(:,2), [], 'all')])

    % Add legend
    gNBLegends = cellstr(repmat("gNB",1,numel(detgNBNums)) + ...
        detgNBNums + [" (reference gNB)" repmat("",1,numel(detgNBNums)-1)]);
    legend([gNBLegends {'Actual UE Position'} repmat({''},1,numel(curveX)) {'Estimated UE Position'} ...
        {'Ellipse'}]);


    function plotgNBAndUEPositions(gNBPos,UEPos,gNBNums)
    %   plotgNBAndUEPositions(GNBPOS,UEPOS,GNBNUMS) plots gNB and UE positions.
    
        numgNBs = numel(gNBNums);
        colors = getColors(numel(gNBPos));
    
        figure;
        hold on;
        legendstr = cell(1,numgNBs);
    
        % Plot position of each gNB
        for gNBIdx = 1:numgNBs
            plot(gNBPos{gNBNums(gNBIdx)}(1),gNBPos{gNBNums(gNBIdx)}(2), ...
                'Color',colors(gNBNums(gNBIdx),:),'Marker','^', ...
                'MarkerSize',11,'LineWidth',2,'MarkerFaceColor',colors(gNBNums(gNBIdx),:));
            legendstr{gNBIdx} = sprintf('gNB%d',gNBIdx);
        end
    
        % Plot UE position
        plot(UEPos(1),UEPos(2),'ko','MarkerSize',10,'LineWidth',2,'MarkerFaceColor','k');
    
        axis equal;
        legend([legendstr 'UE']);
        xlabel('X Position (meters)');
        ylabel('Y Position (meters)');
    
    end


end