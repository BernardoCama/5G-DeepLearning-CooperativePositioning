
function drawaxis(varargin)
% DRAWAXIS - Draws an axis at the specified crossing point. No input
% arguments move the x-axis to cross at zero.
%   
%   DRAWAXIS(axes_handle) - applies the axis crossing to the axes specified by
%   the axes handle. This must be the first input argument.
%    
%   DRAWAXIS('x', x_crossing, ...) - applies the axis crossing to the
%   x-axis, and moves it to the y-value specified by x_crossing.
%    
%   DRAWAXIS('y', y_crossing, ...) - same as 'x' but applied to y-axis.
%   
%   DRAWAXIS('movelabel', ...) - moves the position of the axis label to
%   follow to the new position of the axis.
%   
% Only use this function once the axes are formatted correctly. This
% function only copies the properties of the axes and uses them to create
% line and text objects on the axes. To make the original axis invisible,
% it's properties are changed to remove the tick marks and tick mark
% labels. So only call the function once per axes, because successive calls
% will cause errors and create havoc.


MOVE_X_AXIS = logical(0);
MOVE_Y_AXIS = logical(0);
MOVE_LABEL = logical(0);

myX_Crossing = 0;
myY_Crossing = 0;


axes_color = 'k';

if nargin > 0
    if ishandle(varargin{1})
        h = varargin{1};
        starting_arg = 2;
    else
        h = gca;
        starting_arg = 1;
    end
    
    for iCount = starting_arg:nargin
        
        switch lower(varargin{iCount})
            
            case 'x'
                iCount = iCount + 1;
                MOVE_X_AXIS = logical(1);
                myX_Crossing = varargin{iCount};
                
            case 'y'
                iCount = iCount + 1;
                MOVE_Y_AXIS = logical(1);
                myY_Crossing = varargin{iCount};
                
            case 'movelabel'
                iCount = iCount + 1;
                MOVE_LABEL = logical(varargin{iCount});
                
        end
        
    end
    
else
    
    MOVE_X_AXIS = logical(0);
    MOVE_Y_AXIS = logical(0);
    MOVE_LABEL = logical(0);

    myX_Crossing = 0;
    myY_Crossing = 0;
    
    h = gca;
    
end

props = get(h);

      
                
    
if MOVE_X_AXIS
    
    if (myX_Crossing< props.XLim(1)) | (myX_Crossing> props.XLim(2))
        error('Specified X crossing outside axis limits')
        return
    end
    
    
    tick_bottom = -props.TickLength(1)*diff(props.YLim);
    tick_top = 0;
    
    h_xaxis = line(props.XLim, [0 0] + myX_Crossing, 'color', axes_color);


    if ~isempty(props.XTick)
        xtick_x = repmat(props.XTick, 2, 1);
        xtick_y = repmat([tick_bottom; tick_top] + myX_Crossing, 1, length(props.XTick));
        h_ticks = line(xtick_x, xtick_y, 'color', axes_color);

    end

    %using a for loop so the leading and trailing spaces can be removed from
    %the tick labels
    nTicks = length(props.XTick);
    h_ticklabels = zeros(size(props.XTick));
    for iCount = 1:nTicks
        h_ticklabels(iCount) = text(props.XTick(iCount), tick_bottom + myX_Crossing, ...
                                    strtrim(props.XTickLabel(iCount, :)), ...
                                    'HorizontalAlignment', 'Center', ...
                                    'VerticalAlignment', 'Top', ...
                                    'FontSize', props.FontSize, ...
                                    'FontName', props.FontName, ...
                                    'FontAngle', props.FontAngle, ...
                                    'FontUnits', props.FontUnits, ...
                                    'FontWeight', props.FontWeight);

    end

    if strcmp(lower(props.XMinorTick), 'on')
        nMinorTicks = 10;
        dist_bw_ticks = diff(props.XTick)/nMinorTicks;
        for iCount = 1:nTicks - 1
            xminors_x = repmat(props.XTick(iCount)+dist_bw_ticks(iCount):dist_bw_ticks(iCount):props.XTick(iCount+1) - dist_bw_ticks, 2, 1);
            xminors_y = repmat([tick_bottom; tick_top]*0.5 + myX_Crossing, 1, nMinorTicks - 1);
            h_minors = line(xminors_x, xminors_y, 'color', axes_color);
        end

    end

    if MOVE_LABEL

        label_position = get(props.XLabel, 'Position');
        label_position(2) = label_position(2)+(myX_Crossing - props.YLim(1));
        set(props.XLabel, 'Position', label_position);

    end
    set(h, 'XTick', [], 'XTickLabel', [])
end


if MOVE_Y_AXIS
    
    if (myY_Crossing< props.YLim(1)) | (myY_Crossing> props.YLim(2))
        error('Specified Y crossing outside axis limits')
        return
    end
    
    tick_left = -props.TickLength(1)*diff(props.XLim);
    tick_right = 0;
    
    h_yaxis = line([0 0] + myY_Crossing, props.YLim, 'color', axes_color);


    if ~isempty(props.YTick)
        ytick_y = repmat(props.YTick, 2, 1);
        ytick_x = repmat([tick_left; tick_right] + myY_Crossing, 1, length(props.YTick));
        h_ticks = line(ytick_x, ytick_y, 'color', axes_color);

    end

    %using a for loop so the leading and trailing spaces can be removed from
    %the tick labels
    nTicks = length(props.YTick);
    h_ticklabels = zeros(size(props.YTick));
    for iCount = 1:nTicks
        h_ticklabels(iCount) = text(tick_left + myY_Crossing, props.YTick(iCount), ...
                                    strtrim(props.YTickLabel(iCount, :)), ...
                                    'HorizontalAlignment', 'Right', ...
                                    'VerticalAlignment', 'Middle', ...
                                    'FontSize', props.FontSize, ...
                                    'FontName', props.FontName, ...
                                    'FontAngle', props.FontAngle, ...
                                    'FontUnits', props.FontUnits, ...
                                    'FontWeight', props.FontWeight);

    end

    if strcmp(lower(props.YMinorTick), 'on')
        nMinorTicks = 10;
        dist_bw_ticks = diff(props.YTick)/nMinorTicks;
        for iCount = 1:nTicks - 1
            yminors_y = repmat(props.YTick(iCount)+dist_bw_ticks(iCount):dist_bw_ticks(iCount):props.YTick(iCount+1) - dist_bw_ticks, 2, 1);
            yminors_x = repmat([tick_left; tick_right]*0.5 + myY_Crossing, 1, nMinorTicks - 1);
            h_minors = line(yminors_x, yminors_y, 'color', axes_color);
        end

    end

    if MOVE_LABEL

        label_position = get(props.YLabel, 'Position');
        label_position(1) = label_position(1)+(myY_Crossing - props.XLim(1));
        set(props.YLabel, 'Position', label_position);

    end
    set(h, 'YTick', [], 'YTickLabel', [])
end

