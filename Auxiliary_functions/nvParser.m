function [inputs,userSupplied] = nvParser(defaults, varargin)
    % COMM.INTERNAL.UTILITIES.NVPARSER Parse name-value pair input arguments.
    %
    %   [INPUTS,USRSUP] = COMM.INTERNAL.UTILITIES.NVPARSER(DEFAULTS,
    %   NVPAIRS) parses name-value pairs in NVPAIRS cell array. DEFAULTS
    %   struct has all possible argument names as fieldnames whose values
    %   are set to the corresponding default values. INPUTS is a struct
    %   with same field names as DEFAULTS and the values are set to the
    %   parsed values from NVPAIRS. USRSUP is a struct with the same field
    %   names as INPUTS and has logical values. If a field is set to true,
    %   then that field's value has been supplied by the user.
    %
    %   Partial and case-insensitive name matching is supported. This
    %   function supports MATLAB Coder.
    %
    %   This is an internal function that may be removed in a future
    %   release. The inputs to this function are assumed to meet the
    %   required specifications as documented here and are not validated.
    %
    %   % EXAMPLE 1: Set values for function arguments from name-value
    %   % pair input arguments. Use inside a function which accepts
    %   variable number of input arguments using varargin.
    %   defaults = struct('InputType', 'integer', 'UnitAveragePower', false, ...
    %       'PlotConstellation', false, 'OutputDataType', []);
    %   inputs = comm.internal.utilities.nvParser(defaults, varargin{:});
    %
    %   % EXAMPLE 2: Set values for function arguments from name-value
    %   % pairs
    %   defaults = struct('UnitAveragePower', false, 'OutputType', 'integer', ...
    %       'NoiseVariance', 1);
    %   inputArgs = {'OutputType', 'approxllr', 'NoiseVariance', 0.5};
    %   inputs = comm.internal.utilities.nvParser(defaults, inputArgs{:});
    
    %   Copyright 2021 The MathWorks, Inc.
    
    %#codegen
    
    if coder.target('MATLAB')
        % Simulation
        nArgs = length(varargin);
        if rem(nArgs,2) ~= 0
            error(message('comm:shared:InvalidNumArgs'));
        end
        inputs = defaults;
        argNames = fieldnames(defaults);
        nTotalArgs = length(argNames);
        userSuppliedArgs = false(nTotalArgs, 1);
        for k = 1:2:nArgs
            thisArg = varargin{k};
            % thisArg must be nonempty row char vector or scalar string
            if ~((ischar(thisArg) || isStringScalar(thisArg)) && ...
                    (isrow(thisArg) && (strlength(thisArg) > 0)))
                error(message('comm:shared:InvalidNameType'));
            end
            idx = strncmpi(argNames, thisArg, strlength(thisArg));
            % thisArg must uniquely match argNames
            if sum(idx) ~= 1
                error(message('comm:shared:InvalidName'));
            end            
            inputs.(argNames{idx}) = varargin{k+1};
            userSuppliedArgs(idx) = true;
        end
        if nargout == 2
          userSupplied = defaults([]);
          for k = 1:nTotalArgs
            userSupplied(1).(argNames{k}) = userSuppliedArgs(k);
          end
        end
    else
        % Code generation
        [inputs,userSupplied] = coder.internal.nvparse(defaults, varargin{:});
    end
end