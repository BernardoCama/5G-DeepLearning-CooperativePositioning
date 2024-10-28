function [offset,mag] = nrTimingEstimate_cust(varargin)

    fftinfo_ifft = varargin{end};
    varargin(end) = [];
    fftinfo_fft = varargin{end};
    varargin(end) = [];

    narginchk(3,14);
    
    % Get optional inputs or inputs whose position depends upon the syntax
    [waveform,refGrid,ofdmInfo,initialNSlot,hasSampleRate] = getOptionalInputs(varargin{:});
    
    % Get the number of ports P in the reference resource grid
    P = size(refGrid,3);
    
    % Perform OFDM modulation
    overlapping = false;
    ref = nr5g.internal.OFDMModulate(refGrid,ofdmInfo,initialNSlot,overlapping,hasSampleRate);
    
    % Get the number of time samples T and receive antennas R in the 
    % waveform
    [T,R] = size(waveform);

    % Pad the input waveform if necessary to make it longer than the
    % correlation reference signal; this is required to normalize xcorr
    % behavior as it always pads the shorter input signal
    minlength = size(ref,1);
    if (T < minlength)
        waveformPad = [waveform; zeros([minlength-T R],'like',waveform)];
        T = minlength;
    else
        waveformPad = waveform;
    end

    
    % Create array 'mag' to store correlation magnitude for each time
    % sample, receive antenna and port
    mag = zeros([T R P],'like',waveformPad);

    % For each receive antenna
    for r = 1: 1 % R
        
        % For each port
        for p = 1:P
        
            % Correlate the given antenna of the received signal with the
            % given port of the reference signal
            [refcorr, ~, fftinfo_fft, fftinfo_ifft] = xcorr_cust(waveformPad(:,r),ref(:,p), fftinfo_fft, fftinfo_ifft);
            mag(:,r,p) = abs(refcorr(T:end));
            
        end
        
    end

    
    % Sum the magnitudes of the ports
    mag = sum(mag,3);
    
    % Find timing peak in the sum of the magnitudes of the receive antennas
    [~,peakindex] = max(sum(mag,2));
    offset = peakindex - 1;
    
end

% Parse optional inputs
function [waveform,refGrid,ofdmInfo,initialNSlot,hasSampleRate, fftinfo] = getOptionalInputs(varargin)

    
    coder.extrinsic('nr5g.internal.parseOptions');

    fcnName = 'nrTimingEstimate';
    
    % Determine if syntax with nrCarrierConfig is being used and parse
    % relevant inputs
    isCarrierSyntax = isa(varargin{1},'nrCarrierConfig');
    if (isCarrierSyntax)
        carrier = varargin{1};
        validateattributes(carrier,{'nrCarrierConfig'}, ...
            {'scalar'},fcnName,'Carrier specific configuration object');
        waveform = varargin{2};
        initialNSlot = carrier.NSlot;
        firstrefarg = 3;
    else
        narginchk(5,14);
        waveform = varargin{1};
        NRB = varargin{2};
        SCS = varargin{3};
        initialNSlot = varargin{4};
        firstrefarg = 5;
    end

    
    % Validate waveform
    validateattributes(waveform,{'double','single'}, ...
        {},fcnName,'WAVEFORM');
    coder.internal.errorIf(~ismatrix(waveform), ...
        'nr5g:nrTimingEstimate:InvalidWaveformDims',ndims(waveform));
    
    if (~isCarrierSyntax)
        % Validate the number of resource blocks (1...275)
        validateattributes(NRB,{'numeric'}, ...
            {'real','integer','scalar','>=',1,'<=',275},fcnName,'NRB');

        % Validate subcarrier spacing input in kHz (15/30/60/120/240)
        validateattributes(SCS,{'numeric'}, ...
            {'real','integer','scalar'},fcnName,'SCS');
        validSCS = [15 30 60 120 240];
        coder.internal.errorIf(~any(SCS==validSCS), ...
            'nr5g:nrTimingEstimate:InvalidSCS', ...
            SCS,num2str(validSCS));

        % Validate zero-based initial slot number
        validateattributes(initialNSlot,{'numeric'}, ...
            {'real','nonnegative','scalar','integer'}, ...
            fcnName,'INITIALNSLOT');
    end
    
    % Determine whether the refInd,refSym syntax or refGrid syntax is being
    % used
    isRefGridSyntax = ...
        (nargin==firstrefarg) || ischar(varargin{firstrefarg + 1}) ...
            || isstring(varargin{firstrefarg + 1}) ...
            || isstruct(varargin{firstrefarg + 1});
    if (isRefGridSyntax)
        % nrTimingEstimate(...,refGrid,...)
        firstoptarg = firstrefarg + 1;
    else
        % nrTimingEstimate(...,refInd,refSym,...)
        firstoptarg = firstrefarg + 2;
    end
    
    % Parse options and get OFDM information
    if (isCarrierSyntax)
        optNames = {'Nfft','SampleRate','CarrierFrequency'};
        opts = coder.const(nr5g.internal.parseOptions( ...
            [fcnName '(carrier,...'],optNames,varargin{firstoptarg:end}));
        ofdmInfo = nr5g.internal.OFDMInfo(carrier,opts);
    else
        optNames = {'CyclicPrefix','Nfft','SampleRate','CarrierFrequency'};
        opts = coder.const(nr5g.internal.parseOptions( ...
            fcnName,optNames,varargin{firstoptarg:end}));
        ECP = strcmpi(opts.CyclicPrefix,'extended');
        if firstoptarg > numel(varargin)
            ofdmInfo = nr5g.internal.OFDMInfo(NRB,SCS,ECP,opts);
        else
            % If there are NV pairs, ofdminfo must be constant
            ofdmInfo = coder.const(feval('nr5g.internal.OFDMInfo',NRB,SCS,ECP,opts));
        end
    end
    
    % If performing code generation, the presence of sample rate with the
    % function syntax using nrCarrierConfig triggers a compile-time error
    hasSampleRate = ~isempty(opts.SampleRate);
    if (~isempty(coder.target) && isCarrierSyntax && hasSampleRate)
        coder.internal.errorIf(true,'nr5g:nrTimingEstimate:CompilationCarrierSampleRate',sprintf('%g',opts.SampleRate),'IfNotConst','Fail');
    end
    
    % Get the number of subcarriers K and OFDM symbols L from the OFDM 
    % information
    K = ofdmInfo.NSubcarriers;
    L = ofdmInfo.SymbolsPerSlot;

    % Validate reference inputs
    if (~isRefGridSyntax)
    
        refInd = varargin{firstrefarg};
        refSym = varargin{firstrefarg + 1};

        % Validate reference indices
        validateattributes(refInd,{'numeric'}, ...
            {'positive','finite','2d'},fcnName,'REFIND');
        refIndColumns = size(refInd,2);

        % Validate reference symbols
        validateattributes(refSym,{'double','single'}, ...
            {'finite','2d'},fcnName,'REFSYM');
        refSymColumns = size(refSym,2);

        % Get the number of ports, based on the range of the reference
        % symbol indices
        if (isempty(refInd) && isempty(refSym) && refIndColumns==refSymColumns)
            P = max(refIndColumns,1);
        else
            P = ceil(max(double(refInd(:))/(K*L)));
        end
        
        % Create the reference resource grid
        refGrid = zeros([K L P],'like',waveform);
        
        % Map the reference symbols to the reference grid
        refGrid(refInd) = refSym;
        
    else % One optional input, not including 'cp' if present
        
        refGrid = varargin{firstrefarg};
        
        % Validate reference grid
        validateattributes(refGrid, ...
            {'double','single'},{'finite','3d'},fcnName,'REFGRID');
        coder.internal.errorIf(size(refGrid,1)~=K, ...
            'nr5g:nrTimingEstimate:InvalidRefGridSubcarriers', ...
        size(refGrid,1),K);
        coder.internal.errorIf(size(refGrid,2)==0, ...
            'nr5g:nrTimingEstimate:InvalidRefGridOFDMSymbols');
        
    end

end
