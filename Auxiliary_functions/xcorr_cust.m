function [c,lags, fftinfo_fft, fftinfo_ifft] = xcorr(x,varargin)
%XCORR Cross-correlation function estimates.
%   C = XCORR(A,B), where A and B are length M vectors (M>1), returns
%   the length 2*M-1 cross-correlation sequence C. If A and B are of
%   different length, the shortest one is zero-padded. C will be a
%   row vector if A is a row vector, and a column vector if A is a
%   column vector.
%
%   XCORR produces an estimate of the correlation between two random
%   (jointly stationary) sequences:
%          C(m) = E[A(n+m)*conj(B(n))] = E[A(n)*conj(B(n-m))]
%   It is also the deterministic correlation between two deterministic
%   signals.
%
%   C = XCORR(A), where A is a length M vector, returns the length 2*M-1
%   auto-correlation sequence C. The zeroth lag of the output correlation
%   is in the middle of the sequence, at element M.
%
%   C = XCORR(A), where A is an M-by-N matrix (M>1), returns a large matrix
%   with 2*M-1 rows and N^2 columns containing the cross-correlation
%   sequences for all combinations of the columns of A; the first N columns
%   of C contain the delays and cross correlations using the first column
%   of A as the reference, the next N columns of C contain the delays and
%   cross correlations using the second column of A as the reference, and
%   so on.
%
%   C = XCORR(...,MAXLAG) computes the (auto/cross) correlation over the
%   range of lags:  -MAXLAG to MAXLAG, i.e., 2*MAXLAG+1 lags.
%   If missing, default is MAXLAG = M-1.
%
%   [C,LAGS] = XCORR(...)  returns a vector of lag indices (LAGS).
%
%   XCORR(...,SCALEOPT), normalizes the correlation according to SCALEOPT:
%     'biased' - scales the raw cross-correlation by 1/M.
%     'unbiased' - scales the raw correlation by 1/(M-abs(lags)).
%     'normalized' or 'coeff' - normalizes the sequence so that the 
%                               auto-correlations at zero lag are 
%                               identically 1.0.
%     'none' - no scaling (this is the default).
%
%   % Example:
%   % Compute and plot the cross-correlation of two 16-sample
%   % exponential sequences
%
%   N = 16;
%   n = 0:N-1;
%   a = 0.84;
%   b = 0.92;
%   xa = a.^n;
%   xb = b.^n;
%   [r,lags] = xcorr(xa,xb);
%   stem(lags,r)
%
%   See also XCOV, CORRCOEF, CONV, COV.

%   Copyright 1988-2021 The MathWorks, Inc.

%   References:
%     S.J. Orfanidis, "Optimum Signal Processing. An Introduction"
%     2nd Ed. Macmillan, 1988.


fftinfo_ifft = varargin{end};
varargin(end) = [];
fftinfo_fft = varargin{end};
varargin(end) = [];


narginchk(1,4);
if isnumeric(x)
    if isa(x,'uint64') || isa(x,'int64')
        error(message('MATLAB:xcorr:InvalidInputType'));
    end
end

[ySupplied,maxlagInput,scale] = ...
    matlab.internal.math.parseXcorrOptions(varargin{:});


% Calculate the cross-correlation.

% Cross-correlation of two column vectors.
if ~iscolumn(x)
    error(message('MATLAB:xcorr:MismatchedAB'));
end
y = varargin{1}(:); % y was validated to be a vector, make it a column.

maxlagDefault = max(size(x,1),size(y,1)) - 1;
maxlag = maxlagDefault;



% Perform the cross-correlation.
% 1
% c = crosscorr_cust(x,y,maxlag);
% 2
% tic
% [c, fftinfo_fft, fftinfo_ifft] = crosscorr(x,y,maxlag, fftinfo_fft, fftinfo_ifft);
% toc
% 3
c = crosscorr_original(x,y,maxlag);


% Scale the output.
c = scaleOutput(scale,c,x,y);



% Pad the output with zeros.
if maxlag > maxlagDefault
    zeropad = zeros(maxlag - maxlagDefault,size(c,2),'like',c);
    c = [zeropad; c; zeropad];
end

if nargout >= 2
    lags = -maxlag:maxlag;
end

%--------------------------------------------------------------------------
function [c,fftinfo_fft, fftinfo_ifft] = crosscorr(x,y,maxlag, fftinfo_fft, fftinfo_ifft)

% Compute cross-correlation for vector inputs. Output is clipped based on
% maxlag but not padded if maxlag >= max(size(x,1),size(y,1)).
nx = numel(x);
ny = numel(y);
m = max(nx,ny);
maxlagDefault = m-1;
mxl = min(maxlag,maxlagDefault);

m2 = findTransformLength(m);
fftw('dwisdom',fftinfo_fft);
X = fft(x,m2,1);
Y = fft(y,m2,1);
if isempty(fftinfo_fft)
    fftinfo_fft = fftw('dwisdom');
end
fftw('dwisdom',fftinfo_ifft);
c1 = ifft(X.*conj(Y),[],1);
if isempty(fftinfo_fft)
    fftinfo_ifft = fftw('dwisdom');
end

% Keep only the lags we want and move negative lags before positive
% lags.
c = [c1(m2 - mxl + (1:mxl)); c1(1:mxl+1)];



%--------------------------------------------------------------------------
function m = findTransformLength(m)
m = 2*m;
while true
    r = m;
    for p = [2 3 5 7]
        while (r > 1) && (mod(r, p) == 0)
            r = r / p;
        end
    end
    if r == 1
        break;
    end
    m = m + 1;
end

%--------------------------------------------------------------------------
function c = crosscorr_original(x,y,maxlag)
% Compute cross-correlation for vector inputs. Output is clipped based on
% maxlag but not padded if maxlag >= max(size(x,1),size(y,1)).
nx = numel(x);
ny = numel(y);
m = max(nx,ny);
maxlagDefault = m-1;
mxl = min(maxlag,maxlagDefault);

if any(~isfinite(x),'all') || any(~isfinite(y),'all')
    c1 = conv(x,conj(flip(y)));
    if mxl <= maxlagDefault
        % Clip if maxlag is small. This would be a no-op for larger maxlag.
        % Account for 0 lag corresponding to c1(ny).
        c1 = c1(max(1,ny-mxl):(ny+min(mxl,nx-1)));
    end
    % Pad the head or tail if nx >= ny or nx < ny, respectively.
    % Note that we may need to both clip and pad: xcorr(1:5,1:3,3).
    c = zeros(2*mxl+1,1,'like',c1);
    offset = max(0,mxl-ny+1);
    c(offset+(1:numel(c1))) = c1;
else
    m2 = findTransformLength(m);
    X = fft(x,m2,1);
    Y = fft(y,m2,1);
    if isreal(x) && isreal(y)
        c1 = ifft(X.*conj(Y),[],1,'symmetric');
    else
        c1 = ifft(X.*conj(Y),[],1);
    end
    % Keep only the lags we want and move negative lags before positive
    % lags.
    c = [c1(m2 - mxl + (1:mxl)); c1(1:mxl+1)];
end

%--------------------------------------------------------------------------
function c = scaleOutput(scale,c,x,y)
% Scale correlation as specified.
if strcmp(scale,'none')
    return
end

ySupplied = nargin == 4;
m = size(x,1);
if ySupplied && (m ~= size(y,1))
    error(message('MATLAB:xcorr:NoScale'));
end

if strcmp(scale,'biased')
    % Scales the raw cross-correlation by 1/M.
    c = c./m;
elseif strcmp(scale,'unbiased')
    % Scales the raw correlation by 1/(M-abs(lags)).
    L = (size(c,1) - 1)/2;
    scaleUnbiased = (m - abs(-L:L)).';
    scaleUnbiased(scaleUnbiased <= 0) = 1;
    c = c./scaleUnbiased;
else % 'normalized'/'coeff'
    % Normalizes the sequence so that the auto-correlations
    % at zero lag are identically 1.0.
    if ySupplied
        % Compute autocorrelations at zero lag.
        % scale = norm(x)*norm(y) is numerically superior but slower.
        cxx0 = sum(abs(x).^2);
        cyy0 = sum(abs(y).^2);
        scaleCoeffCross = sqrt(cxx0*cyy0);
        c = c./scaleCoeffCross;
    elseif size(c,2) == 1
        % Autocorrelation of a vector. Normalize by c[0].
        mid = (size(c,1) + 1)/2; % row corresponding to zero lag.
        c = c./c(mid);
    else
        % Compute the indices corresponding to the columns that are
        % autocorrelations.
        [~,n] = size(x);
        % Note that size(c,2) = n^2.
        kvec = 1:n+1:n*n; % a 1-by-n row vector
        % kvec is an index vector such that for an n-by-n matrix A,
        % A(kvec(j)) = A(j,j).
        mid = (size(c,1) + 1)/2; % row index corresponding to zero lag
        trow = sqrt(c(mid,kvec)); % a 1-by-n row vector
        tmat = trow.'*trow; % an n-by-n matrix, tmat(i,j) = trow(i)*trow(j)
        scaleCoeffAuto = tmat(:).'; % a 1-by-n^2 row vector
        % The autocorrelations at zero-lag are normalized to one.
        c = c./scaleCoeffAuto;
    end
end
