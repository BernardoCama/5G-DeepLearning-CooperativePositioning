function y = convertSNR(x,inSNRMode,varargin)
%convertSNR Convert SNR values
%   Y = convertSNR(X,InputMode) converts input signal-to-noise ratio value,
%   X, to SNR, and returns in Y. X must be a row vector of real numeric
%   values. Specify the ratio type for X in InputMode as one of 'ebno',
%   'esno', or 'snr'. If InputMode is 'ebno', X is energy per bit to noise
%   power spectral density ratio (Eb/No). If InputMode is 'esno', X is
%   energy per symbol to noise power spectral density ratio (Es/No). If
%   InputMode is 'snr', X is signal power to noise power ratio (SNR). The
%   conversion uses the default values for BitsPerSymbol, CodingRate, and
%   SamplesPerSymbol, which are all 1. To change the default values, use
%   the Name-Value arguments as defined below.
%   
%   Y = convertSNR(X,InputMode,OutputMode) converts input signal-to-noise
%   ratio value, X, to OutputMode, and returns in Y. Specify the ratio type
%   for Y in OutputMode as one of 'ebno', 'esno', or 'snr'. 
%   
%   Y = convertSNR(X,InputMode,...,Name=Value) specifies additional
%   name-value pair arguments described below. 
%
%   'SamplesPerSymbol'  Specify samples per symbol as a positive, integer
%                       valued scalar. The default is 1. SamplesPerSymbol
%                       is ignored if InputMode is 'ebno' and OutputMode
%                       is 'esno', or vice versa. 
%
%   'BitsPerSymbol'     Specify bits per symbol as a positive, integer
%                       valued scalar. The default is 1. BitsPerSymbol is
%                       ignored if InputMode is 'esno' and OutputMode is
%                       'snr', or vice versa.
%
%   'CodingRate'        Specify coding rate as a floating-point scalar in
%                       the range (0 1]. The default is 1. CodingRate is
%                       ignored if InputMode is 'esno' and OutputMode is
%                       'snr', or vice versa.
%
%   The following is a summary of parameters used in the conversion based
%   on the choice of InputMode and OutputMode.
%
%    Conversion (InputMode <-> OutputMode) | Parameter           | Default
%    ----------------------------------------------------------------------
%    'ebno' <-> 'snr'                      | 'BitsPerSymbol'     |    1
%                                          | 'CodingRate'        |    1
%                                          | 'SamplesPerSymbol'  |    1
%    ----------------------------------------------------------------------
%    'ebno' <-> 'esno'                     | 'BitsPerSymbol'     |    1
%                                          | 'CodingRate'        |    1
%    ----------------------------------------------------------------------
%    'esno' <-> 'snr'                      | 'SamplesPerSymbol'  |    1
%    ----------------------------------------------------------------------
%    'esno' <-> 'ebno'                     | 'BitsPerSymbol'     |    1
%                                          | 'CodingRate'        |    1
%    ----------------------------------------------------------------------
%    'snr' <-> 'ebno'                      | 'BitsPerSymbol'     |    1
%                                          | 'CodingRate'        |    1
%                                          | 'SamplesPerSymbol'  |    1
%    ----------------------------------------------------------------------
%    'snr' <-> 'esno'                      | 'SamplesPerSymbol'  |    1
%    ----------------------------------------------------------------------
%
%   % Example 1:
%   %   Add noise equivalent to Eb/No of 6 dB to an 8-PSK modulated signal.
%   EbNo = 6
%   d = randi([0 7],100,1);
%   x = pskmod(d, 8);
%   SNR = convertSNR(EbNo,"ebno", BitsPerSymbol=3)
%   y = awgn(x,SNR);
%   figure
%   plot(real(x))
%   hold on
%   plot(real(y))
%   legend('Perfect signal', 'Noisy signal')
%
%   % Example 2:
%   %   Convert Eb/No values to SNR values for a system with 16-QAM
%   %   modulation, 8 samples per symbol and a coding rate of 1/3.
%   EbNo = 0:3:9
%   SNR = convertSNR(EbNo, "ebno", BitsPerSymbol=4, ...
%                                   SamplesPerSymbol=8, ...
%                                   CodingRate=1/3)
%
%   % Example 3:
%   %   Convert SNR values to Es/No values for a system with 4 samples per
%   %   symbol over sampling. 
%   SNR = -4:2:4
%   EsNo = convertSNR(SNR, "snr", "esno", SamplesPerSymbol=4)
%
%   % Example 4:
%   %   Add noise equivalent to Eb/No of 6 dB to a 16-QAM modulated signal.
%   %   Soft demodulate.
%   EbNo = 6
%   b = randi([0 1],1000,1);
%   x = qammod(b,16,InputType='bit');
%   SNR = convertSNR(EbNo,"ebno", BitsPerSymbol=4)
%   [y,s2] = awgn(x,SNR);
%   sb = qamdemod(y,16,OutputType='llr',NoiseVariance=s2);
%   figure
%   plot(sb,'.')
%   xlabel('Bit Number')
%   ylabel('LLR')
%
%   See also awgn.

%   Copyright 2021 The MathWorks, Inc.

%#codegen

funcName = mfilename;
% Validate required inputs
validateattributes(x,{'double','single'},{'nonempty','row','real'},...
  funcName,"X")
inSNRMode = validatestring(inSNRMode,{'ebno','esno','snr'},...
  funcName,"InputMode");

if mod(length(varargin),2) == 1
  outSNRMode = validatestring(varargin{1}, ...
    {'ebno', 'esno', 'snr'},funcName,'OutputMode');
  nvStart = 2;
else
  outSNRMode = 'snr';
  nvStart = 1;
end

% Parse and validate optional inputs
defaults = struct(...
  'BitsPerSymbol',1,...
  'CodingRate',1,...
  'SamplesPerSymbol',1);
[inputs,userSupplied] = nvParser(defaults, ...
  varargin{nvStart:end});
if userSupplied.SamplesPerSymbol
  validateattributes(inputs.SamplesPerSymbol,...
    {'numeric'},{'scalar','nonnan','finite','real','integer','positive'},...
    funcName,'SamplesPerSymbol')
end
if userSupplied.BitsPerSymbol
  validateattributes(inputs.BitsPerSymbol,...
    {'numeric'},{'scalar','nonnan','finite','real','integer','positive'},...
    funcName,'BitsPerSymbol')
end
if userSupplied.CodingRate
  validateattributes(inputs.CodingRate,...
    {'double','single'},{'scalar','nonnan','real','positive',"<=",1},...
    funcName,'CodingRate')
end

bps = cast(inputs.BitsPerSymbol,'like',x);
R = cast(inputs.CodingRate,'like',x);
sps = cast(inputs.SamplesPerSymbol,'like',x);

switch inSNRMode
  case 'ebno'
    switch outSNRMode
      case 'snr'
        y = x + 10*log10((bps*R)/sps);
      case 'esno'
        y = x + 10*log10(bps*R);
      otherwise % 'ebno'
        y = x;
    end
  case 'esno'
    switch outSNRMode
      case 'snr'
        y = x - 10*log10(sps);
      case 'ebno'
        y = x - 10*log10(bps*R);
      otherwise % 'esno'
        y = x;
    end
  otherwise % 'snr'
    switch outSNRMode
      case 'esno'
        y = x + 10*log10(sps);
      case 'ebno'
        y = x + 10*log10(sps/(bps*R));
      otherwise % 'snr'
        y = x;
    end
end
