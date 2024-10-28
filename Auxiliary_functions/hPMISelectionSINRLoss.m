%hPMISelectionSINRLoss Compute SINR loss after precoding
%   [LOSS, PMIP] = hPMISelectionSINRLoss(PMI, NLAYERS, HESTP, NVARP) returns
%   an SINR loss LOSS when precoder matrix indicators PMI are used in a
%   codebook-based transmission over a MIMO channel HESTP with AWGN noise
%   variance NVARP. Codebook-based transmission is defined in TS 38.211
%   Section 6.3.1.5. The input PMI must be an NSubbands-by-1 vector
%   containing the estimated precoding matrix indicators. NLAYERS denotes
%   the number of layers (1...4). The SINR is calculated on the resource
%   elements (RE) specified by the channel estimates HESTP of dimensions
%   K-by-L-by-R-by-P where K is the number of subcarriers, L is the number
%   of OFDM symbols, R is the number of receive antennas and P is the
%   number of reference signal ports.
%
%   LOSS is defined as a ratio of the SINR after precoding with PMIs
%   selected using a practical channel estimate and the SINR after
%   precoding with PMIs selected using a perfect channel estimate. The SINR
%   is calculated on the resource elements (RE) specified by the array
%   REFGRID of dimensions K-by-L-by-R-by-P where K is the number of
%   subcarriers, L is the number of OFDM symbols, R is the number of
%   receive antennas and P is the number of reference signal ports.
%
%   PMIP is the vector of precoding matrix indicators selected using the
%   perfect channel estimate HESTP and noise power estimate NVARP.
%
%   See also hPMISelect, nrPUSCHCodebook.

%   Copyright 2019 The MathWorks, Inc.

function [loss, pmip] = hPMISelectionSINRLoss(pmi, nlayers, hEstP, nVarP)
 
    NRB = size(hEstP,1)/12;
    L = size(hEstP,2);   
    nports = size(hEstP,4);

    NumCSISubbands = size(pmi,1);
    r = NRB/NumCSISubbands;
    NumEqualPMISubbands = NumCSISubbands - (ceil(r)~=r);
    
    pmiBandSize = floor(NRB/NumEqualPMISubbands);
    lastBandSize = mod(NRB,pmiBandSize*NumEqualPMISubbands);
    lastBandSize = lastBandSize + pmiBandSize*(lastBandSize==0);
        
    % Perform TPMI selection using perfect channel estimates
    pmip = hPMISelect(nlayers,hEstP,nVarP,pmiBandSize);

    % Initialize SINR matrices
    sinr = zeros(NRB*12,L);
    sinrPerfect = sinr;
    
    % Expand PMI per subband into PMI per RE
    pmiPerRE = kron(pmi(1:end-1),ones(pmiBandSize*12,L));
    
    % Handle last subband (might be of a different size)
    pmiPerRE = [pmiPerRE; pmi(end)*ones(lastBandSize*12,L)];
    
    indices = find(~isnan(pmiPerRE));
    [sc,symb,~,~]= ind2sub(size(hEstP),indices);
    subs = [sc symb];
    
    pmiPerREPerfect = kron(pmip(:),ones(pmiBandSize*12,L));

    % Get all precoding matrices for nlayers and nports
    maxTPMI = hMaxPUSCHPrecodingMatrixIndicator(nlayers,nports);
    W = zeros(nports,nlayers,maxTPMI+1);
    for tpmi = 0:maxTPMI
        W(:,:,tpmi+1) = nrPUSCHCodebook(nlayers,nports,tpmi).';
    end
    
    H = permute(hEstP,[3 4 1 2]); % RxAntPorts-by-TxAntPorts-by-NumSubcarriers-by-NumOFDMSymbols
    sigma = sqrt(nVarP); % Standard deviation of noise
        
    % Get the SINR after precoding with the estimated and perfect TPMI for
    % each RE using the perfect channel estimate
    for s = 1:size(subs,1)
        scNo = subs(s,1);
        symbNo = subs(s,2);
        pmiNo = pmiPerRE(scNo,symbNo);

        sinr(scNo,symbNo) = hPrecodedSINR(H(:,:,scNo,symbNo),sigma,W(:,:,pmiNo+1));
        
        pmiNoPerfect = pmiPerREPerfect(scNo,symbNo);
        sinrPerfect(scNo,symbNo) = hPrecodedSINR(H(:,:,scNo,symbNo),sigma,W(:,:,pmiNoPerfect+1));
    end
    
    % Calculate SINR per subband
    sinrBands = hSINRPerSubband(sinr, pmiBandSize);
    sinrBandsPerfect = hSINRPerSubband(sinrPerfect, pmiBandSize);
    
    % Performance loss
    loss = sinrBandsPerfect./sinrBands;
    loss = 10*log10(loss);  
end
