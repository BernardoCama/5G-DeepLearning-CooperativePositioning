function [csrs,bsrs] = hSRSBandwidthConfiguration(srs, NRB)
% [CSRS,BSRS] = hBandwidthConfiguration(SRS, NRB) returns the SRS
% bandwidth configuration parameters CSRS and BSRS required to transmit an
% SRS in a bandwidth specified by NRB. The function calculates CSRS and
% BSRS considering the SRS properties FrequencyStart and NRRC, so the
% available bandwidth NRB is reduced by the frequency origin of the SRS.
% For frequency hopping cases, the value of BSRS returned is equal to that
% of the BSRS property in the input SRS configuration object.

    f0 = hSRSFrequencyOrigin(srs);
    NRB = NRB - f0;
    if NRB < 4
        error('The available bandwidth is not sufficient to allocate an SRS transmission. Increase the carrier bandwidth or configure the SRS in a lower frequency.' )    
    end
    
    % For frequency hopping configurations
    if srs.BHop >= srs.BSRS && nargout == 2 % No frequency hopping and BSRS is requested
        BSRS = 0:3;
    else
        BSRS = 0;
    end
    
    % Initialize gap between SRS frequency allocation and carrier bandwidth
    NRBGap = NRB; 
    
    % Find the appropriate CSRS for each BSRS that minimizes the gap only
    % in non-hopping cases. For freq. hopping, find the value of CSRS only.
    for b = BSRS
        % NRB allocated to the SRS for BSRS = b and all CSRS
        srsNRBb = srs.BandwidthConfigurationTable{:,2*(b+1)};
        mSRSbMax = max( srsNRBb( srsNRBb <= NRB ));
        
        % Calculate gap between SRS allocation and carrier bandwidth
        gap = NRB - mSRSbMax;
        if gap < NRBGap
            csrs = srs.BandwidthConfigurationTable{ srsNRBb == mSRSbMax ,1};
            bsrs = b;
            NRBGap = gap;
        end
    end
    csrs = csrs(1);
    
    if srs.BHop < bsrs
        bsrs = srs.BSRS;
    end




    function f0 = hSRSFrequencyOrigin(srs)
    % Calculate the frequency origin of the first SRS symbol in a slot
    
        NSBTable = hSRSNumberOfSubbandsOrHoppingPatterns(srs);
        NRBt = srs.NRBPerTransmission;
        
        % Origin of the SRS in frequency in RB
        f0 = srs.FrequencyStart + NRBt*mod(floor(4*srs.NRRC/NRBt),NSBTable);
    end
    
    function [NRRC,NRB] = hNRRCSet(srs)
    % Calculate the values of NRRC that generate a unique set of orthogonal SRS in frequency
        if srs.BHop < srs.BSRS % Frequency-hopping cases
            NRB = srs(1).NRB;  % Hopping bandwidth
        else 
            NRB = nrSRSConfig.BandwidthConfigurationTable{srs(1).CSRS+1,2}; 
        end
        
        % Number of frequency-hopping patterns or SRS subbands depending on the values of BSRS and BHop
        N = hSRSNumberOfSubbandsOrHoppingPatterns(srs);
        
        NRRC = NRB/4*(0:N-1)/N; 
    end


    function out = hSRSNumberOfSubbandsOrHoppingPatterns(srs)
    % N = hSRSNumberOfSubbandsOrHoppingPatterns(SRS) returns the number of
    % unique SRS subbands or frequency-hopping patterns N configurable by the
    % SRS property NRRC. An SRS subband is the frequency band allocated for SRS
    % in a given OFDM symbol (See SRS property NRBPerTransmission). N is
    % calculated using TS 38.211 Table 6.4.1.4.3-1 for the bandwidth
    % configuration parameters CSRS, BSRS, and BHop specified in the SRS
    % configuration object SRS.
    
        bwct = nrSRSConfig.BandwidthConfigurationTable{:,:};
        if srs.BHop < srs.BSRS % Number of unique frequency-hopping patterns
            b0 = srs.BHop+1;
        else % Number of unique SRS subbands
            b0 = 0;
        end
        out = prod(bwct(srs.CSRS+1,2*(b0:srs.BSRS)+3));
    end


end