import os
import torch
from torch.autograd import Variable
import numpy as np
from math import ceil
from scipy import stats
import scipy
from scipy.stats import lognorm
from sklearn.metrics import precision_recall_fscore_support as prf, accuracy_score

def to_var(x, volatile=False):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, volatile=volatile)

def mkdir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
        
def compute_thresholds(thrs, gt, scores, var_to_monitor = 'likelihood1'):
    
    acc = []
    recall = []
    precision = []
    f_score = []

    for thr in thrs:  

        if var_to_monitor == 'likelihood1' or var_to_monitor == 'likelihood2':
            pred = (scores < thr).astype(int)

        elif var_to_monitor == 'energy1' or var_to_monitor == 'energy2':   
            pred = (scores > thr).astype(int)

        accuracy_ = accuracy_score(gt,pred)
        precision_, recall_, f_score_, support = prf(gt, pred, average='binary')
        
        acc.append(accuracy_)
        recall.append(recall_)
        precision.append(precision_)
        f_score.append(f_score_)

    return acc, recall, precision, f_score


# Function to find the maximum in column 'mid'
# 'rows' is number of rows.
def findMax(arr, rows, mid,max):
 
    max_index = 0
    for i in range(rows):
        if (max < arr[i][mid]):
             
            # Saving global maximum and its index
            # to check its neighbours
            max = arr[i][mid]
            max_index = i
 
    return max,max_index
 

# Function to find a peak element
def findPeakRec(arr, rows, columns,mid):
 
    # Evaluating maximum of mid column.
    # Note max is passed by reference.
    max = 0
    max, max_index = findMax(arr, rows, mid, max)
 
    # If we are on the first or last column,
    # max is a peak
    if (mid == 0 or mid == columns - 1):
        return max
 
    # If mid column maximum is also peak
    if (max >= arr[max_index][mid - 1] and
        max >= arr[max_index][mid + 1]):
        return max
 
    # If max is less than its left
    if (max < arr[max_index][mid - 1]):
        return findPeakRec(arr, rows, columns,
                           mid - ceil(mid / 2.0))
 
    # If max is less than its left
    # if (max < arr[max_index][mid+1])
    return findPeakRec(arr, rows, columns,
                       mid + ceil(mid / 2.0))
 
# A wrapper over findPeakRec()
def findPeak(arr, rows, columns):
    return findPeakRec(arr, rows,
                       columns, columns // 2)



def compute_avg_Pearson_coeff(test_sample, ref_samples):

    corr_list = []
    num_ref_samples = len(ref_samples)
    for ref_sample in ref_samples:
        # corr_list.append(np.dot(test_sample,ref_sample.T))
        corr_list.append(stats.pearsonr(test_sample, ref_sample)[0])
    return np.mean(corr_list)

def extract_features_CIR(CIR, angle_CIR):

    # 1
    max_rx_power_over_delay_samples = np.max(CIR)

    # 2
    kurtosis = stats.kurtosis(CIR)

    # 3
    skewness = stats.skew(CIR)

    # 4
    peaks = scipy.signal.find_peaks_cwt(CIR, [30])
    max_peak = max(CIR[peaks])
    rising_time = int(np.where(CIR == max_peak) - peaks[0])

    # 5
    mean_excess_delay = np.sqrt((np.sum(peaks*CIR[peaks]))/np.sum(peaks))
    if np.isnan(mean_excess_delay):
        root_mean_squared_delay_spread = 0
    else:
        root_mean_squared_delay_spread = np.sqrt((np.sum(((peaks-mean_excess_delay)**2)*CIR[peaks]))/np.sum(peaks))

    # 6
    rician_K_factor = max_peak/(2*np.std(CIR))

    # 7
    delay_peaks = scipy.signal.find_peaks_cwt(angle_CIR, [2])
    mean_direction = np.sqrt((np.sum(delay_peaks*angle_CIR[delay_peaks]))/np.sum(delay_peaks))
    if np.isnan(mean_direction):
        angular_spread_of_arrival = 0
    else:
        angular_spread_of_arrival = np.sqrt((np.sum(((delay_peaks-mean_direction)**2)*angle_CIR[delay_peaks]))/np.sum(delay_peaks))

    return [max_rx_power_over_delay_samples,
            kurtosis, 
            skewness,
            rising_time, 
            rician_K_factor, 
            root_mean_squared_delay_spread,
            angular_spread_of_arrival] 

# Fit a log-normal distribution for each features
def mean_var_log_normal(samples):

    samples = np.array(samples)
    num_features = samples.shape[1]
    num_samples = samples.shape[0]
    log_normal_disributions = []
    mean_std_min_max = []
    for i in range(num_features):
        params = stats.lognorm.fit(samples[:,i], loc=0)
        mean_ = np.mean(samples[:,i])
        std_ = np.std(samples[:,i])
        min_ = np.min(samples[:,i])
        max_ = np.max(samples[:,i])
        mean_std_min_max.append([mean_, std_, min_, max_])
        # log_normal_disributions.append(lognorm([std_],loc=mean_))
        # log_normal_disributions.append(lognorm([std_],scale=np.exp(mean_)))
        log_normal_disributions.append(lognorm(params[0],loc=params[1],scale=params[2]))
    return mean_std_min_max, log_normal_disributions


def vector_to_matrix(input_vector):
    input_matrix = torch.repeat_interleave(input_vector.view(-1, 1, 1), repeats=227, dim=1)
    input_matrix = torch.repeat_interleave(input_matrix, repeats=227, dim=2)
    return input_matrix.unsqueeze(1)  # add channel dimension