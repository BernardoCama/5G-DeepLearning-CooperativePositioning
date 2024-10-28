import os
import torch
from torch.autograd import Variable
from math import ceil
import math
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


def findMax(arr, rows, mid,max):
 
    max_index = 0
    for i in range(rows):
        if (max < arr[i][mid]):
             
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


def coord2xyz(lat, lon, h):
    a = 6378137.0  # meters
    b = 6356752.314245  # meters

    f = (a - b) / a  # flattening
    lambda_ = math.radians(lat)
    e2 = f * (2 - f)  # Square of eccentricity
    phi = math.radians(lon)
    sin_lambda = math.sin(lambda_)
    cos_lambda = math.cos(lambda_)
    cos_phi = math.cos(phi)
    sin_phi = math.sin(phi)

    N = a / math.sqrt(1 - e2 * sin_lambda * sin_lambda)

    x = (h + N) * cos_lambda * cos_phi
    y = (h + N) * cos_lambda * sin_phi
    z = (h + (1 - e2) * N) * sin_lambda
    r = [x, y, z]

    return r

def xyz2coord(x, y, z):
    a = 6378137.0  # meters
    b = 6356752.314245  # meters

    f = (a - b) / a  # Flattening
    e2 = f * (2 - f)  # Square of eccentricity

    eps = e2 / (1.0 - e2)

    p = math.sqrt(x * x + y * y)
    q = math.atan2((z * a), (p * b))

    sin_q = math.sin(q)
    cos_q = math.cos(q)

    sin_q_3 = sin_q * sin_q * sin_q
    cos_q_3 = cos_q * cos_q * cos_q

    phi = math.atan2((z + eps * b * sin_q_3), (p - e2 * a * cos_q_3))
    lam = math.atan2(y, x)

    v = a / math.sqrt(1.0 - e2 * math.sin(phi) * math.sin(phi))
    h = (p / math.cos(phi)) - v

    lat = math.degrees(phi)
    lon = math.degrees(lam)

    pos = [lat, lon, h]

    return pos
