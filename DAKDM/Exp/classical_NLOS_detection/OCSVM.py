import numpy as np
import os
import sys
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")
import torch
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_fscore_support as prf, accuracy_score
from sklearn.svm import OneClassSVM
from collections import defaultdict
def nested_dict():
    return defaultdict(nested_dict)

import matplotlib.pyplot as plt
plt.rcParams['image.cmap'] = 'gray'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams.update({'font.size': 22})

# Reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Directories
CLASSES_DIR = os.path.split(os.path.split(os.path.split(os.path.abspath(__file__))[0])[0])[0] + '/Classes'
DIR_EXPERIMENT = os.path.dirname(os.path.abspath(__file__))
cwd = DIR_EXPERIMENT
notebook_DIR = CLASSES_DIR + '/..'
name_EXP = DIR_EXPERIMENT.split('/')[-1]
# print(os.path.abspath(__file__))
# print(os.path.split(os.path.split(os.path.abspath(__file__))[0])[0])
# print(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(CLASSES_DIR))
sys.path.append(os.path.dirname(cwd))

from Classes.utils import mkdir, extract_features_CIR
from Classes.plotting import *
from Classes.dataset import Dataset


# Folders ##############################################################################################################
normal = 'LOS' 
anomaly = 'LOS' if normal == 'NLOS' else 'NLOS'
channel = 'ADCRM'
which_BS = 17   # None = all , 17
channel_element = 'abs_ADCRM' 
normalized_dataset = 1
FR = 1  # 1 or 2

if which_BS == None:
    DB_raw = notebook_DIR + f'/../../../../DB/FR{FR}_allBS_raw'
    if normalized_dataset:
        DB = notebook_DIR + f'/../../../../DB/FR{FR}_allBS_normalized'
        mkdir(DB)
    else:
        DB = DB_raw
else:
    DB_raw = notebook_DIR + f'/../../../../DB/FR{FR}_singleBS_raw/cell{which_BS}'
    if normalized_dataset:
        DB = notebook_DIR + f'/../../../../DB/FR{FR}_singleBS_normalized/cell{which_BS}'
        mkdir(DB)
    else:
        DB = DB_raw

dir_normal_dataset = DB + f'/{normal}'
dir_anomaly_dataset = DB + f'/{anomaly}'

name_training_dataset = DB + f'/train_dataset.pt'
name_validation_dataset = DB + f'/valid_dataset.pt'

name_normal_dataset = DB_raw + f'/{normal}/{channel_element}.mat'
name_anomaly_dataset = DB_raw + f'/{anomaly}/{channel_element}.mat'



# Training ##############################################################################################################
# Create directories if not exist
perform_training = 0

final_metrics = {}
try:
    final_metrics = np.load(DIR_EXPERIMENT + '/final_metrics.npy', allow_pickle=True).tolist()
except:
    pass
NUM_EPOCHS = 30
values = (range(1,11))


# Dataset ##############################################################################################################
# import dataset
load_dataset = 1  # load dataset or reprocess it
recompute_statistics = 0
save_dataset = 0       # save dataset as .pt pytorch tensordataset
new_matlab_format = 0  # mat73

batch_size = 1 # 128 train

shuffle_dataset = True

train_loader, val_loader = Dataset(DB, name_normal_dataset, name_anomaly_dataset, name_training_dataset, name_validation_dataset, load_dataset, 
normalized_dataset, recompute_statistics, save_dataset, new_matlab_format, batch_size, shuffle_dataset)

num_BATCHES = len(train_loader)
print(f'Number batches in training dataset: {num_BATCHES}')

var_to_monitor = 'energy1' 

test_name = name_EXP

test_results_dir = notebook_DIR + '/../../../..' + f'/Results_test/{test_name}'
log_path = test_results_dir + '/log_path'
model_save_path = test_results_dir + '/model_save_path'
mkdir(test_results_dir)
mkdir(log_path)
mkdir(model_save_path)

value = 'OCSVM'

# Compute feature extraction
train_samples = []
for i, (input_data, labels) in enumerate(tqdm(train_loader)):
    sample = np.squeeze(input_data.data.cpu().numpy())
    delay_peak = np.where(sample == np.max(sample))
    delay_peak_start = delay_peak[0][0] - 10
    delay_peak_end = delay_peak_start + 100
    angle_peak = delay_peak[1][0]
    delay_peak = delay_peak[0][0]

    train_samples.append(extract_features_CIR(sample[:, angle_peak], sample[delay_peak, :]))


# One-class SVM
train_samples = np.array(train_samples)
ocsvm = OneClassSVM(gamma='auto').fit(train_samples)

# Test performances
gt = []
score = []
for i, (input_data, labels) in enumerate(tqdm(val_loader)):

    sample = np.squeeze(input_data.data.cpu().numpy())
    delay_peak = np.where(sample == np.max(sample))
    delay_peak_start = delay_peak[0][0] - 10
    delay_peak_end = delay_peak_start + 100
    angle_peak = delay_peak[1][0]
    delay_peak = delay_peak[0][0]
    features = np.array(extract_features_CIR(sample[:, angle_peak], sample[delay_peak, :])).reshape([1,-1])
    score.append(ocsvm.score_samples(features))
    gt.append(int(labels.data.cpu().numpy()[0][0]))


fpr, tpr, thr = roc_curve(gt, score, pos_label=0)  
roc_auc = auc(fpr, tpr)

optimal_idx = np.argmax(tpr - fpr)
optimal_threshold = thr[(np.abs(fpr - 0.2)).argmin()] # thr[optimal_idx]

pred = (score < optimal_threshold).astype(int)
accuracy_ = accuracy_score(gt,pred)
precision_, recall_, f_score_, support = prf(gt, pred, average='binary')

aucs = []
best_fpr = []
max_acc = []
max_recall = []
max_precision = []
max_f_score = []

aucs.append(roc_auc)
best_fpr.append(fpr[optimal_idx])
max_acc.append(accuracy_)
max_recall.append(recall_)
max_precision.append(precision_)
max_f_score.append(f_score_)


final_metrics[value] = {'aucs':aucs,
                        'best_fpr':best_fpr, 
                        'max_acc':max_acc,
                        'max_recall':max_recall,
                        'max_precision':max_precision,
                        'max_f_score':max_f_score}
np.save(DIR_EXPERIMENT + '/final_metrics_OCSVM.npy', final_metrics, allow_pickle = True)

auc_ = final_metrics[value]['aucs'][-1]
p = final_metrics[value]['max_precision'][-1]
r = final_metrics[value]['max_recall'][-1]
f = final_metrics[value]['max_f_score'][-1]
fpr = final_metrics[value]['best_fpr'][-1]
a = final_metrics[value]['max_acc'][-1]
print(f'Value: {value}, AUC: {auc_}, Precision: {p}, Recall: {r}, F_score: {f}, Acc: {a}, FPR: {fpr}')


