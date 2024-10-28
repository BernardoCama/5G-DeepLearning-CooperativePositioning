import numpy as np
import os
import sys
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")
import torch
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_fscore_support as prf, accuracy_score
from sklearn.ensemble import RandomForestClassifier


from collections import defaultdict
def nested_dict():
    return defaultdict(nested_dict)

import matplotlib.pyplot as plt
plt.rcParams['image.cmap'] = 'gray'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams.update({'font.size': 22})

# RANDOM FORESTS

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

from Classes.utils import mkdir, extract_features_CIR, mean_var_log_normal
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
perform_training = 1

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

value = 'RF'
num_samples_training = 100
num_samples_testing = 3124-num_samples_training

# Compute feature extraction
train_LOS_samples = []
train_NLOS_samples = []
train_samples = []
train_output = []
for i, (input_data, labels) in enumerate(tqdm(val_loader)):
    if i<num_samples_training:
        sample = np.squeeze(input_data.data.cpu().numpy())
        delay_peak = np.where(sample == np.max(sample))
        delay_peak_start = delay_peak[0][0] - 10
        delay_peak_end = delay_peak_start + 100
        angle_peak = delay_peak[1][0]
        delay_peak = delay_peak[0][0]

        if labels == 1:
            train_NLOS_samples.append(extract_features_CIR(sample[:, angle_peak], sample[delay_peak, :]))
        elif labels == 0:
            train_LOS_samples.append(extract_features_CIR(sample[:, angle_peak], sample[delay_peak, :]))

        # both classes
        train_samples.append(extract_features_CIR(sample[:, angle_peak], sample[delay_peak, :]))
        train_output.append(labels.item())

mean_std_min_max_NLOS, pdfs_NLOS = mean_var_log_normal(train_NLOS_samples)
mean_std_min_max_LOS, pdfs_LOS = mean_var_log_normal(train_LOS_samples)

# RF
train_samples = np.array(train_samples)
train_output = np.array(train_output)
rf = RandomForestClassifier(n_estimators = 50)
rf.fit(train_samples, train_output)

# Export the first three decision trees from the forest
tree_idx = 0  # you can change this to select another tree
decision_tree = rf.estimators_[tree_idx]

# Test performances
gt = []
score = []
for i, (input_data, labels) in enumerate(tqdm(val_loader)):

    if i>=num_samples_training:
        sample = np.squeeze(input_data.data.cpu().numpy())
        delay_peak = np.where(sample == np.max(sample))
        delay_peak_start = delay_peak[0][0] - 10
        delay_peak_end = delay_peak_start + 100
        angle_peak = delay_peak[1][0]
        delay_peak = delay_peak[0][0]
        features = np.array(extract_features_CIR(sample[:, angle_peak], sample[delay_peak, :])).reshape([1,-1])
        score.append(rf.predict_proba(features)[0,1]) # take only positve class probability
        gt.append(int(labels.data.cpu().numpy()[0][0]))

score = np.array(score)
fpr, tpr, thr = roc_curve(gt, score, pos_label=1)  
roc_auc = auc(fpr, tpr)

optimal_idx = np.argmax(tpr - fpr)
optimal_threshold = thr[(np.abs(fpr - 0.2)).argmin()] # thr[optimal_idx]

pred = (score > optimal_threshold).astype(int)
accuracy_ = accuracy_score(gt,pred)
precision_, recall_, f_score_, support = prf(gt, pred, average='binary')

aucs = []
best_fpr = []
max_acc = []
max_recall = []
max_precision = []
max_f_score = []

aucs.append(roc_auc)
# best_fpr.append(fpr[optimal_idx])
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
np.save(DIR_EXPERIMENT + '/final_metrics_RF.npy', final_metrics, allow_pickle = True)


# print(f'Results\n')     
auc_ = final_metrics[value]['aucs'][-1]
p = final_metrics[value]['max_precision'][-1]
r = final_metrics[value]['max_recall'][-1]
f = final_metrics[value]['max_f_score'][-1]
# fpr = final_metrics[value]['best_fpr'][-1]
fpr = final_metrics[value]['best_fpr']
a = final_metrics[value]['max_acc'][-1]
print(f'Value: {value}, AUC: {auc_}, Precision: {p}, Recall: {r}, F_score: {f}, Acc: {a}, FPR: {fpr}')

