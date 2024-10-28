import numpy as np
import os
import sys
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, accuracy_score, roc_curve
from collections import defaultdict
def nested_dict():
    return defaultdict(nested_dict)

import matplotlib.pyplot as plt
plt.rcParams['image.cmap'] = 'gray'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams.update({'font.size': 22})

# CNN 

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
from Classes.model import CNN_AlexNet


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

value = '1DCNN_AlexNet'
num_samples_training = 2000
num_samples_testing = 3124-num_samples_training

# Compute feature extraction
train_LOS_samples = []
train_NLOS_samples = []
train_samples = []
train_output = []
valid_samples = []
valid_output = []
for i, (input_data, labels) in enumerate(tqdm(val_loader)):

    sample = np.squeeze(input_data.data.cpu().numpy())
    # plot_reconstructed_samples (input_=sample, sample_number=0, label=0, log_path='', save_pdf = 0, save_eps = 0, file_name = '')
    # delay_peak = np.where(sample == findPeak(sample, sample.shape[0], sample.shape[1]))
    delay_peak = np.where(sample == np.max(sample))
    delay_peak_start = delay_peak[0][0] - 10
    delay_peak_end = delay_peak_start + 100
    angle_peak = delay_peak[1][0]
    delay_peak = delay_peak[0][0]

    if i<num_samples_training:

        if labels == 1:
            train_NLOS_samples.append(extract_features_CIR(sample[:, angle_peak], sample[delay_peak, :]))
        elif labels == 0:
            train_LOS_samples.append(extract_features_CIR(sample[:, angle_peak], sample[delay_peak, :]))

        # both classes
        train_samples.append(extract_features_CIR(sample[:, angle_peak], sample[delay_peak, :]))
        train_output.append(labels.item())
    else:
        # both classes
        valid_samples.append(extract_features_CIR(sample[:, angle_peak], sample[delay_peak, :]))
        valid_output.append(labels.item())      

mean_std_min_max_NLOS, pdfs_NLOS = mean_var_log_normal(train_NLOS_samples)
mean_std_min_max_LOS, pdfs_LOS = mean_var_log_normal(train_LOS_samples)

# CNN
train_samples = np.array(train_samples)
train_output = np.array(train_output)
valid_samples = np.array(valid_samples)
valid_output = np.array(valid_output)

train_dataset = TensorDataset(torch.tensor(train_samples).float(), torch.tensor(np.zeros((train_output.shape[0], 1))).float())
valid_dataset = TensorDataset(torch.tensor(valid_samples).float(), torch.tensor(valid_output).float()) # assign labels to validation dataset
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=shuffle_dataset, drop_last=True)
val_loader = DataLoader(valid_dataset, batch_size=32, shuffle=shuffle_dataset, drop_last=True)


# Assuming you have train_dataloader and valid_dataloader already created
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Set the model, loss function, and optimizer
model = CNN_AlexNet(num_classes=2).to(device)
# criterion = nn.CrossEntropyLoss()
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

for epoch in range(NUM_EPOCHS):
    model.train()
    for samples, labels in train_loader:
        samples, labels = samples.to(device), labels.to(device)

        # Forward pass
        outputs = model((samples).view(32, 1, -1))
        # loss = criterion(outputs, labels.squeeze().long())
        loss = criterion(outputs, labels)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        # Validation
        all_outputs = []
        all_labels = []

        for samples, labels in val_loader:
            samples, labels = samples.to(device), labels.to(device)
            outputs = model((samples).view(32, 1, -1))

            all_outputs.extend(outputs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        # Compute AUC
        # all_outputs = torch.tensor(all_outputs)[:, 1].numpy()
        all_outputs = torch.tensor(all_outputs).numpy()
        all_labels = torch.tensor(all_labels)
        auc = roc_auc_score(all_labels, all_outputs)
        print(f"Epoch {epoch + 1}/{NUM_EPOCHS}: AUC = {auc:.4f}")

        # Find threshold with FPR equal to TPR
        fpr, tpr, thresholds = roc_curve(all_labels, all_outputs)
        threshold = thresholds[(fpr - tpr).argmin()]
        # print(f"Threshold = {threshold:.4f}")

        # Compute Precision, Recall, F-score, and Accuracy
        predictions = (all_outputs > threshold).astype(int)
        precision, recall, f_score, _ = precision_recall_fscore_support(all_labels, predictions, average='binary')
        accuracy = accuracy_score(all_labels, predictions)

        print(f"Precision = {precision:.4f}, Recall = {recall:.4f}, F-score = {f_score:.4f}, Accuracy = {accuracy:.4f}")



