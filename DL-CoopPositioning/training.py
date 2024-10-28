import numpy as np
import os
import warnings
warnings.filterwarnings("ignore")
import torch
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
# CLASSES_DIR = os.path.split(os.path.split(os.path.split(os.path.abspath(__file__))[0])[0])[0]
# DIR_EXPERIMENT = os.path.dirname(os.path.abspath(__file__))
# cwd = DIR_EXPERIMENT
# print(os.path.abspath(__file__))
# print(os.path.split(os.path.split(os.path.abspath(__file__))[0])[0])
# print(os.path.dirname(os.path.abspath(__file__)))
# sys.path.append(os.path.dirname(CLASSES_DIR))

from Classes.solver import Solver
from Classes.utils import mkdir
from Classes.plotting import *
from Classes.dataset import Dataset


cwd = os.getcwd()


# # Import Dataset
# # Folders
positioning_dataset = 'Positioning1'
cooperative = '' #  'Cooperative_' # '' or  'Cooperative_'
positioning = f'{cooperative}{positioning_dataset}'
type_ = 'BOTH' # 'LOS' 'NLOS' 'BOTH'
channel = 'ADCRM'
which_BS = None   # None = all , 17
channel_element = 'abs_ADCRM' 
normalized_dataset = 1
FR = 1  # 1 or 2

if which_BS == None:
    DB_raw = cwd + f'/../../../../DB/FR{FR}_allBS_raw_{positioning}/{type_}'
    if normalized_dataset:
        DB = cwd + f'/../../../../DB/FR{FR}_allBS_normalized_{positioning}'
        mkdir(DB)
    else:
        DB = DB_raw
else:
    DB_raw = cwd + f'/../../../../DB/FR{FR}_singleBS_raw_{positioning}/cell{which_BS}'
    if normalized_dataset:
        DB = cwd + f'/../../../../DB/FR{FR}_singleBS_normalized_{positioning}/cell{which_BS}'
        mkdir(DB)
    else:
        DB = DB_raw


name_input_dataset = DB + f'/dataset.mat'
name_training_dataset = DB + f'/train_dataset.pt'
name_validation_dataset = DB + f'/valid_dataset.pt'

load_dataset = 1  # load dataset or reprocess it
recompute_statistics = 0
save_dataset = 0       # save dataset as .pt pytorch tensordataset
new_matlab_format = 0  # mat73

batch_size = 32 
shuffle_dataset = True

train_loader, val_loader = Dataset(DB, name_input_dataset, which_BS, name_training_dataset, name_validation_dataset, load_dataset, 
normalized_dataset, recompute_statistics, save_dataset, new_matlab_format, batch_size, shuffle_dataset, adjust_latlon_geoid = 0, create_global_xyz=0)

num_BATCHES = len(val_loader)
print(f'Number batches in validation dataset: {num_BATCHES}')

num_BATCHES = len(train_loader)
print(f'Number batches in training dataset: {num_BATCHES}')


# # Training
# Create directories if not exist
perform_training = 1
type_model = f'Model_{positioning}'             
test_name = f'prove_{type_model}'

test_results_dir = DB = cwd + '/../../../..' + f'/Results_test/{test_name}'
log_path = test_results_dir + '/log_path'
model_save_path = test_results_dir + '/model_save_path'
mkdir(test_results_dir)
mkdir(log_path)
mkdir(model_save_path)

config = {
    'type_model': type_model,
    'lr' : 1e-4,               
    'weight_decay': 0,      
    'num_epochs' : 30,       
    
    'neurons_per_layer_LOS': [16, 32, 64, 128, 256, 128, 64, 32, 16],
    'neurons_per_layer_NLOS': [16, 32, 64, 128, 256, 128, 64, 32, 16],
    
    'kernel' : 'gaussian',               #   'epanechnikov'
    'bandwidth': 100.0, 
    
    'beta': 10**-3,    
    'num_samples': 10, 
    
    'wadv' : 0.01,  
    'wcon' : 1, 
    'wenc' : 0.01,  
    
    'latent_dim': 8,          
    'lamda_rec_err': 0.1,
    'lamda_uncertainty_pos': 1, 
    'lambda_energy' : 0.1,     # regularizer parameter on log-likelihood, exploration
    'lamda_diffenergy': 0.1,   # converge to density
    'lambda_cov_diag' : 0.005, # regularizer parameter on covariance Gaussian Mixture Model
    'gmm_k' : 5,   

    'pretrained_model' : None ,  #  None 
    'use_tensorboard' : False, # 
    'save_training_info': True, # 
    'to_print_network': False,

    'log_path' : log_path,                # 
    'model_save_path' : model_save_path,  # 
    'plt_show' : 1,

    'log_step' : int(num_BATCHES/30),                   
    'model_save_step' : num_BATCHES-1, #
}   

solver = Solver(config)
if perform_training:
    solver.train(train_loader)

solver.compute_threshold(train_loader, fpr=0.5)
roc_auc, accuracy, precision, recall, f_score, chosen_thr, positioning_error, LOS_predicted, pos_predicted = solver.test(val_loader)

