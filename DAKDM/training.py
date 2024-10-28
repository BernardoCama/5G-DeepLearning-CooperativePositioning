import numpy as np
import os
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")
import torch
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_fscore_support as prf, accuracy_score
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KernelDensity
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
from Classes.utils import mkdir, compute_thresholds
from Classes.plotting import *
from Classes.dataset import Dataset

cwd = os.getcwd()

# # NORMAL label 0
# # ANOMALY label 1

# # Import Dataset
# Folders
normal = 'LOS' 
anomaly = 'LOS' if normal == 'NLOS' else 'NLOS'
channel = 'ADCRM'
which_BS = 17   # None = all , 17
channel_element = 'abs_ADCRM' 
normalized_dataset = 1
FR = 2  # 1 or 2

if which_BS == None:
    DB_raw = cwd + f'/../../../../DB/FR{FR}_allBS_raw'
    if normalized_dataset:
        DB = cwd + f'/../../../../DB/FR{FR}_allBS_normalized'
        mkdir(DB)
    else:
        DB = DB_raw
else:
    DB_raw = cwd + f'/../../../../DB/FR{FR}_singleBS_raw/cell{which_BS}'
    if normalized_dataset:
        DB = cwd + f'/../../../../DB/FR{FR}_singleBS_normalized/cell{which_BS}'
        mkdir(DB)
    else:
        DB = DB_raw

dir_normal_dataset = DB + f'/{normal}'
dir_anomaly_dataset = DB + f'/{anomaly}'

name_training_dataset = DB + f'/train_dataset.pt'
name_validation_dataset = DB + f'/valid_dataset.pt'

name_normal_dataset = DB_raw + f'/{normal}/{channel_element}.mat'
name_anomaly_dataset = DB_raw + f'/{anomaly}/{channel_element}.mat'

load_dataset = 1  # load dataset or reprocess it
recompute_statistics = 0
save_dataset = 0       # save dataset as .pt pytorch tensordataset
new_matlab_format = 0  # mat73

batch_size = 64
shuffle_dataset = True

train_loader, val_loader = Dataset(DB, name_normal_dataset, name_anomaly_dataset, name_training_dataset, name_validation_dataset, load_dataset, 
normalized_dataset, recompute_statistics, save_dataset, new_matlab_format, batch_size, shuffle_dataset)

num_BATCHES = len(val_loader)
print(f'Number batches in validation dataset: {num_BATCHES}')

num_BATCHES = len(train_loader)
print(f'Number batches in training dataset: {num_BATCHES}')


# # Training
# Create directories if not exist
perform_training = 1
type_model = 'GANomaly'          # 'DaGMM', 'Model1', 'Model2', 'Model3', 'BetaVAE_H', 'GANomaly', 'AEGMM', 'AEKDE'
var_to_monitor = 'energy2'  # 'likelihood1' 'energy1' 'energy2'
test_name = 'test_GANomaly' # prove_model3_correct13

test_results_dir = DB = cwd + '/../../../..' + f'/Results_test/{test_name}'
log_path = test_results_dir + '/log_path'
model_save_path = test_results_dir + '/model_save_path'
mkdir(test_results_dir)
mkdir(log_path)
mkdir(model_save_path)


config = {
    'type_model': type_model,
    'var_to_monitor':var_to_monitor,
    'lr' : 1e-4,               
    'weight_decay': 0,      
    'num_epochs' : 30,        
    
    'kernel' : 'gaussian',               #   'epanechnikov'
    'bandwidth': 100.0, 
    
    'beta': 10**-3,    # BetaVAE_H
    'num_samples': 10, # BetaVAE_H 
    
    'wadv' : 0.01,  # GANomaly
    'wcon' : 1, # GANomaly
    'wenc' : 0.01,  # GANomaly
    
    'latent_dim': 8,           # 
    'lamda_rec_err': 1,
    'lambda_energy' : 0.1,     # regularizer parameter on log-likelihood, exploration
    'lamda_diffenergy': 0.1,   # converge to density
    'lambda_cov_diag' : 0.005, # regularizer parameter on covariance Gaussian Mixture Model
    'gmm_k' : 5,   # KDEGMM

    'pretrained_model' : None ,  #  None 
    'use_tensorboard' : False, # 
    'save_training_info': True, # 
    'to_print_network': False,

    'log_path' : log_path,                # 
    'model_save_path' : model_save_path,  # 
    'plt_show' : 1,

    'log_step' : int(num_BATCHES/1),                   
    'model_save_step' : num_BATCHES-1, 
}  

solver = Solver(config)
if perform_training:
    solver.train(train_loader)

plot_training_metrics(config, log_path, batches_per_epoch = num_BATCHES, file_name='', save_eps = 0)


solver.model_save_path = model_save_path
torch.save(solver.model.state_dict(), os.path.join(solver.model_save_path, '{}_{}_model.pth'.format(180, 1)))


# # Number Epochs
epochs_to_test = np.arange(1, 31, 1)
dir_energy = log_path + '/energy_per_epoch'

aucs = []
best_fpr = []
max_acc = []
max_recall = []
max_precision = []
max_f_score = []
for e in epochs_to_test:
    
    # Load results
    ris = np.load(dir_energy + f'/_epoch_{e}.npy', allow_pickle = True).tolist()
    solver.train_z1= ris['train_z1'] 
    solver.train_z2= ris['train_z2']
    solver.train_likelihood1= ris['train_likelihood1'] 
    solver.train_energy1= ris['train_energy1']
    solver.train_energy2=ris['train_energy2'] 
    solver.test_z1= ris['test_z1']
    solver.test_z2= ris['test_z2']
    solver.test_likelihood1= ris['test_likelihood1']
    solver.test_energy1= ris['test_energy1']
    solver.test_energy2= ris['test_energy2']
    solver.test_labels= ris['test_labels']

    gt = solver.test_labels.astype(int)
    if config['var_to_monitor'] == 'likelihood1':
        score = solver.test_likelihood1        
    elif config['var_to_monitor'] == 'energy1':   
        score = solver.test_energy1
    elif config['var_to_monitor'] == 'energy2':   
        score = solver.test_energy2

    fpr, tpr, thr = roc_curve(gt, score, pos_label=0) 
    roc_auc = auc(fpr, tpr)
    
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thr[(np.abs(fpr - 0.2)).argmin()]

    if config['var_to_monitor'] == 'energy2' or config['var_to_monitor'] == 'energy1': #likelihood
        pred = (score < optimal_threshold).astype(int)

    elif config['var_to_monitor'] == 'likelihood1':   
        pred = (score > optimal_threshold).astype(int)
    accuracy_ = accuracy_score(gt,pred)
    precision_, recall_, f_score_, support = prf(gt, pred, average='binary')
    
    aucs.append(roc_auc)
    best_fpr.append(fpr[optimal_idx])
    max_acc.append(accuracy_)
    max_recall.append(recall_)
    max_precision.append(precision_)
    max_f_score.append(f_score_)

plot_roc_auc_acc(config, aucs, best_fpr, max_acc, max_recall, max_precision, max_f_score, epochs_to_test, sw = 1,
                 title='AUC vs Epochs components',  file_name = 'AUC_vs_epochs', dir = log_path)

auc_ = aucs[-2]
p = max_precision[-2]
r = max_recall[-2]
f = max_f_score[-2]
fpr = best_fpr[-2]
a = max_acc[-2]
print(f'AUC: {auc_}, Precision: {p}, Recall: {r}, F_score: {f}, Acc: {a}, FPR: {fpr}')


# # Validation single case
solver.compute_threshold(train_loader, fpr=0.1)
fpr = 0.2

if config['var_to_monitor'] == 'likelihood1':
    solver.thresh = np.percentile(solver.train_likelihood1, fpr*100)
elif config['var_to_monitor'] == 'energy1':   
    solver.thresh = np.percentile(solver.train_energy1, (1-fpr)*100)
elif config['var_to_monitor'] == 'likelihood2':
    solver.thresh = np.percentile(solver.train_likelihood2, fpr*100)
elif config['var_to_monitor'] == 'energy2':   
    solver.thresh = np.percentile(solver.train_energy2, (1-fpr)*100)
accuracy, precision, recall, f_score = solver.test(val_loader)


# # Visualize likelihood
if config['var_to_monitor'] == 'likelihood1':
    len(solver.train_likelihood1)
    len(np.unique(solver.train_likelihood1))
    print(min(solver.train_likelihood1), max(solver.train_likelihood1))

    len(solver.test_likelihood1)
    len(np.unique(solver.test_likelihood1))
    print(min(solver.test_likelihood1),max(solver.test_likelihood1))
elif config['var_to_monitor'] == 'likelihood2':
    len(solver.train_likelihood2)
    len(np.unique(solver.train_likelihood2))
    print(min(solver.train_likelihood2), max(solver.train_likelihood2))

    len(solver.test_likelihood2)
    len(np.unique(solver.test_likelihood2))
    print(min(solver.test_likelihood2),max(solver.test_likelihood2))
elif config['var_to_monitor'] == 'energy1':   
    len(solver.train_energy1)
    len(np.unique(solver.train_energy1))
    print(min(solver.train_energy1), max(solver.train_energy1))

    len(solver.test_energy1)
    len(np.unique(solver.test_energy1))
    print(min(solver.test_energy1),max(solver.test_energy1))
elif config['var_to_monitor'] == 'energy2':   
    len(solver.train_energy2)
    len(np.unique(solver.train_energy2))
    print(min(solver.train_energy2), max(solver.train_energy2))

    len(solver.test_energy2)
    len(np.unique(solver.test_energy2))
    print(min(solver.test_energy2),max(solver.test_energy2))


dir_energy = log_path + '/energy_per_epoch'       
e = 30
ris = np.load(dir_energy + f'/_epoch_{e}.npy', allow_pickle = True).tolist()
solver.train_z1= ris['train_z1'] 
solver.train_z2= ris['train_z2']
solver.train_likelihood1= ris['train_likelihood1'] 
solver.train_energy1= ris['train_energy1']
solver.train_energy2=ris['train_energy2'] 
solver.test_z1= ris['test_z1']
solver.test_z2= ris['test_z2']
solver.test_likelihood1= ris['test_likelihood1']
solver.test_energy1= ris['test_energy1']
solver.test_energy2= ris['test_energy2']
solver.test_labels= ris['test_labels']

try:
    if config['var_to_monitor'] == 'likelihood1':
        plot_likelihood(train_score=solver.train_likelihood1.ravel(), test_score=solver.test_likelihood1.ravel(), test_labels=solver.test_labels.ravel(), 
                    log_path=dir_energy, file_name=f'/epoch_{e}', prove = 0,
                    ax = None, save_pdf = 1, plt_show = 0, var_to_monitor='likelihood1')

    elif config['var_to_monitor'] == 'energy1':
        plot_likelihood(train_score=solver.train_energy1.ravel(), test_score=solver.test_energy1.ravel(), test_labels=solver.test_labels.ravel(), 
                    log_path=dir_energy, file_name=f'/epoch_{e}', prove = 0,
                     ax = None, save_pdf = 1, plt_show = 0, var_to_monitor='energy1')
    elif config['var_to_monitor'] == 'likelihood2':
        plot_likelihood(train_score=solver.train_likelihood2.ravel(), test_score=solver.test_likelihood2.ravel(), test_labels=solver.test_labels.ravel(), 
                    log_path=dir_energy, file_name=f'/epoch_{e}', prove = 0,
                    ax = None, save_pdf = 1, plt_show = 0, var_to_monitor='likelihood2')

    elif config['var_to_monitor'] == 'energy2':
        plot_likelihood(train_score=solver.train_energy2.ravel(), test_score=solver.test_energy2.ravel(), test_labels=solver.test_labels.ravel(), 
                    log_path=dir_energy, file_name=f'/epoch_{e}', prove = 0,
                    ax = None, save_pdf = 1, plt_show = 0, var_to_monitor='energy2')
except:
    pass

