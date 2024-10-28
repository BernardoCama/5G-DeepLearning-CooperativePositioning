import numpy as np
import os
import sys
import warnings
warnings.filterwarnings("ignore")
import torch
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_fscore_support as prf, accuracy_score
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

from Classes.solver import Solver
from Classes.utils import mkdir
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


type_model = 'AEGMM'          # 'DaGMM', 'Model1', 'Model2', 'Model3', 'BetaVAE_H', 'GANomaly'

# Dataset ##############################################################################################################
# import dataset
load_dataset = 1  # load dataset or reprocess it
recompute_statistics = 0
save_dataset = 0       # save dataset as .pt pytorch tensordataset
new_matlab_format = 0  # mat73

batch_size = 64 

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


# # Models and Parameters ##############################################################################################################
config = {
    'type_model': type_model,
    'var_to_monitor':var_to_monitor,
    'lr' : 1e-4,               
    'weight_decay': 0,      
    'num_epochs' : NUM_EPOCHS,        
    
    'gmm_k' : 5,
    
    'kernel' : 'gaussian',               #   'epanechnikov'
    'bandwidth': 0.1, 
    
    'beta': 10**-3,    # BetaVAE_H
    'num_samples': 10, # BetaVAE_H 
    
    'wadv' : 0.1,  # GANomaly
    'wcon' : 1, # GANomaly
    'wenc' : 0.1,  # GANomaly
    
    'latent_dim': 8,           #   real used
    'lamda_rec_err': 1,
    'lambda_energy' : 0.1,     #  regularizer parameter on log-likelihood, exploration
    'lamda_diffenergy': 0.1,   # converge to density
    'lambda_cov_diag' : 0.005, #  regularizer parameter on covariance Gaussian Mixture Model

    'pretrained_model' : None ,  #  None 
    'use_tensorboard' : False, # 
    'save_training_info': True, # 
    'to_print_network': False,

    'log_path' : log_path,                # 
    'model_save_path' : model_save_path,  # 
    'plt_show' : 0,

    'log_step' : int(num_BATCHES/1),                  
    'model_save_step' : num_BATCHES, 
}  

solver = Solver(config)

epochs_to_test = np.arange(1, NUM_EPOCHS+1, 1)
if perform_training:
    solver.train(train_loader)

    print(f'Training complete\n')

    for value in values:
 
        print(f'Value: {value}')
        config['gmm_k'] = value
    
        prove = 0
        log_scale = False
        bins = 10
        solver_class = Solver
        warnings.filterwarnings("default")
        energy_per_epoch(config, epochs_to_test, train_loader, \
            val_loader, solver_class, log_path = DIR_EXPERIMENT, model_save_path = model_save_path, \
                model_save_step = config['model_save_step'], \
                    prove = prove, log_scale = log_scale, bins=bins, \
                        threshold = None, remove_outliers = None, \
                            xlim = None, save_eps = 1, file_name=f'{value}', plt_show=0)

for value in values:
    
    print(f'Value: {value}')
    config['gmm_k'] = value
                
    dir_energy = DIR_EXPERIMENT + '/energy_per_epoch'

    aucs = []
    best_fpr = []
    max_acc = []
    max_recall = []
    max_precision = []
    max_f_score = []
    for e in epochs_to_test:
        
        # Load results
        ris = np.load(dir_energy + f'/{value}_epoch_{e}.npy', allow_pickle = True).tolist()
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

        if type_model == 'AEKDE' or type_model == 'AEGMM':
            fpr, tpr, thr = roc_curve(gt, score, pos_label=1)  
        else:
            fpr, tpr, thr = roc_curve(gt, score, pos_label=0)  
        roc_auc = auc(fpr, tpr)

        
        optimal_idx = np.argmax(tpr - fpr)
        optimal_threshold = thr[optimal_idx]

        if type_model == 'AEKDE' or type_model == 'AEGMM':
            if config['var_to_monitor'] == 'energy1' or config['var_to_monitor'] == 'energy2': #likelihood
                pred = (score > optimal_threshold).astype(int)

            elif config['var_to_monitor'] == 'likelihood1' or config['var_to_monitor'] == 'likelihood2':   
                pred = (score < optimal_threshold).astype(int)
            accuracy_ = accuracy_score(gt,pred)
            precision_, recall_, f_score_, support = prf(gt, pred, average='binary')
        
        else:
            if config['var_to_monitor'] == 'energy1' or config['var_to_monitor'] == 'energy2': #likelihood
                pred = (score < optimal_threshold).astype(int)

            elif config['var_to_monitor'] == 'likelihood1' or config['var_to_monitor'] == 'likelihood2':   
                pred = (score > optimal_threshold).astype(int)
            accuracy_ = accuracy_score(gt,pred)
            precision_, recall_, f_score_, support = prf(gt, pred, average='binary')

        
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
    np.save(DIR_EXPERIMENT + '/final_metrics.npy', final_metrics, allow_pickle = True)

    plot_roc_auc_acc(config, aucs, best_fpr, max_acc, max_recall, max_precision, max_f_score, epochs_to_test, sw = 1,
                    title='AUC vs Epochs components',  file_name = f'{value}_AUC_vs_epochs', dir = DIR_EXPERIMENT)

    auc_ = final_metrics[value]['aucs'][-1]
    p = final_metrics[value]['max_precision'][-1]
    r = final_metrics[value]['max_recall'][-1]
    f = final_metrics[value]['max_f_score'][-1]
    fpr = final_metrics[value]['best_fpr'][-1]
    print(f'Value: {value}, AUC: {auc_}, Precision: {p}, Recall: {r}, F_score: {f}, FPR: {fpr}')


