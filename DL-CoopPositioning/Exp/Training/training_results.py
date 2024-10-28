## Import Packages
import numpy as np
import os
import sys
from scipy.signal import medfilt
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # -1 to not use GPU
import matplotlib.pyplot as plt

CLASSES_DIR = os.path.split(os.path.split(os.path.split(os.path.split(os.path.abspath(__file__))[0])[0])[0])[0] + '/Classes'
DIR_EXPERIMENT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(CLASSES_DIR))
sys.path.append(os.path.dirname(DIR_EXPERIMENT))

from Classes.plotting import *
from Classes.train_validate_fun import *
plt.rcParams.update({'font.size': 22})

EPOCHS = 120
ris_LOS = np.load(DIR_EXPERIMENT + '/LOS/log_loss.npy', allow_pickle=True).tolist()
ris_NLOS = np.load(DIR_EXPERIMENT + '/NLOS/log_loss.npy', allow_pickle=True).tolist()
ris_BOTH = np.load(DIR_EXPERIMENT + '/BOTH/log_loss.npy', allow_pickle=True).tolist()


batches_per_epoch = len(ris_LOS['recon_error'])/EPOCHS
config = {'num_epochs': EPOCHS, 'log_step': 1}
plot_training_metrics ({'num_epochs': EPOCHS, 'log_step': 1}, log_path = DIR_EXPERIMENT, batches_per_epoch = batches_per_epoch, file_name='ego_training_results', save_eps = 1)
save_pdf = 1
log_path = DIR_EXPERIMENT
file_name='ego_training_results'
save_eps = 1
plt_show = 0


save_pdf = 1
log_path = DIR_EXPERIMENT
file_name='ego_classification_results'
save_eps = 1
plt_show = 0
fig = plt.figure(figsize=(10, 10))
for (k_LOS,v_LOS),(k_NLOS,v_NLOS), (k_BOTH,v_BOTH) in zip(ris_LOS.items(), ris_NLOS.items(),  ris_BOTH.items()):
    if k_BOTH == 'accuracy' or  k_BOTH == 'precision' or  k_BOTH == 'recall':
        plt.plot(np.arange(0,60, 1/30), np.array(v_BOTH)[:len(np.arange(0,60, 1/30))], lw=2, linestyle='-') # color='lightblue'


plt.xlim([0, 60])
plt.ylim([0.5, 1])
plt.legend(['Accuracy', 'Precision', 'Recall'], shadow=False)
if save_pdf:
    plt.savefig(log_path + f'/{file_name}_{k_LOS}.pdf', bbox_inches='tight')
if save_eps:
    plt.savefig(log_path + f'/{file_name}_{k_LOS}.eps', format='eps', bbox_inches='tight')
if plt_show:
    plt.show() 