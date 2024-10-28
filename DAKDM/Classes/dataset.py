import torch
import numpy as np
import warnings
warnings.filterwarnings("ignore")
import mat73
from torch.utils.data import TensorDataset, DataLoader
import scipy.io

def Dataset(DB, name_normal_dataset, name_anomaly_dataset, name_training_dataset, name_validation_dataset, load_dataset, 
normalized_dataset, recompute_statistics, save_dataset, new_matlab_format, batch_size, shuffle_dataset):
        
    if not load_dataset:
        
        # import datasets
        if new_matlab_format:
            raw_normal = mat73.loadmat(name_normal_dataset)['dataset']
        else: 
            raw_normal = scipy.io.loadmat(name_normal_dataset)['dataset']
        Ng = raw_normal.shape[0]
        number_OFDM_symbols = raw_normal.shape[1]
        RXants = raw_normal.shape[2]

        # raw_normal = raw_normal.reshape((-1,1,Ng,RXants))
        raw_normal = np.rollaxis(raw_normal, 1, 0)
        raw_normal = raw_normal.reshape((number_OFDM_symbols, 1, Ng, RXants))
        print(f'normal dataset shape: {raw_normal.shape}')
        np.random.shuffle(raw_normal)

        train_dataset = raw_normal[:int(0.7*number_OFDM_symbols),:,:,:]
        raw_normal = raw_normal[int(0.7*number_OFDM_symbols):,:,:,:]
        
        # raw_anomaly = N_samples (number_OFDM_symbols) x channels (1) x Ng x RXants
        if new_matlab_format:
            raw_anomaly = mat73.loadmat(name_anomaly_dataset)['dataset']
        else:
            raw_anomaly = scipy.io.loadmat(name_anomaly_dataset)['dataset']

        number_OFDM_symbols = raw_anomaly.shape[1]
        # raw_normal = raw_normal.reshape((-1,1,Ng,RXants))
        raw_anomaly = np.rollaxis(raw_anomaly, 1, 0)
        raw_anomaly = raw_anomaly.reshape((number_OFDM_symbols, 1, Ng, RXants))
        print(f'anomaly dataset shape: {raw_anomaly.shape}')
        valid_dataset_x = np.concatenate((raw_normal, raw_anomaly), axis = 0)
        valid_dataset_y = np.concatenate((np.zeros((raw_normal.shape[0], 1)), np.ones((raw_anomaly.shape[0], 1))), axis = 0)

        print(f'train dataset shape: {train_dataset.shape}')
        print(f'valid dataset features shape: {valid_dataset_x.shape}')
        print(f'valid dataset labels shape: {valid_dataset_y.shape}')

    else:
        train_dataset = torch.load(name_training_dataset)
        valid_dataset = torch.load(name_validation_dataset)
        
    if recompute_statistics:
        # compute mean and var of training dataset, normalize data
        temp_loader = DataLoader(torch.tensor(train_dataset).float(), batch_size=train_dataset.shape[0])
        data = next(iter(temp_loader))
        mean_train = data[0].mean() # torch.mean(data[0], 0, keepdim=True) #data[0].mean()
        std_train =  data[0].std() # torch.std(data[0], 0, keepdim=True) # data[0].std()
        mean_train = mean_train.cpu().detach().numpy()
        std_train = std_train.cpu().detach().numpy()
        np.save(DB + '/mean_var_train.npy', [mean_train, std_train], allow_pickle = True)
        del temp_loader
    else:
        if normalized_dataset:
            mean_var = np.load(DB + '/mean_var_train.npy',  allow_pickle = True).tolist()
            mean_train = mean_var[0]
            std_train = mean_var[1]
            print(f'mean train: {mean_train}', f'std train: {std_train}')

    if not load_dataset:
        if normalized_dataset:
            # get normalized dataset
            train_dataset = (train_dataset - mean_train)/std_train
            valid_dataset_x = (valid_dataset_x - mean_train)/std_train

        train_dataset = TensorDataset(torch.tensor(train_dataset).float(), torch.tensor(np.zeros((train_dataset.shape[0], 1))).float())
        valid_dataset = TensorDataset(torch.tensor(valid_dataset_x).float(), torch.tensor(valid_dataset_y).float()) # assign labels to validation dataset
        
    if save_dataset:
        torch.save(train_dataset, name_training_dataset)
        torch.save(valid_dataset, name_validation_dataset)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle_dataset)
    val_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=shuffle_dataset)

    return train_loader, val_loader