import torch
import numpy as np
import os 
import sys
import math
import warnings
warnings.filterwarnings("ignore")
import mat73
from torch.utils.data import TensorDataset, DataLoader
import scipy.io
from scipy.optimize import least_squares
DATASET_DIR = os.path.join(os.path.split(os.path.split(os.path.split(os.path.split(os.path.split(os.path.abspath(__file__))[0])[0])[0])[0])[0], 'Dataset')
sys.path.append(DATASET_DIR)
import pygeodesy
from pygeodesy.ellipsoidalKarney import LatLon
ginterpolator = pygeodesy.GeoidKarney(os.path.join(DATASET_DIR, "Geodetic_map/geoids/egm2008-5.pgm"))
CLASSES_DIR = os.path.split(os.path.split(os.path.split(os.path.abspath(__file__))[0])[0])[0]
sys.path.append(os.path.dirname(CLASSES_DIR))
sys.path.append(os.path.dirname(os.path.split(os.path.split(os.path.abspath(__file__))[0])[0]))
from Classes.utils import coord2xyz


def Dataset(DB, name_input_dataset, which_BS, name_training_dataset, name_validation_dataset, load_dataset, 
normalized_dataset, recompute_statistics, save_dataset, new_matlab_format, batch_size, shuffle_dataset, adjust_latlon_geoid = 0, create_global_xyz = 1):
       
    if not load_dataset:
        
        # import datasets
        try:
            raw_dataset = mat73.loadmat(name_input_dataset)['dataset']
        except: 
            raw_dataset = scipy.io.loadmat(name_input_dataset)['dataset']

        print('Loaded dataset')

        if which_BS is not None:
            raw_dataset = raw_dataset[f'cell_{which_BS}']

            input_dataset = raw_dataset['input']

            los_index = raw_dataset['LOS']
            pos_index = raw_dataset['pos']
            cell_index = np.ones(raw_dataset['pos'].shape[0])*int(which_BS)

            realPos_latlon = raw_dataset['realPos_latlon']
            realPos_xyz_cell = raw_dataset['realPos_xyz_cell']
            estimatedPos_xyz_cell = raw_dataset['estimatedPos_xyz_cell']
            ML_estPos_xyz_cell = raw_dataset['ML_estPos_xyz_cell']

            est_ToA = raw_dataset['est_ToA']
            est_AoA_cell = raw_dataset['est_AoA_cell']
            true_ToA = raw_dataset['true_ToA']
            true_AoA_cell = raw_dataset['true_AoA_cell']

            Ng = input_dataset.shape[0]
            number_OFDM_symbols = input_dataset.shape[1]
            RXants = input_dataset.shape[2]

            del raw_dataset

            input_dataset = np.rollaxis(input_dataset, 1, 0)
            input_dataset = input_dataset.reshape((number_OFDM_symbols, 1, Ng, RXants))

        else:

            if 'Cooperative' in name_input_dataset:
                numPos = len(list(filter((-1).__ne__, [k if v['LOS'].shape[0]!=2 else -1 for k,v in raw_dataset.items()])))
                positions = [f'pos_{pos}'for pos in range(1, numPos + 1)] 
                # positions = positions[:10] 
                num_OFDM_symbols_per_pos = raw_dataset[positions[0]]['LOS'].shape[0] 
                Ng = raw_dataset[positions[0]]['input'].shape[0] 
                RXants = raw_dataset[positions[0]]['input'].shape[2]
                num_BS_per_pos =  np.array([raw_dataset[pos]['LOS'].reshape([num_OFDM_symbols_per_pos,-1]).shape[1] for pos in positions])

                print('Creating input dataset')
                input_dataset = np.concatenate([np.pad(raw_dataset[pos]['input'].reshape([Ng, num_OFDM_symbols_per_pos,RXants,-1]),((0, 0), (0, 0), (0,0), (0, np.max(num_BS_per_pos)-num_BS_per_pos[i])),mode='constant',constant_values=(np.nan,)) for i,pos in enumerate(positions)], 1)
                print('Created input dataset')

                los_index =  np.concatenate([np.pad(raw_dataset[pos]['LOS'].reshape([num_OFDM_symbols_per_pos,-1]),((0, 0), (0, np.max(num_BS_per_pos)-num_BS_per_pos[i])),mode='constant',constant_values=(np.nan,)) for i,pos in enumerate(positions)], 0)
                pos_index =  np.concatenate([np.pad(raw_dataset[pos]['pos'].reshape([num_OFDM_symbols_per_pos,-1]),((0, 0), (0, np.max(num_BS_per_pos)-num_BS_per_pos[i])),mode='constant',constant_values=(np.nan,)) for i,pos in enumerate(positions)], 0)
                cell_index = np.concatenate([np.pad(raw_dataset[pos]['cell_index'].reshape([num_OFDM_symbols_per_pos,-1]),((0, 0), (0, np.max(num_BS_per_pos)-num_BS_per_pos[i])),mode='constant',constant_values=(np.nan,)) for i,pos in enumerate(positions)], 0)

                realPos_latlon = np.concatenate([np.pad(raw_dataset[pos]['realPos_latlon'].reshape([num_OFDM_symbols_per_pos,2,-1]),((0, 0), (0, 0), (0, np.max(num_BS_per_pos)-num_BS_per_pos[i])),mode='constant',constant_values=(np.nan,)) for i,pos in enumerate(positions)], 0)
                realPos_xyz_cell = np.concatenate([np.pad(raw_dataset[pos]['realPos_xyz_cell'].reshape([num_OFDM_symbols_per_pos,3,-1]),((0, 0), (0, 0), (0, np.max(num_BS_per_pos)-num_BS_per_pos[i])),mode='constant',constant_values=(np.nan,)) for i,pos in enumerate(positions)], 0)
                estimatedPos_xyz_cell = np.concatenate([np.pad(raw_dataset[pos]['estimatedPos_xyz_cell'].reshape([num_OFDM_symbols_per_pos,3,-1]),((0, 0), (0, 0), (0, np.max(num_BS_per_pos)-num_BS_per_pos[i])),mode='constant',constant_values=(np.nan,)) for i,pos in enumerate(positions)], 0)
                ML_estPos_xyz_cell = np.concatenate([np.pad(raw_dataset[pos]['ML_estPos_xyz_cell'].reshape([num_OFDM_symbols_per_pos,3,-1]),((0, 0), (0, 0), (0, np.max(num_BS_per_pos)-num_BS_per_pos[i])),mode='constant',constant_values=(np.nan,)) for i,pos in enumerate(positions)], 0)

                est_ToA =  np.concatenate([np.pad(raw_dataset[pos]['est_ToA'].reshape([num_OFDM_symbols_per_pos,-1]),((0, 0), (0, np.max(num_BS_per_pos)-num_BS_per_pos[i])),mode='constant',constant_values=(np.nan,)) for i,pos in enumerate(positions)], 0)
                est_AoA_cell =  np.concatenate([np.pad(raw_dataset[pos]['est_AoA_cell'].reshape([num_OFDM_symbols_per_pos, 2,-1]),((0, 0), (0, 0), (0, np.max(num_BS_per_pos)-num_BS_per_pos[i])),mode='constant',constant_values=(np.nan,)) for i,pos in enumerate(positions)], 0)
                true_ToA =  np.concatenate([np.pad(raw_dataset[pos]['true_ToA'].reshape([num_OFDM_symbols_per_pos,-1]),((0, 0), (0, np.max(num_BS_per_pos)-num_BS_per_pos[i])),mode='constant',constant_values=(np.nan,)) for i,pos in enumerate(positions)], 0)
                true_AoA_cell =  np.concatenate([np.pad(raw_dataset[pos]['true_AoA_cell'].reshape([num_OFDM_symbols_per_pos, 2, -1]),((0, 0), (0, 0), (0, np.max(num_BS_per_pos)-num_BS_per_pos[i])),mode='constant',constant_values=(np.nan,)) for i,pos in enumerate(positions)], 0)

                Ng = input_dataset.shape[0]
                number_OFDM_symbols = input_dataset.shape[1]
                RXants = input_dataset.shape[2]

                print('Deleting raw dataset')
                del raw_dataset
                print('Deleted raw dataset')

                input_dataset = np.rollaxis(input_dataset, 1, 0)
                input_dataset = input_dataset.reshape((number_OFDM_symbols, 1, Ng, RXants, np.max(num_BS_per_pos)))

            else:
                cells = [k for k,v in raw_dataset.items()]
                input_dataset = np.concatenate([raw_dataset[cell]['input'] for cell in cells], 1)

                los_index = np.concatenate([raw_dataset[cell]['LOS'] for cell in cells], 0)
                pos_index = np.concatenate([raw_dataset[cell]['pos'] for cell in cells], 0)
                cell_index = np.concatenate([np.ones(raw_dataset[cell]['pos'].shape[0])*int(cell.split('_')[1]) for cell in cells], 0)

                realPos_latlon = np.concatenate([raw_dataset[cell]['realPos_latlon'] for cell in cells], 0)
                realPos_xyz_cell = np.concatenate([raw_dataset[cell]['realPos_xyz_cell'] for cell in cells], 0)
                estimatedPos_xyz_cell = np.concatenate([raw_dataset[cell]['estimatedPos_xyz_cell'] for cell in cells], 0)
                ML_estPos_xyz_cell = np.concatenate([raw_dataset[cell]['ML_estPos_xyz_cell'] for cell in cells], 0)

                est_ToA = np.concatenate([raw_dataset[cell]['est_ToA'] for cell in cells], 0)
                est_AoA_cell = np.concatenate([raw_dataset[cell]['est_AoA_cell'] for cell in cells], 0)
                true_ToA = np.concatenate([raw_dataset[cell]['true_ToA'] for cell in cells], 0)
                true_AoA_cell = np.concatenate([raw_dataset[cell]['true_AoA_cell'] for cell in cells], 0)

                Ng = input_dataset.shape[0]
                number_OFDM_symbols = input_dataset.shape[1]
                RXants = input_dataset.shape[2]

                print('Deleting raw dataset')
                del raw_dataset
                print('Deleted raw dataset')

                input_dataset = np.rollaxis(input_dataset, 1, 0)
                input_dataset = input_dataset.reshape((number_OFDM_symbols, 1, Ng, RXants))

        print(f'dataset shape: {input_dataset.shape}')

        # shuffle
        print('Random shuffling')
        rng_state = np.random.get_state()
        np.random.shuffle(input_dataset)

        np.random.set_state(rng_state)
        np.random.shuffle(los_index)
        np.random.set_state(rng_state)
        np.random.shuffle(pos_index)
        np.random.set_state(rng_state)
        np.random.shuffle(cell_index)
        np.random.set_state(rng_state)
        np.random.shuffle(realPos_latlon)
        np.random.set_state(rng_state)
        np.random.shuffle(realPos_xyz_cell)
        np.random.set_state(rng_state)
        np.random.shuffle(estimatedPos_xyz_cell)
        np.random.set_state(rng_state)
        np.random.shuffle(ML_estPos_xyz_cell)
        np.random.set_state(rng_state)
        np.random.shuffle(est_ToA)
        np.random.set_state(rng_state)
        np.random.shuffle(est_AoA_cell)
        np.random.set_state(rng_state)
        np.random.shuffle(true_ToA)
        np.random.set_state(rng_state)
        np.random.shuffle(true_AoA_cell)
        print('Random shuffling complete')

        input_dataset_train = input_dataset[:int(0.7*number_OFDM_symbols),:,:,:]
        input_dataset_valid = input_dataset[int(0.7*number_OFDM_symbols):,:,:,:]
        del input_dataset
        #
        los_index_train = los_index[:int(0.7*number_OFDM_symbols)]
        los_index_valid = los_index[int(0.7*number_OFDM_symbols):]
        del los_index
        #
        pos_index_train = pos_index[:int(0.7*number_OFDM_symbols)]
        pos_index_valid = pos_index[int(0.7*number_OFDM_symbols):]
        del pos_index
        #
        cell_index_train = cell_index[:int(0.7*number_OFDM_symbols)]
        cell_index_valid = cell_index[int(0.7*number_OFDM_symbols):]
        del cell_index
        #
        realPos_latlon_train = realPos_latlon[:int(0.7*number_OFDM_symbols),:]
        realPos_latlon_valid = realPos_latlon[int(0.7*number_OFDM_symbols):,:]
        del realPos_latlon
        #
        realPos_xyz_cell_train = realPos_xyz_cell[:int(0.7*number_OFDM_symbols),:]
        realPos_xyz_cell_valid = realPos_xyz_cell[int(0.7*number_OFDM_symbols):,:]
        del realPos_xyz_cell
        #
        estimatedPos_xyz_cell_train = estimatedPos_xyz_cell[:int(0.7*number_OFDM_symbols),:]
        estimatedPos_xyz_cell_valid = estimatedPos_xyz_cell[int(0.7*number_OFDM_symbols):,:]
        del estimatedPos_xyz_cell
        #
        ML_estPos_xyz_cell_train = ML_estPos_xyz_cell[:int(0.7*number_OFDM_symbols),:]
        ML_estPos_xyz_cell_valid = ML_estPos_xyz_cell[int(0.7*number_OFDM_symbols):,:]
        del ML_estPos_xyz_cell
        #
        est_ToA_train = est_ToA[:int(0.7*number_OFDM_symbols)]
        est_ToA_valid = est_ToA[int(0.7*number_OFDM_symbols):]
        del est_ToA
        #
        est_AoA_cell_train = est_AoA_cell[:int(0.7*number_OFDM_symbols),:]
        est_AoA_cell_valid = est_AoA_cell[int(0.7*number_OFDM_symbols):,:]
        del est_AoA_cell
        #
        true_ToA_train = true_ToA[:int(0.7*number_OFDM_symbols)]
        true_ToA_valid = true_ToA[int(0.7*number_OFDM_symbols):]
        del true_ToA
        #
        true_AoA_cell_train = true_AoA_cell[:int(0.7*number_OFDM_symbols),:]
        true_AoA_cell_valid = true_AoA_cell[int(0.7*number_OFDM_symbols):,:]
        del true_AoA_cell

        print(f'train dataset shape: {input_dataset_train.shape}')
        print(f'valid dataset shape: {input_dataset_valid.shape}')

    else:
        train_dataset = torch.load(name_training_dataset)
        valid_dataset = torch.load(name_validation_dataset)

    if adjust_latlon_geoid:
        num_samples = train_dataset.tensors[9].shape[0] 
        num_BS = train_dataset.tensors[9].shape[2] 
        tmp = torch.full([num_samples, 3, num_BS], torch.nan)
        for i in range(num_samples):
            for j in range(num_BS):
                if not torch.isnan(train_dataset.tensors[9][i,:,j]).any():
                    tmp[i,:,j] = torch.cat((train_dataset.tensors[9][i,:,j], torch.tensor([ginterpolator(LatLon(train_dataset.tensors[9][i,0,j], train_dataset.tensors[9][i,1,j]))]).float()), 0)
        train_dataset = TensorDataset(train_dataset.tensors[0], 
                                    train_dataset.tensors[1], 
                                    train_dataset.tensors[2], 

                                    train_dataset.tensors[3],
                                    train_dataset.tensors[4],
                                    train_dataset.tensors[5],
                                    train_dataset.tensors[6],

                                    train_dataset.tensors[7],
                                    train_dataset.tensors[8],
                                    tmp,
                                    train_dataset.tensors[10],
                                    train_dataset.tensors[11],
                                    )

        num_samples = valid_dataset.tensors[9].shape[0] 
        num_BS = valid_dataset.tensors[9].shape[2] 
        tmp = torch.full([num_samples, 3, num_BS], torch.nan)
        for i in range(num_samples):
            for j in range(num_BS):
                if not torch.isnan(valid_dataset.tensors[9][i,:,j]).any():
                    tmp[i,:,j] = torch.cat((valid_dataset.tensors[9][i,:,j], torch.tensor([ginterpolator(LatLon(valid_dataset.tensors[9][i,0,j], valid_dataset.tensors[9][i,1,j]))]).float()), 0)
        valid_dataset = TensorDataset(valid_dataset.tensors[0], 
                                    valid_dataset.tensors[1], 
                                    valid_dataset.tensors[2], 

                                    valid_dataset.tensors[3],
                                    valid_dataset.tensors[4],
                                    valid_dataset.tensors[5],
                                    valid_dataset.tensors[6],

                                    valid_dataset.tensors[7],
                                    valid_dataset.tensors[8],
                                    tmp,
                                    valid_dataset.tensors[10],
                                    valid_dataset.tensors[11],
                                    )

    if create_global_xyz:
        num_samples = train_dataset.tensors[9].shape[0] 
        num_BS = train_dataset.tensors[9].shape[2] 
        realPos_global_xyz_train = torch.full([num_samples, 3, num_BS], torch.nan)
        for i in range(num_samples):
            for j in range(num_BS):
                if not torch.isnan(train_dataset.tensors[9][i,:,j]).any():
                    realPos_global_xyz_train[i,:,j] = torch.tensor(coord2xyz(train_dataset.tensors[9][i,0,j], train_dataset.tensors[9][i,1,j], train_dataset.tensors[9][i,2,j]))
        train_dataset = TensorDataset(train_dataset.tensors[0], 
                                    train_dataset.tensors[1], 
                                    train_dataset.tensors[2], 

                                    train_dataset.tensors[3],
                                    train_dataset.tensors[4],
                                    train_dataset.tensors[5],
                                    train_dataset.tensors[6],

                                    train_dataset.tensors[7],
                                    train_dataset.tensors[8],
                                    train_dataset.tensors[9],
                                    train_dataset.tensors[10],
                                    train_dataset.tensors[11],
                                    realPos_global_xyz_train.float()
                                    )
        num_samples = valid_dataset.tensors[9].shape[0] 
        num_BS = valid_dataset.tensors[9].shape[2] 
        realPos_global_xyz_valid = torch.full([num_samples, 3, num_BS], torch.nan)
        for i in range(num_samples):
            for j in range(num_BS):
                if not torch.isnan(valid_dataset.tensors[9][i,:,j]).any():
                    realPos_global_xyz_valid[i,:,j] = torch.tensor(coord2xyz(valid_dataset.tensors[9][i,0,j], valid_dataset.tensors[9][i,1,j], valid_dataset.tensors[9][i,2,j]))
        valid_dataset = TensorDataset(valid_dataset.tensors[0], 
                                    valid_dataset.tensors[1], 
                                    valid_dataset.tensors[2], 

                                    valid_dataset.tensors[3],
                                    valid_dataset.tensors[4],
                                    valid_dataset.tensors[5],
                                    valid_dataset.tensors[6],

                                    valid_dataset.tensors[7],
                                    valid_dataset.tensors[8],
                                    valid_dataset.tensors[9],
                                    valid_dataset.tensors[10],
                                    valid_dataset.tensors[11],
                                    realPos_global_xyz_valid.float()
                                    )

    if recompute_statistics:
        # compute mean and var of training dataset, normalize data
        temp_loader = DataLoader(torch.tensor(input_dataset_train).double(), batch_size=input_dataset_train.shape[0])
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
            input_dataset_train = (input_dataset_train - mean_train)/std_train
            input_dataset_valid = (input_dataset_valid - mean_train)/std_train

        train_dataset = TensorDataset(torch.tensor(input_dataset_train).float(), 
                                    torch.tensor(est_ToA_train).float(), 
                                    torch.tensor(est_AoA_cell_train).float(), 

                                    torch.tensor(los_index_train).float(),
                                    torch.tensor(realPos_xyz_cell_train).float(),
                                    torch.tensor(true_ToA_train).float(),
                                    torch.tensor(true_AoA_cell_train).float(),

                                    torch.tensor(pos_index_train).float(),
                                    torch.tensor(cell_index_train).float(),
                                    torch.tensor(realPos_latlon_train).float(),
                                    torch.tensor(estimatedPos_xyz_cell_train).float(),
                                    torch.tensor(ML_estPos_xyz_cell_train).float(),
                                    )
        valid_dataset = TensorDataset(torch.tensor(input_dataset_valid).float(), 
                                    torch.tensor(est_ToA_valid).float(), 
                                    torch.tensor(est_AoA_cell_valid).float(), 

                                    torch.tensor(los_index_valid).float(),
                                    torch.tensor(realPos_xyz_cell_valid).float(),
                                    torch.tensor(true_ToA_valid).float(),
                                    torch.tensor(true_AoA_cell_valid).float(),

                                    torch.tensor(pos_index_valid).float(),
                                    torch.tensor(cell_index_valid).float(),
                                    torch.tensor(realPos_latlon_valid).float(),
                                    torch.tensor(estimatedPos_xyz_cell_valid).float(),
                                    torch.tensor(ML_estPos_xyz_cell_valid).float(),
                                    )    
    if save_dataset:
        torch.save(train_dataset, name_training_dataset)
        torch.save(valid_dataset, name_validation_dataset)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle_dataset)
    val_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=shuffle_dataset)

    return train_loader, val_loader


def Dataset_synthetic(type_, params):

    if type_ == 'y = ax':
        X = np.random.uniform(low=params['x_low'], high=params['x_high'], size=(params['num_samples'],1))
        Y = X*params['alpha']

        rng_state = np.random.get_state()
        np.random.shuffle(X)
        np.random.set_state(rng_state)
        np.random.shuffle(Y)

        X_train_dataset = X[:int(params['training_split']*params['num_samples']),:]
        X_valid_normal = X[int(params['training_split']*params['num_samples']):,:]
        Y_train_dataset = Y[:int(params['training_split']*params['num_samples']),:]
        Y_valid_normal = Y[int(params['training_split']*params['num_samples']):,:]

        train_dataset = TensorDataset(torch.tensor(X_train_dataset).float(), torch.tensor(Y_train_dataset).float())
        valid_dataset = TensorDataset(torch.tensor(X_valid_normal).float(), torch.tensor(Y_valid_normal).float())

    if type_ == 'y = bsin(x)':
        X = np.random.uniform(low=params['x_low'], high=params['x_high'], size=(params['num_samples'],1))
        Y = params['beta']*np.sin(X)

        rng_state = np.random.get_state()
        np.random.shuffle(X)
        np.random.set_state(rng_state)
        np.random.shuffle(Y)

        X_train_dataset = X[:int(params['training_split']*params['num_samples']),:]
        X_valid_normal = X[int(params['training_split']*params['num_samples']):,:]
        Y_train_dataset = Y[:int(params['training_split']*params['num_samples']),:]
        Y_valid_normal = Y[int(params['training_split']*params['num_samples']):,:]

        train_dataset = TensorDataset(torch.tensor(X_train_dataset).float(), torch.tensor(Y_train_dataset).float())
        valid_dataset = TensorDataset(torch.tensor(X_valid_normal).float(), torch.tensor(Y_valid_normal).float())

    if type_ == 'sqrt(y1**2+y2**2)=x1, y2 = y1 tan(x2)':
        X1 = np.random.uniform(low=params['x1_low'], high=params['x1_high'], size=(params['num_samples'],1))
        X2 = np.random.uniform(low=params['x2_low'], high=params['x2_high'], size=(params['num_samples'],1))

        Y2 = X1*np.sin(X2*math.pi/180)
        Y1 = X1*np.cos(X2*math.pi/180)

        X = np.squeeze(np.stack((X1, X2), 1))
        Y = np.squeeze(np.stack((Y1, Y2), 1))

        rng_state = np.random.get_state()
        np.random.shuffle(X)
        np.random.set_state(rng_state)
        np.random.shuffle(Y)

        X_train_dataset = X[:int(params['training_split']*params['num_samples']),:]
        X_valid_normal = X[int(params['training_split']*params['num_samples']):,:]
        Y_train_dataset = Y[:int(params['training_split']*params['num_samples']),:]
        Y_valid_normal = Y[int(params['training_split']*params['num_samples']):,:]

        train_dataset = TensorDataset(torch.tensor(X_train_dataset).float(), torch.tensor(Y_train_dataset).float())
        valid_dataset = TensorDataset(torch.tensor(X_valid_normal).float(), torch.tensor(Y_valid_normal).float())

    if type_ == 'sqrt(y1**2+y2**2)=xi':
        angle = 360/int(params['number_cells']) * math.pi/180
        params_xi_high = params['center_high']
        for center_index in range(1, int(params['number_cells'])+1):
            specific_angle = (center_index)*angle
            exec(f"center{center_index} = np.array([{params_xi_high}*np.cos({specific_angle}), {params_xi_high}*np.sin({specific_angle})]).reshape([1, 2])")

        # center1 = np.array([params['xi_high']*np.cos(angle), params['xi_high']*np.sin(angle)]).reshape([1, 2])
        # center2 = np.array([params['xi_high']*np.cos(2*angle), params['xi_high']*np.sin(2*angle)]).reshape([1, 2])
        # center3 = np.array([params['xi_high']*np.cos(3*angle), params['xi_high']*np.sin(3*angle)]).reshape([1, 2])

        Y = np.random.uniform(low=-params['xi_high'], high=params['xi_high'], size=(params['num_samples'],2))

        for center_index in range(1, int(params['number_cells'])+1):
            ris = np.power(np.linalg.norm(Y-locals()[f'center{center_index}'], axis = 1), 1)
            exec(f"X_{center_index} = ris")
        # X1 = np.power(np.linalg.norm(Y-center1, axis = 1), 1)
        # X2 = np.power(np.linalg.norm(Y-center2, axis = 1), 1)
        # X3 = np.power(np.linalg.norm(Y-center3, axis = 1), 1)

        X = np.reshape(locals()[f'X_1'], [-1, 1])
        for center_index in range(2, int(params['number_cells'])+1):
            X = np.squeeze(np.concatenate((X, np.reshape(locals()[f'X_{center_index}'], [-1, 1])), 1))

        # X = np.squeeze(np.stack(([locals()[f'X_{center_index}'] for center_index in range(1, int(params['number_cells'])+1)]), 1))
        # X = np.squeeze(np.stack(([exec(f'X_{center_index}') for center_index in range(1, int(params['number_cells'])+1)]), 1))
        # X = np.squeeze(np.stack((X1, X2, X3), 1))

        rng_state = np.random.get_state()
        np.random.shuffle(X)
        np.random.set_state(rng_state)
        np.random.shuffle(Y)

        X_train_dataset = X[:int(params['training_split']*params['num_samples']),:]
        X_valid_normal = X[int(params['training_split']*params['num_samples']):,:]
        Y_train_dataset = Y[:int(params['training_split']*params['num_samples']),:]
        Y_valid_normal = Y[int(params['training_split']*params['num_samples']):,:]

        train_dataset = TensorDataset(torch.tensor(X_train_dataset).float(), torch.tensor(Y_train_dataset).float())
        valid_dataset = TensorDataset(torch.tensor(X_valid_normal).float(), torch.tensor(Y_valid_normal).float())
        
    train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=1)
    val_loader = DataLoader(valid_dataset, batch_size=params['batch_size'], shuffle=1)

    center_return = []
    try:
        for center_index in range(1, int(params['number_cells'])+1):
            center_return.append(locals()[f'center{center_index}'])
    except:
        pass

    return train_loader, val_loader, center_return


def intersectionPoint(p1,p2,p3):
    
    x1, y1, dist_1 = (p1[0], p1[1], p1[2])
    x2, y2, dist_2 = (p2[0], p2[1], p2[2])
    x3, y3, dist_3 = (p3[0], p3[1], p3[2])

    def eq(g):
        x, y = g

        return (
            (x - x1)**2 + (y - y1)**2 - dist_1**2,
            (x - x2)**2 + (y - y2)**2 - dist_2**2,
            (x - x3)**2 + (y - y3)**2 - dist_3**2)

    guess = (x1, y1 + dist_1)

    ans = least_squares(eq, guess, ftol=None, xtol=None)

    return ans
             