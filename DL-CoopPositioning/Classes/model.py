import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple
from torch.autograd import Variable
from torch import linalg as LA
import sys 
import os
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_fscore_support as prf, accuracy_score
from torch.distributions import MultivariateNormal, Normal
from torch.distributions.distribution import Distribution
import torch.nn.init as init
# Directories
CLASSES_DIR = os.path.split(os.path.split(os.path.split(os.path.abspath(__file__))[0])[0])[0]
DIR_EXPERIMENT = os.path.dirname(os.path.abspath(__file__))
cwd = DIR_EXPERIMENT
# print(os.path.abspath(__file__))
# print(os.path.split(os.path.split(os.path.abspath(__file__))[0])[0])
# print(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(CLASSES_DIR))
sys.path.append(os.path.dirname(os.path.split(os.path.split(os.path.abspath(__file__))[0])[0]))

from Classes.utils import to_var

class Model_Positioning1(nn.Module):

    def __init__(self, latent_dim=8, in_channels=1, out_channels=1, neurons_per_layer_LOS = [16, 32, 16], neurons_per_layer_NLOS = [16, 32, 16]):
        super(Model_Positioning1, self).__init__()

        self.is_unpooling = True
        self.neurons_per_layer_LOS = neurons_per_layer_LOS
        self.neurons_per_layer_NLOS = neurons_per_layer_NLOS

        # Encoder1
        self.down1 = segnetDown2(in_channels, 64)
        self.down2 = segnetDown2(64, 128)
        self.down3 = segnetDown3(128, 256)
        self.down4 = segnetDown3(256, 512)
        self.down5 = segnetDown3(512, 512)
        self.down6 = nn.Linear(13312,latent_dim)

        # Decoder
        self.up6 = nn.Linear(latent_dim,13312)
        self.up5 = segnetUp3(512, 512)
        self.up4 = segnetUp3(512, 256)
        self.up3 = segnetUp3(256, 128)
        self.up2 = segnetUp2(128, 64)
        self.up1 = segnetUp2(64, out_channels)


        # Identification network
        layers = []
        layers += [nn.Linear(latent_dim,latent_dim)]
        layers += [nn.Tanh()]  
        layers += [nn.Dropout(p=0.5)]  

        layers += [nn.Linear(latent_dim,5)]
        layers += [nn.Tanh()] 
        layers += [nn.Dropout(p=0.5)]   

        layers += [nn.Linear(5,1)] 
        layers += [nn.BatchNorm1d(int(1))] 
        layers += [nn.Sigmoid()]
        self.identification_network = nn.Sequential(*layers)

        # LOS network
        self.LOS_network = NN_function_approx(latent_dim, 3, neurons_per_layer_LOS)

        # NLOS network
        self.NLOS_network = NN_function_approx(latent_dim, 3, neurons_per_layer_NLOS)



    def relative_euclidean_distance(self, a, b):
        return (a-b).norm(2, dim=1) / a.norm(2, dim=1)

    def forward(self, x, initialize=0, test = 0, cooperative = 0, cooperative_predict_pos_LOS = 0, z_LOS = None):

        self.batch_size = x.shape[0]

        if not cooperative:

            down1, indices_1, unpool_shape1 = self.down1(x)
            down2, indices_2, unpool_shape2 = self.down2(down1)
            down3, indices_3, unpool_shape3 = self.down3(down2)    
            down4, indices_4, unpool_shape4 = self.down4(down3)
            down5, indices_5, unpool_shape5 = self.down5(down4)
            down5 = torch.flatten(down5, 1)
            enc1 = self.down6(down5)

            if test == 0:
                up6 = self.up6(enc1)
                # modify
                up6 = torch.reshape(up6, (-1, 512, 13, 2))
                up5 = self.up5(up6, indices_5, unpool_shape5)
                up4 = self.up4(up5, indices_4, unpool_shape4)
                up3 = self.up3(up4, indices_3, unpool_shape3)
                up2 = self.up2(up3, indices_2, unpool_shape2)
                dec = self.up1(up2, indices_1, unpool_shape1)

            # Latent features LOS identification
            z_identification = enc1
            LOS_identification = self.identification_network(z_identification)

            # Latent features LOS positioning
            z_LOS = enc1
            pos_LOS = self.LOS_network(z_LOS)

            # Latent features NLOS positioning
            z_NLOS = enc1
            pos_NLOS = self.NLOS_network(z_NLOS)

        else:

            # step 1 
            if not cooperative_predict_pos_LOS:

                down1, indices_1, unpool_shape1 = self.down1(x)
                down2, indices_2, unpool_shape2 = self.down2(down1)
                down3, indices_3, unpool_shape3 = self.down3(down2)    
                down4, indices_4, unpool_shape4 = self.down4(down3)
                down5, indices_5, unpool_shape5 = self.down5(down4)
                down5 = torch.flatten(down5, 1)
                enc1 = self.down6(down5)

                if test == 0:
                    up6 = self.up6(enc1)
                    up6 = torch.reshape(up6, (-1, 512, 13, 2))
                    up5 = self.up5(up6, indices_5, unpool_shape5)
                    up4 = self.up4(up5, indices_4, unpool_shape4)
                    up3 = self.up3(up4, indices_3, unpool_shape3)
                    up2 = self.up2(up3, indices_2, unpool_shape2)
                    dec = self.up1(up2, indices_1, unpool_shape1)

                # Latent features LOS identification
                z_identification = enc1
                LOS_identification = self.identification_network(z_identification)

                # Dont'care
                pos_LOS = 0

                # Latent features NLOS positioning
                z_NLOS = enc1
                pos_NLOS = self.NLOS_network(z_NLOS)
  
            # step 2 
            else:
                # Dont'care
                enc1 = 0

                # Dont'care
                dec = 0

                # Dont'care
                LOS_identification = 0

                # Latent features LOS positioning
                pos_LOS = self.LOS_network(z_LOS)

                # Dont'care
                pos_NLOS = 0

        return enc1, dec, LOS_identification, pos_LOS, pos_NLOS


    def loss_function(self, x, x_hat, LOS_identification, true_pos, LOS_identification_hat, pos_LOS_hat, pos_NLOS_hat, lamda_rec_err, lamda_uncertainty_pos):

        LOS_identification = to_var(LOS_identification.reshape([-1, 1]))
        true_pos = to_var(true_pos)

        # Loss 1
        recon_error = torch.mean((x - x_hat) ** 2)
        # Loss 2
        # Define Balancing weight
        positive_vals = LOS_identification.sum()
        if positive_vals > 0:
            if (self.batch_size - positive_vals) == 0:
                pos_weight = to_var(torch.tensor([1], dtype=torch.float))
            else:
                pos_weight = to_var(torch.tensor([(self.batch_size - positive_vals) / positive_vals], dtype=torch.float))
        else: # If there are no positives labels, avoid dividing by zero
            pos_weight = to_var(torch.tensor([0], dtype=torch.float))
        # print(pos_weight)

        loss_positioning_LOS = torch.mean(
                pos_weight * LOS_identification * (-torch.log(LOS_identification_hat) + lamda_uncertainty_pos * ((true_pos - pos_LOS_hat) ** 2).sum(1).reshape([-1, 1]) ) + 
                (1 - LOS_identification) * (-torch.log(1 - LOS_identification_hat) + lamda_uncertainty_pos * ((true_pos - pos_NLOS_hat) ** 2).sum(1).reshape([-1, 1]) )
        )

        total_loss = lamda_rec_err * recon_error + loss_positioning_LOS

        return total_loss, recon_error, loss_positioning_LOS

    def evaluate(self, x, x_hat, LOS_identification, true_pos, LOS_identification_hat, pos_LOS_hat, pos_NLOS_hat, chosen_thr = None, test = 0):
        # test = 0 for training, return only one mean_error
        # test = 1 for testing, return many mean_error and the chosen threshold

        try:
            LOS_identification = LOS_identification.data.cpu().numpy().astype(int).reshape([-1,1])
        except:
            LOS_identification = LOS_identification.astype(int).reshape([-1,1])
        try:
            LOS_identification_hat = LOS_identification_hat.data.cpu().numpy().reshape([-1,1])
        except:
            LOS_identification_hat = LOS_identification_hat.reshape([-1,1])

        fpr, tpr, thr = roc_curve(LOS_identification, LOS_identification_hat, pos_label=1) 
        roc_auc = auc(fpr, tpr)

        if chosen_thr == None:
            if np.isnan(fpr).any():
                chosen_thr = np.array([0])
            elif np.isnan(tpr).any():
                chosen_thr = np.array([1])
            else:
                optimal_idx = np.argmax(tpr - fpr)
                chosen_thr = thr[optimal_idx]

        LOS_identification_binary_hat = (LOS_identification_hat > chosen_thr).astype(int)

        accuracy = accuracy_score(LOS_identification,LOS_identification_binary_hat)
        precision, recall, f_score, support = prf(LOS_identification,LOS_identification_binary_hat, average='binary')

        try:
            pos_predicted = np.where(LOS_identification_binary_hat==0, pos_NLOS_hat.data.cpu().numpy(), 0) + \
                np.where(LOS_identification_binary_hat==1, pos_LOS_hat.data.cpu().numpy(), 0)
        except:
            pos_predicted = np.where(LOS_identification_binary_hat==0, pos_NLOS_hat, 0) + \
                np.where(LOS_identification_binary_hat==1, pos_LOS_hat, 0)

        if test == 0:
            mean_error = np.linalg.norm(true_pos-pos_predicted, axis = 1).mean()
            # mean_error = mean_squared_error(true_pos.data.cpu().numpy(), pos_predicted, squared=False)
        else:
            mean_error = np.linalg.norm(true_pos-pos_predicted, axis = 1)

        return roc_auc, accuracy, precision, recall, f_score, chosen_thr, mean_error, LOS_identification_binary_hat,  pos_predicted


class NN_function_approx(nn.Module):
    def __init__(self, in_size, out_size, neurons_per_layer):
        super(NN_function_approx, self).__init__()

        self.in_size = in_size
        self.out_size = out_size
        self.neurons_per_layer = neurons_per_layer

        self.fcin = nn.Linear(in_size, neurons_per_layer[0])
        self.num_layers = len(self.neurons_per_layer)
        for layer in range(0, self.num_layers-1):
            name_layer = f'fc{layer}'
            neurons_in = self.neurons_per_layer[layer]
            neurons_exit = self.neurons_per_layer[layer+1]
            exec(f"self.{name_layer} = nn.Linear({neurons_in},{neurons_exit})")
        self.fcout = nn.Linear(self.neurons_per_layer[-1], out_size)
    
    def forward(self, x):

        self.x = x.reshape([x.shape[0], self.in_size])
        try:
            self.x = torch.flatten(self.x, 1)
        except:
            pass
        self.x = F.gelu(self.fcin(self.x))
        for layer in range(0, self.num_layers-1):
            name_layer = f'fc{layer}'
            exec(f"self.x = F.gelu(self.{name_layer}(self.x))")
        self.x = self.fcout(self.x)

        return self.x

############################################################################################################################################
############################################################################################################################################
# Multi-path Res-Inception
class MPRI(nn.Module):
    
    def __init__(self, in_channels=1):
        super(MPRI, self).__init__()

        self.input_module = conv2DBatchNormRelu(in_channels=1, n_filters=3, k_size=3)

        self.bottleneck = nn.Conv2d(3,1,kernel_size=3)
        self.upper_MPRI_path_1 = conv2DBatchNormRelu(in_channels=1, n_filters=64, k_size=1)
        self.upper_MPRI_path_2 = nn.Sequential(conv2DBatchNormRelu(in_channels=1, n_filters=32, k_size=(5,1)), nn.Conv2d(32,32,kernel_size=(1,3)))
        self.upper_MPRI_path_3 = nn.Sequential(nn.MaxPool2d(2, 2, return_indices=True),conv2DBatchNormRelu(in_channels=1, n_filters=32, k_size=3))
        self.upper_MPRI_path_4 = nn.Sequential(conv2DBatchNormRelu(in_channels=1, n_filters=32, k_size=1), conv2DBatchNormRelu(in_channels=32, n_filters=128, k_size=1))

    def forward(self, x):

        output_input_module = self.input_module(x)
        output_upper_MPRI_path_1 = self.upper_MPRI_path_1(output_input_module)
        output_upper_MPRI_path_2 = self.upper_MPRI_path_2(output_input_module)
        output_upper_MPRI_path_3 = self.upper_MPRI_path_3(output_input_module)
        output_upper_MPRI_path_4 = self.upper_MPRI_path_4(output_input_module)

        output_upper_MPRI = torch.cat([output_upper_MPRI_path_1, output_upper_MPRI_path_2, output_upper_MPRI_path_3, output_upper_MPRI_path_4]) + output_input_module


# Neural Additive Models: Interpretable Machine Learning with Neural Nets
############################################################################################################################################
############################################################################################################################################

def truncated_normal_(tensor, mean: float = 0., std: float = 1.):
    size = tensor.shape
    tmp = tensor.new_empty(size + (4,)).normal_()
    valid = (tmp < 2) & (tmp > -2)
    ind = valid.max(-1, keepdim=True)[1]
    tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
    tensor.data.mul_(std).add_(mean)


class ActivationLayer(torch.nn.Module):
    def __init__(self,
                 in_features: int,
                 out_features: int):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.empty((in_features, out_features)))
        self.bias = torch.nn.Parameter(torch.empty(in_features))

    def forward(self, x):
        raise NotImplementedError("abstract method called")


class ExULayer(ActivationLayer):
    def __init__(self,
                 in_features: int,
                 out_features: int):
        super().__init__(in_features, out_features)
        truncated_normal_(self.weight, mean=4.0, std=0.5)
        truncated_normal_(self.bias, std=0.5)

    def forward(self, x):
        exu = (x - self.bias) @ torch.exp(self.weight)
        return torch.clip(exu, 0, 1)


class ReLULayer(ActivationLayer):
    def __init__(self,
                 in_features: int,
                 out_features: int):
        super().__init__(in_features, out_features)
        torch.nn.init.xavier_uniform_(self.weight)
        truncated_normal_(self.bias, std=0.5)

    def forward(self, x):
        return F.relu((x - self.bias) @ self.weight)


class FeatureNN(torch.nn.Module): 
    def __init__(self,
                shallow_units: int,
                 hidden_units: Tuple = (),
                 shallow_layer: ActivationLayer = ExULayer,
                 hidden_layer: ActivationLayer = ReLULayer,
                 dropout: float = .5,
                 ):
        super().__init__()
        self.layers = torch.nn.ModuleList([
            hidden_layer(shallow_units if i == 0 else hidden_units[i - 1], hidden_units[i])
            for i in range(len(hidden_units))
        ])
        self.layers.insert(0, shallow_layer(1, shallow_units))

        self.dropout = torch.nn.Dropout(p=dropout)
        self.linear = torch.nn.Linear(shallow_units if len(hidden_units) == 0 else hidden_units[-1], 1, bias=False)
        torch.nn.init.xavier_uniform_(self.linear.weight)


    def forward(self, x):
        x = x.unsqueeze(1)
        for layer in self.layers:
            x = layer(x)
            x = self.dropout(x)
        return self.linear(x)


class MultiFeatureNN(torch.nn.Module):
    def __init__(
        self,
        num_subnets: int,
        num_tasks: int,
        dropout: float,
        hidden_units: list = [64, 32],
        activation: str = 'relu'
    ):
        """Initializes FeatureNN hyperparameters.
        Args:
            dropout: Coefficient for dropout regularization.
            feature_num: Feature Index used for naming the hidden layers.
        """
        super(MultiFeatureNN, self).__init__()
        subnets = [
            NN_function_approx(1, 1, hidden_units)
            for i in range(num_subnets)
        ]
        self.feature_nns = nn.ModuleList(subnets)
        self.linear = torch.nn.Linear(num_subnets, num_tasks, bias=False)

    def forward(self, inputs):
        """Computes FeatureNN output with either evaluation or training mode."""
        individual_outputs = []
        for fnn in self.feature_nns:
            individual_outputs.append(fnn(inputs)) 

        # (batch_size, num_subnets)
        stacked = torch.stack(individual_outputs, dim=-1)
        # stacked = stacked.squeeze()
        # (batch_size, num_tasks)
        weighted = self.linear(stacked)
        return weighted



class NeuralAdditiveModel(torch.nn.Module): # shallow_units: int,
    def __init__(self,
                 input_size: int, 
                 out_size: int,
                 hidden_units: Tuple = (),
                 shallow_layer: ActivationLayer = ExULayer,
                 hidden_layer: ActivationLayer = ReLULayer,
                 feature_dropout: float = 0.,
                 hidden_dropout: float = 0.,
                 ):
        super().__init__()
        self.input_size = input_size
        self.hidden_units = hidden_units
        self.out_size = out_size
        self.num_subnets = 1
        self.num_tasks = out_size

        self.feature_nns = nn.ModuleList([
            MultiFeatureNN(
                    num_subnets=self.num_subnets,
                    num_tasks=self.num_tasks,
                    dropout=feature_dropout,
                    hidden_units=self.hidden_units
                )
            for i in range(self.input_size)
        ])
        self.feature_dropout = torch.nn.Dropout(p=feature_dropout)
        self._bias = torch.nn.Parameter(data=torch.zeros(1, self.num_tasks))



    def forward(self, x):
        # tuple: (batch, num_tasks) x num_inputs
        individual_outputs = self._feature_nns(x)
        # (batch, num_tasks, num_inputs)
        stacked_out = torch.stack(individual_outputs, dim=-1).squeeze(dim=1)
        dropout_out = self.feature_dropout(stacked_out)

        # (batch, num_tasks)
        summed_out = torch.sum(dropout_out, dim=2) + self._bias
        return summed_out


    def _feature_nns(self, x):
        return [self.feature_nns[i](x[:, i]) for i in range(self.input_size)]
        

############################################################################################################################################
############################################################################################################################################
class View(nn.Module):
    def __init__(self, size):
        super(View, self).__init__()
        self.size = size

    def forward(self, tensor):
        return tensor.view(self.size)       

class conv2DBatchNormRelu(nn.Module):
    def __init__(
            self,
            in_channels,
            n_filters,
            k_size,
            stride=1,
            padding=1,
            bias=True,
            dilation=1,
            with_bn=True,
    ):
        super(conv2DBatchNormRelu, self).__init__()

        conv_mod = nn.Conv2d(int(in_channels),
                             int(n_filters),
                             kernel_size=k_size,
                             padding=padding,
                             stride=stride,
                             bias=bias,
                             dilation=dilation, )

        if with_bn:
            self.cbr_unit = nn.Sequential(conv_mod,
                                          nn.BatchNorm2d(int(n_filters)),
                                          nn.ReLU(inplace=True))
        else:
            self.cbr_unit = nn.Sequential(conv_mod, nn.ReLU(inplace=True))
    def forward(self, inputs):
        outputs = self.cbr_unit(inputs)
        return outputs


class segnetDown2(nn.Module):
    def __init__(self, in_size, out_size):
        super(segnetDown2, self).__init__()
        self.conv1 = conv2DBatchNormRelu(in_size, out_size, 3, 1, 1)
        self.conv2 = conv2DBatchNormRelu(out_size, out_size, 3, 1, 1)
        self.maxpool_with_argmax = nn.MaxPool2d(2, 2, return_indices=True)

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        unpooled_shape = outputs.size()
        outputs, indices = self.maxpool_with_argmax(outputs)
        return outputs, indices, unpooled_shape


class segnetDown3(nn.Module):
    def __init__(self, in_size, out_size):
        super(segnetDown3, self).__init__()
        self.conv1 = conv2DBatchNormRelu(in_size, out_size, 3, 1, 1)
        self.conv2 = conv2DBatchNormRelu(out_size, out_size, 3, 1, 1)
        self.conv3 = conv2DBatchNormRelu(out_size, out_size, 3, 1, 1)
        self.maxpool_with_argmax = nn.MaxPool2d(2, 2, return_indices=True)

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        outputs = self.conv3(outputs)
        unpooled_shape = outputs.size()
        outputs, indices = self.maxpool_with_argmax(outputs)
        return outputs, indices, unpooled_shape


class segnetUp2(nn.Module):
    def __init__(self, in_size, out_size):
        super(segnetUp2, self).__init__()
        self.unpool = nn.MaxUnpool2d(2, 2)
        self.conv1 = conv2DBatchNormRelu(in_size, in_size, 3, 1, 1)
        self.conv2 = conv2DBatchNormRelu(in_size, out_size, 3, 1, 1)

    def forward(self, inputs, indices, output_shape):
        outputs = self.unpool(input=inputs, indices=indices, output_size=output_shape)
        outputs = self.conv1(outputs)
        outputs = self.conv2(outputs)
        return outputs


class segnetUp3(nn.Module):
    def __init__(self, in_size, out_size):
        super(segnetUp3, self).__init__()
        self.unpool = nn.MaxUnpool2d(2, 2)
        self.conv1 = conv2DBatchNormRelu(in_size, in_size, 3, 1, 1)
        self.conv2 = conv2DBatchNormRelu(in_size, in_size, 3, 1, 1)
        self.conv3 = conv2DBatchNormRelu(in_size, out_size, 3, 1, 1)

    def forward(self, inputs, indices, output_shape):
        outputs = self.unpool(input=inputs, indices=indices, output_size=output_shape)
        outputs = self.conv1(outputs)
        outputs = self.conv2(outputs)
        outputs = self.conv3(outputs)
        return outputs


class Cholesky(torch.autograd.Function):
    def forward(ctx, a):
        l = torch.cholesky(a, False)
        ctx.save_for_backward(l)
        return l
    def backward(ctx, grad_output):
        l, = ctx.saved_variables
        linv = l.inverse()
        inner = torch.tril(torch.mm(l.t(), grad_output)) * torch.tril(
            1.0 - Variable(l.data.new(l.size(1)).fill_(0.5).diag()))
        s = torch.mm(linv.t(), torch.mm(inner, linv))
        return s
    

class CustomKDE(Distribution):
    def __init__(self, kernel = 'gaussian', bandwidth = 0.1 ):
        """
        X : tensor (n, d)
          `n` points with `d` dimensions to which KDE will be fit
        bandwidth : numeric
          bandwidth for kernel
        """
        if torch.cuda.is_available():
            self.bandwidth = torch.tensor(bandwidth, requires_grad=True).to(device=torch.device("cuda"))
        else:
            self.bandwidth = torch.tensor(bandwidth, requires_grad=True).to(device=torch.device("cpu"))

    def fit(self, X):
        """
        X : tensor (n, d)
          `n` points with `d` dimensions to which KDE will be fit
        """
        self.X = X
        self.dims = X.shape[-1]
        self.n = X.shape[0]
        if torch.cuda.is_available():
            self.mvn = MultivariateNormal(loc=torch.zeros(self.dims, requires_grad=True).to(device=torch.device("cuda")),
                                        covariance_matrix=torch.eye(self.dims).to(device=torch.device("cuda")))
        else:
            self.mvn = MultivariateNormal(loc=torch.zeros(self.dims, requires_grad=True).to(device=torch.device("cpu")),
                                        covariance_matrix=torch.eye(self.dims).to(device=torch.device("cpu")))

    def sample(self, num_samples):
        idxs = (np.random.uniform(0, 1, num_samples) * self.n).astype(int)
        norm = Normal(loc=self.X[idxs], scale=self.bandwidth)
        return norm.sample()

    def score_samples(self, Y, X=None):
        """Returns the kernel density estimates of each point in `Y`.

        Parameters
        ----------
        Y : tensor (m, d)
          `m` points with `d` dimensions for which the probability density will
          be calculated
        X : tensor (n, d), optional
          `n` points with `d` dimensions to which KDE will be fit. Provided to
          allow batch calculations in `log_prob`. By default, `X` is None and
          all points used to initialize KernelDensityEstimator are included.


        Returns
        -------
        log_probs : tensor (m)
          log probability densities for each of the queried points in `Y`
        """
        if X == None:
            X = self.X
        a = self.bandwidth**(-self.dims) 
        c = to_var(X.unsqueeze(1) - Y)
        d = self.mvn.log_prob( c/ (self.bandwidth))
        b = torch.exp(d)
        log_probs = torch.log((a* b).sum(dim=0) / self.n)

        return log_probs

    def log_prob(self, Y):
        """Returns the total log probability of one or more points, `Y`, using
        a Multivariate Normal kernel fit to `X` and scaled using `bandwidth`.

        Parameters
        ----------
        Y : tensor (m, d)
          `m` points with `d` dimensions for which the probability density will
          be calculated

        Returns
        -------
        log_prob : numeric
          total log probability density for the queried points, `Y`
        """

        X_chunks = self.X.split(1000)
        Y_chunks = Y.split(1000)

        log_prob = 0

        for x in X_chunks:
            for y in Y_chunks:
                log_prob += self.score_samples(y, x).sum(dim=0)

        return log_prob

############################################################################################################################################
############################################################################################################################################
def log_relu(x):
    return torch.log(nn.ReLU(x) + 1)

############################################################################################################################################
############################################################################################################################################
class AE(nn.Module):
    """Residual Block."""
    def __init__(self, latent_dim=8, in_channels=1, out_channels=1):
        super(AE, self).__init__()

        self.is_unpooling = True

        # Encoder1
        self.down1 = segnetDown2(in_channels, 64)
        self.down2 = segnetDown2(64, 128)
        self.down3 = segnetDown3(128, 256)
        self.down4 = segnetDown3(256, 512)
        self.down5 = segnetDown3(512, 512)
        self.down6 = nn.Linear(11264,latent_dim)

        # Decoder
        self.up6 = nn.Linear(latent_dim,11264)
        self.up5 = segnetUp3(512, 512)
        self.up4 = segnetUp3(512, 256)
        self.up3 = segnetUp3(256, 128)
        self.up2 = segnetUp2(128, 64)
        self.up1 = segnetUp2(64, out_channels)


    def forward(self, x, initialize=0, test = 0):

        batch_size = x.shape[0]

        down1, indices_1, unpool_shape1 = self.down1(x)
        down2, indices_2, unpool_shape2 = self.down2(down1)
        down3, indices_3, unpool_shape3 = self.down3(down2)    
        down4, indices_4, unpool_shape4 = self.down4(down3)
        down5, indices_5, unpool_shape5 = self.down5(down4)
        down5 = torch.flatten(down5, 1)
        enc1 = self.down6(down5)

        up6 = self.up6(enc1)
        up6 = torch.reshape(up6, (-1, 512, 11, 2))
        up5 = self.up5(up6, indices_5, unpool_shape5)
        up4 = self.up4(up5, indices_4, unpool_shape4)
        up3 = self.up3(up4, indices_3, unpool_shape3)
        up2 = self.up2(up3, indices_2, unpool_shape2)
        dec = self.up1(up2, indices_1, unpool_shape1)

        # For matrices
        rec_frobenius = LA.matrix_norm(x-dec, ord='fro') # ‖A‖_2 loss 1
        rec_l1 = LA.matrix_norm(x-dec, ord=1)   # max(sum(abs(x), dim=0)) loss 2

        # Score network
        z1 = torch.cat([enc1, rec_frobenius, rec_l1], dim=1)

        return enc1, dec, z1

    def loss_function(self, x, x_hat, lamda_rec_err):

        recon_error = torch.mean((x - x_hat) ** 2)

        loss = lamda_rec_err * recon_error 

        return loss, recon_error




############################################################################################################################################
############################################################################################################################################
def reparametrize(mu, logvar):
    std = logvar.div(2).exp()
    eps = Variable(std.data.new(std.size()).normal_())
    return mu + std*eps

def sample_latent(mu, logvar, num_samples):
    ris = [] 
    for n in range(num_samples):
        ris.append(reparametrize(mu, logvar))
    ris = torch.stack(ris,axis=0)

    return ris

def kaiming_init(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        init.kaiming_normal(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)

    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)

def normal_init(m, mean, std):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        m.weight.data.normal_(mean, std)
        if m.bias.data is not None:
            m.bias.data.zero_()

    elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
        m.weight.data.fill_(1)
        if m.bias.data is not None:
            m.bias.data.zero_()

    def weight_init(self):
        for block in self._modules:
            for m in self._modules[block]:
                kaiming_init(m)

def kl_divergence(mu, logvar):
    batch_size = mu.size(0)
    assert batch_size != 0
    if mu.data.ndimension() == 4:
        mu = mu.view(mu.size(0), mu.size(1))
    if logvar.data.ndimension() == 4:
        logvar = logvar.view(logvar.size(0), logvar.size(1))

    klds = -0.5*(1 + logvar - mu.pow(2) - logvar.exp())
    total_kld = klds.sum(1).mean(0, True)
    dimension_wise_kld = klds.mean(0)
    mean_kld = klds.mean(1).mean(0, True)

    return total_kld, dimension_wise_kld, mean_kld

class BetaVAE_H(nn.Module):
    """Model proposed in original beta-VAE paper(Higgins et al, ICLR, 2017)."""
    def __init__(self, beta = 10**-3, num_samples = 10, latent_dim=8, in_channels=1, out_channels=1):
        super(BetaVAE_H, self).__init__()

        self.z_dim = latent_dim
        self.nc = in_channels
        self.beta = beta
        self.num_samples = num_samples

        self.is_unpooling = True

        # Encoder1
        self.down1 = segnetDown2(in_channels, 64)
        self.down2 = segnetDown2(64, 128)
        self.down3 = segnetDown3(128, 256)
        self.down4 = segnetDown3(256, 512)
        self.down5 = segnetDown3(512, 512)
        self.down6 = nn.Linear(11264,latent_dim*2) # mean and var 

        # Decoder
        self.up6 = nn.Linear(latent_dim,11264)
        self.up5 = segnetUp3(512, 512)
        self.up4 = segnetUp3(512, 256)
        self.up3 = segnetUp3(256, 128)
        self.up2 = segnetUp2(128, 64)
        self.up1 = segnetUp2(64, out_channels)

    def relative_euclidean_distance(self, a, b):
        return (a-b).norm(2, dim=1) / a.norm(2, dim=1)

    def forward(self, x, test = 0):

        batch_size = x.size(0)

        down1, indices_1, unpool_shape1 = self.down1(x)
        down2, indices_2, unpool_shape2 = self.down2(down1)
        down3, indices_3, unpool_shape3 = self.down3(down2)
        down4, indices_4, unpool_shape4 = self.down4(down3)
        down5, indices_5, unpool_shape5 = self.down5(down4)
        down5 = torch.flatten(down5, 1)
        enc1 = self.down6(down5)

        mu = enc1[:, :self.z_dim]
        logvar = enc1[:, self.z_dim:] 
        z1 = reparametrize(mu, logvar)

        up6 = self.up6(z1)
        up6 = torch.reshape(up6, (-1, 512, 11, 2))
        up5 = self.up5(up6, indices_5, unpool_shape5)
        up4 = self.up4(up5, indices_4, unpool_shape4)
        up3 = self.up3(up4, indices_3, unpool_shape3)
        up2 = self.up2(up3, indices_2, unpool_shape2)
        dec = self.up1(up2, indices_1, unpool_shape1)

        if test == 1:
            z1_sampled = sample_latent(mu, logvar, self.num_samples)
            sample_energy1 = [] 
            for n in range(self.num_samples):
                up6 = self.up6(z1_sampled[n,:])
                up6 = torch.reshape(up6, (-1, 512, 11, 2))
                up5 = self.up5(up6, indices_5, unpool_shape5)
                up4 = self.up4(up5, indices_4, unpool_shape4)
                up3 = self.up3(up4, indices_3, unpool_shape3)
                up2 = self.up2(up3, indices_2, unpool_shape2)
                temp_dec = self.up1(up2, indices_1, unpool_shape1)
                sample_energy1.append(((x - temp_dec) ** 2).mean(-1).mean(-1))
            sample_energy1 = torch.stack(sample_energy1, axis = 0)
            sample_energy1 = torch.mean(sample_energy1, axis = 0)

        else:
            sample_energy1 = F.mse_loss(dec, x, size_average=False).div(batch_size)

        return enc1, dec, z1, sample_energy1, mu, logvar

    def loss_function(self, x, x_hat, mu, logvar):

        recon_error = torch.mean((x - x_hat) ** 2)

        total_kld, dim_wise_kld, mean_kld = kl_divergence(mu, logvar)

        loss = recon_error + self.beta*total_kld

        return loss, total_kld, recon_error

