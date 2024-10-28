import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
from torch import linalg as LA
import sys 
import os
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
            stride,
            padding,
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
class DaGMM(nn.Module):
    """Residual Block."""
    def __init__(self, n_gmm = 2, latent_dim=8, in_channels=1, out_channels=1):
        super(DaGMM, self).__init__()

        self.n_gmm = n_gmm

        self.is_unpooling = True

        self.down1 = segnetDown2(in_channels, 64)
        self.down2 = segnetDown2(64, 128)
        self.down3 = segnetDown3(128, 256)
        self.down4 = segnetDown3(256, 512)
        self.down5 = segnetDown3(512, 512)
        self.down6 = nn.Linear(11264,latent_dim)

        self.up6 = nn.Linear(latent_dim,11264)
        self.up5 = segnetUp3(512, 512)
        self.up4 = segnetUp3(512, 256)
        self.up3 = segnetUp3(256, 128)
        self.up2 = segnetUp2(128, 64)
        self.up1 = segnetUp2(64, out_channels)


        layers = []
        layers += [nn.Linear(latent_dim+2,10)] # [nn.Linear(latent_dim+2,10)]
        layers += [nn.Tanh()]        
        layers += [nn.Dropout(p=0.5)]        
        layers += [nn.Linear(10,n_gmm)]
        layers += [nn.Softmax(dim=1)]
        self.estimation = nn.Sequential(*layers)

        self.register_buffer("phi", torch.zeros(n_gmm))
        self.register_buffer("mu", torch.zeros(n_gmm,latent_dim+2))
        self.register_buffer("cov", torch.zeros(n_gmm,latent_dim+2,latent_dim+2))

    def relative_euclidean_distance(self, a, b):
        return (a-b).norm(2, dim=1) / a.norm(2, dim=1)

    def forward(self, x):

        down1, indices_1, unpool_shape1 = self.down1(x)
        down2, indices_2, unpool_shape2 = self.down2(down1)
        down3, indices_3, unpool_shape3 = self.down3(down2)
        down4, indices_4, unpool_shape4 = self.down4(down3)
        down5, indices_5, unpool_shape5 = self.down5(down4)
        down5 = torch.flatten(down5, 1)
        enc = self.down6(down5)

        # torch.Size([-1, 8])
        # print(enc.shape)
        up6 = self.up6(enc)
        up6 = torch.reshape(up6, (-1, 512, 11, 2))
        up5 = self.up5(up6, indices_5, unpool_shape5)
        up4 = self.up4(up5, indices_4, unpool_shape4)
        up3 = self.up3(up4, indices_3, unpool_shape3)
        up2 = self.up2(up3, indices_2, unpool_shape2)
        dec = self.up1(up2, indices_1, unpool_shape1)

        # For matrices
        rec_frobenius = LA.matrix_norm(x-dec, ord='fro') # ‖A‖_2 loss 1
        # rec_l2 = LA.matrix_norm(x-dec, ord=2)   # largest singular value loss 2
        rec_l1 = LA.matrix_norm(x-dec, ord=1)   # max(sum(abs(x), dim=0)) loss 2
        # rec_nuclear = LA.matrix_norm(x-dec, ord='nuc')   # sum of its singular values loss 2

        # With compression network
        # For matrices
        z = torch.cat([enc, rec_frobenius, rec_l1], dim=1)
        # Without compression network
        # z = enc

        gamma = self.estimation(z)

        return enc, dec, z, gamma

    def compute_gmm_params(self, z, gamma):
        N = gamma.size(0)
        # K
        sum_gamma = torch.sum(gamma, dim=0)

        # K
        phi = (sum_gamma / N)

        self.phi = phi.data

 
        # K x D
        mu = torch.sum(gamma.unsqueeze(-1) * z.unsqueeze(1), dim=0) / sum_gamma.unsqueeze(-1)
        self.mu = mu.data
        D = self.mu.size(1)
        # z = N x D
        # mu = K x D
        # gamma N x K

        # OLD  
        # z_mu = N x K x D
        z_mu = (z.unsqueeze(1)- mu.unsqueeze(0))

        # z_mu_outer = N x K x D x D
        z_mu_outer = z_mu.unsqueeze(-1) * z_mu.unsqueeze(-2)

        # K x D x D
        cov = torch.sum(gamma.unsqueeze(-1).unsqueeze(-1) * z_mu_outer, dim = 0) / sum_gamma.unsqueeze(-1).unsqueeze(-1)
        self.cov = cov.data

        # print(f'gamma: {gamma}', f'phi: {phi}', f'mu: {mu}', f'cov: {cov}')
        return phi, mu, cov
        
    def compute_energy(self, z, phi=None, mu=None, cov=None, size_average=True):
        # size_average=True in Training set to average the energy of all samples in the batch
        # size_average=Flase in Testing set

        if phi is None:
            phi = to_var(self.phi)
        if mu is None:
            mu = to_var(self.mu)
        if cov is None:
            cov = to_var(self.cov)

        k, D, _ = cov.size()

        z_mu = (z.unsqueeze(1)- mu.unsqueeze(0))

        cov_inverse = []
        det_cov = []
        cov_diag = 0
        eps = 1e-12
        # Compute covariance inverse, covariance diagonal, determinat of covariance
        for i in range(k):
            # K x D x D
            cov_k = cov[i] + to_var(torch.eye(D)*eps)

            # symmetrize to correct minor numerical errors
            # cov_k = (cov_k + cov_k.t())/2

            # print(f'k: {i}', f'cov_k: {cov_k}\n')
            cov_inverse.append(torch.inverse(cov_k).unsqueeze(0))

            # Compute determinant
            # Method 1
            det_cov.append(np.linalg.det(cov_k.data.cpu().numpy()* (2*np.pi)))
            # Method 2
            # det_cov.append((Cholesky.apply(cov_k.cpu() * (2*np.pi)).diag().prod()).unsqueeze(0))

            # w, v = np.linalg.eig(cov_k.data.cpu().numpy()* (2*np.pi))
            # print(w)

            cov_diag = cov_diag + torch.sum(1 / cov_k.diag())

        # K x D x D
        cov_inverse = torch.cat(cov_inverse, dim=0)
        # K
        # det_cov = to_var(det_cov)
        # det_cov = torch.from_numpy(np.asarray(det_cov))
        # Method 1
        det_cov = to_var(torch.abs(torch.from_numpy(np.float32(np.array(det_cov)))))
        # det_cov = to_var(torch.from_numpy(np.float32(np.array(det_cov))))
        # Method 2
        # if torch.cuda.is_available():
        #     det_cov = torch.cat(det_cov).cuda()
        # else:
        #     det_cov = torch.cat(det_cov)

        # N x K
        exp_term_tmp = -0.5 * torch.sum(torch.sum(z_mu.unsqueeze(-1) * cov_inverse.unsqueeze(0), dim=-2) * z_mu, dim=-1)
        # for stability (logsumexp)
        max_val = torch.max((exp_term_tmp).clamp(min=0), dim=1, keepdim=True)[0]

        exp_term = torch.exp(exp_term_tmp - max_val)

        sample_energy = -max_val.squeeze() - torch.log(torch.sum(phi.unsqueeze(0) * exp_term / (torch.sqrt(det_cov)).unsqueeze(0), dim = 1) + eps)

        if size_average:
            sample_energy = torch.mean(sample_energy)

        return sample_energy, cov_diag


    def loss_function(self, x, x_hat, z, gamma, lambda_energy, lambda_cov_diag):

        recon_error = torch.mean((x - x_hat) ** 2)

        phi, mu, cov = self.compute_gmm_params(z, gamma)

        sample_energy, cov_diag = self.compute_energy(z, phi, mu, cov)

        loss = recon_error + lambda_energy * sample_energy + lambda_cov_diag * cov_diag

        return loss, sample_energy, recon_error, cov_diag



############################################################################################################################################
############################################################################################################################################
class Model1(nn.Module):
    """Residual Block."""
    def __init__(self, n_gmm = 2, latent_dim=8, in_channels=1, out_channels=1):
        super(Model1, self).__init__()

        self.n_gmm = n_gmm

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

        # Encoder2
        self.down1_ = segnetDown2(out_channels, 64)
        self.down2_ = segnetDown2(64, 128)
        self.down3_ = segnetDown3(128, 256)
        self.down4_ = segnetDown3(256, 512)
        self.down5_ = segnetDown3(512, 512)
        self.down6_ = nn.Linear(11264,latent_dim)

        # Estimation network for DAGMM
        layers = []
        layers += [nn.Linear(latent_dim+2,10)] # [nn.Linear(latent_dim+2,10)]
        layers += [nn.Tanh()]        
        layers += [nn.Dropout(p=0.5)]        
        layers += [nn.Linear(10,n_gmm)]
        layers += [nn.Softmax(dim=1)]
        self.estimation = nn.Sequential(*layers)

        # Likelihood network
        layers = []
        layers += [nn.Linear(latent_dim,latent_dim)]
        layers += [nn.Tanh()]        
        layers += [nn.Dropout(p=0.5)]  

        layers += [nn.Linear(latent_dim,5)]
        layers += [nn.Tanh()]        
        layers += [nn.Dropout(p=0.5)]   

        layers += [nn.Linear(5,1)]
        layers += [nn.ReLU()]
        self.likelihood_network = nn.Sequential(*layers)

        self.register_buffer("phi", torch.zeros(n_gmm))
        self.register_buffer("mu", torch.zeros(n_gmm,latent_dim+2))
        self.register_buffer("cov", torch.zeros(n_gmm,latent_dim+2,latent_dim+2))

    def relative_euclidean_distance(self, a, b):
        return (a-b).norm(2, dim=1) / a.norm(2, dim=1)

    def forward(self, x):

        down1, indices_1, unpool_shape1 = self.down1(x)
        down2, indices_2, unpool_shape2 = self.down2(down1)
        down3, indices_3, unpool_shape3 = self.down3(down2)
        down4, indices_4, unpool_shape4 = self.down4(down3)
        down5, indices_5, unpool_shape5 = self.down5(down4)
        down5 = torch.flatten(down5, 1)
        enc1 = self.down6(down5)

        # torch.Size([-1, 8])
        # print(enc.shape)
        up6 = self.up6(enc1)
        up6 = torch.reshape(up6, (-1, 512, 11, 2))
        up5 = self.up5(up6, indices_5, unpool_shape5)
        up4 = self.up4(up5, indices_4, unpool_shape4)
        up3 = self.up3(up4, indices_3, unpool_shape3)
        up2 = self.up2(up3, indices_2, unpool_shape2)
        dec = self.up1(up2, indices_1, unpool_shape1)

        down1_, indices_1_, unpool_shape1_ = self.down1_(dec)
        down2_, indices_2_, unpool_shape2_ = self.down2_(down1_)
        down3_, indices_3_, unpool_shape3_ = self.down3_(down2_)
        down4_, indices_4_, unpool_shape4_ = self.down4_(down3_)
        down5_, indices_5_, unpool_shape_5 = self.down5_(down4_)
        down5_ = torch.flatten(down5_, 1)
        enc2 = self.down6_(down5_)

        # For matrices
        rec_frobenius = LA.matrix_norm(x-dec, ord='fro') # ‖A‖_2 loss 1
        # rec_l2 = LA.matrix_norm(x-dec, ord=2)   # largest singular value loss 2
        rec_l1 = LA.matrix_norm(x-dec, ord=1)   # max(sum(abs(x), dim=0)) loss 2
        # rec_nuclear = LA.matrix_norm(x-dec, ord='nuc')   # sum of its singular values loss 2

        # Score network
        z1 = enc1
        likelihood1 = self.likelihood_network(enc1)

        # With compression network
        # For matrices
        z2 = torch.cat([enc2, rec_frobenius, rec_l1], dim=1)

        # Without compression network
        # z = enc

        gamma = self.estimation(z2)

        return enc1, enc2, dec, z1, z2, gamma, likelihood1

    def compute_gmm_params(self, z, gamma):
        N = gamma.size(0)
        # K
        sum_gamma = torch.sum(gamma, dim=0)

        # K
        phi = (sum_gamma / N)

        self.phi = phi.data

        # K x D
        mu = torch.sum(gamma.unsqueeze(-1) * z.unsqueeze(1), dim=0) / sum_gamma.unsqueeze(-1)
        self.mu = mu.data
        D = self.mu.size(1)
        # z = N x D
        # mu = K x D
        # gamma N x K

        # OLD  
        # z_mu = N x K x D
        z_mu = (z.unsqueeze(1)- mu.unsqueeze(0))

        # z_mu_outer = N x K x D x D
        z_mu_outer = z_mu.unsqueeze(-1) * z_mu.unsqueeze(-2)

        # K x D x D
        cov = torch.sum(gamma.unsqueeze(-1).unsqueeze(-1) * z_mu_outer, dim = 0) / sum_gamma.unsqueeze(-1).unsqueeze(-1)
        self.cov = cov.data

        # print(f'gamma: {gamma}', f'phi: {phi}', f'mu: {mu}', f'cov: {cov}')
        return phi, mu, cov
        
    def compute_energy2(self, z, phi=None, mu=None, cov=None, size_average=True):
        if phi is None:
            phi = to_var(self.phi)
        if mu is None:
            mu = to_var(self.mu)
        if cov is None:
            cov = to_var(self.cov)

        k, D, _ = cov.size()

        z_mu = (z.unsqueeze(1)- mu.unsqueeze(0))

        cov_inverse = []
        det_cov = []
        cov_diag = 0
        eps = 1e-12
        # Compute covariance inverse, covariance diagonal, determinat of covariance
        for i in range(k):
            # K x D x D
            cov_k = cov[i] + to_var(torch.eye(D)*eps)

            # symmetrize to correct minor numerical errors
            # cov_k = (cov_k + cov_k.t())/2

            # print(f'k: {i}', f'cov_k: {cov_k}\n')
            cov_inverse.append(torch.inverse(cov_k).unsqueeze(0))

            # Compute determinant
            # print(f'det1: {np.linalg.det(cov_k.data.cpu().numpy()* (2*np.pi))}')
            # print(f'det2: {Cholesky.apply(cov_k.cpu() * (2*np.pi))}')
            # Method 1
            det_cov.append(np.linalg.det(cov_k.data.cpu().numpy()* (2*np.pi)))
            # Method 2
            # det_cov.append((Cholesky.apply(cov_k.cpu() * (2*np.pi)).diag().prod()).unsqueeze(0))

            # w, v = np.linalg.eig(cov_k.data.cpu().numpy()* (2*np.pi))
            # print(w)

            cov_diag = cov_diag + torch.sum(1 / cov_k.diag())

        # K x D x D
        cov_inverse = torch.cat(cov_inverse, dim=0)
        # K
        # det_cov = to_var(det_cov)
        # det_cov = torch.from_numpy(np.asarray(det_cov))
        # Method 1
        det_cov = to_var(torch.abs(torch.from_numpy(np.float32(np.array(det_cov)))))
        # det_cov = to_var(torch.from_numpy(np.float32(np.array(det_cov))))
        # Method 2
        # if torch.cuda.is_available():
        #     det_cov = torch.cat(det_cov).cuda()
        # else:
        #     det_cov = torch.cat(det_cov)

        # N x K
        exp_term_tmp = -0.5 * torch.sum(torch.sum(z_mu.unsqueeze(-1) * cov_inverse.unsqueeze(0), dim=-2) * z_mu, dim=-1)
        # for stability (logsumexp)
        max_val = torch.max((exp_term_tmp).clamp(min=0), dim=1, keepdim=True)[0]

        exp_term = torch.exp(exp_term_tmp - max_val)

        sample_energy = -max_val.squeeze() - torch.log(torch.sum(phi.unsqueeze(0) * exp_term / (torch.sqrt(det_cov)).unsqueeze(0), dim = 1) + eps)

        if size_average:
            sample_energy = torch.mean(sample_energy)

        return sample_energy, cov_diag


    def loss_function(self, x, x_hat, likelihood1, z2, gamma, lambda_energy, lambda_cov_diag, lamda_diffenergy):

        recon_error = torch.mean((x - x_hat) ** 2)

        phi, mu, cov = self.compute_gmm_params(z2, gamma)

        sample_energy2, cov_diag = self.compute_energy2(z2, phi, mu, cov)

        score_error = torch.mean((-torch.log(likelihood1) - sample_energy2) ** 2)

        loss = recon_error + \
               lambda_energy * (-torch.log(likelihood1)+sample_energy2) + \
               lamda_diffenergy * score_error + \
               lambda_cov_diag * cov_diag

        return loss, score_error, recon_error, cov_diag



############################################################################################################################################
############################################################################################################################################
class Model2(nn.Module):
    """Residual Block."""
    def __init__(self, kernel = 'epanechnikov', bandwidth = 0.1, latent_dim=8, in_channels=1, out_channels=1):
        super(Model2, self).__init__()

        self.kernel = kernel
        self.bandwidth = bandwidth

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

        # Encoder2
        self.down1_ = segnetDown2(out_channels, 64)
        self.down2_ = segnetDown2(64, 128)
        self.down3_ = segnetDown3(128, 256)
        self.down4_ = segnetDown3(256, 512)
        self.down5_ = segnetDown3(512, 512)
        self.down6_ = nn.Linear(11264,latent_dim)

        # Likelihood network
        layers = []
        layers += [nn.Linear(latent_dim,latent_dim)]
        layers += [nn.Tanh()]        
        layers += [nn.Dropout(p=0.5)]  

        layers += [nn.Linear(latent_dim,5)]
        layers += [nn.Tanh()]        
        layers += [nn.Dropout(p=0.5)]   

        layers += [nn.Linear(5,1)]
        layers += [nn.ReLU()]
        self.likelihood_network = nn.Sequential(*layers)

        # KDE
        # self.kd = KernelDensity(kernel = kernel, bandwidth = bandwidth)
        self.kd = CustomKDE(kernel = kernel, bandwidth = bandwidth)

    def relative_euclidean_distance(self, a, b):
        return (a-b).norm(2, dim=1) / a.norm(2, dim=1)

    def forward(self, x, initialize=0):

        # torch.Size([-1, 1, 352, 64])
        # print(x.shape)
        # print(f'x: {x}')

        down1, indices_1, unpool_shape1 = self.down1(x)
        down2, indices_2, unpool_shape2 = self.down2(down1)
        down3, indices_3, unpool_shape3 = self.down3(down2)
        down4, indices_4, unpool_shape4 = self.down4(down3)
        down5, indices_5, unpool_shape5 = self.down5(down4)
        # torch.Size([-1, 512, 11, 2])
        # print(down5.shape)
        down5 = torch.flatten(down5, 1)
        enc1 = self.down6(down5)

        # torch.Size([-1, 8])
        # print(enc.shape)
        up6 = self.up6(enc1)
        up6 = torch.reshape(up6, (-1, 512, 11, 2))
        up5 = self.up5(up6, indices_5, unpool_shape5)
        up4 = self.up4(up5, indices_4, unpool_shape4)
        up3 = self.up3(up4, indices_3, unpool_shape3)
        up2 = self.up2(up3, indices_2, unpool_shape2)
        dec = self.up1(up2, indices_1, unpool_shape1)

        down1_, indices_1_, unpool_shape1_ = self.down1_(dec)
        down2_, indices_2_, unpool_shape2_ = self.down2_(down1_)
        down3_, indices_3_, unpool_shape3_ = self.down3_(down2_)
        down4_, indices_4_, unpool_shape4_ = self.down4_(down3_)
        down5_, indices_5_, unpool_shape_5 = self.down5_(down4_)
        # torch.Size([-1, 512, 11, 2])
        # print(down5.shape)
        down5_ = torch.flatten(down5_, 1)
        enc2 = self.down6_(down5_)

        # For matrices
        rec_frobenius = LA.matrix_norm(x-dec, ord='fro') # ‖A‖_2 loss 1
        rec_l1 = LA.matrix_norm(x-dec, ord=1)   # max(sum(abs(x), dim=0)) loss 2

        # Score network
        z1 = enc1
        likelihood1 = self.likelihood_network(enc1)

        # With compression network
        # For matrices
        z2 = torch.cat([enc2, rec_frobenius, rec_l1], dim=1)

        if initialize:
            # self.kd.fit(z2.data.cpu().numpy())
            self.kd.fit(to_var(z2.requires_grad_()))

        sample_energy2 = -self.kd.score_samples(to_var(z2.requires_grad_()))

        return enc1, enc2, dec, z1, z2, sample_energy2, likelihood1


    def loss_function(self, x, x_hat, likelihood1, z2, sample_energy2, lambda_energy, lamda_diffenergy):

        recon_error = torch.mean((x - x_hat) ** 2)

        kl_loss = torch.nn.KLDivLoss(size_average=None, reduce=None, reduction='batchmean', log_target=True)
        score_error = kl_loss(torch.log(likelihood1), -sample_energy2)

        loss = recon_error + \
                 lambda_energy * (- sample_energy2) +\
              lamda_diffenergy * score_error  \


        return loss, score_error, recon_error



############################################################################################################################################
############################################################################################################################################
def log_relu(x):
    return torch.log(nn.ReLU(x) + 1)

class Model3(nn.Module):
    """Residual Block."""
    def __init__(self, kernel = 'epanechnikov', bandwidth = 0.1, latent_dim=8, in_channels=1, out_channels=1):
        super(Model3, self).__init__()

        self.kernel = kernel
        self.bandwidth = bandwidth

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


        # Likelihood network
        layers = []
        layers += [nn.Linear(latent_dim,latent_dim)]
        layers += [nn.Tanh()]  
        layers += [nn.Dropout(p=0.5)]  

        layers += [nn.Linear(latent_dim,5)]
        layers += [nn.Tanh()] 
        layers += [nn.Dropout(p=0.5)]   

        layers += [nn.Linear(5,1)] 
        layers += [nn.BatchNorm1d(int(1))] 
        layers += [nn.ReLU()]
        self.likelihood_network = nn.Sequential(*layers)

        # KDE
        self.kd = CustomKDE(kernel = kernel, bandwidth = bandwidth)

    def relative_euclidean_distance(self, a, b):
        return (a-b).norm(2, dim=1) / a.norm(2, dim=1)

    def forward(self, x, initialize=0, test = 0):

        batch_size = x.shape[0]

        down1, indices_1, unpool_shape1 = self.down1(x)
        down2, indices_2, unpool_shape2 = self.down2(down1)
        down3, indices_3, unpool_shape3 = self.down3(down2)    
        down4, indices_4, unpool_shape4 = self.down4(down3)
        down5, indices_5, unpool_shape5 = self.down5(down4)
        down5 = torch.flatten(down5, 1)
        enc1 = self.down6(down5)

        if test == 0:
            up6 = self.up6(enc1)
            up6 = torch.reshape(up6, (-1, 512, 11, 2))
            up5 = self.up5(up6, indices_5, unpool_shape5)
            up4 = self.up4(up5, indices_4, unpool_shape4)
            up3 = self.up3(up4, indices_3, unpool_shape3)
            up2 = self.up2(up3, indices_2, unpool_shape2)
            dec = self.up1(up2, indices_1, unpool_shape1)


        # With compression network
        # For matrices
        z2 = enc1
        # Score network
        z1 = enc1
        likelihood1 = self.likelihood_network(z2)
        likelihood1 = likelihood1 + to_var(torch.tensor(1))

        if test == 0:
            if initialize:
                self.kd.fit(to_var(z2.requires_grad_()))
            sample_energy2 = -self.kd.score_samples(to_var(z2.requires_grad_())).requires_grad_()
        else:
            dec = None
            sample_energy2 = None

        return enc1, dec, z1, z2, sample_energy2,-torch.log(likelihood1)


    def loss_function(self, x, x_hat, sample_energy1, z2, sample_energy2, lambda_energy, lamda_diffenergy, lamda_rec_err):

        recon_error = torch.mean((x - x_hat) ** 2)

        kl_loss = torch.nn.KLDivLoss(size_average=None, reduce=None, reduction='batchmean', log_target=True)
        score_error = kl_loss(-sample_energy1, -sample_energy2)
        

        loss = lamda_rec_err * recon_error + \
            lamda_diffenergy * score_error + \
            lambda_energy * (torch.mean(sample_energy2))

        return loss, score_error, recon_error


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
    # print(f'ris:{ris.shape}')

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
            # print(f'z1_sampled:{z1_sampled.shape}')
            sample_energy1 = [] 
            for n in range(self.num_samples):
                up6 = self.up6(z1_sampled[n,:])
                up6 = torch.reshape(up6, (-1, 512, 11, 2))
                up5 = self.up5(up6, indices_5, unpool_shape5)
                up4 = self.up4(up5, indices_4, unpool_shape4)
                up3 = self.up3(up4, indices_3, unpool_shape3)
                up2 = self.up2(up3, indices_2, unpool_shape2)
                temp_dec = self.up1(up2, indices_1, unpool_shape1)
                # print(f'temp_dec:{temp_dec.shape}')
                sample_energy1.append(((x - temp_dec) ** 2).mean(-1).mean(-1))
            sample_energy1 = torch.stack(sample_energy1, axis = 0)
            # print(f'sample_energy1:{sample_energy1.shape}')
            sample_energy1 = torch.mean(sample_energy1, axis = 0)
            # print(f'sample_energy1:{sample_energy1.shape}')

        else:
            sample_energy1 = F.mse_loss(dec, x, size_average=False).div(batch_size)

        return enc1, dec, z1, sample_energy1, mu, logvar

    def loss_function(self, x, x_hat, mu, logvar):

        recon_error = torch.mean((x - x_hat) ** 2)

        total_kld, dim_wise_kld, mean_kld = kl_divergence(mu, logvar)

        loss = recon_error + self.beta*total_kld

        return loss, total_kld, recon_error


############################################################################################################################################
############################################################################################################################################
class GANomaly(nn.Module):
    def __init__(self, wadv = 1, wcon = 50, wenc = 1, latent_dim=8, in_channels=1, out_channels=1):
        super(GANomaly, self).__init__()

        self.z_dim = latent_dim
        self.nc = in_channels
        self.wadv = wadv   #|f(x)-f(x_hat)|_2
        self.wcon = wcon   #|x-x_hat|_1
        self.wenc = wenc   #|z1-z2|_2

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

        # Encoder2
        self.down1_ = segnetDown2(out_channels, 64)
        self.down2_ = segnetDown2(64, 128)
        self.down3_ = segnetDown3(128, 256)
        self.down4_ = segnetDown3(256, 512)
        self.down5_ = segnetDown3(512, 512)
        self.down6_ = nn.Linear(11264,latent_dim)

        # Discriminator 
        self.down1_dis = segnetDown2(out_channels, 64)
        self.down2_dis = segnetDown2(64, 128)
        self.down3_dis = segnetDown3(128, 256)
        self.down4_dis = segnetDown3(256, 512)
        self.down5_dis = segnetDown3(512, 512)
        self.down6_dis = nn.Linear(11264,latent_dim)

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

        up6 = self.up6(enc1)
        up6 = torch.reshape(up6, (-1, 512, 11, 2))
        up5 = self.up5(up6, indices_5, unpool_shape5)
        up4 = self.up4(up5, indices_4, unpool_shape4)
        up3 = self.up3(up4, indices_3, unpool_shape3)
        up2 = self.up2(up3, indices_2, unpool_shape2)
        dec = self.up1(up2, indices_1, unpool_shape1)

        down1_, indices_1_, unpool_shape1_ = self.down1_(dec)
        down2_, indices_2_, unpool_shape2_ = self.down2_(down1_)
        down3_, indices_3_, unpool_shape3_ = self.down3_(down2_)
        down4_, indices_4_, unpool_shape4_ = self.down4_(down3_)
        down5_, indices_5_, unpool_shape_5 = self.down5_(down4_)
        down5_ = torch.flatten(down5_, 1)
        enc2 = self.down6_(down5_)


        # Discrinator x
        down1_dis_x, indices_1_, unpool_shape1_ = self.down1_dis(x)
        down2_dis_x, indices_2_, unpool_shape2_ = self.down2_dis(down1_dis_x)
        down3_dis_x, indices_3_, unpool_shape3_ = self.down3_dis(down2_dis_x)
        down4_dis_x, indices_4_, unpool_shape4_ = self.down4_dis(down3_dis_x)
        down5_dis_x, indices_5_, unpool_shape_5 = self.down5_dis(down4_dis_x)
        down5_dis_x = torch.flatten(down5_dis_x, 1)
        dis_x = self.down6_dis(down5_dis_x)

        # Discrinator dec
        down1_dis_dec, indices_1_, unpool_shape1_ = self.down1_dis(dec)
        down2_dis_dec, indices_2_, unpool_shape2_ = self.down2_dis(down1_dis_dec)
        down3_dis_dec, indices_3_, unpool_shape3_ = self.down3_dis(down2_dis_dec)
        down4_dis_dec, indices_4_, unpool_shape4_ = self.down4_dis(down3_dis_dec)
        down5_dis_dec, indices_5_, unpool_shape_5 = self.down5_dis(down4_dis_dec)
        # torch.Size([-1, 512, 11, 2])
        # print(down5.shape)
        down5_dis_dec = torch.flatten(down5_dis_dec, 1)
        dis_dec = self.down6_dis(down5_dis_dec)

        z1 = enc1
        z2 = enc2

        if test:
            # l1 
            # sample_energy2 = torch.mean( torch.abs(z1-z2), 1).reshape((-1, 1))
            # l2
            sample_energy2 = torch.mean((z1 - z2) ** 2, 1).reshape((-1, 1))
        else:
            # l1 
            # l1 = nn.L1Loss()
            # sample_energy2 = l1(z1, z2)
            # l2
            l2 = nn.MSELoss()
            sample_energy2 = l2(z1, z2)
        # print(f'z1.shape:{z1.shape}')
        # print(f'sample_energy2.shape:{sample_energy2.shape}')

        return z1, z2, dec, dis_x, dis_dec, sample_energy2

    def loss_function(self, x, x_hat, z1, z2, dis_x, dis_dec):

        l1 = nn.L1Loss()
        recon_error = l1(x, x_hat)

        z_err = torch.mean((z1 - z2) ** 2)

        l_adv = torch.mean((dis_x - dis_dec) ** 2)

        loss = self.wcon * recon_error + \
                self.wenc * z_err + \
                self.wadv * l_adv 

        return loss, z_err, recon_error, l_adv
    
############################################################################################################################################
############################################################################################################################################
class CNN_AlexNet(nn.Module):
    def __init__(self, num_classes):
        super(CNN_AlexNet, self).__init__()
        
        self.conv_layers = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=2, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 128, kernel_size=1),
            nn.ReLU(inplace=True),
        )

        self.fc_layers = nn.Sequential(
            nn.Dropout(),
            nn.Linear(128, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = torch.flatten(x, 1)
        x = self.fc_layers(x)
        return x





