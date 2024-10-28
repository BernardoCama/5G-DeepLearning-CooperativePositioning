import torch
torch.autograd.set_detect_anomaly(True)
torch.set_printoptions(sci_mode=True)
import numpy as np
import os
import time
import datetime
from torch.autograd import Variable
import sys
from sklearn.neighbors import KernelDensity
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support as prf, accuracy_score

# Directories
CLASSES_DIR = os.path.split(os.path.split(os.path.split(os.path.abspath(__file__))[0])[0])[0]
DIR_EXPERIMENT = os.path.dirname(os.path.abspath(__file__))
cwd = DIR_EXPERIMENT
# print(os.path.abspath(__file__))
# print(os.path.split(os.path.split(os.path.abspath(__file__))[0])[0])
# print(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(CLASSES_DIR))
sys.path.append(os.path.dirname(os.path.split(os.path.split(os.path.abspath(__file__))[0])[0]))

from Classes.model import DaGMM, Model1, Model2, Model3, BetaVAE_H, GANomaly, AE
import IPython
from tqdm import tqdm



############################################################################################################################################
############################################################################################################################################
class Solver(object):
    DEFAULTS = {}   
    def __init__(self, config):
        # Data loader
        self.__dict__.update(Solver.DEFAULTS, **config)

        # Build tensorboard if use
        self.build_model()
        if self.use_tensorboard:
            self.build_tensorboard()

        # Start with trained model
        if self.pretrained_model:
            self.load_pretrained_model()

    def build_model(self):

        # Define model
        if self.type_model == 'DaGMM':
            self.model = DaGMM(self.gmm_k, self.latent_dim)
        elif self.type_model == 'Model1':
            self.model = Model1(self.gmm_k, self.latent_dim)
        elif self.type_model == 'Model2':
            self.model = Model2(self.kernel, self.bandwidth)
        elif self.type_model == 'Model3':
            self.model = Model3(self.kernel, self.bandwidth)
        elif self.type_model == 'BetaVAE_H':
            self.model = BetaVAE_H(beta = self.beta, num_samples = self.num_samples, latent_dim = self.latent_dim)
        elif self.type_model == 'GANomaly':
            self.model = GANomaly(wadv = self.wadv, wcon = self.wcon, wenc = self.wenc, latent_dim = self.latent_dim)
        elif self.type_model == 'AEGMM' or self.type_model == 'AEKDE':
            self.model = AE(self.latent_dim)

        # Optimizers
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        # Print networks
        if self.to_print_network:
            self.print_network(self.model, self.type_model)

        if torch.cuda.is_available():
            self.model.cuda()

    def print_network(self, model, name):
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        print(name)
        print(model)
        print("The number of parameters: {}".format(num_params))

    def load_pretrained_model(self):
        self.model.load_state_dict(torch.load(os.path.join(
            self.model_save_path, '{}_model.pth'.format(self.pretrained_model))))

        if self.type_model == 'DaGMM' or self.type_model == 'Model1':
            print("phi", self.model.phi,"mu",self.model.mu, "cov",self.model.cov)

        print('loaded trained models (step: {})..!'.format(self.pretrained_model))

    def build_tensorboard(self):
        from Classes.logger import Logger
        self.logger = Logger(self.log_path)

    def reset_grad(self):
        self.model.zero_grad()

    def to_var(self, x, volatile=False):
        if torch.cuda.is_available():
            x = x.cuda()
        return Variable(x, volatile=volatile)

    def train(self, train_data_loader):
        self.train_data_loader = train_data_loader
        iters_per_epoch = len(self.train_data_loader)

        # Start with trained model if exists
        if self.pretrained_model:
            start = int(self.pretrained_model.split('_')[0])
        else:
            start = 0

        # Start training
        iter_ctr = 0
        start_time = time.time()

        for e in range(start, self.num_epochs):
            for i, (input_data, labels) in enumerate(tqdm(self.train_data_loader)):
                iter_ctr += 1
                start = time.time()

                input_data = self.to_var(input_data)

                return_model_step =  self.model_step(input_data, iter_ctr)

                # Logging
                loss = {}
                for k,v in return_model_step.items():
                    loss[k] = v.data.item()

                # Print out log info
                if (i+1) % self.log_step == 0:
                    elapsed = time.time() - start_time
                    total_time = ((self.num_epochs*iters_per_epoch)-(e*iters_per_epoch+i)) * elapsed/(e*iters_per_epoch+i+1)
                    epoch_time = (iters_per_epoch-i)* elapsed/(e*iters_per_epoch+i+1)
                    
                    epoch_time = str(datetime.timedelta(seconds=epoch_time))
                    total_time = str(datetime.timedelta(seconds=total_time))
                    elapsed = str(datetime.timedelta(seconds=elapsed))

                    lr_tmp = []
                    for param_group in self.optimizer.param_groups:
                        lr_tmp.append(param_group['lr'])
                    tmplr = np.squeeze(np.array(lr_tmp))

                    log = "Elapsed {}/{} -- {} , Epoch [{}/{}], Iter [{}/{}], lr {}".format(
                        elapsed,epoch_time,total_time, e+1, self.num_epochs, i+1, iters_per_epoch, tmplr)

                    for tag, value in loss.items():
                        log += ", {}: {:.4f}".format(tag, value)

                    IPython.display.clear_output()
                    print(log)

                    columns_subplot = np.ceil(len(loss.keys())/2)

                    if self.use_tensorboard:
                        for tag, value in loss.items():
                            self.logger.scalar_summary(tag, value, e * iters_per_epoch + i + 1)
                    else:
                        fig = plt.figure(figsize=(10, 10))
                        plt_ctr = 1
                        if not hasattr(self,"loss_logs"):
                            self.loss_logs = {}
                            for loss_key in loss:
                                self.loss_logs[loss_key] = [loss[loss_key]]
                                plt.subplot(2,columns_subplot,plt_ctr)
                                plt.plot(np.array(self.loss_logs[loss_key]), label=loss_key)
                                plt.legend(fontsize = 'xx-small')
                                plt_ctr += 1
                        else:
                            for loss_key in loss:
                                self.loss_logs[loss_key].append(loss[loss_key])
                                plt.subplot(2,columns_subplot,plt_ctr)
                                plt.plot(np.array(self.loss_logs[loss_key]), label=loss_key)
                                plt.legend(fontsize = 'xx-small')
                                plt_ctr += 1

                        if self.save_training_info:
                            plt.savefig(self.log_path + '/training_epochs.pdf', bbox_inches='tight')
                            plt.savefig(self.log_path + '/training_epochs.eps', format='eps', bbox_inches='tight')
                            np.save(self.log_path + '/log_loss.npy', self.loss_logs, allow_pickle = True)
                            
                        if self.plt_show:
                            plt.show()

                    if self.type_model == 'DaGMM' or self.type_model == 'Model1':
                        if self.to_print_network:
                            print("phi", self.model.phi,"mu",self.model.mu, "cov",self.model.cov)

                # Save model checkpoints
                if (i+1) % self.model_save_step == 0:
                    torch.save(self.model.state_dict(),
                        os.path.join(self.model_save_path, '{}_{}_model.pth'.format(e+1, i+1)))


    def compute_gmm_params_after_train(self, train_data_loader):

        self.model.eval()

        with torch.no_grad():
            # Incrementally compute for the GMM parameters across all training data for a better estimate 
            N = 0
            mu_sum = 0
            cov_sum = 0
            gamma_sum = 0

            for it, (input_data, labels) in enumerate(self.train_data_loader):
                input_data = self.to_var(input_data)
                enc, dec, z, gamma = self.model(input_data)
                phi, mu, cov = self.model.compute_gmm_params(z, gamma)
                
                batch_gamma_sum = torch.sum(gamma, dim=0)
                
                gamma_sum += batch_gamma_sum
                mu_sum += mu * batch_gamma_sum.unsqueeze(-1) # keep sums of the numerator only
                cov_sum += cov * batch_gamma_sum.unsqueeze(-1).unsqueeze(-1) # keep sums of the numerator only
                
                N += input_data.size(0)
                
            self.train_phi = gamma_sum / N
            self.train_mu = mu_sum / gamma_sum.unsqueeze(-1)
            self.train_cov = cov_sum / gamma_sum.unsqueeze(-1).unsqueeze(-1)

            print("N:",N)
            print("phi :\n",self.train_phi)
            print("mu :\n",self.train_mu)
            print("cov :\n",self.train_cov)

            self.train_z1 = []  
            self.train_energy1 = []
            self.train_labels = []
            for it, (input_data, labels) in enumerate(self.train_data_loader):
                input_data = self.to_var(input_data)
                enc, dec, z, gamma = self.model(input_data)
                sample_energy, cov_diag = self.model.compute_energy(z, phi=self.train_phi, mu=self.train_mu, cov=self.train_cov, size_average=False)

                self.train_z1.append(z.data.cpu().numpy())    
                self.train_energy1.append(sample_energy.data.cpu().numpy())
                self.train_labels.append(labels.numpy())

            self.train_z1 = np.concatenate(self.train_z1,axis=0)
            self.train_energy1 = np.concatenate(self.train_energy1,axis=0)
            self.train_labels = np.concatenate(self.train_labels,axis=0)  

    def model_step(self, input_data, i = None):

        self.model.train()

        if self.type_model == 'DaGMM':
            enc, dec, z, gamma = self.model(input_data)
            total_loss, sample_energy, recon_error, cov_diag = self.model.loss_function(input_data, dec, z, gamma, self.lambda_energy, self.lambda_cov_diag)
       
        elif self.type_model == 'Model1':
            enc1, enc2, dec, z1, z2, gamma, likelihood1 = self.model(input_data) 
            likelihood1 = torch.mean(likelihood1)
            total_loss, score_error, recon_error, cov_diag = self.model.loss_function(input_data, dec, likelihood1, z2, gamma, self.lambda_energy, self.lambda_cov_diag, self.lamda_diffenergy)
       
        elif self.type_model == 'Model2':
            # Forward
            enc1, enc2, dec, z1, z2, sample_energy2, likelihood1 = self.model(input_data, initialize =[1 if i == 1 else 0]) 
            # Fit KDE with batch
            self.model.kd.fit(z2.requires_grad_())
            likelihood1 = torch.mean(likelihood1)
            sample_energy2 = torch.mean(sample_energy2)
            total_loss, score_error, recon_error = self.model.loss_function(input_data, dec, likelihood1, z2, sample_energy2, self.lambda_energy, self.lamda_diffenergy)
        
        elif self.type_model == 'Model3':
            # Forward
            enc1, dec, z1, z2, sample_energy2, sample_energy1 = self.model(input_data, initialize =[1 if i == 1 else 0]) 
            sample_energy2 = sample_energy2.reshape([-1, 1]) 
            total_loss, score_error, recon_error = self.model.loss_function(input_data, dec, sample_energy1, z2, sample_energy2, self.lambda_energy, self.lamda_diffenergy, self.lamda_rec_err)
            sample_energy1 = torch.mean(sample_energy1)
            # likelihood1 = torch.mean(likelihood1)
            sample_energy2 = torch.mean(sample_energy2)

        elif self.type_model == 'BetaVAE_H':
            enc1, dec, z1, sample_energy1, mu, logvar = self.model(input_data) 
            # likelihood1 = torch.mean(likelihood1)
            total_loss, total_kld, recon_error = self.model.loss_function(input_data, dec, mu, logvar)

        elif self.type_model == 'GANomaly':
            z1, z2, dec, dis_x, dis_dec, sample_energy2 = self.model(input_data) 
            total_loss, z_err, recon_error, l_adv  = self.model.loss_function(input_data, dec, z1, z2, dis_x, dis_dec)

        elif self.type_model == 'AEGMM' or self.type_model == 'AEKDE':
            enc1, dec, z1 = self.model(input_data) 
            total_loss, recon_error = self.model.loss_function(input_data, dec, self.lamda_rec_err)

        self.reset_grad()
        total_loss.backward()

        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5)
        self.optimizer.step()

        if self.type_model == 'DaGMM':
            return {'total_loss':total_loss, 'sample_energy':sample_energy, 'recon_error':recon_error, 'cov_diag':cov_diag}

        elif self.type_model == 'Model1':
            return {'total_loss':total_loss, 'likelihood1':likelihood1, 'score_error':score_error, 'recon_error':recon_error, 'cov_diag':cov_diag}

        elif self.type_model == 'Model2':
            return {'total_loss':total_loss, 'likelihood1':likelihood1, 'score_error':score_error, 'recon_error':recon_error, 'sample_energy2':sample_energy2}

        elif self.type_model == 'Model3':
            # Fit KDE with batch
            # self.model.kd.fit(z2.data.cpu().numpy())
            self.model.kd.fit(z2.requires_grad_(True))
            return {'total_loss':total_loss, 'sample_energy1':sample_energy1, 'score_error':score_error, 'recon_error':recon_error, 'sample_energy2':sample_energy2}
        
        elif self.type_model == 'BetaVAE_H':
            return {'total_loss':total_loss, 'total_kld':total_kld, 'recon_error':recon_error, 'sample_energy1':sample_energy1}

        elif self.type_model == 'GANomaly':
            return {'total_loss':total_loss, 'z_err':z_err, 'recon_error':recon_error, 'l_adv':l_adv, 'sample_energy2':sample_energy2}

        elif self.type_model == 'AEGMM' or self.type_model == 'AEKDE':
            return {'total_loss':total_loss, 'recon_error':recon_error}

    def compute_threshold (self, train_data_loader, fpr):

        # compute threshold from normal data (train_data_loader) given a 
        # specific FPR (false positive rate)
        self.model.eval()

        with torch.no_grad():

            self.train_enc1 = []            
            self.train_z1 = []
            self.train_z2 = []
            self.train_likelihood1 = []
            self.train_energy1 = []
            self.train_energy2 = []
            self.train_dis_x = []
            self.train_dis_dec = []
            for it, (input_data, labels) in enumerate(train_data_loader):
                input_data = self.to_var(input_data)

                if self.type_model == 'DaGMM':
                    enc1, dec, z1, gamma = self.model(input_data)
                    sample_energy1, cov_diag = self.model.compute_energy(z1, size_average=False)
                    self.train_z1.append(z1.data.cpu().numpy())
                    self.train_energy1.append(sample_energy1.data.cpu().numpy())

                elif self.type_model == 'Model1':
                    enc1, enc2, dec, z1, z2, sample_energy2, likelihood1 = self.model(input_data, initialize = 1) 
                    self.train_z1.append(z1.data.cpu().numpy())
                    self.train_z2.append(z2.data.cpu().numpy())
                    self.train_likelihood1.append(likelihood1.data.cpu().numpy())
                    self.train_energy2.append(sample_energy2.data.cpu().numpy())

                elif self.type_model == 'Model2':
                    enc1, enc2, dec, z1, z2, sample_energy2, likelihood1 = self.model(input_data, initialize = 1)
                    self.train_z1.append(z1.data.cpu().numpy())
                    self.train_z2.append(z2.data.cpu().numpy())
                    self.train_likelihood1.append(likelihood1.data.cpu().numpy())
                    self.train_energy2.append(sample_energy2.data.cpu().numpy())  

                elif self.type_model == 'Model3':
                    enc1, dec, z1, z2, sample_energy2, sample_energy1 = self.model(input_data, initialize = 1) 
                    self.train_z1.append(z1.data.cpu().numpy())
                    self.train_z2.append(z2.data.cpu().numpy())
                    self.train_energy1.append(sample_energy1.data.cpu().numpy())
                    self.train_energy2.append(sample_energy2.data.cpu().numpy())  

                elif self.type_model == 'BetaVAE_H':
                    enc1, dec, z1, sample_energy1, mu, logvar = self.model(input_data, test = 1) 
                    # print(sample_energy1)
                    self.train_enc1.append(enc1.data.cpu().numpy())
                    self.train_z1.append(z1.data.cpu().numpy())
                    self.train_energy1.append(sample_energy1.data.cpu().numpy())

                elif self.type_model == 'GANomaly':
                    # dis_x: discriminator of x
                    # dis_dec: discriminator of dec
                    z1, z2, dec, dis_x, dis_dec, sample_energy2 = self.model(input_data, test = 1) 
                    self.train_z1.append(z1.data.cpu().numpy())
                    self.train_z2.append(z2.data.cpu().numpy())
                    self.train_dis_x.append(dis_x.data.cpu().numpy())
                    self.train_dis_dec.append(dis_dec.data.cpu().numpy())
                    self.train_energy2.append(sample_energy2.data.cpu().numpy())

                elif self.type_model == 'AEGMM' or self.type_model == 'AEKDE':
                    enc1, dec, z1 = self.model(input_data) 
                    self.train_z1.append(z1.data.cpu().numpy())

            if self.type_model == 'DaGMM':
                self.train_z1 = np.concatenate(self.train_z1,axis=0)
                self.train_energy1 = np.concatenate(self.train_energy1,axis=0)

            elif self.type_model == 'Model1':
                self.train_z1 = np.concatenate(self.train_z1,axis=0)
                self.train_z2 = np.concatenate(self.train_z2,axis=0)
                self.train_likelihood1 = np.concatenate(self.train_likelihood1,axis=0)
                self.train_energy2 = np.concatenate(self.train_energy2,axis=0)

            elif self.type_model == 'Model2':
                self.train_z1 = np.concatenate(self.train_z1,axis=0)
                self.train_z2 = np.concatenate(self.train_z2,axis=0)
                self.train_likelihood1 = np.concatenate(self.train_likelihood1,axis=0)
                self.train_energy2 = np.concatenate(self.train_energy2,axis=0)

            elif self.type_model == 'Model3':
                self.train_z1 = np.concatenate(self.train_z1,axis=0)
                self.train_z2 = np.concatenate(self.train_z2,axis=0)
                self.train_energy1 = np.concatenate(self.train_energy1,axis=0)
                self.train_energy2 = np.concatenate(self.train_energy2,axis=0)

            elif self.type_model == 'BetaVAE_H':
                # print(self.train_energy1)
                self.train_enc1 = np.concatenate(self.train_enc1,axis=0)
                self.train_z1 = np.concatenate(self.train_z1,axis=0)
                self.train_energy1 = np.concatenate(self.train_energy1,axis=0)

            elif self.type_model == 'GANomaly':
                self.train_z1 = np.concatenate(self.train_z1,axis=0)
                self.train_z2 = np.concatenate(self.train_z2,axis=0)
                self.train_dis_x = np.concatenate(self.train_dis_x,axis=0)
                self.train_dis_dec = np.concatenate(self.train_dis_dec,axis=0)
                self.train_energy2 = np.concatenate(self.train_energy2,axis=0)

            elif self.type_model == 'AEGMM' or self.type_model == 'AEKDE':
                self.train_z1 = np.concatenate(self.train_z1,axis=0)

                if self.type_model == 'AEGMM':
                    self.gm = GaussianMixture(n_components = self.gmm_k, random_state=0).fit(self.train_z1)   
                    self.train_energy1 = -self.gm.score_samples(self.train_z1)
                elif self.type_model == 'AEKDE':
                    self.kd = KernelDensity(kernel = self.kernel, bandwidth = self.bandwidth).fit(self.train_z1)
                    self.train_energy1 = -self.kd.score_samples(self.train_z1)

            if self.var_to_monitor == 'likelihood1':
                    self.thresh = np.percentile(self.train_likelihood1, fpr*100)
            elif self.var_to_monitor == 'energy1':   
                    self.thresh = np.percentile(self.train_energy1, (1-fpr)*100)
            elif self.var_to_monitor == 'energy2': 
                    self.thresh = np.percentile(self.train_energy2, (1-fpr)*100)   

            print("Threshold :", self.thresh)


    def test(self, test_data_loader):
        print("======================TEST MODE======================")
        self.model.eval()

        with torch.no_grad():

            self.test_data_loader = test_data_loader
            self.test_enc1 = []
            self.test_z1 = []
            self.test_z2 = []
            self.test_likelihood1 = []
            self.test_energy1 = []
            self.test_energy2 = []
            self.test_labels = []
            self.test_dis_x = []
            self.test_dis_dec = []
            for it, (input_data, labels) in enumerate(test_data_loader):
                input_data = self.to_var(input_data)

                if self.type_model == 'DaGMM':
                    enc1, dec, z1, gamma = self.model(input_data)
                    sample_energy1, cov_diag = self.model.compute_energy(z1, size_average=False)
                    self.test_z1.append(z1.data.cpu().numpy())
                    self.test_energy1.append(sample_energy1.data.cpu().numpy())

                elif self.type_model == 'Model1':
                    enc1, enc2, dec, z1, z2, sample_energy2, likelihood1 = self.model(input_data, initialize = 1) 
                    self.test_z1.append(z1.data.cpu().numpy())
                    self.test_z2.append(z2.data.cpu().numpy())
                    self.test_likelihood1.append(likelihood1.data.cpu().numpy())
                    self.test_energy2.append(sample_energy2.data.cpu().numpy())

                elif self.type_model == 'Model2':
                    enc1, enc2, dec, z1, z2, sample_energy2, likelihood1 = self.model(input_data, initialize = 1)
                    self.test_z1.append(z1.data.cpu().numpy())
                    self.test_z2.append(z2.data.cpu().numpy())
                    self.test_likelihood1.append(likelihood1.data.cpu().numpy())
                    self.test_energy2.append(sample_energy2.data.cpu().numpy())  

                elif self.type_model == 'Model3':
                    enc1, dec, z1, z2, sample_energy2, sample_energy1 = self.model(input_data, initialize = 1) 
                    self.test_z1.append(z1.data.cpu().numpy())
                    self.test_z2.append(z2.data.cpu().numpy())
                    self.test_energy1.append(sample_energy1.data.cpu().numpy())
                    self.test_energy2.append(sample_energy2.data.cpu().numpy())  

                elif self.type_model == 'BetaVAE_H':
                    enc1, dec, z1, sample_energy1, mu, logvar = self.model(input_data, test = 1) 
                    self.test_enc1.append(enc1.data.cpu().numpy())
                    self.test_z1.append(z1.data.cpu().numpy())
                    self.test_energy1.append(sample_energy1.data.cpu().numpy())

                elif self.type_model == 'GANomaly':
                    # dis_x: discriminator of x
                    # dis_dec: discriminator of dec
                    z1, z2, dec, dis_x, dis_dec, sample_energy2 = self.model(input_data, test = 1) 
                    self.test_z1.append(z1.data.cpu().numpy())
                    self.test_z2.append(z2.data.cpu().numpy())
                    self.test_dis_x.append(dis_x.data.cpu().numpy())
                    self.test_dis_dec.append(dis_dec.data.cpu().numpy())
                    self.test_energy2.append(sample_energy2.data.cpu().numpy())

                elif self.type_model == 'AEGMM' or self.type_model == 'AEKDE':
                    enc1, dec, z1 = self.model(input_data) 
                    self.test_z1.append(z1.data.cpu().numpy())

                self.test_labels.append(labels.numpy())

            if self.type_model == 'DaGMM':
                self.test_z1 = np.concatenate(self.test_z1,axis=0)
                self.test_energy1 = np.concatenate(self.test_energy1,axis=0)

            elif self.type_model == 'Model1':
                self.test_z1 = np.concatenate(self.test_z1,axis=0)
                self.test_z2 = np.concatenate(self.test_z2,axis=0)
                self.test_likelihood1 = np.concatenate(self.test_likelihood1,axis=0)
                self.test_energy2 = np.concatenate(self.test_energy2,axis=0)

            elif self.type_model == 'Model2':
                self.test_z1 = np.concatenate(self.test_z1,axis=0)
                self.test_z2 = np.concatenate(self.test_z2,axis=0)
                self.test_likelihood1 = np.concatenate(self.test_likelihood1,axis=0)
                self.test_energy2 = np.concatenate(self.test_energy2,axis=0)

            elif self.type_model == 'Model3':
                self.test_z1 = np.concatenate(self.test_z1,axis=0)
                self.test_z2 = np.concatenate(self.test_z2,axis=0)
                self.test_energy1 = np.concatenate(self.test_energy1,axis=0)
                self.test_energy2 = np.concatenate(self.test_energy2,axis=0)

            elif self.type_model == 'BetaVAE_H':
                self.test_enc1 = np.concatenate(self.test_enc1,axis=0)
                self.test_z1 = np.concatenate(self.test_z1,axis=0)
                self.test_energy1 = np.concatenate(self.test_energy1,axis=0)

            elif self.type_model == 'GANomaly':
                self.test_z1 = np.concatenate(self.test_z1,axis=0)
                self.test_z2 = np.concatenate(self.test_z2,axis=0)
                self.test_dis_x = np.concatenate(self.test_dis_x,axis=0)
                self.test_dis_dec = np.concatenate(self.test_dis_dec,axis=0)
                self.test_energy2 = np.concatenate(self.test_energy2,axis=0)

            elif self.type_model == 'AEGMM' or self.type_model == 'AEKDE':
                self.test_z1 = np.concatenate(self.test_z1,axis=0)

                if self.type_model == 'AEGMM':
                    self.test_energy1 = -self.gm.score_samples(self.test_z1)
                elif self.type_model == 'AEKDE':
                    self.test_energy1 = -self.kd.score_samples(self.test_z1)

            self.test_labels = np.concatenate(self.test_labels,axis=0)

            if self.var_to_monitor == 'likelihood1':
                pred = (self.test_likelihood1 < self.thresh).astype(int)
            elif self.var_to_monitor == 'energy1': 
                pred = (self.test_energy1 > self.thresh).astype(int)
            elif self.var_to_monitor == 'energy2': 
                pred = (self.test_energy2 > self.thresh).astype(int) 

            gt = self.test_labels.astype(int)

            accuracy = accuracy_score(gt,pred)
            precision, recall, f_score, support = prf(gt, pred, average='binary')

            print("Accuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f}".format(accuracy, precision, recall, f_score))
        
        return accuracy, precision, recall, f_score 



