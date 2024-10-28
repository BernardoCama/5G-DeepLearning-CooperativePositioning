import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

def train(device, model, optimizer, train_loader):
    
    type_ = 'train'
    
    model.train()

    logs_all = []
    
    for i, (input_data, gt) in enumerate((train_loader)):
        
        input_data = input_data.to(device)
        gt = gt.to(device)
        num_feat = input_data.shape[1]
        
        optimizer.zero_grad()
        
        output = model(input_data)
        
        loss = _compute_loss(output, gt)
        
        metrics= compute_perform_metrics(output, gt, num_feat)
        
        logs = {**metrics, **{'loss': loss.item()}}
        
        log = {key + f'/{type_}': val for key, val in logs.items()}
        
        loss.backward()

        logs_all.append(log)
        
        optimizer.step()
    
    return epoch_end(logs_all)


def _compute_loss(outputs, gt):
    l2 = nn.MSELoss()
    loss = l2(outputs, gt)
    return loss

def evaluate(device, model, loader):
    type_ = 'val'
    model.eval()

    logs_all = []

    with torch.no_grad():
        for i, (input_data, gt) in enumerate((loader)):

            input_data = input_data.to(device)
            gt = gt.to(device)  
            num_feat = input_data.shape[1]
            
            output = model(input_data)
            
            loss = _compute_loss(output, gt)
            
            metrics= compute_perform_metrics(output, gt, num_feat)
            
            logs = {**metrics, **{'loss': loss.item()}}
            
            log = {key + f'/{type_}': val for key, val in logs.items()}

            logs_all.append(log)            
            
    return epoch_end(logs_all)
    
def test(device, model, loader):
    type_ = 'test'
    model.eval()

    X = []
    Y = []
    Y_hat = []

    with torch.no_grad():
        for i, (input_data, gt) in enumerate((loader)):

            input_data = input_data.to(device)
            gt = gt.to(device)  
            
            output = model(input_data)
                      
            X.append(input_data.data.cpu().numpy())
            Y.append(gt.data.cpu().numpy())
            Y_hat.append(output.data.cpu().numpy())

    X = np.concatenate(X,axis=0)
    Y = np.concatenate(Y,axis=0)
    Y_hat = np.concatenate(Y_hat,axis=0)
    return X, Y, Y_hat

def epoch_end(outputs):
    metrics = pd.DataFrame(outputs).mean(axis=0).to_dict()
    metrics = {metric_name: torch.as_tensor(metric).item() for metric_name, metric in metrics.items()}
    return metrics


def compute_perform_metrics(output, gt, num_feat):
    """
    Returns:
        dictionary with metrics summary
    """
    output = output.detach().numpy()
    gt = gt.detach().numpy()

    r2 = r2_score(output, gt)
    n=len(gt)
    adj_r2_score = 1 - (1-r2)*(n-1)/(n-num_feat-1)

    return {'adj_r2_score':adj_r2_score}





















































































































































































































































































































































































































































