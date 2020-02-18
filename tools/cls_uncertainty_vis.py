import numpy as np 
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch

NUM_SAMPLES = 1000
pred_values = np.arange(-20,19.5,0.5)
var_arr     = [0.02,0.1,0.8,0.9,1,1.5,2,3,10]
fig, axs = plt.subplots(3)
eps = 1e-9
#e^0=1
wrong_val = 0


def custom_ce(logits,sel,reduction='mean'):
    pred_logit = torch.index_select(logits,1,sel)
    #pred_logit = logits[sel]
    all_logit = torch.log(torch.sum(torch.exp(logits),dim=1))
    ce_loss = torch.exp(pred_logit - all_logit)
    if(reduction == 'mean'):
        ce_loss = -torch.log(torch.mean(ce_loss))
    #Need to add another layer of summation if a 3D tensor.
    return ce_loss

def elu(variance,mean):
    #Mean is in normal domain, log is in exp domain, need to transform.
    log_var = -torch.log(variance) - mean
    log_var = torch.pow(log_var,2)
    log_elu_var = -torch.nn.functional.elu(-log_var)
    log_elu_loss = log_elu_var + mean
    #Transform back for variable consistency
    return torch.exp(-log_elu_loss)


for var in var_arr:
    dist_vector = []
    norm_vector = []
    elu_dist_vector = []
    custom_vector = []
    for value in pred_values:
        dist = torch.distributions.Normal(0,np.sqrt(var))







        distorted_logit = dist.sample((NUM_SAMPLES,)) + value
        distorted_logit = distorted_logit.unsqueeze(1)
        dummy_val       = torch.tensor([wrong_val],dtype=torch.float32).repeat(NUM_SAMPLES,1)
        dummy_sel       = torch.tensor([0])
        #Normal classification CE loss
        logit_val       = torch.tensor([value,wrong_val],dtype=torch.float32).unsqueeze(1).permute(1,0) #permute to 1,C
        dist_logit_val  = torch.cat((distorted_logit,dummy_val),dim=1)
        #Distorted loss (Kendall 2017)
        ce          = F.cross_entropy(logit_val,torch.tensor([0],dtype=torch.long),reduction='none')
        dist_ce     = F.cross_entropy(dist_logit_val,dummy_sel.repeat(NUM_SAMPLES),reduction='none')
        dist_loss   = -torch.log(torch.mean(torch.exp(-dist_ce),dim=0))
        #log_diff    = dist_ce - ce
        elu_diff    = -torch.nn.functional.elu(-torch.exp(-dist_ce))
        #elu_loss    = torch.exp(-(elu_diff))
        elu_loss    = -torch.log(torch.mean(elu_diff))

        #ce_mean = custom_ce(logit_val,torch.tensor([0],dtype=torch.long))
        #ce_distorted_loss_vec = custom_ce(dist_logit_val,dummy_sel,reduction='none')
        #ce_dist_loss = -torch.log(torch.mean(ce_distorted_loss_vec))
        #Modify further with ELU
        #ce_elu       = elu(ce_distorted_loss_vec,ce_mean)
        #elu_loss = -torch.log(torch.mean(ce_elu))
        #custom_dist = -torch.nn.functional.elu(-dist_var)
        #custom_loss = torch.log(torch.mean(ce_mean+custom_dist)+eps)
        #Regularizer testing
        #dist_var = -torch.log(ce_distorted_loss_vec) - ce_mean
        #regularizer = torch.mean(torch.abs(dist_var)) - 1
        norm_vector.append(torch.mean(ce))
        dist_vector.append(dist_loss)
        elu_dist_vector.append(elu_loss)
        #custom_vector.append(elu_loss+regularizer)
    axs[0].plot(pred_values,norm_vector)
    axs[1].plot(pred_values,dist_vector,label='var {}'.format(var))
    axs[2].plot(pred_values,elu_dist_vector,label='var {}'.format(var))
    #axs[3].plot(pred_values,custom_vector,label='var {}'.format(var))
#axs[3].set_title('elu distorted loss w/ regularizer')
#axs[3].set_ylabel('loss')
#axs[3].set_xlabel('diff(pred_logit-wrong_logit)')
#axs[3].grid(True)
axs[2].set_title('elu distorted loss')
axs[2].set_ylabel('loss')
#axs[2].set_xlabel('diff(pred_logit-wrong_logit)')
axs[2].grid(True)
axs[1].set_title('distorted loss')
axs[1].set_ylabel('loss')
#axs[1].set_xlabel('diff(pred_logit-wrong_logit)')
axs[1].grid(True)
axs[0].set_title('normal CE loss')
axs[0].set_ylabel('loss')
#axs[0].set_xlabel('diff(pred_logit-wrong_logit)')
axs[0].grid(True)
plt.legend()
plt.show()