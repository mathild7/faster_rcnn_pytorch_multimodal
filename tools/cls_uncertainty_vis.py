import numpy as np 
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch

NUM_SAMPLES = 10000
pred_values = np.arange(-8,8,0.5)
var_arr     = [2.5,1.75,1,0.5,0.25]
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
        value = value
        #Normal classification CE loss
        logit_pair = torch.tensor([value,wrong_val],dtype=torch.float32).unsqueeze(1).permute(1,0) #permute to 1,C
        ce_loss    = F.cross_entropy(logit_pair,torch.tensor([0],dtype=torch.long),reduction='none')
        softmax    = F.softmax(logit_pair)

        #Experimental sampling CE loss
        #Step 1: Sample logits
        dist = torch.distributions.Normal(0,np.sqrt(var))
        logit_sampled = dist.sample((NUM_SAMPLES,)) + value
        logit_sampled = logit_sampled.unsqueeze(1) - 1.5
        #Get a similar sized array for other logit
        dummy_val_sampled = torch.tensor([wrong_val],dtype=torch.float32).repeat(NUM_SAMPLES,1)
        dummy_sel        = torch.tensor([0])
        logit_pair_sampled  = torch.cat((logit_sampled,dummy_val_sampled),dim=1)
        #Step 2: Get softmax of distorted(sampled) logit pair
        softmax_sampled      = F.softmax(logit_pair_sampled)
        dist_softmax         = torch.mean(softmax_sampled,dim=0).unsqueeze(0)
        #Step 3: Obtain negative log likelihood (loss)
        dist_loss      = F.nll_loss(torch.log(dist_softmax),dummy_sel)




        #Experimental ELU sampling CE loss
        #softmax_diff_sampled = softmax_sampled - softmax.repeat(NUM_SAMPLES,1)
        #elu_diff_sampled     = -torch.nn.functional.elu(-softmax_diff_sampled)
        #elu_sampled          = elu_diff_sampled + softmax.repeat(NUM_SAMPLES,1)
        #elu_softmax          = F.softmax(elu_sampled)
        #elu_avg_softmax      = torch.mean(torch.log(elu_sampled),dim=0).unsqueeze(0)
        #elu_loss             = F.nll_loss(elu_avg_softmax,dummy_sel)
        #     = F.cross_entropy(dist_logit_val,dummy_sel.repeat(NUM_SAMPLES),reduction='none')
        #dist_loss   = -torch.log(torch.mean(torch.exp(-dist_ce),dim=0))
        #log_diff    = dist_ce - ce
        #elu_diff    = -torch.nn.functional.elu(-dist_softmax,alpha=0.1)
        #elu_loss    = torch.exp(-(elu_diff))
        #elu_avg_diff = torch.mean(elu_diff,dim=0).unsqueeze(0)
        #elu_loss    = F.nll_loss(torch.log(elu_avg_diff),dummy_sel)

        regularizer = var*0.1
        norm_vector.append(torch.mean(ce_loss))
        dist_vector.append(dist_loss)
        elu_dist_vector.append(dist_loss+regularizer)
        #custom_vector.append(elu_loss-dist_loss)
    axs[0].plot(pred_values,norm_vector)
    axs[1].plot(pred_values,dist_vector,label='var {}'.format(var))
    axs[2].plot(pred_values,elu_dist_vector,label='var {}'.format(var))
    #axs[3].plot(pred_values,custom_vector,label='var {}'.format(var))
#axs[3].set_title('distorted loss w/ regularizer')
#axs[3].set_ylabel('loss')
#axs[3].set_xlabel('diff(pred_logit-wrong_logit)')
#axs[3].grid(True)
axs[2].set_title('distorted loss + regularizers')
axs[2].set_ylabel('loss')
axs[2].set_xlabel('diff(pred_logit-wrong_logit)')
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