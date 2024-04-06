import torch
from torch_scatter import scatter_sum

import numpy as np

def reward_improve(data):
    # Our default reward scheme: Difference to previous best clipped at 0
    with torch.no_grad():
        batch_sum_of_weights = data.get_batch_weights()
        reward = data.all_f_val / batch_sum_of_weights
        # reward = np.log(data.all_f_val)
        
        # print(reward)
        
        # test=data.batch_num_cst.view(-1, 1) - data.all_num_unsat
        # test /= data.batch_num_cst.view(-1, 1)
        # print(test)

        # reward = data.batch_num_cst.view(-1, 1) - data.all_num_unsat # get number of satisfied constraints (all_num_sat); [[x1], [x2], ...]
        # reward /= data.batch_num_cst.view(-1, 1) + 1.0e-8
        reward = reward - reward[:, 0].view(-1, 1) # "subtractive baseline"

        max_prior = torch.cummax(reward, dim=1)[0]
        reward[:, 1:] -= max_prior[:, :-1]
        reward[reward < 0.0] = 0.0
        reward[:, 0] = 0.0
        # print(reward)
        return reward


def reward_quality(data):
    # Naive reward scheme used in our ablation study
    with torch.no_grad():
        reward = data.batch_num_cst.view(-1, 1) - data.all_num_unsat
        reward /= data.batch_num_cst.view(-1, 1) + 1.0e-8
        reward = reward - reward[:, 0].view(-1, 1)
        return reward


def reinforce_loss(data, config):
    # get reward in each step t
    assert config['reward'] in {'improve', 'quality'}
    if config['reward'] == 'improve':
        reward = reward_improve(data)
    else:
        reward = reward_quality(data)

    # accumulate discounted future rewards
    with torch.no_grad():
        discount = config['discount']
        return_disc = torch.zeros((reward.shape[0], reward.shape[1]-1), device=data.device)
        weights = discount ** torch.arange(0, reward.shape[1], device=data.device)
        weights = weights.view(1, -1)
        for i in range(return_disc.shape[1]):
            r = reward[:, i+1:]
            w = weights[:, :r.shape[1]]
            return_disc[:, i] = (r * w).sum(dim=1)

    # loss term for the SGD optimizer
    loss = - (data.all_log_probs * return_disc).mean()
    return loss
