import numpy as np
import pyro
import torch


def condCausal(their_tensors, absent_tensors, movie_inds):
    their_cond = pyro.condition(model, data = {"x" : their_tensors})
    absent_cond = pyro.condition(model, data = {"x" : absent_tensors})
    
    their_y = []
    for _ in range(1000):
        their_y.append(torch.sum(their_cond(p2)['y'][movie_inds]).item())
    
    absent_y = []
    for _ in range(1000):
        absent_y.append(torch.sum(absent_cond(p2)['y'][movie_inds]).item())
    
    their_mean = np.mean(their_y)
    absent_mean = np.mean(absent_y)
    causal_effect_noconf = their_mean - absent_mean
    
    return causal_effect_noconf


def doCausal(their_tensors, absent_tensors, movie_inds):
    # With confounding
    their_do = pyro.do(model, data = {"x" : their_tensors})
    absent_do = pyro.do(model, data = {"x" : absent_tensors})
    
    their_do_y = []
    for _ in range(1000):
        their_do_y.append(torch.sum(their_do(p2)['y'][movie_inds]).item())
    
    absent_do_y = []
    for _ in range(1000):
        absent_do_y.append(torch.sum(absent_do(p2)['y'][movie_inds]).item())
    
    their_do_mean = np.mean(their_do_y)
    absent_do_mean = np.mean(absent_do_y)
    causal_effect_conf = their_do_mean - absent_do_mean
    
    return causal_effect_conf

def causal_effects(actor):
    # Get all movies where that actor is present
    # Make him/her absent, and then get conditional expectation
    
    actor_tensor = pd.DataFrame(x_train_tensors.numpy(), columns=actors)
    # All movies where actor is present
    movie_inds = actor_tensor.index[actor_tensor[actor] == 1.0]

    absent_movies = actor_tensor.copy()
    absent_movies[actor] = 0
    
    their_tensors = x_train_tensors
    absent_tensors = torch.tensor(absent_movies.to_numpy(dtype = 'float32'))
    
    cond_effect_mean = condCausal(their_tensors, absent_tensors, movie_inds)
    do_effect_mean = doCausal(their_tensors, absent_tensors, movie_inds)
#     print(their_tensors.shape, absent_tensors.shape)
    diff_mean = cond_effect_mean - do_effect_mean
    if diff_mean > 0:
        status = "Overvalued"
    else: status = "Undervalued"
    

    return cond_effect_mean, do_effect_mean, diff_mean, status