import torch
import torch.nn as nn
import os
import pickle
import yaml
import ml_collections

def get_time_vector(size: int, length: int) -> torch.Tensor:
    return torch.linspace(1/length, 1, length).reshape(1, -1, 1).repeat(size, 1, 1)


def AddTime(x):
    t = get_time_vector(x.shape[0], x.shape[1]).to(x.device)
    return torch.cat([t, x], dim=-1)

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(
            m.weight.data, gain=nn.init.calculate_gain('relu'))
        try:
            # m.bias.zero_()#, gain=nn.init.calculate_gain('relu'))
            nn.init.zeros_(m.bias)
        except:
            pass
    elif isinstance(m, nn.LSTM):
        for name, param in m.named_parameters():
            if 'weight_ih' in name:
                nn.init.kaiming_normal_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)
    elif isinstance(m, nn.GRU):
        for name, param in m.named_parameters():
            if 'weight_ih' in name:
                nn.init.kaiming_normal_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)

        try:
            # m.bias.zero_()#, gain=nn.init.calculate_gain('relu'))
            nn.init.zeros_(m.bias)
        except:
            pass


def track_gradient_norms(model):
    total_norm = 0
    for p in model.parameters():
        param_norm = p.grad.data.norm(2)
        total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    return total_norm

def track_norm(model):
    total_norm = 0
    for p in model.parameters():
        param_norm = p.data.norm(2)
        total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    return total_norm


def to_numpy(x):
    """
    Casts torch.Tensor to a numpy ndarray.

    The function detaches the tensor from its gradients, then puts it onto the cpu and at last casts it to numpy.
    """
    return x.detach().cpu().numpy()


def construct_future_dev_path(pcf, path, steps):
    with torch.no_grad():
        lie_degree = pcf.degree
        N, T, D = path.shape
        dev_list = []
        for step in range(steps):
            if step == steps-1:
                dev = torch.eye(lie_degree).repeat(N, 1, 1, 1).to(dtype=dev_list[0].dtype, device=dev_list[0].device)
                dev_list.append(dev)
            else:
                dev_list.append(pcf.unitary_development(path[:, step:]))
    return torch.cat(dev_list, dim = 1)

def construct_past_dev_path(pcf, path, steps):
    with torch.no_grad():
        lie_degree = pcf.degree
        N, T, D = path.shape
        dev_list = []
        for step in range(1, steps+1):
            if step == 1:
                dev = torch.eye(lie_degree).repeat(N, 1, 1, 1)
                dev_list.append(dev)
            else:
                dev_list.append(pcf.unitary_development(path[:, :step]))
        dev_list[0] = dev_list[0].to(dtype=dev_list[-1].dtype, device=dev_list[-1].device)
    return torch.cat(dev_list, dim = 1)

def get_experiment_dir(config):
    exp_dir = './numerical_results/{dataset}/algo_{gan}_G_{generator}_D_lie_degree_{liedeg}_n_lag_{n_lags}_{seed}_comment_{comment}_{lie_group}'.format(
        dataset=config.dataset, gan=config.gan_algo, generator=config.generator,
        liedeg=config.Rank_2_lie_degree, n_lags=config.n_lags, seed=config.seed, comment=config.comment, lie_group=config.lie_group)
    os.makedirs(exp_dir, exist_ok=True)
    if config.train and os.path.exists(exp_dir):
        print("WARNING! The model exists in directory and will be overwritten")
    config.exp_dir = exp_dir


def save_obj(obj: object, filepath: str):
    """ Generic function to save an object with different methods. """
    if filepath.endswith('pkl'):
        saver = pickle.dump
    elif filepath.endswith('pt'):
        saver = torch.save
    else:
        raise NotImplementedError()
    with open(filepath, 'wb') as f:
        saver(obj, f)
    return 0


def load_obj(filepath):
    """ Generic function to load an object. """
    if filepath.endswith('pkl'):
        loader = pickle.load
    elif filepath.endswith('pt'):
        loader = torch.load
    elif filepath.endswith('json'):
        import json
        loader = json.load
    else:
        raise NotImplementedError()
    with open(filepath, 'rb') as f:
        return loader(f)


def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(
            m.weight.data, gain=nn.init.calculate_gain('relu'))
        try:
            # m.bias.zero_()#, gain=nn.init.calculate_gain('relu'))
            nn.init.zeros_(m.bias)
        except:
            pass
    elif isinstance(m, nn.LSTM):
        for name, param in m.named_parameters():
            if 'weight_ih' in name:
                nn.init.kaiming_normal_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)
    elif isinstance(m, nn.GRU):
        for name, param in m.named_parameters():
            if 'weight_ih' in name:
                nn.init.kaiming_normal_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)

        try:
            # m.bias.zero_()#, gain=nn.init.calculate_gain('relu'))
            nn.init.zeros_(m.bias)
        except:
            pass

def load_config(file_dir: str):
    with open(file_dir) as file:
        config = ml_collections.ConfigDict(yaml.safe_load(file))
    return config