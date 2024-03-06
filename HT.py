import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
# import higherOrderKME
# from higherOrderKME import sigkernel
import src
from src.path_characteristic_function import char_func_path
from src.model import LSTMRegressor
from src.utils import AddTime
from torch.utils.data import DataLoader
from src.high_level_pcf import expected_dev
import ml_collections
from torch.distributions import Bernoulli
import seaborn as sns
import os
from fbm import FBM
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
sns.set()
torch.manual_seed(0)
device = 'cuda'


def FBM_data(num_samples, dim, length, h):
    fbm_paths = []
    for i in range(num_samples*dim):
        f = FBM(n=length, hurst=h, method='daviesharte')
        fbm_paths.append(f.fbm())
    data = torch.FloatTensor(np.array(fbm_paths)).reshape(
        num_samples, dim, length+1).permute(0, 2, 1)
    return data

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


def train_regressor(regressor, iterations, X_dl, Y_dl):
    best_loss = 10000.
    loss = []
    regressor_optimizer = torch.optim.Adam(regressor.parameters(), betas=(0, 0.9), lr=0.001)
    regressor.train()
    for i in tqdm(range(iterations)):
        regressor_optimizer.zero_grad()
        batch_X = next(iter(X_dl))
        batch_X_dev = next(iter(Y_dl))
        #         print(batch_X.shape, batch_X_dev.shape)
        reg_dev = regressor(batch_X, device)
        #         print(reg_dev.shape, batch_X_dev.shape)

        regressor_loss = torch.norm(reg_dev - batch_X_dev, dim=(2, 3)).sum(1).mean()
        loss.append(regressor_loss.item())
        if regressor_loss < best_loss:
            print("Loss updated: {}".format(regressor_loss), " at iteration {}".format(i))
            #             with torch.no_grad():
            #                 print(torch.norm(reg_dev - batch_X_dev, dim = [2,3]).mean(0))
            best_loss = regressor_loss
            trained_regressor = regressor

        regressor_loss.backward()
        regressor_optimizer.step()

    return trained_regressor, loss


def run_HT(h, i):
    torch.manual_seed(0)
    device = 'cuda'
    # Construct fbm path with different Hurst parameter
    samples = 5000
    steps = 50

    bm = FBM_data(5000, dim=3, length=steps, h=0.5)
    fbm_h = FBM_data(5000, dim=3, length=steps, h=h)

    bm_test = FBM_data(5000, dim=3, length=steps, h=0.5)
    fbm_h_test = FBM_data(5000, dim=3, length=steps, h=h)

    fbm_h = fbm_h.to(device)
    bm = bm.to(device)
    fbm_h_test = fbm_h_test.to(device)
    bm_test = bm_test.to(device)

    config = {'G_input_dim': 4,
              'G_hidden_dim': 32,
              'G_num_layers': 2,
              'G_output_dim': 5}
    config = ml_collections.ConfigDict(config)

    # Construct rank 1 pcf discriminator
    input_dim = fbm_h.shape[-1]
    num_samples_1 = 1
    lie_degree_1 = 5
    pcf_level_1 = char_func_path(num_samples=num_samples_1,
                                 hidden_size=lie_degree_1,
                                 input_dim=input_dim,
                                 add_time=True,
                                 include_initial=False,
                                 return_sequence=False).to(device)

    batch_size = 512

    with torch.no_grad():
        future_dev_path_X = construct_future_dev_path(pcf_level_1, AddTime(bm).to(device), steps + 1)
        future_dev_path_Y = construct_future_dev_path(pcf_level_1, AddTime(fbm_h).to(device), steps + 1)

    joint_train_X_dl = DataLoader(AddTime(torch.cat([bm, fbm_h])), batch_size, shuffle=True)
    joint_train_X_dev_dl = DataLoader(torch.cat([future_dev_path_X, future_dev_path_Y]), batch_size, shuffle=True)

    train_X_dl = DataLoader(AddTime(bm), batch_size, shuffle=True)
    train_Y_dl = DataLoader(AddTime(fbm_h), batch_size, shuffle=True)

    test_X_dl = DataLoader(AddTime(bm_test), batch_size, shuffle=True)
    test_Y_dl = DataLoader(AddTime(fbm_h_test), batch_size, shuffle=True)

    regressor_for_X = LSTMRegressor(
        input_dim=config.G_input_dim,
        hidden_dim=config.G_hidden_dim,
        output_dim=config.G_output_dim,
        n_layers=config.G_num_layers
    )
    regressor_for_X.to(device)

    trained_regressor_X, loss_X = train_regressor(regressor_for_X, 2000, joint_train_X_dl, joint_train_X_dev_dl)

    expected_devx = expected_dev(regressor=trained_regressor_X,
                                 lie_degree_1=5, lie_degree_2=5, num_samples_2=10)

    losses = expected_devx.train_M(train_X_dl, train_Y_dl, 6000)

    plt.plot(losses)
    plt.savefig("./examples/HT/rank2_loss_reg_h={}_iter={}.png".format(h, i))

    MMD_1, MMD_2 = expected_devx.evaluate(test_X_dl, test_Y_dl)

    expected_devx.print_hist(MMD_1, MMD_2, "rank2_mmd_reg_h={}_iter={}.png".format(h, i))

    power, type1_error, MMD_1, MMD_2 = expected_devx.permutation_test(AddTime(bm_test), AddTime(fbm_h_test), sample_size=batch_size, num_permutations=100)

    expected_devx.print_hist(MMD_1, MMD_2, "rank2_permutation_mmd_reg_h={}_iter={}.png".format(h, i))

    return power, type1_error


if __name__ == "__main__":
    import ml_collections
    import yaml
    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    print(os.environ["CUDA_VISIBLE_DEVICES"])

    sns.set()
    h_list = [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.475, 0.5, 0.525, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]
    # h_list = [0.45]
    df_dict = {}
    for h in h_list:
        power_list = []
        type1_error_list = []
        for i in range(5):
            power, type1_error = run_HT(h, i)
            power_list.append(power)
            type1_error_list.append(type1_error)
        df_dict["power_{}".format(h)] = power_list
        df_dict["typeI_error_{}".format(h)] = type1_error_list
    df = pd.DataFrame(df_dict)

    df.to_csv("./examples/HT/metric_fbm.csv")