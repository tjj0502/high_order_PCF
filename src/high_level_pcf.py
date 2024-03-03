import torch
import numpy as np
from src.path_characteristic_function import char_func_path
from tqdm import tqdm
import matplotlib.pyplot as plt

class expected_dev():

    def __init__(self, regressor, lie_degree_1, lie_degree_2, num_samples_2, add_time=True, device='cuda'):
        super(expected_dev, self).__init__()
        """ Generator base class. All generators should be children of this class. """
        self.device = device
        self.regressor_X = regressor
        self.regressor_X.to(device)

        self.lie_degree_1 = lie_degree_1
        self.add_time = add_time
        self.num_samples_2 = num_samples_2
        self.lie_degree_2 = lie_degree_2
        self.pcf_level_2 = char_func_path(num_samples=self.num_samples_2,
                                          hidden_size=self.lie_degree_2,
                                          input_dim=2 * self.lie_degree_1 ** 2,
                                          add_time=add_time,
                                          include_initial=False,
                                          return_sequence=False)
        self.pcf_level_2.to(device)

    def train_M(self, X_dl, Y_dl, iterations):
        best_loss = 0.

        char_2_optimizer = torch.optim.Adam(self.pcf_level_2.parameters(), betas=(0, 0.9), lr=0.002)
        losses = []
        print('start opitmize charateristics function')
        self.regressor_X.eval()
        self.pcf_level_2.train()
        for i in tqdm(range(iterations)):

            X = next(iter(X_dl))
            Y = next(iter(Y_dl))
            with torch.no_grad():
                exp_dev_X = self.regressor_X(X, self.device).reshape([-1, X.shape[1], self.lie_degree_1 ** 2])
                exp_dev_Y = self.regressor_X(Y, self.device).reshape([-1, Y.shape[1], self.lie_degree_1 ** 2])

                exp_dev_X = torch.cat([exp_dev_X.real, exp_dev_X.imag], -1)
                exp_dev_Y = torch.cat([exp_dev_Y.real, exp_dev_Y.imag], -1)

            char_2_optimizer.zero_grad()
            char_loss = - self.pcf_level_2.distance_measure(exp_dev_X, exp_dev_Y, Lambda=0)
            losses.append(-char_loss.item())
            if -char_loss > best_loss:
                print("Loss updated: {}".format(-char_loss))
                best_loss = -char_loss
            if i % 100 == 0:
                print("Iteration {} :".format(i), " loss = {}".format(-char_loss))
            char_loss.backward()
            char_2_optimizer.step()

        return losses

    def evaluate(self, X_dl, Y_dl):
        self.pcf_level_2.eval()
        self.regressor_X.eval()
        repeats = 100
        MMD_1 = np.zeros((repeats))
        MMD_2 = np.zeros((repeats))
        with torch.no_grad():
            for i in tqdm(range(repeats)):
                X = next(iter(X_dl))
                Y = next(iter(Y_dl))
                X_ = next(iter(X_dl))
                exp_dev_X = self.regressor_X(X, self.device).reshape([-1, X.shape[1], self.lie_degree_1 ** 2])
                exp_dev_X_ = self.regressor_X(X_, self.device).reshape([-1, X.shape[1], self.lie_degree_1 ** 2])
                exp_dev_Y = self.regressor_X(Y, self.device).reshape([-1, Y.shape[1], self.lie_degree_1 ** 2])

                exp_dev_X = torch.cat([exp_dev_X.real, exp_dev_X.imag], -1)
                exp_dev_Y = torch.cat([exp_dev_Y.real, exp_dev_Y.imag], -1)
                exp_dev_X_ = torch.cat([exp_dev_X_.real, exp_dev_X_.imag], -1)

                MMD_1[i] = self.pcf_level_2.distance_measure(exp_dev_X, exp_dev_X_, Lambda=0)
                MMD_2[i] = self.pcf_level_2.distance_measure(exp_dev_X, exp_dev_Y, Lambda=0)
        # self.print_hist(MMD_1, MMD_2)
        return MMD_1, MMD_2

    def permutation_test(self, X, Y, sample_size, num_permutations=500):
        with torch.no_grad():
            self.pcf_level_2.eval()
            self.regressor_X.eval()

            #             X = self.subsample(X, sample_size).to(self.device)
            #             Y = self.subsample(Y, sample_size).to(self.device)

            # print(t1)
            n, m = X.shape[0], Y.shape[0]
            combined = torch.cat([X, Y])
            H0_stats = np.zeros((num_permutations))
            H1_stats = np.zeros((num_permutations))

            for i in tqdm(range(num_permutations)):
                X_sample = self.subsample(X, sample_size).to(self.device)
                Y_sample = self.subsample(Y, sample_size).to(self.device)

                exp_dev_X = self.regressor_X(X_sample, self.device).reshape(
                    [-1, X_sample.shape[1], self.lie_degree_1 ** 2])
                exp_dev_Y = self.regressor_X(Y_sample, self.device).reshape(
                    [-1, Y_sample.shape[1], self.lie_degree_1 ** 2])

                exp_dev_X = torch.cat([exp_dev_X.real, exp_dev_X.imag], -1)
                exp_dev_Y = torch.cat([exp_dev_Y.real, exp_dev_Y.imag], -1)

                n, m = exp_dev_X.shape[0], exp_dev_Y.shape[0]
                combined = torch.cat([exp_dev_X, exp_dev_Y])

                idx = torch.randperm(n + m)
                H0_stats[i] = self.pcf_level_2.distance_measure(combined[idx[:n]], combined[idx[n:]], Lambda=0)
                H1_stats[i] = self.pcf_level_2.distance_measure(exp_dev_X, exp_dev_Y, Lambda=0)
            Q_a = np.quantile(np.array(H0_stats), q=0.95)
            Q_b = np.quantile(np.array(H1_stats), q=0.05)

            # print(statistics)
            # print(np.array(statistics))
            power = 1 - (Q_a > np.array(H1_stats)).sum() / num_permutations
            type1_error = (Q_b < np.array(H0_stats)).sum() / num_permutations

            # self.print_hist(H0_stats, H1_stats)
        return power, type1_error, H0_stats, H1_stats

    def subsample(self, data, sample_size):
        idx = torch.randint(low=0, high=data.shape[0], size=[sample_size])
        return data[idx]

    def print_hist(self, hist_1, hist_2, title):
        fig, ax = plt.subplots(1, 1, figsize=(15, 5))

        ax.hist(hist_1, bins=25, label='H_0', edgecolor='#E6E6E6')
        ax.hist(hist_2, bins=25, label='H_A', edgecolor='#E6E6E6')

        ax.legend(loc='upper right', ncol=2, fontsize=22)
        ax.set_xlabel('D_1(X, Y)^2', labelpad=10)
        ax.set_ylabel('Count', labelpad=10)

        plt.tight_layout(pad=3.0)
        plt.savefig('./examples/HT/' + title)
        plt.close()