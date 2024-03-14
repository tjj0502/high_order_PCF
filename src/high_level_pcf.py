import torch
import numpy as np
from src.model.discriminator.path_characteristic_function import pcf
from tqdm import tqdm
import matplotlib.pyplot as plt

class high_order_pcf():

    def __init__(self,
                 regressor_X,
                 lie_degree_1,
                 lie_degree_2,
                 num_samples_2,
                 config,
                 whole_dev = False,
                 regressor_Y = None,
                 add_time=True,
                 device='cuda'):
        super(high_order_pcf, self).__init__()
        """ Generator base class. All generators should be children of this class. """
        self.device = device
        self.regressor_X = regressor_X
        self.regressor_X.to(device)

        if regressor_Y:
            self.regressor_Y = regressor_Y
            self.regressor_Y.to(device)
        else:
            self.regressor_Y = regressor_X
            self.regressor_Y.to(device)

        self.config = config
        self.lie_degree_1 = lie_degree_1
        self.add_time = add_time
        self.num_samples_2 = num_samples_2
        self.lie_degree_2 = lie_degree_2
        self.pcf_level_2 = pcf(num_samples=self.num_samples_2,
                               hidden_size=self.lie_degree_2,
                               input_dim=2 * self.lie_degree_1 ** 2,
                               add_time=add_time,
                               include_initial=False,
                               )
        self.pcf_level_2.to(device)
        self.whole_dev = whole_dev


    def train_M(self, X_dl, Y_dl, X_test_dl, Y_test_dl, iterations):
        best_loss = 0.

        char_2_optimizer = torch.optim.Adam(self.pcf_level_2.parameters(), betas=(0, 0.9), lr=self.config.lr_D)
        losses = {"R1X_R2Y_loss": [], "R1X_R2X_loss": [], "R1Y_R2Y_loss": [], "R1Y_R2X_loss": [],
                  "Out-of-sample-loss": []}
        print('start opitmize charateristics function')
        self.regressor_X.eval()
        self.regressor_Y.eval()
        self.pcf_level_2.train()
        for i in tqdm(range(iterations)):

            X, past_dev_X = next(iter(X_dl))
            Y, past_dev_Y = next(iter(Y_dl))
            with torch.no_grad():

                exp_dev_X, exp_dev_Y = self.path_to_dev(X, Y, past_dev_X, past_dev_Y)
                # if self.whole_dev:
                #     exp_dev_X = self.regressor_X(X, self.device)
                #     exp_dev_Y = self.regressor_Y(Y, self.device)
                #     exp_dev_X = (past_dev_X @ exp_dev_X).reshape([-1, X.shape[1], self.lie_degree_1 ** 2])
                #     exp_dev_Y = (past_dev_Y @ exp_dev_Y).reshape([-1, X.shape[1], self.lie_degree_1 ** 2])
                # else:
                #     exp_dev_X = self.regressor_X(X, self.device).reshape([-1, X.shape[1], self.lie_degree_1 ** 2])
                #     exp_dev_Y = self.regressor_Y(Y, self.device).reshape([-1, X.shape[1], self.lie_degree_1 ** 2])

                # exp_dev_X = torch.cat([exp_dev_X.real, exp_dev_X.imag], -1)
                # exp_dev_Y = torch.cat([exp_dev_Y.real, exp_dev_Y.imag], -1)

                if i % 5 == 0:
                    X_test, past_dev_X_test = next(iter(X_test_dl))
                    Y_test, past_dev_Y_test = next(iter(Y_test_dl))
                    exp_dev_X_test, exp_dev_Y_test = self.path_to_dev(X_test, Y_test, past_dev_X_test, past_dev_Y_test)
                    exp_dev_XY, exp_dev_YX = self.path_to_dev(Y_test, X_test, past_dev_Y_test, past_dev_X_test)
                    # if self.whole_dev:
                    #     exp_dev_X_test = self.regressor_X(X_test, self.device)
                    #     exp_dev_Y_test = self.regressor_Y(Y_test, self.device)
                    #     exp_dev_X_test = (past_dev_X_test @ exp_dev_X_test).reshape(
                    #         [-1, X.shape[1], self.lie_degree_1 ** 2])
                    #     exp_dev_Y_test = (past_dev_Y_test @ exp_dev_Y_test).reshape(
                    #         [-1, X.shape[1], self.lie_degree_1 ** 2])
                    #     exp_dev_XY = self.regressor_X(Y, self.device)
                    #     exp_dev_YX = self.regressor_Y(X, self.device)
                    #     exp_dev_XY = (past_dev_Y @ exp_dev_XY).reshape([-1, X.shape[1], self.lie_degree_1 ** 2])
                    #     exp_dev_YX = (past_dev_X @ exp_dev_YX).reshape([-1, X.shape[1], self.lie_degree_1 ** 2])
                    #
                    # else:
                    #     exp_dev_X_test = self.regressor_X(X_test, self.device).reshape(
                    #         [-1, X.shape[1], self.lie_degree_1 ** 2])
                    #     exp_dev_Y_test = self.regressor_Y(Y_test, self.device).reshape(
                    #         [-1, X.shape[1], self.lie_degree_1 ** 2])
                    #     exp_dev_XY = self.regressor_X(Y, self.device).reshape(
                    #         [-1, X.shape[1], self.lie_degree_1 ** 2])
                    #     exp_dev_YX = self.regressor_Y(X, self.device).reshape(
                    #         [-1, X.shape[1], self.lie_degree_1 ** 2])

                    # exp_dev_X_test = torch.cat([exp_dev_X_test.real, exp_dev_X_test.imag], -1)
                    # exp_dev_Y_test = torch.cat([exp_dev_Y_test.real, exp_dev_Y_test.imag], -1)
                    # exp_dev_XY = torch.cat([exp_dev_XY.real, exp_dev_XY.imag], -1)
                    # exp_dev_YX = torch.cat([exp_dev_YX.real, exp_dev_YX.imag], -1)

            char_2_optimizer.zero_grad()
            char_loss = - self.pcf_level_2.distance_measure(exp_dev_X, exp_dev_Y, Lambda=0)

            losses["R1X_R2Y_loss"].append(-char_loss.item())
            with torch.no_grad():
                if i % 5 == 0:
                    char_loss_ = self.pcf_level_2.distance_measure(exp_dev_XY, exp_dev_Y, Lambda=0)
                    losses["R1Y_R2Y_loss"].append(char_loss_.item())
                    char_loss_ = self.pcf_level_2.distance_measure(exp_dev_X, exp_dev_YX, Lambda=0)
                    losses["R1X_R2X_loss"].append(char_loss_.item())
                    char_loss_ = self.pcf_level_2.distance_measure(exp_dev_XY, exp_dev_YX, Lambda=0)
                    losses["R1Y_R2X_loss"].append(char_loss_.item())

                    char_loss_ = self.pcf_level_2.distance_measure(exp_dev_X_test, exp_dev_Y_test, Lambda=0)
                    losses["Out-of-sample-loss"].append(char_loss_.item())

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
        self.regressor_Y.eval()
        repeats = 100
        MMD_1 = np.zeros((repeats))
        MMD_2 = np.zeros((repeats))
        with torch.no_grad():
            for i in tqdm(range(repeats)):
                X, past_dev_X = next(iter(X_dl))
                Y, past_dev_Y = next(iter(Y_dl))
                X_, past_dev_X_ = next(iter(X_dl))

                if self.whole_dev:
                    exp_dev_X = self.regressor_X(X, self.device)
                    exp_dev_X_ = self.regressor_X(X_, self.device)
                    exp_dev_Y = self.regressor_Y(Y, self.device)
                    exp_dev_X = (past_dev_X @ exp_dev_X).reshape([-1, X.shape[1], self.lie_degree_1 ** 2])
                    exp_dev_X_ = (past_dev_X_ @ exp_dev_X_).reshape([-1, X.shape[1], self.lie_degree_1 ** 2])
                    exp_dev_Y = (past_dev_Y @ exp_dev_Y).reshape([-1, X.shape[1], self.lie_degree_1 ** 2])
                else:
                    exp_dev_X = self.regressor_X(X, self.device).reshape([-1, X.shape[1], self.lie_degree_1 ** 2])
                    exp_dev_X_ = self.regressor_X(X_, self.device).reshape([-1, X.shape[1], self.lie_degree_1 ** 2])
                    exp_dev_Y = self.regressor_Y(Y, self.device).reshape([-1, X.shape[1], self.lie_degree_1 ** 2])

                # exp_dev_X = torch.cat([exp_dev_X.real, exp_dev_X.imag], -1)
                # exp_dev_Y = torch.cat([exp_dev_Y.real, exp_dev_Y.imag], -1)
                # exp_dev_X_ = torch.cat([exp_dev_X_.real, exp_dev_X_.imag], -1)

                MMD_1[i] = self.pcf_level_2.distance_measure(exp_dev_X, exp_dev_X_, Lambda=0)
                MMD_2[i] = self.pcf_level_2.distance_measure(exp_dev_X, exp_dev_Y, Lambda=0)
        self.print_hist(MMD_1, MMD_2, 'HT_test.png')
        return MMD_1, MMD_2

    def permutation_test(self, X, Y, sample_size, num_permutations=500):
        with torch.no_grad():
            self.pcf_level_2.eval()
            self.regressor_X.eval()
            self.regressor_Y.eval()
            #             X = self.subsample(X, sample_size).to(self.device)
            #             Y = self.subsample(Y, sample_size).to(self.device)

            # print(t1)
            #             n, m = X.shape[0], Y.shape[0]
            #             combined = torch.cat([X, Y])
            H0_stats = np.zeros((num_permutations))
            H1_stats = np.zeros((num_permutations))

            for i in tqdm(range(num_permutations)):
                X_sample, past_dev_X = self.subsample(X, sample_size)
                Y_sample, past_dev_Y = self.subsample(Y, sample_size)

                exp_dev_X, exp_dev_Y = self.path_to_dev(X_sample, Y_sample, past_dev_X, past_dev_Y)

                # if self.whole_dev:
                #     exp_dev_X = self.regressor_X(X_sample, self.device)
                #     exp_dev_Y = self.regressor_Y(Y_sample, self.device)
                #     exp_dev_X = (past_dev_X @ exp_dev_X).reshape([-1, X.shape[1], self.lie_degree_1 ** 2])
                #     exp_dev_Y = (past_dev_Y @ exp_dev_Y).reshape([-1, X.shape[1], self.lie_degree_1 ** 2])
                # else:
                #     exp_dev_X = self.regressor_X(X_sample, self.device).reshape(
                #         [-1, X.shape[1], self.lie_degree_1 ** 2])
                #     exp_dev_Y = self.regressor_Y(Y_sample, self.device).reshape(
                #         [-1, X.shape[1], self.lie_degree_1 ** 2])
                #
                # exp_dev_X = torch.cat([exp_dev_X.real, exp_dev_X.imag], -1)
                # exp_dev_Y = torch.cat([exp_dev_Y.real, exp_dev_Y.imag], -1)

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

            # self.print_hist(H0_stats, H1_stats, 'permutation_test.png')
        return power, type1_error, H0_stats, H1_stats

    def subsample(self, data, sample_size):
        idx = torch.randint(low=0, high=data.shape[0], size=[sample_size])
        return data[idx]


    def path_to_dev(self, X, Y, past_dev_X = None, past_dev_Y = None):
        if self.whole_dev:
            exp_dev_X = self.regressor_X(X, self.device)
            exp_dev_Y = self.regressor_Y(Y, self.device)
            exp_dev_X = (past_dev_X @ exp_dev_X).reshape([-1, X.shape[1], self.lie_degree_1 ** 2])
            exp_dev_Y = (past_dev_Y @ exp_dev_Y).reshape([-1, X.shape[1], self.lie_degree_1 ** 2])
        else:
            exp_dev_X = self.regressor_X(X, self.device).reshape(
                [-1, X.shape[1], self.lie_degree_1 ** 2])
            exp_dev_Y = self.regressor_Y(Y, self.device).reshape(
                [-1, X.shape[1], self.lie_degree_1 ** 2])

        exp_dev_X = torch.cat([exp_dev_X.real, exp_dev_X.imag], -1)
        exp_dev_Y = torch.cat([exp_dev_Y.real, exp_dev_Y.imag], -1)
        return exp_dev_X, exp_dev_Y

    def print_hist(self, hist_1, hist_2, title):
        fig, ax = plt.subplots(1, 1, figsize=(15, 5))

        ax.hist(hist_1, bins=25, label='H_0', edgecolor='#E6E6E6')
        ax.hist(hist_2, bins=25, label='H_A', edgecolor='#E6E6E6')

        ax.legend(loc='upper right', ncol=2, fontsize=22)
        ax.set_xlabel('D_1(X, Y)^2', labelpad=10)
        ax.set_ylabel('Count', labelpad=10)

        plt.tight_layout(pad=3.0)
        plt.savefig(self.config.exp_dir + title)
        plt.close()