import torch
import numpy as np
from Path_Char.path_characteristic_function import char_func_path
from tqdm.notebook import tqdm
class expected_dev():

    def __init__(self, regressor_X, regressor_Y, lie_degree_1, lie_degree_2, num_samples_2, add_time=True, device='cuda'):
        super(expected_dev, self).__init__()
        """ Generator base class. All generators should be children of this class. """
        self.device = device
        self.regressor_X = regressor_X
        self.regressor_Y = regressor_Y
        self.regressor_X.to(device)
        self.regressor_Y.to(device)

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



    def train_M(self, X_dl, Y_dl):
        iterations = 10000
        best_loss = 0.

        char_2_optimizer = torch.optim.Adam(self.pcf_level_2.parameters(), betas=(0, 0.9), lr=0.002)

        print('start opitmize charateristics function')
        self.regressor_X.eval()
        self.regressor_Y.eval()
        self.pcf_level_2.train()
        for i in tqdm(range(iterations)):

            X = next(iter(X_dl))
            Y = next(iter(Y_dl))
            with torch.no_grad():
                exp_dev_X = self.regressor_X(X, self.device).reshape([-1, X.shape[1], self.lie_degree_1 ** 2])
                exp_dev_Y = self.regressor_Y(Y, self.device).reshape([-1, Y.shape[1], self.lie_degree_1 ** 2])

                exp_dev_X = torch.cat([exp_dev_X.real, exp_dev_X.imag], -1)
                exp_dev_Y = torch.cat([exp_dev_Y.real, exp_dev_Y.imag], -1)

            char_2_optimizer.zero_grad()
            char_loss = - self.pcf_level_2.distance_measure(exp_dev_X, exp_dev_Y, Lambda=0)
            if -char_loss > best_loss:
                print("Loss updated: {}".format(-char_loss))
                best_loss = -char_loss
            if i % 100 == 0:
                print("Iteration {} :".format(i), " loss = {}".format(-char_loss))
            char_loss.backward()
            char_2_optimizer.step()

    def evaluate(self, X_dl, Y_dl):
        self.pcf_level_2.eval()
        self.regressor_X.eval()
        self.regressor_Y.eval()
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
                exp_dev_Y = self.regressor_Y(Y, self.device).reshape([-1, Y.shape[1], self.lie_degree_1 ** 2])

                exp_dev_X = torch.cat([exp_dev_X.real, exp_dev_X.imag], -1)
                exp_dev_Y = torch.cat([exp_dev_Y.real, exp_dev_Y.imag], -1)
                exp_dev_X_ = torch.cat([exp_dev_X_.real, exp_dev_X_.imag], -1)

                MMD_1[i] = self.pcf_level_2.distance_measure(exp_dev_X, exp_dev_X_, Lambda=0)
                MMD_2[i] = self.pcf_level_2.distance_measure(exp_dev_X, exp_dev_Y, Lambda=0)
        return MMD_1, MMD_2