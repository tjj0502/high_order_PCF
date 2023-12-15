import torch
import torch.nn as nn
from Path_Char.unitary_representation import development_layer
from Path_Char.utils import AddTime

class char_func_path(nn.Module):
    def __init__(self, num_samples, hidden_size, input_dim, add_time: bool, init_range: float = 1, include_initial: bool = False):
        """
        Path characteristic function class from paths
        Args:
            num_samples: the number of linear maps L(R^d, u(n))
            hidden_size: the degree of the unitary Lie algebra
            input_dim: the path dimension, R^d
            add_time: Apply time augmentation
        """
        super(char_func_path, self).__init__()
        self.num_samples = num_samples
        self.degree = hidden_size
        self.input_dim = input_dim
        if add_time:
            self.input_dim = input_dim + 1
        else:
            self.input_dim = input_dim + 0
        self.unitary_development = development_layer(input_size=self.input_dim,
                                                     hidden_size=self.degree,
                                                     channels=self.num_samples,
                                                     include_initial=include_initial,
                                                     return_sequence=False,
                                                     init_range=init_range)

        for param in self.unitary_development.parameters():
            param.requires_grad = True
        self.add_time = add_time

    def reset_parameters(self):
        pass

    @staticmethod
    def HS_norm(X: torch.tensor, Y: torch.Tensor):
        """_summary_

        Args:
            X (torch.Tensor): (C,m,m) complexed valued
        """
        if len(X.shape) == 4:

            m = X.shape[-1]
            X = X.reshape(-1, m, m)

        else:
            pass
        D = torch.bmm(X, torch.conj(Y).permute(0, 2, 1))
        return ((torch.einsum('bii->b', D)).mean().real)

    def distance_measure(self, X1: torch.tensor, X2: torch.tensor, Lambda=0.1) -> torch.float:
        """distance measure given by the Hilbert-schmidt inner product
           d_hs(A,B) = trace[(A-B)(A-B)*]**(0.5)
           measure = \integral d_hs(\phi_{x1}(m),\phi_{x2}(m)) dF_M(m)
           let m be the linear map sampled from F_M(m)
        Args:
            X1 (torch.tensor): time series samples with shape (N_1,T,d)
            X2 (torch.tensor): time series samples with shape (N_2,T,d)
        Returns:
            torch.float: distance measure between two batch of samples
        """
        # print(X1.shape)
        if self.add_time:
            X1 = AddTime(X1)
            X2 = AddTime(X2)
        else:
            pass
        # print(X1.shape)
        dev1, dev2 = self.unitary_development(X1), self.unitary_development(X2)
        N, T, d = X1.shape

        # initial_dev = self.unitary_development_initial()
        CF1, CF2 = dev1.mean(
            0), dev2.mean(0)

        if Lambda != 0:
            initial_incre_X1 = torch.cat(
                [torch.zeros((N, 1, d)).to(X1.device), X1[:, 0, :].unsqueeze(1)], dim=1)
            initial_incre_X2 = torch.cat(
                [torch.zeros((N, 1, d)).to(X1.device), X2[:, 0, :].unsqueeze(1)], dim=1)
            initial_CF_1 = self.unitary_development(initial_incre_X1).mean(0)
            initial_CF_2 = self.unitary_development(initial_incre_X2).mean(0)
            return self.HS_norm(CF1-CF2, CF1-CF2) + Lambda*self.HS_norm(initial_CF_1-initial_CF_2, initial_CF_1-initial_CF_2)
        else:
            return self.HS_norm(CF1-CF2, CF1-CF2)