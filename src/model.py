import torch
import torch.nn as nn
from src.utils import init_weights
from typing import Tuple
import torch.multiprocessing as mp


class RegressionBase(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(RegressionBase, self).__init__()
        """ Generator base class. All generators should be children of this class. """
        self.input_dim = input_dim
        self.output_dim = output_dim

    # @abstractmethod
    def forward_(self, batch_size: int, n_lags: int, device: str):
        """Implement here generation scheme."""
        # ...
        pass

    def forward(self, batch_size: int, n_lags: int, device: str):
        x = self.forward_(batch_size, n_lags, device)
        x = self.pipeline.inverse_transform(x)
        return x


class LSTMRegressor(RegressionBase):
    def __init__(
            self,
            input_dim: int,
            output_dim: int,
            hidden_dim: int,
            n_layers: int,
            activation=nn.Tanh(),
    ):
        super(LSTMRegressor, self).__init__(input_dim, output_dim)
        # LSTM
        self.rnn = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
        ).to(torch.float)
        self.rnn.apply(init_weights)
        self.linear = nn.Linear(hidden_dim, 2 * output_dim ** 2, bias=False).to(torch.float)
        self.linear.apply(init_weights)

        self.activation = activation


    def forward(
            self, input_path: torch.Tensor, device: str, z=None
    ) -> torch.Tensor:
        batch_size, n_lags, _ = input_path.shape

        z0 = input_path.to(device)

        h1, _ = self.rnn(z0)
        X = self.linear(self.activation(h1)).reshape(batch_size, n_lags, 2, self.output_dim, self.output_dim)

        X_complex = X[:, :, 0, :, :] + torch.tensor([1j], device=X.device) * X[:, :, 1, :, :]

        # X_complex_unitary = (X_complex - torch.conj(X_complex.transpose(-2, -1))) / 2
        # assert X_complex_unitary.shape[1] == n_lags

        return X_complex


# class LSTMRegressor_with_M(RegressionBase):
#     def __init__(
#             self,
#             input_dim: int,
#             output_dim: int,
#             hidden_dim: int,
#             n_layers: int,
#             M: torch.Tensor,
#             activation=nn.Tanh(),
#     ):
#         super(LSTMRegressor_with_M, self).__init__(input_dim, output_dim)
#         # LSTM
#         input_dim_, n, lie_degree, lie_degree = M.shape
#         self.M = torch.cat([M.flatten().real, M.flatten().imag]).detach()
#         assert input_dim_ == input_dim, "Input dimension does not agree."
#
#         self.rnn = nn.LSTM(
#             input_size=input_dim + self.M.shape[0],
#             hidden_size=hidden_dim,
#             num_layers=n_layers,
#             batch_first=True,
#         ).to(torch.float)
#         self.rnn.apply(init_weights)
#         self.linear = nn.Linear(hidden_dim, 2 * output_dim ** 2, bias=False).to(torch.float)
#         self.linear.apply(init_weights)
#
#         self.activation = activation
#
#
#     def forward(
#             self, input_path: torch.Tensor, device: str, z=None
#     ) -> torch.Tensor:
#         batch_size, n_lags, _ = input_path.shape
#
#         z0 = input_path.to(device)
#
#         h1, _ = self.rnn(torch.cat([z0, self.M.repeat([batch_size, n_lags, 1])], dim = 2))
#         X = self.linear(self.activation(h1)).reshape(batch_size, n_lags, 2, self.output_dim, self.output_dim)
#
#         X_complex = X[:, :, 0, :, :] + torch.tensor([1j], device=X.device) * X[:, :, 1, :, :]
#
#         # X_complex_unitary = (X_complex - torch.conj(X_complex.transpose(-2, -1))) / 2
#         # assert X_complex_unitary.shape[1] == n_lags
#
#         return X_complex
#
#
# class LSTMRegressor_with_M_2(RegressionBase):
#     def __init__(
#             self,
#             input_dim: int,
#             output_dim: int,
#             hidden_dim: int,
#             n_layers: int,
#             M: torch.Tensor,
#             activation=nn.Tanh(),
#     ):
#         super(LSTMRegressor_with_M_2, self).__init__(input_dim, output_dim)
#         # LSTM
#         input_dim_, n, lie_degree, lie_degree = M.shape
#         self.M = torch.cat([M.flatten().real, M.flatten().imag]).detach()
#         self.n_layers = n_layers
#         assert input_dim_ == input_dim, "Input dimension does not agree."
#
#         self.rnn = nn.LSTM(
#             input_size=input_dim,
#             hidden_size=hidden_dim,
#             num_layers=n_layers,
#             batch_first=True,
#         ).to(torch.float)
#         self.init_hidden_layer = nn.Linear(self.M.shape[0], hidden_dim)
#         self.rnn.apply(init_weights)
#         self.linear = nn.Linear(hidden_dim, 2 * output_dim ** 2, bias=False).to(torch.float)
#         self.linear.apply(init_weights)
#
#         self.activation = activation
#
#
#     def forward(
#             self, input_path: torch.Tensor, device: str, z=None
#     ) -> torch.Tensor:
#         batch_size, n_lags, _ = input_path.shape
#
#         init_hidden = self.init_hidden_layer(self.M).repeat([self.n_layers, batch_size, 1])
#         # print(init_hidden.shape)
#         init_cell = torch.zeros_like(init_hidden)
#         z0 = input_path.to(device)
#
#         h1, _ = self.rnn(z0, (init_hidden, init_cell))
#         X = self.linear(self.activation(h1)).reshape(batch_size, n_lags, 2, self.output_dim, self.output_dim)
#
#         X_complex = X[:, :, 0, :, :] + torch.tensor([1j], device=X.device) * X[:, :, 1, :, :]
#
#         # X_complex_unitary = (X_complex - torch.conj(X_complex.transpose(-2, -1))) / 2
#         # assert X_complex_unitary.shape[1] == n_lags
#
#         return X_complex
#
class LSTMRegressor_(RegressionBase):
    def __init__(
            self,
            input_dim: int,
            output_dim: int,
            hidden_dim: int,
            n_layers: int,
            n_linear_maps: int,
            activation=nn.Tanh(),
    ):
        super(LSTMRegressor_, self).__init__(input_dim, output_dim)
        self.n_linear_maps = n_linear_maps
        self.rnns = nn.ModuleList([nn.LSTM(input_size=input_dim,
                                           hidden_size=hidden_dim,
                                           num_layers=n_layers,
                                           batch_first=True,
                                           ).to(torch.float) for _ in range(n_linear_maps)])
        self.rnns.apply(init_weights)
        self.linears = nn.ModuleList([nn.Linear(hidden_dim, 2 * output_dim ** 2, bias=False).to(torch.float)
                                     for _ in range(n_linear_maps)])
        self.linears.apply(init_weights)

        self.activation = activation

    def forward(
            self, input_path: torch.Tensor, idx: int, device: str = 'cuda', z=None
    ) -> torch.Tensor:
        batch_size, n_lags, _ = input_path.shape

        z0 = input_path.to(device)

        h1, _ = self.rnns[idx](z0)
        X = self.linears[idx](self.activation(h1)).reshape(batch_size, n_lags, 2, self.output_dim, self.output_dim)

        X_complex = X[:, :, 0, :, :] + torch.tensor([1j], device=X.device) * X[:, :, 1, :, :]

        return X_complex


class GeneratorBase(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(GeneratorBase, self).__init__()
        """ Generator base class. All generators should be children of this class. """
        self.input_dim = input_dim
        self.output_dim = output_dim

    # @abstractmethod
    def forward_(self, batch_size: int, n_lags: int, device: str):
        """Implement here generation scheme."""
        # ...
        pass

    def forward(self, batch_size: int, n_lags: int, device: str):
        x = self.forward_(batch_size, n_lags, device)
        x = self.pipeline.inverse_transform(x)
        return x


class LSTMGenerator(GeneratorBase):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int,
        n_layers: int,
        noise_scale=0.1,
        BM=False,
        activation=nn.Tanh(),
    ):
        super(LSTMGenerator, self).__init__(input_dim, output_dim)
        # LSTM
        self.rnn = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
        )
        self.rnn.apply(init_weights)
        self.linear = nn.Linear(hidden_dim, output_dim, bias=False)
        self.linear.apply(init_weights)

        self.initial_nn = nn.Sequential(
            ResFNN(input_dim, hidden_dim * n_layers, [hidden_dim, hidden_dim]),
            nn.Tanh(),
        )  # we use a simple residual network to learn the distribution at the initial time step.
        self.initial_nn1 = nn.Sequential(
            ResFNN(input_dim, hidden_dim * n_layers, [hidden_dim, hidden_dim]),
            nn.Tanh(),
        )
        self.initial_nn.apply(init_weights)
        self.initial_nn1.apply(init_weights)

        self.BM = BM
        if BM:
            self.noise_scale = noise_scale
        else:
            self.noise_scale = 0.3
        self.activation = activation

    def forward(
        self, batch_size: int, n_lags: int, device: str, z=None
    ) -> torch.Tensor:
        if z == None:
            z = (self.noise_scale * torch.randn(batch_size, n_lags, self.input_dim)).to(
                device
            )  # cumsum(1)
            if self.BM:
                z = z.cumsum(1)
            else:
                pass
            # z[:, 0, :] *= 0  # first point is fixed
            #
        else:
            z = z
        z0 = self.noise_scale * torch.randn(batch_size, self.input_dim, device=device)

        h0 = (
            self.initial_nn(z0)
            .view(batch_size, self.rnn.num_layers, self.rnn.hidden_size)
            .permute(1, 0, 2)
            .contiguous()
        )
        c0 = (
            self.initial_nn1(z0)
            .view(batch_size, self.rnn.num_layers, self.rnn.hidden_size)
            .permute(1, 0, 2)
            .contiguous()
        )
        # c0 = torch.zeros_like(h0)

        h1, _ = self.rnn(z, (h0, c0))
        x = self.linear(self.activation(h1))

        assert x.shape[1] == n_lags

        return x


class ResidualBlock(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super(ResidualBlock, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.activation = nn.LeakyReLU()
        self.create_residual_connection = True if input_dim == output_dim else False

    def forward(self, x):
        y = self.activation(self.linear(x))
        if self.create_residual_connection:
            y = x + y
        return y


class ResFNN(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: Tuple[int],
        flatten: bool = False,
    ):
        """
        Feedforward neural network with residual connection.
        Args:
            input_dim: integer, specifies input dimension of the neural network
            output_dim: integer, specifies output dimension of the neural network
            hidden_dims: list of integers, specifies the hidden dimensions of each layer.
                in above definition L = len(hidden_dims) since the last hidden layer is followed by an output layer
        """
        super(ResFNN, self).__init__()
        blocks = list()
        self.input_dim = input_dim
        self.flatten = flatten
        input_dim_block = input_dim
        for hidden_dim in hidden_dims:
            blocks.append(ResidualBlock(input_dim_block, hidden_dim))
            input_dim_block = hidden_dim
        blocks.append(torch.nn.Tanh())
        blocks.append(nn.Linear(input_dim_block, output_dim))
        self.network = nn.Sequential(*blocks)
        self.blocks = blocks

    def forward(self, x):
        if self.flatten:
            x = x.reshape(x.shape[0], -1)
        out = self.network(x)
        return out