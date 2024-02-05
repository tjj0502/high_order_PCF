import torch
import torch.nn as nn
from Path_Char.utils import init_weights
from typing import Tuple
import torch.multiprocessing as mp


def matrix_power_two_batch(A, k):
    """
    Computes the matrix power of A for each element in k using batch processing.

    Args:
        A (torch.Tensor): Input tensor of shape (..., m, m).
        k (torch.Tensor): Exponent tensor of shape (...).

    Returns:
        torch.Tensor: Resulting tensor of shape (..., m, m).
    """
    orig_size = A.size()
    A, k = A.flatten(0, -3), k.flatten()
    ksorted, idx = torch.sort(k)
    # Abusing bincount...
    count = torch.bincount(ksorted)
    nonzero = torch.nonzero(count, as_tuple=False)
    A = torch.matrix_power(A, 2 ** ksorted[0])
    last = ksorted[0]
    processed = count[nonzero[0]]
    for exp in nonzero[1:]:
        new, last = exp - last, exp
        A[idx[processed:]] = torch.matrix_power(A[idx[processed:]], 2 ** new.item())
        processed += count[exp]
    return A.reshape(orig_size)


def rescaled_matrix_exp(f, A):
    """
    Computes the rescaled matrix exponential of A.
    By following formula exp(A) = (exp(A/k))^k

    Args:
        f (callable): Function to compute the matrix exponential.
        A (torch.Tensor): Input tensor of shape (..., m, m).

    Returns:
        torch.Tensor: Resulting tensor of shape (..., m, m).
    """
    normA = torch.max(torch.sum(torch.abs(A), axis=-2), axis=-1).values
    more = normA > 1
    s = torch.ceil(torch.log2(normA)).long()
    s = normA.new_zeros(normA.size(), dtype=torch.long)
    s[more] = torch.ceil(torch.log2(normA[more])).long()
    A_1 = torch.pow(0.5, s.float()).unsqueeze_(-1).unsqueeze_(-1).expand_as(A) * A
    return matrix_power_two_batch(f(A_1), s)


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


class UnitaryLSTMGenerator(GeneratorBase):
    def __init__(
            self,
            input_dim: int,
            output_dim: int,
            hidden_dim: int,
            n_layers: int,
            activation=nn.Tanh(),
    ):
        super(UnitaryLSTMGenerator, self).__init__(input_dim, output_dim)
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
        self.triv = torch.linalg.matrix_exp

    def forward(
            self, input_path: torch.Tensor, device: str, z=None
    ) -> torch.Tensor:
        batch_size, n_lags, _ = input_path.shape

        z0 = input_path.to(device)

        h1, _ = self.rnn(z0)
        X = self.linear(self.activation(h1)).reshape(batch_size, n_lags, 2, self.output_dim, self.output_dim)

        X_complex = X[:, :, 0, :, :] + torch.tensor([1j], device=X.device) * X[:, :, 1, :, :]

        X_complex_unitary = (X_complex - torch.conj(X_complex.transpose(-2, -1))) / 2
        assert X_complex_unitary.shape[1] == n_lags

        return rescaled_matrix_exp(self.triv, X_complex_unitary)


class LSTMGenerator(GeneratorBase):
    def __init__(
            self,
            input_dim: int,
            output_dim: int,
            hidden_dim: int,
            n_layers: int,
            activation=nn.Tanh(),
    ):
        super(LSTMGenerator, self).__init__(input_dim, output_dim)
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

    # def forward(
    #         self, input_path: torch.Tensor, device: str, z=None
    # ) -> torch.Tensor:
    #     batch_size, n_lags, _ = input_path.shape
    #
    #     z0 = input_path.to(device)
    #
    #     h1, _ = self.rnn(z0)
    #     X = self.linear(self.activation(h1))
    #
    #     assert X.shape[1] == n_lags
    #
    #     return X

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


class UnitaryLSTMRegressor(GeneratorBase):
    def __init__(
            self,
            input_dim: int,
            output_dim: int,
            hidden_dim: int,
            n_layers: int,
            n_linear_maps: int,
            activation=nn.Tanh(),
    ):
        super(UnitaryLSTMRegressor, self).__init__(input_dim, output_dim)
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
        self.triv = torch.linalg.matrix_exp

    def forward(
            self, input_path: torch.Tensor, idx: int, device: str = 'cuda', z=None
    ) -> torch.Tensor:
        batch_size, n_lags, _ = input_path.shape

        z0 = input_path.to(device)

        h1, _ = self.rnns[idx](z0)
        X = self.linears[idx](self.activation(h1)).reshape(batch_size, n_lags, 2, self.output_dim, self.output_dim)

        X_complex = X[:, :, 0, :, :] + torch.tensor([1j], device=X.device) * X[:, :, 1, :, :]

        X_complex_unitary = (X_complex - torch.conj(X_complex.transpose(-2, -1))) / 2
        assert X_complex_unitary.shape[1] == n_lags

        return rescaled_matrix_exp(self.triv, X_complex_unitary)