import math
from functools import partial

import torch
import torch.nn as nn



def matrix_power_two_batch(A, k):
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
        A[idx[processed:]] = torch.matrix_power(
            A[idx[processed:]], 2 ** new.item())
        processed += count[exp]
    return A.reshape(orig_size)


def rescaled_matrix_exp(f, A):
    """
    An efficient way of computing the tensor exponential
    :param f:
    :param A:
    :return:
    """
    normA = torch.max(torch.sum(torch.abs(A), axis=-2), axis=-1).values
    more = normA > 1
    s = torch.ceil(torch.log2(normA)).long()
    s = normA.new_zeros(normA.size(), dtype=torch.long)
    s[more] = torch.ceil(torch.log2(normA[more])).long()
    A_1 = torch.pow(
        0.5, s.float()).unsqueeze_(-1).unsqueeze_(-1).expand_as(A) * A
    # print(A_1.shape)
    return matrix_power_two_batch(f(A_1), s)


class unitary_diag(nn.Module):
    def __init__(self):
        super().__init__()

    @ staticmethod
    def frame(X: torch.tensor) -> torch.tensor:
        M, C, n = X.shape
        matrix = torch.zeros((M, C, n+1, n+1)).to(X.device).to(X.dtype)
        indices = torch.arange(0, n)
        matrix[:, :, indices, indices + 1] = X
        matrix = (matrix - torch.conj(matrix.transpose(-2, -1))) / 2
        return matrix

    def forward(self, X: torch.tensor) -> torch.tensor:
        if len(X.size()) < 2:
            raise ValueError('weights has dimension < 2')
        return self.frame(X)

    @ staticmethod
    def in_lie_algebra(X, eps=1e-5):
        return (X.dim() >= 2
                and X.size(-2) == X.size(-1)
                and torch.allclose(torch.conj(X.transpose(-2, -1)), -X, atol=eps))


def unitary_diag_lie_init_(tensor: torch.tensor, init_=None):
    """
    Fills in the input tensor in place with initialization on the unitary Lie algebra.

    Args:
        tensor (torch.Tensor): A multi-dimensional tensor.
        init_ (callable): Optional. A function that initializes the tensor according to some distribution.

    Raises:
        ValueError: If the tensor has less than 2 dimensions or the last two dimensions are not square.

    """
    with torch.no_grad():
        if tensor.ndim < 2:
            raise ValueError(
                "Only tensors with 2 or more dimensions are supported. "
                "Got a tensor of shape {}".format(tuple(tensor.size()))
            )

        if init_ is None:
            torch.nn.init.uniform_(tensor, -math.pi, math.pi)
        else:
            init_(tensor)

    return tensor


class unitary_diag_projection(nn.Module):
    def __init__(self, input_size, hidden_size, channels=1, init_range=1, **kwargs):
        """
        Projection module used to project the path increments to the Lie group path increments
        using trainable weights from the Lie algebra.

        Args:
            input_size (int): Input size.
            hidden_size (int): Size of the hidden Lie algebra matrix.
            channels (int, optional): Number of channels to produce independent Lie algebra weights. Defaults to 1.
            init_range (int, optional): Range for weight initialization. Defaults to 1.
        """
        self.__dict__.update(kwargs)

        A = torch.empty(
            input_size, channels, hidden_size-1, dtype=torch.cfloat
        )
        self.channels = channels
        super(unitary_diag_projection, self).__init__()
        self.param_map = unitary_diag()
        self.A = nn.Parameter(A)

        self.triv = torch.linalg.matrix_exp
        self.init_range = init_range
        self.reset_parameters()

        self.hidden_size = hidden_size

    def reset_parameters(self):
        upper_triangular_lie_init_(self.A, partial(nn.init.normal_, std=1))

    def forward(self, dX: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the projection module.

        Args:
            dX (torch.Tensor): Tensor of shape (N, input_size).

        Returns:
            torch.Tensor: Tensor of shape (N, channels, hidden_size, hidden_size).
        """
        A = self.param_map(self.A).permute(1, 2, -1, 0)  # C,m,m,in
        AX = A.matmul(dX.T).permute(-1, 0, 1, 2)  # ->C,m,m,N->N,C,m,m

        return rescaled_matrix_exp(self.triv, AX)


def upper_triangular_lie_init_(tensor: torch.tensor, init_=None):
    """
    Fills in the input tensor in place with initialization on the unitary Lie algebra.

    Args:
        tensor (torch.Tensor): A multi-dimensional tensor.
        init_ (callable): Optional. A function that initializes the tensor according to some distribution.

    Raises:
        ValueError: If the tensor has less than 2 dimensions or the last two dimensions are not square.

    """
    with torch.no_grad():
        if tensor.ndim < 2:
            raise ValueError(
                "Only tensors with 2 or more dimensions are supported. "
                "Got a tensor of shape {}".format(tuple(tensor.size()))
            )

        if init_ is None:
            torch.nn.init.uniform_(tensor, -math.pi, math.pi)
        else:
            init_(tensor)

    return tensor