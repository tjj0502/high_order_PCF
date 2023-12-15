import numpy as np
import torch
import torch.cuda

class PCFKernel():
    """PCF kernel"""

    def __init__(self, pcf_kernel):
        # self.num_samples = num_samples
        # self.degree = degree
        # self.input_dim = input_dim

        if isinstance(pcf_kernel, list):
            self.pcf_kernel = pcf_kernel[0]
            self.pcf_kernel_higher_order = pcf_kernel[1]

            # Sanity check
            assert self.pcf_kernel_higher_order.input_dim == self.pcf_kernel.degree ** 2, "The high order PCF input dimension must agree with the previous PCF degree."
        else:
            self.pcf_kernel = pcf_kernel
            self.pcf_kernel_higher_order = pcf_kernel

        # self.pcf_kernel = char_func_path(num_samples=self.num_samples,
        #                                  hidden_size=self.degree,
        #                                  input_size=self.input_dim,
        #                                  add_time=add_time,
        #                                  include_initial=False)

    def HS_norm(self, X: torch.tensor, Y: torch.Tensor):
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

    def Gram_matrix(self, X, Y, lambda_):
        """Input:
                  - X: torch tensor of shape (batch_X, length_X, dim),
                  - Y: torch tensor of shape (batch_Y, length_Y, dim)
           Output:
                  - matrix k(X^i_s,Y^j_t) of shape (batch_X, batch_Y, length_X, length_Y)
        """
        batch_X, batch_Y, length_X, length_Y = X.shape[0], Y.shape[0], X.shape[1], Y.shape[1]
        dim = X.shape[-1]

        assert dim == self.pcf_kernel.input_dim, "The dimension must agree"
        assert batch_X == batch_Y, "The size of the paths must agree"
        # X = x_{0,s}, Y = x_{0,T}
        # print(X.shape, Y.shape)
        dev_x = self.pcf_kernel.unitary_development(X)  # Shape: (batch_X, num_samples, Lie, Lie)
        dev_y = self.pcf_kernel.unitary_development(Y)  # Shape: (batch_Y, num_samples, Lie, Lie)

        dev_x_reshaped = dev_x.unsqueeze(1)  # Shape: (batch_X, 1, num_samples, Lie, Lie)

        dev_x_ct_reshaped = dev_x.conj().permute([0, 1, 3, 2]).unsqueeze(0)  # Shape: (1, batch_X, num_samples, Lie, Lie)
        product_xx = torch.matmul(dev_x_ct_reshaped, dev_x_reshaped) # Shape: (batch_X, batch_X, num_samples, Lie, Lie)
        KXX = product_xx.diagonal(dim1=-2, dim2=-1).sum(-1).permute([2, 0, 1]) # Shape: (num_samples, batch_X, batch_X)

        # product_xy = torch.matmul(dev_x_ct_reshaped, dev_y_reshaped)
        # KXY = product_xy.diagonal(dim1=-2, dim2=-1).sum(-1)

        idm = tile(torch.eye(batch_X).unsqueeze(0), 0, self.pcf_kernel.num_samples).to(KXX.device)

        # Gram_matrix = torch.matmul(KXX.t(), torch.linalg.inv(KXX + batch_X * self.lambda_ * torch.eye(batch_X).to(KXX.device))) # Shape:

        Gram_matrix = torch.matmul(KXX.permute([0, 2, 1]), torch.linalg.inv(KXX + batch_X * lambda_ * idm)) # Shape: (num_samples, batch_X, batch_X)

        dev_y_reshaped = dev_y.permute([1, 0, 2, 3])  # Shape: (batch_Y, num_samples, Lie, Lie)

        mu_y_given_x = torch.matmul(Gram_matrix,
                                    dev_y_reshaped.reshape([self.pcf_kernel.num_samples,
                                                            batch_Y, -1])).reshape(self.pcf_kernel.num_samples*batch_Y, -1)
            # .reshape([self.pcf_kernel.num_samples,
            #                                                                         batch_Y,
            #                                                                         self.pcf_kernel.degree,
            #                                                                         self.pcf_kernel.degree]) # Shape: (num_samples, batch_Y, Lie, Lie)
        # print(mu_y_given_x.shape)
        return mu_y_given_x

    def compute_mmd(self, X, Y, order=1, lambda_=1.):
        """
            Corresponds to Algorithm 3 or 5 in "Higher Order Kernel Mean Embeddings to Capture Filtrations of Stochastic Processes"

            Input:
                  - X: torch tensor of shape (batch_X, length_X, dim),
                  - Y: torch tensor of shape (batch_Y, length_Y, dim),
                  - order: (int) the order of the MMD
                  - lambda_: (float) hyperparameter for the conditional KME estimator (to be specified if order=2)
           Output:
                  - scalar: MMD signature distance between samples X and samples Y
        """

        assert not Y.requires_grad, "the second input should not require grad"
        assert order == 1 or order == 2, "order>2 have not been implemented yet"

        if order == 2:
            return self._compute_higher_order_mmd(X, Y, lambda_=lambda_)

        pcf_x = self.pcf_kernel.unitary_development(X).mean(0)  # Shape: (batch_X, num_samples, Lie, Lie)
        pcf_y = self.pcf_kernel.unitary_development(Y).mean(0)  # Shape: (batch_Y, num_samples, Lie, Lie)

        return self.HS_norm(pcf_x - pcf_y, pcf_x - pcf_y)

    def _compute_higher_order_mmd(self, X, Y, lambda_=1.):
        '''Corresponds to Algorithm 2 in "Higher Order Kernel Mean Embeddings to Capture Filtrations of Stochastic Processes" '''

        # - X: torch tensor of shape(batch_X, length_X, dim), the truncated path
        # - Y: torch tensor of shape(batch_Y, length_Y, dim), the whole path
        # - Output: torch tensor of shape (batch_X, batch_X, length_X, length_X)

        batch_X = X.shape[0]  # A
        batch_Y = Y.shape[0]  # B
        length_X = X.shape[1]  # M
        length_Y = Y.shape[1]  # N
        feat_dim = X.shape[2]  # D

        # computing dsdt k(X^i_s,Y^j_t)
        mu_path_X = torch.zeros([self.pcf_kernel.num_samples * batch_X, length_X, self.pcf_kernel_higher_order.input_dim]).to(X.device)
        mu_path_Y = torch.zeros([self.pcf_kernel.num_samples * batch_Y, length_Y, self.pcf_kernel_higher_order.input_dim]).to(X.device)

        for s in range(length_X):
            X_filtration = X[:, :s + 1, :]
            mu_X = self.Gram_matrix(X_filtration, X, lambda_=lambda_)
            mu_path_X[:, s, :] = mu_X

        for s in range(length_Y):
            Y_filtration = Y[:, :s + 1, :]
            mu_Y = self.Gram_matrix(Y_filtration, Y, lambda_=lambda_)
            mu_path_Y[:, s, :] = mu_Y



        pcf_x = self.pcf_kernel_higher_order.unitary_development(mu_path_X).reshape([self.pcf_kernel.num_samples,
                                                                                     batch_X,
                                                                                     self.pcf_kernel_higher_order.num_samples,
                                                                                     self.pcf_kernel_higher_order.degree,
                                                                                     self.pcf_kernel_higher_order.degree]).mean(1).reshape([-1, self.pcf_kernel_higher_order.degree, self.pcf_kernel_higher_order.degree])  # Shape: (num_samples_1, num_samples_high_order, Lie, Lie)
        pcf_y = self.pcf_kernel_higher_order.unitary_development(mu_path_Y).reshape([self.pcf_kernel.num_samples,
                                                                                     batch_X,
                                                                                     self.pcf_kernel_higher_order.num_samples,
                                                                                     self.pcf_kernel_higher_order.degree,
                                                                                     self.pcf_kernel_higher_order.degree]).mean(1).reshape([-1, self.pcf_kernel_higher_order.degree, self.pcf_kernel_higher_order.degree])   # Shape: (num_samples_1, num_samples_high_order, Lie, Lie)

        return self.HS_norm(pcf_x - pcf_y, pcf_x - pcf_y)

def tile(a, dim, n_tile):
    """
    Tile a PyTorch tensor along a specific dimension.

    Args:
    a (torch.Tensor): The input tensor to be tiled.
    dim (int): The dimension along which to tile the tensor.
    n_tile (int): The number of times to tile the tensor along the specified dimension.

    Returns:
    torch.Tensor: The tiled tensor.

    Example:
    If a is a tensor of shape [2, 3] and we call tile(a, 0, 2),
    the output will be a tensor of shape [4, 3], where the original
    tensor is repeated along the first dimension (dim=0).
    """

    # Get the size of the tensor along the specified dimension
    init_dim = a.size(dim)

    # Create a repeat pattern where all dimensions are repeated only once except the specified 'dim'
    repeat_idx = [1] * a.dim()
    repeat_idx[dim] = n_tile

    # Repeat the tensor along each dimension as specified in repeat_idx
    a_repeated = a.repeat(*(repeat_idx))

    # Create an index tensor to rearrange the repeated elements into the correct order
    order_index = torch.LongTensor(np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)])).to(a.device)

    # Reorder the repeated tensor according to the order_index and return
    return torch.index_select(a_repeated, dim, order_index)

