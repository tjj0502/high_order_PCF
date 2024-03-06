import torch
from tqdm import tqdm
from src.utils import AddTime, to_numpy, track_gradient_norms, track_norm, construct_past_dev_path
from src.path_characteristic_function import char_func_path
import torch.optim.swa_utils as swa_utils
import matplotlib.pyplot as plt
from os import path as pt
from collections import defaultdict
import seaborn as sns


def toggle_grad(model, requires_grad):
    for p in model.parameters():
        p.requires_grad_(requires_grad)



class HighRankPCFGANTrainer:
    def __init__(self, G, train_dl, rank_1_pcf, config, regression_module, **kwargs):
        """
        Trainer class for the basic PCF-GAN, without time serier embedding module.

        Args:
            G (torch.nn.Module): PCFG generator model.
            train_dl (torch.utils.data.DataLoader): Training data loader.
            config: Configuration object containing hyperparameters and settings.
            **kwargs: Additional keyword arguments for the base trainer class.
        """
        super(HighRankPCFGANTrainer, self).__init__()

        self.G = G
        self.G_optimizer = torch.optim.Adam(
                G.parameters(), lr=config.lr_G, betas=(0, 0.9))
        self.config = config
        self.add_time = config.add_time
        self.train_dl = train_dl
        self.D_steps_per_G_step = config.D_steps_per_G_step
        self.G_steps_per_D_step = config.G_steps_per_D_step

        self.rank_1_pcf = rank_1_pcf
        self.regression_module = regression_module
        self.D = char_func_path(num_samples=config.Rank_2_num_samples,
                                         hidden_size=config.Rank_2_lie_degree,
                                         input_dim=2 * config.Rank_1_lie_degree ** 2,
                                         add_time=self.add_time,
                                         include_initial=False,
                                         return_sequence=False)

        self.D_optimizer = torch.optim.Adam(
            self.D.parameters(), lr=config.lr_D, betas=(0, 0.9)
        )
        self.averaged_G = swa_utils.AveragedModel(G)
        self.G_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            self.G_optimizer, gamma=config.gamma
        )
        self.D_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            self.D_optimizer, gamma=config.gamma
        )
        self.n_gradient_steps = config.steps
        self.batch_size = config.batch_size
        self.losses_history = defaultdict(list)
        self.device = config.device

    def fit(self, device):
        """
        Trains the PCFGAN model.

        Args:
            device: Device to perform training on.
        """

        self.G.to(device)
        self.D.to(device)
        toggle_grad(self.regression_module, False)
        toggle_grad(self.rank_1_pcf, False)

        for i in tqdm(range(self.n_gradient_steps)):
            self.step(device, i)
            if i > self.config.swa_step_start:
                self.averaged_G.update_parameters(self.G)

    def step(self, device, step):
        """
        Performs one training step.

        Args:
            device: Device to perform training on.
            step (int): Current training step.
        """
        # for i in range(self.D_steps_per_G_step):
        #     # generate x_fake
        #
        with torch.no_grad():
            x_real_batch = next(iter(self.train_dl))[0].to(device)
        #         x_fake = self.G(
        #             batch_size=self.batch_size,
        #             n_lags=self.config.n_lags,
        #             device=device,
        #         )
        #
        #     D_loss = self.D_trainstep(x_fake, x_real_batch)
        #     if i == 0:
        #         self.losses_history["D_loss"].append(D_loss)

        for i in range(self.G_steps_per_D_step):
            G_loss = self.G_trainstep(x_real_batch, device, step, i)
            self.losses_history["G_loss"].append(G_loss)
        torch.cuda.empty_cache()
        # G_loss = self.G_trainstep(x_real_batch, device, step)
        if step % 500 == 0:
            self.G_lr_scheduler.step()
            for param_group in self.G_optimizer.param_groups:
                print("Learning Rate: {}".format(param_group["lr"]))
        else:
            pass

    def G_trainstep(self, x_real, device, step, i=0):
        """
        Performs one training step for the generator.

        Args:
            x_real: Real samples for training.
            device: Device to perform training on.
            step (int): Current training step.

        Returns:
            float: Generator loss value.
        """
        toggle_grad(self.G, True)
        toggle_grad(self.regression_module, False)
        self.G.train()
        self.G_optimizer.zero_grad()
        self.regression_module.train()
        self.D.train()
        x_fake = self.G(
            batch_size=self.batch_size,
            n_lags=self.config.n_lags,
            device=device,
        )
        # if self.loss == "both":
        #     G_loss = self.char_func.distance_measure(x_real, x_fake, Lambda=0.1)

        # On real data, use regression module to compute the expected development
        exp_dev_real = self.regression_module(AddTime(x_real), self.device)
        past_dev_real = construct_past_dev_path(self.rank_1_pcf,  AddTime(x_real), x_real.shape[1])

        # On fake data, do the same
        exp_dev_fake = self.regression_module(AddTime(x_fake), self.device)
        past_dev_fake = construct_past_dev_path(self.rank_1_pcf, AddTime(x_fake), x_real.shape[1])

        G_loss = torch.norm(past_dev_real @ exp_dev_real - past_dev_fake@ exp_dev_fake, dim=(2, 3)).mean(0).sum()

        # expected_dev_real = self.exp_dev_by_regression(x_real)
        # expected_dev_fake = self.exp_dev_by_regression(x_fake)
        #
        # G_loss = self.D.distance_measure(expected_dev_real, expected_dev_fake, Lambda=0.1)  # (T)
        # print(G_loss.shape)
        # self.losses_history['G_loss_dyadic'].append(G_loss)
        # G_loss = G_loss.mean()
        G_loss.backward()
        self.losses_history["G_loss"].append(G_loss.item())

        if i == 0:
            grad_norm_G = track_gradient_norms(self.G)
            # grad_norm_D = track_gradient_norms(self.D)
            norm_G = track_norm(self.G)
            # norm_D = track_norm(self.D)
            self.losses_history['grad_norm_G'].append(grad_norm_G)
            # self.losses_history['grad_norm_D'].append(grad_norm_D)
            self.losses_history['norm_G'].append(norm_G)
            # self.losses_history['norm_D'].append(norm_D)
        torch.nn.utils.clip_grad_norm_(self.G.parameters(), self.config.grad_clip)
        self.G_optimizer.step()
        toggle_grad(self.G, False)
        if step % self.config.evaluate_every == 0 and i==0:
            self.plot_sample(x_real, x_fake, self.config, step)
            # # print(torch.stack(self.losses_history['G_loss_dyadic']).shape)
            # plt.plot(to_numpy(torch.stack(self.losses_history['G_loss_dyadic'])))
            # plt.savefig(
            #     pt.join(self.config.exp_dir, "G_loss_dyadic_" + str(step) + ".png")
            # )
            # plt.close()
            self.plot_losses(loss_item="G_loss", step=step)
            self.plot_losses(loss_item="grad_norm_G", step=step)
            # self.plot_losses(loss_item="grad_norm_D", step=step)
            # self.plot_losses(loss_item="SigMMD", step=step)

        return G_loss.item()

    def D_trainstep(self, x_fake, x_real):
        """
        Performs one training step for the discriminator.

        Args:
            x_fake: Fake samples generated by the generator.
            x_real: Real samples for training.

        Returns:
            float: Discriminator loss value.
        """
        x_real.requires_grad_()
        toggle_grad(self.D, True)
        toggle_grad(self.regression_module, False)
        self.regression_module.train()
        self.D.train()
        self.D_optimizer.zero_grad()
        # print(x_real.shape, x_fake.shape)
        # exp_dev_real = self.exp_dev_by_regression(x_real)
        # exp_dev_fake = self.exp_dev_by_regression(x_fake)

        # On real data, use regression module to compute the expected development
        exp_dev_real = self.regression_module(AddTime(x_real), self.device)
        past_dev_real = construct_past_dev_path(self.pcf_rank_1,  AddTime(x_real), x_real.shape[1]+1)

        # On fake data, do the same
        exp_dev_fake = self.regression_module(AddTime(x_fake), self.device)
        past_dev_fake = construct_past_dev_path(self.pcf_rank_1, AddTime(x_fake), x_real.shape[1] + 1)



        # exp_dev_real = (past_dev_real @ exp_dev_real).reshape([-1, x_real.shape[1], self.config.lie_degree_1 ** 2])
        # exp_dev_fake = (past_dev_fake @ exp_dev_real).reshape([-1, x_real.shape[1], self.config.lie_degree_1 ** 2])

        d_loss = -self.D.distance_measure(exp_dev_real, exp_dev_fake, Lambda=0.1)

        d_loss.backward()

        # Step discriminator params
        self.D_optimizer.step()

        # Toggle gradient to False
        toggle_grad(self.D, False)

        return d_loss.item()

    def exp_dev_by_regression(self, x):
        # with torch.no_grad():
        x_with_time = AddTime(x)
        exp_dev = self.regression_module(x_with_time, self.device)
        past_dev = construct_past_dev_path(self.rank_1_pcf, x_with_time, x.shape[1])
        exp_dev = (past_dev @ exp_dev).reshape([-1, x.shape[1], self.config.Rank_1_lie_degree ** 2])
        exp_dev= torch.cat([exp_dev.real, exp_dev.imag], -1)
        return exp_dev

    def plot_losses(self, loss_item: str, step: int = 0):
        plt.plot(self.losses_history[loss_item])
        plt.savefig(
            pt.join(self.config.exp_dir, loss_item + "_" + str(step) + ".png")
        )
        plt.close()

    @staticmethod
    def plot_sample(real_X, fake_X, config, step):
        sns.set()

        x_real_dim = real_X.shape[-1]
        for i in range(x_real_dim):
            plt.plot(
                to_numpy(fake_X[: config.batch_size, :, i]).T, "C%s" % i, alpha=0.3
            )
        plt.savefig(pt.join(config.exp_dir, "x_fake_" + str(step) + ".png"))
        plt.close()

        for i in range(x_real_dim):
            random_indices = torch.randint(0, real_X.shape[0], (config.batch_size,))
            plt.plot(to_numpy(real_X[random_indices, :, i]).T, "C%s" % i, alpha=0.3)
        plt.savefig(pt.join(config.exp_dir, "x_real_" + str(step) + ".png"))
        plt.close()