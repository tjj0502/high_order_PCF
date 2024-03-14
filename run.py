import os

from os import path as pt
from src.utils import get_experiment_dir, save_obj, load_config
import torch
from torch import nn
import matplotlib
from src.model.discriminator.path_characteristic_function import pcf
from src.datasets.data_preparation import prepare_dl
from src.model import LSTMGenerator
from src.model.regressor.regressor import LSTMRegressor
from src.train_regressor import train_regressor, plot_reg_losses

def main(config):
    """
    Main function for training a synthetic data generator.

    Args:
        config (object): Configuration object containing the experiment settings.

    Returns:
        tuple: A tuple containing the discriminative score, predictive score, and signature MMD.
    """
    # print(config)
    torch.manual_seed(config.seed)
    matplotlib.use('Agg')
    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu_id
    print(os.environ["CUDA_VISIBLE_DEVICES"])
    print(config)
    if config.device == "cuda" and torch.cuda.is_available():
        config.update({"device": "cuda:0"}, allow_val_change=True)
    else:
        config.update({"device": "cpu"}, allow_val_change=True)

    get_experiment_dir(config)

    from src.datasets.fbm_ import FBM_data

    samples = 500
    steps = 20

    fbm_h = FBM_data(samples, dim=3, length=steps, h=0.2).to(config.device)
    fbm_h_test = FBM_data(40, dim=3, length=steps, h=0.2).to(config.device)

    print(fbm_h.shape)
    config.R_input_dim = fbm_h.shape[-1]+1
    config.data_feat_dim = fbm_h.shape[-1]
    config.n_lags = fbm_h.shape[1]

    rank_1_pcf = pcf(num_samples=config.Rank_1_num_samples,
                     hidden_size=config.Rank_1_lie_degree,
                     input_dim=fbm_h.shape[-1],
                     add_time=True,
                     include_initial=False,
                     return_sequence=False).to(config.device)

    train_reg_X_dl, test_reg_X_dl, train_pcf_X_dl, test_pcf_X_dl = prepare_dl(config, rank_1_pcf, fbm_h, fbm_h_test)

    regressor = LSTMRegressor(
        input_dim=config.R_input_dim,
        hidden_dim=config.R_hidden_dim,
        output_dim=config.R_output_dim,
        n_layers=config.R_num_layers
    )
    regressor.to(config.device)

    trained_regressor, loss, test_loss = train_regressor(regressor, config, train_reg_X_dl, test_reg_X_dl)

    save_obj(
        trained_regressor.state_dict(), pt.join(config.exp_dir, "regressor_state_dict.pt")
    )

    plot_reg_losses(loss, config, 'train')
    plot_reg_losses(test_loss, config, 'test')


    from src.trainers.High_rank_PCFGAN import HighRankPCFGANTrainer
    generator = LSTMGenerator(input_dim=config.G_input_dim,
                              hidden_dim=config.G_hidden_dim,
                              output_dim=config.data_feat_dim,
                              n_layers=config.G_num_layers,
                              noise_scale=config.noise_scale,
                              BM=config.BM,
                              activation=nn.Tanh(),).to(config.device)

    trainer = HighRankPCFGANTrainer(G=generator,train_dl=train_pcf_X_dl, rank_1_pcf=rank_1_pcf, config=config, regression_module=trained_regressor)

    save_obj(config, pt.join(config.exp_dir, "config.pkl"))

    # Train the model
    if config.train:
        # Print arguments (Sanity check)
        print(config)
        import datetime

        print(datetime.datetime.now())
        trainer.fit(config.device)
        save_obj(
            trainer.G.state_dict(), pt.join(config.exp_dir, "generator_state_dict.pt")
        )

        save_obj(
            trainer.averaged_G.module.state_dict(),
            pt.join(config.exp_dir, "ave_generator_state_dict.pt"),
        )


    return


if __name__ == "__main__":
    config_dir = pt.join("configs/configs.yaml")
    main(load_config(config_dir))