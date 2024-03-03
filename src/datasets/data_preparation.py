import torch
from torch.utils.data import DataLoader, Dataset
from src.utils import construct_past_dev_path, construct_future_dev_path, AddTime

class XYDataset(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
        self.shape = X.shape

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index], self.Y[index]

def prepare_dl(config, rank_1_pcf, X_train, X_Test):
    device = X_train.device
    steps = X_train.shape[1]
    print(steps)
    with torch.no_grad():
        future_dev_path_X = construct_future_dev_path(rank_1_pcf, AddTime(X_train).to(device), steps)
        future_dev_path_X_test = construct_future_dev_path(rank_1_pcf, AddTime(X_Test).to(device), steps)
        past_dev_path_X = construct_past_dev_path(rank_1_pcf, AddTime(X_train).to(device), steps)
        past_dev_path_X_test = construct_past_dev_path(rank_1_pcf, AddTime(X_Test).to(device), steps)

    """
    Regression dataset
    """
    train_reg_X_ds = XYDataset(AddTime(X_train), future_dev_path_X)
    test_reg_X_ds = XYDataset(AddTime(X_Test), future_dev_path_X_test)
    train_reg_X_dl = DataLoader(train_reg_X_ds, config.batch_size, shuffle=True)
    test_reg_X_dl = DataLoader(test_reg_X_ds, config.batch_size, shuffle=True)

    """
    PCF dataset
    """
    train_pcf_X_ds = XYDataset(X_train, past_dev_path_X)
    test_pcf_X_ds = XYDataset(X_Test, past_dev_path_X_test)
    train_pcf_X_dl = DataLoader(train_pcf_X_ds, config.batch_size, shuffle=True)
    test_pcf_X_dl = DataLoader(test_pcf_X_ds, config.batch_size, shuffle=True)

    return train_reg_X_dl, test_reg_X_dl, train_pcf_X_dl, test_pcf_X_dl