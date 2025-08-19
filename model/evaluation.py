from statistics import mean
from sklearn.metrics import mean_squared_error
import torch

def inference_from_pred(pred, threshold=0.5):
    """Convert ordinal predictions to class labels."""
    # if ((pred > threshold).cumprod(axis=1).sum(axis=1) == 0):
    #     raise ValueError(f"{pred}: Warning: All predictions are below the threshold. Returning zeros.")
    return torch.clamp((pred > threshold).cumprod(axis=1).sum(axis=1), min=1)

def get_acc1_macro(y_true, y_pred):
    """Calculate macro accuracy with a tolerance of +/- 1 for each class."""
    acc_plusless_1_each_class = []
    for true_class in set(y_true):
        matches = [1 if pp in [tt - 1, tt, tt + 1] else 0 for tt, pp in zip(y_true, y_pred) if tt == true_class]
        acc_plusless_1_each_class.append(sum(matches) / len(matches))
    return mean(acc_plusless_1_each_class)


def get_mse_macro(y_true, y_pred):
    """Calculate macro mean squared error for each class."""
    mse_each_class = []
    for true_class in set(y_true):
        tt, pp = zip(*[[tt, pp] for tt, pp in zip(y_true, y_pred) if tt == true_class])
        mse_each_class.append(mean_squared_error(y_true=tt, y_pred=pp))
    return mean(mse_each_class)