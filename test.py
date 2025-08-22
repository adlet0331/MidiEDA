import torch, json
from model import RubricNet
import numpy as np

def get_scaled_features_data_from_splits(runs_name = 'p-est-250822-144330'):
    model_snapshot_path = f'/Users/simhyeongju/AVAPT/EDA/runs/{runs_name}/model_snapshots/model_bestvalidation.pt'
    cipi_cached_path = '/Users/simhyeongju/AVAPT/data/CIPI/features/features_v1.json'
    cipi_label_path = '/Users/simhyeongju/AVAPT/data/CIPI/index.json'

    rubricnet = RubricNet()
    rubricnet.load_state_dict(torch.load(model_snapshot_path, weights_only=False))
    mean, std = rubricnet.get_scaler_infos()
    mean, std = mean.numpy(), std.numpy()

    cipi_features = json.load(open(cipi_cached_path, 'r'))
    features_mem = cipi_features['features_mem']
    features_name = cipi_features['features_names']
    cipi_labels = json.load(open(cipi_label_path, 'r'))

    label_features = [None for _ in range(9)] # [난이도][feature index]
    for key, features in features_mem.items():
        features = np.array(features)
        label = int(cipi_labels[key]["henle"]) - 1
        scaled_features = (features - mean) / std
        if label_features[label] is None:
            label_features[label] = np.expand_dims(scaled_features, axis=0)
        else:
            label_features[label] = np.concatenate((label_features[label], np.expand_dims(scaled_features, axis=0)), axis=0)

    scaled_label_features_count = [features.shape[0] for features in label_features]
    scaled_label_features_mean = [np.mean(features, axis=0) for features in label_features]

    return features_name, scaled_label_features_count, scaled_label_features_mean