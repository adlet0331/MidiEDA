import torch, json, os
from model import RubricNet, MidiFeatures
import numpy as np
import matplotlib.pyplot as plt

def get_mid_files_statistics_per_segments(runs_name = "p-est-250822-160040", mid_folder_path = "/Users/simhyeongju/AVAPT/data/pianovam/midi"):
    # Load Model
    model_snapshot_path = f'/Users/simhyeongju/AVAPT/EDA/runs/{runs_name}/model_snapshots/model_bestvalidation.pt'
    rubricnet = RubricNet()
    rubricnet.load_state_dict(torch.load(model_snapshot_path, weights_only=False))

    mid_file_list = [f for f in os.listdir(mid_folder_path) if f.endswith(('.mid', '.xml'))]
    difficulty_scores = [0 for _ in range(9)]

    # ====================== Scaling Features======================

    features_list = []
    for filename in mid_file_list:
        midi_path = os.path.join(mid_folder_path, filename)
        midi_features = MidiFeatures(midi_path)
        features = midi_features.get_numeric_features()
        features_list.append(features)
    features_list = np.array(features_list)

    feature_mean_std_list = [[], []]
    for idx in range(14):
        feature_list = features_list[:, idx].reshape(-1, 1)
        feature_mean_std_list[0].append(np.mean(feature_list))
        feature_mean_std_list[1].append(np.std(feature_list))
    rubricnet.set_scaler_parameter_manually(feature_mean_std_list[0], feature_mean_std_list[1])

    # ====================== Scaling Features Ends ======================

    for idx, filename in enumerate(mid_file_list):
        midi_path = os.path.join(mid_folder_path, filename)
        midi_features = MidiFeatures(midi_path).extract_features_segments(segment_length_sec=120)
        # Extract features
        features = torch.tensor(midi_features)
        # Get predictions
        with torch.no_grad():
            predictions = rubricnet.predict(features).numpy().tolist()
        # Save statistics
        # print(f"File: {filename}, Prediction: {predictions}")
        for i in range(len(predictions)):
            difficulty_scores[predictions[i]-1] += 1
            # difficulty_scores[predictions[i]-1] += 1

    return difficulty_scores

def get_mid_files_statistics(runs_name = "p-est-250822-160040", mid_folder_path = "/Users/simhyeongju/AVAPT/data/pianovam/midi"):
    # Load Model
    model_snapshot_path = f'/Users/simhyeongju/AVAPT/EDA/runs/{runs_name}/model_snapshots/model_bestvalidation.pt'
    rubricnet = RubricNet()
    rubricnet.load_state_dict(torch.load(model_snapshot_path, weights_only=False))

    mid_file_list = [f for f in os.listdir(mid_folder_path) if f.endswith(('.mid', '.xml'))]
    difficulty_scores = [0 for _ in range(9)]

    # ====================== Scaling Features======================

    features_list = []
    for filename in mid_file_list:
        midi_path = os.path.join(mid_folder_path, filename)
        midi_features = MidiFeatures(midi_path)
        features = midi_features.get_numeric_features()
        features_list.append(features)
    features_list = np.array(features_list)

    feature_mean_std_list = [[], []]
    for idx in range(14):
        feature_list = features_list[:, idx].reshape(-1, 1)
        feature_mean_std_list[0].append(np.mean(feature_list))
        feature_mean_std_list[1].append(np.std(feature_list))
    rubricnet.set_scaler_parameter_manually(feature_mean_std_list[0], feature_mean_std_list[1])

    # ====================== Scaling Features Ends ======================

    for idx, filename in enumerate(mid_file_list):
        midi_path = os.path.join(mid_folder_path, filename)
        midi_features = MidiFeatures(midi_path)
        # Extract features
        features = torch.tensor(midi_features.get_numeric_features())
        # Get predictions
        with torch.no_grad():
            predictions = rubricnet.predict(features)[0].item()
        # Save statistics
        #print(f"File: {filename}, Prediction: {predictions}")
        difficulty_scores[predictions-1] += 1

    return difficulty_scores

if __name__ == "__main__":
    # stats = get_mid_files_statistics("/Users/simhyeongju/AVAPT/data/omaps/OMAPS2/midi")
    stats = get_mid_files_statistics()
    #print(stats)