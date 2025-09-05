import torch, json, os
from model import RubricNet, MidiFeatures
import numpy as np
import matplotlib.pyplot as plt

def get_mid_files_statistics(mid_folder_path = "/Users/simhyeongju/AVAPT/data/pianovam/midi"):
    # Load Model
    runs_name = 'runs/p-est-250827-144050'
    model_snapshot_path = f'/Users/simhyeongju/AVAPT/EDA/{runs_name}/model_snapshots/model_bestvalidation.pt'
    rubricnet = RubricNet()
    rubricnet.load_state_dict(torch.load(model_snapshot_path, weights_only=False))

    mid_file_list = [f for f in os.listdir(mid_folder_path) if f.endswith(('.mid', '.xml'))]
    difficulty_scores = [0 for _ in range(9)]
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