import json, os, random, sys
from torch.utils.data import Dataset
import numpy as np

sys.path.append(os.path.abspath("../EDA/"))
from constants import *
from MidiFeatures import MidiFeatures

'''
최종 데이터셋: x:Midi의 Numeric Features, y: Performance의 점수

Midi 종류 (연주 난이도 기반 분류): 
- Mikrokosmos
- CIPI 

Read metadata
'''

class MidiFeaturesPerformanceDataset(Dataset):
    def __init__(self, dataset_path=MIKROKOSMOS_PATH, device=DEFAULT_DEVICE):
        self.dataset_path = dataset_path
        self.device = device
        self.metadata = self.load_metadata()
        self.size = len(self.metadata)

    def load_metadata(self):
        assert "Need to be implemented in subclass"

    def get_piece_path(self, index):
        assert "Need to be implemented in subclass"

    def get_label(self, index):
        assert "Need to be implemented in subclass"

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        if index >= self.size:
            raise IndexError("Index out of range")
        features = MidiFeatures(midi_path=self.get_piece_path(index)).get_numeric_features_np()
        label = self.get_label(index)
        return features, label

class MirkokosmosDataset(MidiFeaturesPerformanceDataset):
    def __init__(self, dataset_path=MIKROKOSMOS_PATH, device=DEFAULT_DEVICE):
        super().__init__(dataset_path=dataset_path, device=device)

    def load_metadata(self):
        metadata_path = os.path.join(self.dataset_path, 'metadata', 'henle_mikrokosmos.json')
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
            return list(metadata.values())
        
    def get_label(self, index):
        return self.metadata[index]
    
    def get_piece_path(self, index):
        return os.path.join(self.dataset_path, 'musicxml', f"{index + 1}.xml")

class CipiDataset(MidiFeaturesPerformanceDataset):
    def __init__(self, dataset_path=CIPI_PATH, device=DEFAULT_DEVICE):
        super().__init__(dataset_path=dataset_path, device=device)

    def load_metadata(self):
        metadata_path = os.path.join(self.dataset_path, 'index.json')
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
            return list(metadata.values())

    def get_label(self, index):
        return self.metadata[index]['henle']

    def get_piece_paths(self, index):
        paths = []
        for i, path in enumerate(self.metadata[index]['path'].values()):
            paths.append(os.path.join(self.dataset_path, 'scores',path))
        return paths
    
    def __getitem__(self, index):
        if index >= self.size:
            raise IndexError("Index out of range")
        features = []
        for path in self.get_piece_paths(index):
            features.append(MidiFeatures(midi_path=path).get_numeric_features_np())
        features = np.mean(features, axis=0)  # 평균을 내어 하나의 피쳐 벡터로 만듦
        return features, self.get_label(index)