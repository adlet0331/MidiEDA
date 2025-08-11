import json, os
from constants import *
from torch.utils.data import Dataset
from MidiFeatures import MidiFeatures

class MirkokosmosDataset(Dataset):
    def __init__(self, midi_path=MIKROKOSMOS_MIDI_FOLDER, label_path=MIKROKOSMOS_METADATA_FILE, device=DEFAULT_DEVICE):
        """
        Mirkokosmos 데이터셋을 초기화합니다.
        
        Args:
            midi_path (str): MIDI 파일이 저장된 디렉토리 경로.
        """
        self.midi_path = midi_path
        self.label_path = label_path
        self.device = device
        metadata = json.load(open(self.label_path, 'r'))
        self.data = []
        for midi_file in os.listdir(self.midi_path):
            if midi_file.endswith(".mid"):
                midi_path = os.path.join(self.midi_path, midi_file)
                midi_data = MidiFeatures(midi_path, self.device)
                self.data.append(midi_data)