import numpy as np
import pretty_midi
import os

"""
Basic MIDI feature extraction class.
악보 자체의 기본적인 정보를 추출하는 클래스
"""
class MidiFeatures:
    def __init__(self, midi_path):
        self.available = True
        if not os.path.exists(midi_path):
            print(f"MIDI file not found: {midi_path}")
            self.available = False
            return
        self.midi = pretty_midi.PrettyMIDI(midi_path)
        if self.midi.instruments is None or len(self.midi.instruments) == 0:
            print("MIDI file does not contain any instruments or notes.")
            self.available = False
            return
        self.notes = self.midi.instruments[0].notes if len(self.midi.instruments) > 0 else []
        self.features = {}
        self.numeric_features = {}
        self.extract_features()
    
    # MIDI 파일에서 Feature를 추출하는 메소드
    def extract_features(self):
        ### 논문에서 제안된 Feature들
        # Towards Explainable and Interpretable Musical Difficulty Estimation: A parameter-efficient approach
        
        # 음정 분포
        pitch_dist = [self.notes[i].pitch for i in range(len(self.notes))]
        self.features['pitch_dist'] = np.bincount(pitch_dist, minlength=88) / len(pitch_dist)
        pitch_dist_nonzero = self.features['pitch_dist'][self.features['pitch_dist'] > 0]
        self.numeric_features['pitch_entropy'] = - np.sum(np.where(pitch_dist_nonzero > 0, pitch_dist_nonzero * np.log2(pitch_dist_nonzero), 0))
        # Pitch Range 대신 Highest Pitch와 Lowest Pitch를 사용
        #self.numeric_features['pitch_range'] = self.numeric_features['highest_pitch'] - self.numeric_features['lowest_pitch']
        self.numeric_features['highest_pitch'] = np.max([note.pitch for note in self.notes])
        self.numeric_features['lowest_pitch'] = np.min([note.pitch for note in self.notes])
        self.numeric_features['average_pitch'] = np.mean([note.pitch for note in self.notes])

        # Timing 관련 분포
        onsets = [note.start for note in self.notes]
        onsets.sort(reverse=False)
        iois = np.diff(onsets)
        iois = iois[iois > 0]  # Remove non-positive intervals
        if len(iois) == 0:
            self.numeric_features['ioi_mean'] = 0
            self.numeric_features['ioi_entropy'] = 0
        else:
            self.numeric_features['ioi_mean'] = np.mean(iois)
            iois = iois / np.sum(iois)  # Normalize to sum to 1
            self.numeric_features['ioi_entropy'] = - np.sum(iois * np.log2(iois))

        ### 추가한 Audio Model 관련 Feature들
        ### 1. Note-based Feature 추출
        # 음표 길이
        self.features['note_lengths'] = [note.end - note.start for note in self.notes]
        self.numeric_features['average_note_length'] = np.mean(self.features['note_lengths'])
        self.numeric_features['max_note_length'] = np.max(self.features['note_lengths'])
        self.numeric_features['min_note_length'] = np.min(self.features['note_lengths'])

        # 음표 밀도
        self.numeric_features['total_duration'] = self.midi.get_end_time()
        self.numeric_features['note_density'] = len(self.notes) / self.numeric_features['total_duration'] if self.numeric_features['total_duration'] > 0 else 0

        # Polyphony (동시 발음 음 개수)
        piano_roll = self.midi.get_piano_roll()
        self.numeric_features['average_polyphony'] = np.mean(np.count_nonzero(piano_roll, axis=0))
        self.numeric_features['max_polyphony'] = np.max(np.count_nonzero(piano_roll, axis=0))

        # Interval (인접 노트간 음정 간격)
        self.features['intervals'] = [abs(self.notes[i+1].pitch - self.notes[i].pitch) for i in range(len(self.notes)-1)]
        self.numeric_features['interval_mean'] = np.mean(self.features['intervals'])

        # Velocity 분포
        velocities = [note.velocity for note in self.notes]
        self.features['velocity_hist'] = np.bincount(velocities, minlength=128) / len(velocities)

    def get_numeric_features_np(self):
        if not self.available:
            return None
        # Feature들을 numpy array로 변환하여 반환
        features_np = [numeric_feature for numeric_feature in self.numeric_features.values()]
        # 14개의 feature를 numpy array로 변환
        return np.array(features_np)

    # MIDI Feature 정보를 string으로 요약해서 반환
    def get_features_info_string(self):
        if not self.available:
            return "MIDI features are not available due to missing or invalid MIDI file."
        info = "MIDI Features:\n"
        for key, value in self.features.items():
            info += f"  {key}: {value}\n"
        return info