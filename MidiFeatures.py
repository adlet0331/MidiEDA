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
        self.extract_features()
    
    def extract_features(self):
        # 1. 기본 MIDI Feature, Note-based Feature 추출
        # 음정 분포
        pitch_classes = [self.notes[i].pitch % 12 for i in range(len(self.notes))]
        self.features['pitch_classes'] = pitch_classes
        self.features['pitch_hist'] = np.bincount(pitch_classes, minlength=12) / len(pitch_classes)

        # 음역대
        self.features['highest_pitch'] = np.max([note.pitch for note in self.notes])
        self.features['lowest_pitch'] = np.min([note.pitch for note in self.notes])
        self.features['pitch_range'] = self.features['highest_pitch'] - self.features['lowest_pitch']

        # 음표 길이
        note_lengths = [note.end - note.start for note in self.notes]
        self.features['average_note_length'] = np.mean(note_lengths)
        self.features['max_note_length'] = np.max(note_lengths)
        self.features['min_note_length'] = np.min(note_lengths)

        # 음표 밀도
        total_duration = self.midi.get_end_time()
        self.features['note_density'] = len(self.notes) / total_duration if total_duration > 0 else 0

        # Polyphony (동시 발음 음 개수)
        piano_roll = self.midi.get_piano_roll()
        self.features['polyphony'] = np.mean(np.count_nonzero(piano_roll, axis=0))

        # Interval (인접 노트간 음정 간격)
        self.features['intervals'] = [abs(self.notes[i+1].pitch - self.notes[i].pitch) for i in range(len(self.notes)-1)]
        self.features['interval_mean'] = np.mean(self.features['intervals'])
        self.features['interval_std'] = np.std(self.features['intervals'])

        # 3. Performance-based Feature 추출
        # Velocity 분포
        velocities = [note.velocity for note in self.notes]
        self.features['velocity_hist'] = np.bincount(velocities, minlength=128) / len(velocities)

        # Timing 관련 분포
        onsets = [note.start for note in self.notes]
        iois = np.diff(onsets)
        self.features['ioi_mean'] = np.mean(iois)
        self.features['ioi_std'] = np.std(iois)

    # MIDI Feature 정보를 string으로 요약해서 반환
    def get_features_info(self):
        if not self.available:
            return "MIDI features are not available due to missing or invalid MIDI file."
        info = "MIDI Features:\n"
        for key, value in self.features.items():
            info += f"  {key}: {value}\n"
        return info