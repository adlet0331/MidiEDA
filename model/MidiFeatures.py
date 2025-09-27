import numpy as np
import pretty_midi
pretty_midi.pretty_midi.MAX_TICK = 1.1e9 # pretty_midi의 MAX_TICK 값을 늘려서 MIDI 파일의 최대 tick 수를 증가시킴
import os, io
from midi import load_midi, slice_midi, notes_to_piano_roll
import music21  # .xml 처리를 위해 music21 라이브러리 추가
import warnings
from music21.musicxml.xmlToM21 import MusicXMLWarning
warnings.filterwarnings("ignore", category=MusicXMLWarning)

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
            
        self.midi_path = midi_path
        # 파일 확장자 확인
        file_extension = os.path.splitext(self.midi_path)[1].lower()

        try:
            # 확장자에 따라 다른 방식으로 파일 로드
            if file_extension in ['.mid', '.midi']:
                self.midi = pretty_midi.PrettyMIDI(self.midi_path)
            elif file_extension in ['.xml', '.musicxml']:
                # music21을 사용해 .xml 파일 로드 후 pretty_midi 객체로 변환
                score = music21.converter.parse(self.midi_path)
                midi_file_obj = score.write('midi')
                with open(midi_file_obj, 'rb') as f:
                    midi_data = io.BytesIO(f.read())
                self.midi = pretty_midi.PrettyMIDI(midi_data)
            else:
                print(f"Unsupported file format: {file_extension}. Please use .mid or .xml.")
                self.available = False
                return

        except Exception as e:
            print(f"Error processing file {self.midi_path}: {e}")
            self.available = False
            return

        if self.midi.instruments is None or len(self.midi.instruments) == 0:
            print("MIDI file does not contain any instruments or notes.")
            self.available = False
            return
            
        self.notes = self.midi.instruments[0].notes if len(self.midi.instruments) > 0 else []
        self._extract_features(self.notes, midi=self.midi)

    # MIDI 파일에서 Feature를 추출하는 메소드
    def _extract_features(self, notes, midi=None, clipped=False, clipped_time=0.0):
        ### 논문에서 제안된 Feature들
        # Towards Explainable and Interpretable Musical Difficulty Estimation: A parameter-efficient approach
        self.iterable_features = {}
        self.numeric_features = {}

        if len(notes) == 0:
            return {}, {}

        # 음정 분포
        pitch_hist = np.bincount([notes[i].pitch for i in range(len(notes))], minlength=88) / len(notes)
        self.iterable_features['pitch_hist'] = pitch_hist[pitch_hist > 0]
        self.numeric_features['pitch_entropy'] = - np.sum(np.where(self.iterable_features['pitch_hist'] > 0, self.iterable_features['pitch_hist'] * np.log2(self.iterable_features['pitch_hist']), 0))
        # Pitch Range 대신 Highest Pitch와 Lowest Pitch를 사용
        #self.numeric_features['pitch_range'] = self.numeric_features['highest_pitch'] - self.numeric_features['lowest_pitch']
        self.numeric_features['highest_pitch'] = np.max([note.pitch for note in notes])
        self.numeric_features['lowest_pitch'] = np.min([note.pitch for note in notes])
        self.numeric_features['average_pitch'] = np.mean([note.pitch for note in notes])

        # Timing 관련 분포
        onsets = [note.start for note in notes]
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
        self.iterable_features['note_lengths'] = [note.end - note.start for note in notes]
        self.numeric_features['average_note_length'] = np.mean(self.iterable_features['note_lengths'])
        self.numeric_features['max_note_length'] = np.max(self.iterable_features['note_lengths'])
        self.numeric_features['min_note_length'] = np.min(self.iterable_features['note_lengths'])

        # 음표 밀도
        self.numeric_features['total_duration'] = midi.get_end_time() if not clipped else clipped_time
        self.numeric_features['note_density'] = len(notes) / self.numeric_features['total_duration'] if self.numeric_features['total_duration'] > 0 else 0

        # Polyphony (동시 발음 음 개수)
        piano_roll = notes_to_piano_roll(notes, fs=120, max_time=self.numeric_features['total_duration'])
        self.numeric_features['average_polyphony'] = np.mean(np.count_nonzero(piano_roll, axis=0))
        self.numeric_features['max_polyphony'] = np.max(np.count_nonzero(piano_roll, axis=0))

        # Interval (인접 노트간 음정 간격)
        self.iterable_features['intervals'] = [abs(notes[i+1].pitch - notes[i].pitch) for i in range(len(notes)-1)]
        interval_np = np.array(self.iterable_features['intervals'])
        interval_np = interval_np[~np.isnan(interval_np)]  # Remove NaN values
        self.numeric_features['interval_mean'] = np.nanmean(interval_np) if len(interval_np) > 0 else 0

        # Velocity 분포
        velocities = [note.velocity for note in notes]
        velocity_hist = np.bincount(velocities, minlength=128) / len(velocities)
        self.iterable_features['velocity_hist'] = velocity_hist[velocity_hist > 0]

        return self.iterable_features, self.numeric_features
    
    def extract_features_segments(self, segment_length_sec):
        """
        MIDI 파일을 segment 단위로 나누어 각 segment의 Feature를 추출합니다.
        :param segment_length_sec: 각 segment의 길이 (초 단위)
        :return: segment별 Feature 딕셔너리
        """
        if not self.available:
            return None
        
        total_duration = self.midi.get_end_time()
        num_segments = int(np.ceil(total_duration / segment_length_sec))

        midi = load_midi(self.midi_path)
        intervals = midi[:,:2]
        pitches = midi[:,2]
        velocities = midi[:,3]
        segments = slice_midi(pitches, intervals, velocities, num_segments, segment_length_sec)

        segment_features = []
        for i, segment in enumerate(segments):
            # Make Notes List from segment
            segment_notes = []
            for j in range(len(segment['pitches'])):
                segment_notes.append(pretty_midi.Note(
                    velocity=segment['velocities'][j],
                    pitch=segment['pitches'][j],
                    start=segment['intervals'][j][0],
                    end=segment['intervals'][j][1]
                ))
            segment_feature, segment_numeric_feature = self._extract_features(segment_notes, clipped=True, clipped_time=segment_length_sec)
            segment_features.append([numeric_feature for numeric_feature in segment_numeric_feature.values()])
            # segment_features[i] = {
            #     'numeric_features': segment_numeric_feature,
            #     'features': segment_feature
            # }
        
        return segment_features

    def get_numeric_features(self):
        if not self.available:
            return None
        # Feature들을 numpy array로 변환하여 반환
        features = [numeric_feature for numeric_feature in self.numeric_features.values()]
        # 14개의 feature를 numpy array로 변환
        return features
    
    def get_numeric_features_names(self):
        # Feature들의 이름을 리스트로 반환
        return list(self.numeric_features.keys())