import numpy as np
import pretty_midi
from constants import *

class MidiLoader:
    """MIDI 파일을 로드하고 노트 정보를 mir_eval 형식으로 반환하는 클래스입니다."""
    def load_full_midi_notes(self, midi_path: str) -> tuple[np.ndarray, np.ndarray]:
        """MIDI 파일에서 노트 정보를 'mir_eval'이 요구하는 형식으로 불러옵니다."""
        try:
            midi_data = pretty_midi.PrettyMIDI(midi_path)
            notes = []
            for instrument in midi_data.instruments:
                for note in instrument.notes:
                    notes.append([note.start, note.end, note.pitch])
            
            if not notes:
                return np.empty((0, 2)), np.empty((0,))

            notes = np.array(notes)
            return notes[:, :2], notes[:, 2]
        except Exception as e:
            print(f"⚠️ MIDI 파일을 불러오는 데 실패했습니다: {midi_path}, 오류: {e}")
            return np.empty((0, 2)), np.empty((0,))


    """MIDI 파일에서 start, end 구간의 노트 정보만을 반환합니다."""
    def load_seg_midi_notes(self, midi_path: str, start: float, end: float) -> tuple[np.ndarray, np.ndarray]:
        """MIDI 파일에서 주어진 구간의 노트 정보를 'mir_eval'이 요구하는 형식으로 불러옵니다."""
        if start >= end:
            print(f"⚠️ 잘못된 구간: start({start}) >= end({end})")
            return np.empty((0, 2)), np.empty((0,))
        try:
            midi_data = pretty_midi.PrettyMIDI(midi_path)
            notes = []
            for instrument in midi_data.instruments:
                for note in instrument.notes:
                    if note.start <= end and note.end >= start:
                        if max(note.start, start) == min(note.end, end):
                            notes.append([max(note.start, start) - start, min(note.end, end) + 1/SAMPLE_RATE, note.pitch])
                        else:
                            notes.append([max(note.start, start) - start, min(note.end, end) - start, note.pitch])

            if not notes:
                return np.empty((0, 2)), np.empty((0,))

            notes = np.array(notes)
            return notes[:, :2], notes[:, 2]
        except Exception as e:
            print(f"⚠️ MIDI 파일을 불러오는 데 실패했습니다: {midi_path}, 오류: {e}")
            return np.empty((0, 2)), np.empty((0,))