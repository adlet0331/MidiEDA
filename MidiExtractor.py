# Functions
from mel import melspectrogram
from mir_eval.util import midi_to_hz
import numpy as np
import torch, os, soundfile

from midi import save_midi, slice_midi
from constants import *

class AudioLoader:
    """
    Raw 오디오를 모델 입력에 적합한 Mel Spectrogram으로 변환합니다.
    """
    def __init__(self, sample_rate=SAMPLE_RATE):
        self.audio_transcriptor = melspectrogram
        self.sample_rate = sample_rate

    def read_audio(self, input_audio_path):
        try:
            audio, sr = soundfile.read(input_audio_path, dtype='int16')
            if sr != self.sample_rate:
                raise ValueError(f"Sample rate mismatch for {input_audio_path}")
        except Exception as e:
            print(f"Error reading file {input_audio_path}: {e}")
            return
        return audio

    def trascribe_audio(self, audio):
        """
        오디오(numpy array 또는 torch tensor)를 Mel Spectrogram으로 변환합니다.
        """
        if isinstance(audio, np.ndarray):
            # 오디오 데이터가 여러 개일 경우(batch)를 대비해 차원 확인
            if audio.ndim == 1:
                audio = np.expand_dims(audio, 0)
            audio_tensor = torch.from_numpy(audio.astype(np.float32)).div_(32768.0)
        elif isinstance(audio, torch.Tensor):
            if audio.dtype == torch.int16:
                audio_tensor = audio.float().div_(32768.0)
            elif audio.dtype == torch.float32 or audio.dtype == torch.float64:
                if audio.max() > 1.0 or audio.min() < -1.0:
                    audio_tensor = audio.float().div_(audio.abs().max())
                else:
                    audio_tensor = audio.float()
            else:
                audio_tensor = audio.float()
        else:
            raise TypeError("Unsupported audio input type. Expected numpy.ndarray or torch.Tensor.")
        
        # [:, :-1] 부분은 모델의 특성에 따라 다를 수 있으므로 유지, 이걸 빼면 총 MIDI 길이가 늘어난다
        audio_mel = self.audio_transcriptor(audio_tensor[:, :-1]).transpose(-1, -2)
        return audio_mel

from enum import Enum
class ExtractMode(Enum):
    # Segment size = Midi Size, No Combining State
    INPUT_CUT_AND_EXTRACT = 1
    # Segment size = Midi Size * Batch Size
    EXTRACT_AND_CUT = 2

class MidiExtractor:
    """
    하나의 파일을 입력받아, 지정된 모드에 따라 처리한 후
    여러 개의 분할된 MIDI 파일로 출력합니다.
    """
    def __init__(self, model, extract_mode: ExtractMode, segment_length_sec=2.5,
                 transcribe_batch_size=AUDIO_TRANSCRIBE_BATCH_SIZE, save_total_midi: bool = True,
                 input_transcriptor=AudioLoader(), hop_size=HOP_LENGTH,
                 sample_rate=SAMPLE_RATE, min_midi=MIN_MIDI, max_midi=MAX_MIDI):

        self.model = model
        self.extract_mode = extract_mode
        self.segment_length_sec = segment_length_sec
        self.batch_size = transcribe_batch_size
        if extract_mode == ExtractMode.INPUT_CUT_AND_EXTRACT:
            self.batch_size = 1
        self.save_total_midi = save_total_midi
        self.input_transcriptor = input_transcriptor
        self.hop_size = hop_size
        self.sample_rate = sample_rate
        self.min_midi = min_midi
        self.max_midi = max_midi

    def transcribe(self, input_path, output_dir):
        """
        변환을 처리하는 메인 메서드입니다.
        모드에 따라 내부적으로 다른 처리 함수를 호출합니다.
        """
        print(f"Processing '{os.path.basename(input_path)}' with mode: {self.extract_mode.name}")
        os.makedirs(output_dir, exist_ok=True)
        
        audio = self.input_transcriptor.read_audio(input_path)
        if audio is None:
            print(f"Audio reading failed for {input_path}.")
            return

        if self.extract_mode == ExtractMode.INPUT_CUT_AND_EXTRACT:
            self.batch_size = 1
            self._process_by_batch_extract_and_cut(audio, output_dir)
        elif self.extract_mode == ExtractMode.EXTRACT_AND_CUT:
            self._process_by_batch_extract_and_cut(audio, output_dir)
        else:
            raise ValueError(f"Unknown ExtractMode: {self.extract_mode}")

    def _process_by_batch_extract_and_cut(self, audio, output_dir):
        """
        Input 전체를 먼저 변환하여 하나의 MIDI를 만들고,
        그 MIDI를 segment 단위로 잘라 저장합니다.

        * Mode 1인 경우: self.batch_size = 1
        """        
        # 1. 오디오를 큰 덩어리(Transcription Chunk)로 잘라 모델로 처리
        samples_per_batch = int(self.sample_rate * self.segment_length_sec * self.batch_size)
        all_onsets, all_frames, all_velocities = [], [], []
        num_batches = int(np.ceil(len(audio) / samples_per_batch))
        print(f"Audio will be processed in {num_batches} large chunk(s).")

        for i in range(num_batches):
            start = i * samples_per_batch
            end = start + samples_per_batch
            audio_chunk = audio[start:end]
            
            print(f"  - Processing chunk {i+1}/{num_batches}...")
            mel = self.input_transcriptor.trascribe_audio(audio_chunk)
            
            with torch.no_grad():
                onset_pred, _, _, frame_pred, velocity_pred = self.model(mel)
            
            all_onsets.append(onset_pred.squeeze(0))
            all_frames.append(frame_pred.squeeze(0))
            all_velocities.append(velocity_pred.squeeze(0))

        # 2. 모든 예측 결과를 하나로 합쳐 전체 곡의 예측값 생성
        full_onset_pred = torch.cat(all_onsets, dim=0)
        full_frame_pred = torch.cat(all_frames, dim=0)
        full_velocity_pred = torch.cat(all_velocities, dim=0)
        
        # 3. 전체 예측값에서 모든 노트를 한 번에 추출
        print("Extracting notes from full transcription...")
        pitches, intervals_frames, velocities = self._extract_notes(
            full_onset_pred, full_frame_pred, full_velocity_pred
        )
        
        if len(pitches) == 0:
            print("No notes found in the audio.")
            return

        intervals_sec = intervals_frames * (self.hop_size / self.sample_rate)
        pitches_hz = np.array([midi_to_hz(self.min_midi + p) for p in pitches])
        
        # 3.5. [추가] 전체 MIDI 파일 저장 옵션 처리
        if self.save_total_midi:
            print("Saving the total MIDI file...")
            total_midi_filename = os.path.join(output_dir, "total.mid")
            save_midi(total_midi_filename, pitches_hz, intervals_sec, velocities)
            print(f"  - Saved total MIDI to {total_midi_filename}")
        
        # 4. 전체 노트를 segment 단위로 잘라 MIDI 파일로 저장
        num_segments = int(np.ceil(len(audio) / (self.sample_rate * self.segment_length_sec)))
        print(f"Cutting master MIDI into {num_segments} segment(s)...")

        segment_midi = slice_midi(pitches_hz, intervals_sec, velocities, num_segments, self.segment_length_sec)
        print("Slice completed. Saving segments...")

        for i, midi in enumerate(segment_midi):
            segment_pitches, segment_intervals, segment_velocities = midi['pitches'], midi['intervals'], midi['velocities']
            output_filename = os.path.join(output_dir, f"{i+1}.mid")
            save_midi(output_filename, 
                    np.array(segment_pitches), 
                    np.array(segment_intervals), 
                    np.array(segment_velocities))
            
            print(f"  - Saved segment {i+1} with {len(segment_pitches)} notes to {output_filename}")

        print(f"Finished processing. Total {num_segments} MIDI files saved in '{output_dir}'.")

    def _extract_notes(self, onsets, frames, velocity, onset_threshold=0.4, frame_threshold=0.4):
        onsets = (onsets > onset_threshold).cpu().to(torch.uint8)
        frames = (frames > frame_threshold).cpu().to(torch.uint8)
        onset_diff = torch.cat([onsets[:1, :], onsets[1:, :] - onsets[:-1, :]], dim=0) == 1
        pitches, intervals, velocities = [], [], []
        for nonzero in onset_diff.nonzero():
            frame, pitch = nonzero[0].item(), nonzero[1].item()
            
            onset_frame = frame
            offset_frame = frame + 1

            while (offset_frame < onsets.shape[0] 
                   and frames[offset_frame, pitch].item()):
                offset_frame += 1

            if offset_frame > onset_frame:
                pitches.append(pitch)
                intervals.append([onset_frame, offset_frame])
                onset_velocity = np.clip(velocity[onset_frame, pitch].item(), 0, 127)
                velocities.append(onset_velocity)

        return np.array(pitches), np.array(intervals), np.array(velocities)