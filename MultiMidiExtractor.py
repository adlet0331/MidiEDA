import os
import json
import soundfile
import MidiExtractor

# 이전 답변의 MidiExtractor 클래스가 정의되어 있다고 가정합니다.
# from midi_midi_extractor import MidiExtractor 

class MultiMidiExtractor:
    """
    폴더 내의 여러 오디오 파일을 처리합니다.
    모든 변환 로직은 MidiExtractor 인스턴스에 위임합니다.
    """
    def __init__(self, midi_extractor: MidiExtractor, input_folder: str, midi_output_folder: str):
        self.midi_extractor = midi_extractor
        self.input_folder = input_folder
        
        # 프로세서의 설정값을 참조하여 출력 폴더 이름을 생성
        model_name = self.midi_extractor.model.__class__.__name__
        mode_name = self.midi_extractor.extract_mode.value
        segment_len = self.midi_extractor.segment_length_sec
        self.midi_output_folder = os.path.join(midi_output_folder, f"{model_name}_{mode_name}_{segment_len}sec")

    def process_folder(self, audio_file_list=None):
        """
        입력 폴더의 모든 오디오 파일을 처리하고,
        MidiExtractor를 호출하여 MIDI 변환 및 저장을 수행합니다.
        _16020Hz.wav 확장자를 가진 파일만 처리합니다.
        """
        if audio_file_list is None:
            audio_file_list = [f for f in os.listdir(self.input_folder) if f.endswith(('_16020Hz.wav'))]
        
        os.makedirs(self.midi_output_folder, exist_ok=True)
        
        # 변환 프로세스에 대한 메타데이터 준비
        metadata = {
            'info': 'Transcribed MIDI files from audio input',
            'midi_extractor_settings': {
                'model': self.midi_extractor.model.__class__.__name__,
                'extract_mode': self.midi_extractor.extract_mode.name,
                'segment_length_sec': self.midi_extractor.segment_length_sec,
                'sample_rate': self.midi_extractor.sample_rate,
            },
            'items' : {}
        }
        
        print(f'Transcribed MIDI files will be saved to: {self.midi_output_folder}')
        print(f"MIDI 변환 수행 중: 파일 수: {len(audio_file_list)}")
        
        total_seg_count = 0
        for idx, filename in enumerate(audio_file_list):
            print(f"\n[{idx + 1}/{len(audio_file_list)}]--------------------")
            
            input_path = os.path.join(self.input_folder, filename)
            # 각 오디오 파일의 이름으로 하위 출력 폴더 생성
            output_subdir = os.path.join(self.midi_output_folder, os.path.splitext(filename)[0])
            
            # ✅ 핵심 로직: MidiExtractor에 파일 처리 위임
            self.midi_extractor.transcribe(input_path, output_subdir)
            
            # 처리 후 결과 확인 및 메타데이터 기록
            try:
                # 생성된 미디 파일 수 계산
                num_segments = len([f for f in os.listdir(output_subdir) if f.endswith('.mid') and f[:-4].isdigit()])
                audio_info = soundfile.info(input_path)
                
                metadata['items'][os.path.splitext(filename)[0]] = {
                    'audio_length_seconds': round(audio_info.duration, 2),
                    'num_midi_segments_created': num_segments
                }
                total_seg_count += num_segments
            except Exception as e:
                print(f"Could not read info or count segments for {os.path.splitext(filename)[0]}: {e}")
                metadata['items'][os.path.splitext(filename)[0]] = {'error': str(e)}

            # 각 파일 처리 후 메타데이터 저장
            with open(os.path.join(self.midi_output_folder, 'metadata.json'), 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=4, ensure_ascii=False)
        
        print(f"\n=========================================")
        print(f"총 {total_seg_count}개의 MIDI 세그먼트가 변환되었습니다.")
        print(f"자세한 정보는 {os.path.join(self.midi_output_folder, 'metadata.json')} 파일을 참고하세요.")