import os
import json
import numpy as np
import matplotlib.pyplot as plt

from midi import load_midi, slice_midi
from MidiEvaluator import MidiEvaluator

class NumpyFloatEncoder(json.JSONEncoder):
    """numpy float 타입을 json으로 저장하기 위한 인코더 클래스"""
    def default(self, obj):
        if isinstance(obj, np.floating):
            return float(round(obj, 4))
        return json.JSONEncoder.default(self, obj)

class MultiMidiEvaluator:
    """
    분할된 예측 MIDI 파일들을 원본(Ground Truth) MIDI 파일과 비교하여
    성능(Precision, Recall, F1-Score)을 평가하고 결과를 JSON 파일로 저장합니다.
    또한, 특정 세그먼트를 개별적으로 평가하고 시각화하는 기능을 제공합니다.
    """
    def __init__(self, predicted_midi_folder: str, ground_truth_midi_folder: str):
        """
        MultiMidiEvaluator를 초기화합니다.
        """
        self.predicted_midi_folder = predicted_midi_folder
        self.ground_truth_midi_folder = ground_truth_midi_folder
        self.metadata_path = os.path.join(predicted_midi_folder, 'metadata.json')

        if not os.path.exists(self.predicted_midi_folder):
            raise FileNotFoundError(f"예측 MIDI 폴더를 찾을 수 없습니다: {self.predicted_midi_folder}")
        if not os.path.exists(self.ground_truth_midi_folder):
            raise FileNotFoundError(f"정답 MIDI 폴더를 찾을 수 없습니다: {self.ground_truth_midi_folder}")
        if not os.path.exists(self.metadata_path):
            raise FileNotFoundError(f"메타데이터 파일을 찾을 수 없습니다: {self.metadata_path}")

        with open(self.metadata_path, 'r') as f:
            self.transcription_metadata = json.load(f)

        # 새로운 클래스 인스턴스 생성
        self.midi_evaluator = MidiEvaluator()

        print("✅ MultiMidiEvaluator가 성공적으로 초기화되었습니다.")

    def _plot_piano_roll(self, ax, intervals, pitches, title, segment_duration, color='blue', alpha=0.8):
        """피아노롤을 그리는 헬퍼 함수"""
        if intervals.size == 0:
            ax.text(0.5, 0.5, "No Notes", horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize=20, color='gray')
            ax.set_ylim(40, 80) # 기본 범위 설정
        else:
            for (start, end), pitch in zip(intervals, pitches):
                ax.add_patch(plt.Rectangle((start, pitch - 0.4), end - start, 0.8, color=color, alpha=alpha))
            ax.set_ylim(np.min(pitches) - 3, np.max(pitches) + 3)

        ax.set_xlim(0, segment_duration)
        ax.set_xlabel("Time (seconds within segment)", fontsize=12)
        ax.set_ylabel("MIDI Pitch", fontsize=12)
        ax.set_title(title, fontsize=16)
        ax.grid(True, which='both', linestyle=':', linewidth=0.5)

        # C1-C8 라벨 추가
        c_note_locs = np.arange(24, 109, 12)  # MIDI 24(C1) to 108(C8)
        c_note_labels = [f'C{i}' for i in range(1, 9)]
        min_y, max_y = ax.get_ylim()
        visible_ticks = []
        visible_labels = []
        for loc, label in zip(c_note_locs, c_note_labels):
            if min_y <= loc <= max_y:
                visible_ticks.append(loc)
                visible_labels.append(label)
        if visible_ticks:
            ax.set_yticks(visible_ticks)
            ax.set_yticklabels(visible_labels, fontsize=10)

    def _get_segment_midi_paths(self, audio_filename: str, segment_num: int):
        """
        주어진 오디오 파일 이름과 세그먼트 번호에 해당하는
        원본(Ground Truth) 및 예측(Predicted) MIDI 파일 경로를 반환합니다.
        """
        if audio_filename not in self.transcription_metadata['items']:
            print(f"  - ❌ 오류: 메타데이터에서 '{audio_filename}'을 찾을 수 없습니다.")
            return None, None, None, None
        file_meta = self.transcription_metadata['items'][audio_filename]
        if not (1 <= segment_num <= file_meta['num_midi_segments_created']):
            print(f"  - ❌ 오류: 세그먼트 번호({segment_num})가 유효한 범위(1~{file_meta['num_midi_segments_created']})를 벗어났습니다.")
            return None, None, None, None
            
        segment_duration = self.transcription_metadata['midi_extractor_settings']['segment_length_sec']
        segment_start_time = (segment_num - 1) * segment_duration

        base_name = os.path.splitext(audio_filename)[0].replace('_16020Hz', '')
        gt_midi_path = os.path.join(self.ground_truth_midi_folder, f"{base_name}.mid")
        pred_midi_path = os.path.join(self.predicted_midi_folder, audio_filename, f"{segment_num}.mid")

        return gt_midi_path, pred_midi_path, segment_duration, segment_start_time

    def _prepare_and_evaluate_segment(self, gt_midi_path: str, pred_midi_path: str, 
                                       segment_start_time: float, segment_duration: float,
                                       onset_tolerance=0.05, pitch_tolerance=0.5, offset_ratio=None):
        """
        세그먼트의 MIDI 노트를 로드하고, mir_eval을 사용하여 평가를 수행하며,
        TP/TN/FP 분류에 필요한 매칭 정보를 반환합니다.
        
        반환값: ref_intervals_segment_relative, ref_pitches_segment, 
               est_intervals_segment_relative, est_pitches_segment_relative, 
               scores, matched_ref_indices, matched_est_indices
        """
        # MidiLoader를 사용하여 MIDI 노트 로드
        ref_intervals_segment, ref_pitches_segment = self.midi_loader.load_seg_midi_notes(gt_midi_path, segment_start_time, segment_start_time + segment_duration)
        est_intervals_segment, est_pitches_segment = self.midi_loader.load_full_midi_notes(pred_midi_path)
        
        # MidiEvaluator를 사용하여 평가 수행
        scores, matched_ref_indices, matched_est_indices = self.midi_evaluator.evaluate_notes(
            ref_intervals_segment, ref_pitches_segment,
            est_intervals_segment, est_pitches_segment,
            onset_tolerance=onset_tolerance,
            pitch_tolerance=pitch_tolerance,
            offset_ratio=offset_ratio
        )

        return (ref_intervals_segment, ref_pitches_segment,
                est_intervals_segment, est_pitches_segment,
                scores, matched_ref_indices, matched_est_indices)

    def evaluate_all(self, onset_tolerance=0.05, pitch_tolerance=0.5, offset_ratio=None):
        """
        모든 분할된 MIDI 파일의 성능을 평가하고, 각 오디오 파일의 폴더별로
        결과를 'evaluation.json' 파일에 저장합니다.
        """
        print(f"\n MIDI 전체 평가를 시작합니다 (offset_ratio={offset_ratio})... 🎶")

        all_precisions, all_recalls, all_f1s = [], [], []
        file_evaluation_total_data = {
            'info': 'Transcription evaluation scores for Folder file',
            'model': self.transcription_metadata['midi_extractor_settings'].get('model', 'Unknown'),
            'predicted_midi_folder': self.predicted_midi_folder,
            'ground_truth_midi_path': self.ground_truth_midi_folder,
            'file_scores': {}
        }
        cnt = 1
        for audio_filename, file_meta in self.transcription_metadata['items'].items():
            print(f"[{cnt}/{len(self.transcription_metadata['items'])}] 파일 평가 중: {audio_filename}")
            cnt += 1

            base_name = os.path.splitext(audio_filename)[0].replace('_16020Hz', '')
            gt_midi_path = os.path.join(self.ground_truth_midi_folder, f"{base_name}.mid")

            if not os.path.exists(gt_midi_path):
                print(f"  - ❌ 정답 MIDI 파일을 찾을 수 없습니다: {gt_midi_path}. 이 파일을 건너뜁니다.")
                continue

            file_evaluation_data = {
                'info': 'Transcription evaluation scores for a single file',
                'model': self.transcription_metadata['midi_extractor_settings'].get('model', 'Unknown'),
                'audio_filename': audio_filename,
                'predicted_midi_folder': os.path.join(self.predicted_midi_folder, audio_filename),
                'ground_truth_midi_path': gt_midi_path,
                'segments': {},
                'total_scores': {},
                'average_scores': {}
            }

            
            file_precisions, file_recalls, file_f1s = [], [], []
            
            num_segments = file_meta['num_midi_segments_created']
            segments_len_secs = self.transcription_metadata['midi_extractor_settings']['segment_length_sec']

            ground_truth_midi = load_midi(gt_midi_path)
            ground_truth_intervals = ground_truth_midi[:, :2]
            ground_truth_pitches = ground_truth_midi[:, 2]
            ground_truth_velocities = ground_truth_midi[:, 3]

            predicted_midi = load_midi(os.path.join(self.predicted_midi_folder, audio_filename, 'total.mid'))
            if predicted_midi is None:
                print(f"  - ❌ 예측 MIDI 파일을 찾을 수 없습니다: {os.path.join(self.predicted_midi_folder, audio_filename, 'total.mid')}. 이 파일을 건너뜁니다.")
                continue
            est_intervals, est_pitches, est_velocities = predicted_midi[:, :2], predicted_midi[:, 2], predicted_midi[:, 3]
            scores, matched_ref_indices, matched_est_indices = self.midi_evaluator.evaluate_notes(
                ground_truth_intervals, ground_truth_pitches,
                est_intervals, est_pitches,
                onset_tolerance=onset_tolerance,
                pitch_tolerance=pitch_tolerance,
                offset_ratio=offset_ratio
            )
            file_evaluation_data['total_scores'] = scores
            file_evaluation_total_data['file_scores'][audio_filename] = scores

            sliced_ground_truth_midis = slice_midi(
                ground_truth_pitches, ground_truth_intervals, ground_truth_velocities,
                num_segments, segments_len_secs
            )

            for i in range(num_segments):
                segment_num = i + 1
                gt_path, pred_path, seg_dur, seg_start_time = self._get_segment_midi_paths(audio_filename, segment_num)
                
                if gt_path is None or pred_path is None: # _get_segment_midi_paths에서 에러 발생 시
                    continue

                ref_intervals_segment, ref_pitches_segment = sliced_ground_truth_midis[i]['intervals'], sliced_ground_truth_midis[i]['pitches']
                pred_midi = load_midi(pred_path)
                est_intervals_segment = np.array([]).reshape(0, 2)
                est_pitches_segment = np.array([])
                if pred_midi.ndim == 1 and pred_midi.size == 0:
                    print(f"  - ❌ 예측 MIDI 파일이 비어 있습니다: {pred_path}. 이 파일을 건너뜁니다.")
                    continue
                elif ref_intervals_segment.ndim == 1 and ref_intervals_segment.size == 0:
                    print(f"  - ❌ 기준 MIDI 파일이 비어 있습니다: {gt_path}.")
                    ref_intervals_segment = np.array([]).reshape(0, 2)
                    ref_pitches_segment = np.array([])
                else:
                    est_intervals_segment, est_pitches_segment = pred_midi[:, :2], pred_midi[:, 2]

                # MidiEvaluator를 사용하여 평가 수행
                scores, matched_ref_indices, matched_est_indices = self.midi_evaluator.evaluate_notes(
                    ref_intervals_segment, ref_pitches_segment,
                    est_intervals_segment, est_pitches_segment,
                    onset_tolerance=onset_tolerance,
                    pitch_tolerance=pitch_tolerance,
                    offset_ratio=offset_ratio
                )
                
                file_evaluation_data['segments'][str(segment_num)] = scores
                file_precisions.append(scores['Precision'])
                file_recalls.append(scores['Recall'])
                file_f1s.append(scores['F1-Score'])

            if file_precisions:
                avg_p = np.mean(file_precisions)
                avg_r = np.mean(file_recalls)
                avg_f = np.mean(file_f1s)
                file_evaluation_data['total_average_scores'] = {'Precision': avg_p, 'Recall': avg_r, 'F1-Score': avg_f}
                all_precisions.extend(file_precisions)
                all_recalls.extend(file_recalls)
                all_f1s.extend(file_f1s)
                print(f"  - ✅ 완료. 평균 F1-Score: {avg_f:.4f}")

            output_dir = os.path.join(self.predicted_midi_folder, audio_filename)
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, 'evaluation.json')
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(file_evaluation_data, f, indent=4, cls=NumpyFloatEncoder)
            print(f"  - 💾 결과가 다음 파일에 저장되었습니다: {output_path}")
            
        output_dir = os.path.join(self.predicted_midi_folder)
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, 'evaluation.json')
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(file_evaluation_total_data, f, indent=4, cls=NumpyFloatEncoder)
        print(f"최종 결과가 다음 파일에 저장되었습니다: {output_path}")
        
        
        print(f"\n🎉 모든 평가가 완료되었습니다! 각 오디오 폴더에 결과가 저장되었습니다.")
        return 