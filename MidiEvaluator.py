import numpy as np
import mir_eval

class MidiEvaluator:
    """
    mir_eval 라이브러리를 사용하여 MIDI 노트의 평가를 수행하는 클래스입니다.
    Precision, Recall, F1-Score 및 매칭 정보를 계산합니다.
    """
    def evaluate_notes(self, ref_intervals: np.ndarray, ref_pitches: np.ndarray,
                       est_intervals: np.ndarray, est_pitches: np.ndarray,
                       onset_tolerance: float = 0.05, pitch_tolerance: float = 0.5,
                       offset_ratio: float = None) -> dict:
        """
        주어진 정답(reference) 및 예측(estimated) 노트들을 비교하여
        Precision, Recall, F1-Score를 계산하고 매칭 정보를 반환합니다.
        
        반환 값: 점수(dict), 매칭된 정답 인덱스(set), 매칭된 예측 인덱스(set)
        """
        if ref_intervals.size == 0 and est_intervals.size == 0:
            p, r, f = 1.0, 1.0, 1.0
        else:
            p, r, f, _ = mir_eval.transcription.precision_recall_f1_overlap(
                ref_intervals, ref_pitches,
                est_intervals, est_pitches,
                onset_tolerance=onset_tolerance,
                pitch_tolerance=pitch_tolerance,
                offset_ratio=offset_ratio
            )
        scores = {'Precision': p, 'Recall': r, 'F1-Score': f}

        matching = mir_eval.transcription.match_notes(
            ref_intervals, ref_pitches,
            est_intervals, est_pitches,
            onset_tolerance=onset_tolerance,
            pitch_tolerance=pitch_tolerance,
            offset_ratio=offset_ratio
        )
        matched_ref_indices = {match[0] for match in matching}
        matched_est_indices = {match[1] for match in matching}

        return scores, matched_ref_indices, matched_est_indices