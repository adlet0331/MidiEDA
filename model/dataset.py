import json, os, sys, tempfile
from typing import Any, List, Dict
from torch.utils.data import Dataset
import numpy as np
import torch

from constants import *
from .MidiFeatures import MidiFeatures

'''
최종 데이터셋: x:Midi의 Numeric Features, y: Performance의 점수

Midi 종류 (연주 난이도 기반 분류): 
- Mikrokosmos
- CIPI 

Read metadata
'''

def _atomic_write_json(path: str, obj: Any, indent: int = 2) -> None:
    """JSON을 안전하게 덮어쓰기 (윈도우/유닉스 공통)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".tmp_", suffix=".json", dir=os.path.dirname(path))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=indent)
        os.replace(tmp, path)  # atomic
    finally:
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
        except Exception:
            pass


class MidiFeaturesPerformanceDataset(Dataset):
    def __init__(self, dataset_path=MIKROKOSMOS_PATH, numeric_versions=1):
        self.dataset_path = dataset_path
        self.numeric_versions = numeric_versions
        self.feature_version_filename = f"features_v{numeric_versions}"  # Numeric Features 버전
        self.features_names = None

        # 서브클래스가 제공해야 하는 메타데이터 파일 경로
        self.metadata_path = self.get_metadata_path()
        # 메타데이터 로드
        self.metadata = self.load_metadata()  # List 형태로 로드

        # Feature 캐시 로드
        self.cached_feature = self._load_cached_features(self.feature_version_filename)
        # 완전한 계산된 캐시가 없는 경우, 새로 계산 후 저장해주기
        if len(self.cached_feature) == 0:
            print(f"{self.dataset_path} Numeric Features v{numeric_versions}가 캐시되지 않았습니다. 새로 계산합니다.")
            # str: metadata에서의 key
            # List[float]: 해당 key의 Numeric Features
            _features_mem: Dict[str, List[float]] = {}
            _valid_size: int = 0
            _valid_indices: List[str] = []
            _skipped = 0
            _skipped_indices: List[str] = []
            for i in range(len(self.metadata)):
                feats = self.compute_features_for_index(i)
                metadata_i = self.metadata[i]
                key = metadata_i["key"]
                if isinstance(feats, np.ndarray):
                    _features_mem[key] = feats.tolist()
                    _valid_size += 1
                    _valid_indices.append({
                        "key": key,
                        "index": i
                    })
                elif isinstance(feats, Exception) or True:
                    _skipped += 1
                    _skipped_indices.append({
                        "key": key,
                        "index": i,
                        "error": str(feats)
                    })
                    print(f"Index {i}에서 Numeric Features 계산 실패: {feats}")
            print(f"총 {len(self.metadata)}개의 파일 중 {_valid_size}개에서 Numeric Features를 성공적으로 계산했습니다. {_skipped}개는 실패했습니다.")
            if _valid_size + _skipped != len(self.metadata):
                raise ValueError(f"경고: 유효한 크기와 스킵된 크기의 합이 전체 크기와 일치하지 않습니다. {len(self.metadata)} != {_valid_size + _skipped}")
            self.cached_feature["version"] = numeric_versions
            self.cached_feature["valid_size"] = _valid_size
            self.cached_feature["skipped_size"] = _skipped
            self.cached_feature["valid_indices"] = _valid_indices
            self.cached_feature["skipped_indices"] = _skipped_indices
            self.cached_feature["features_names"] = self.features_names
            self.cached_feature["features_mem"] = _features_mem
            self._save_cache_features(self.feature_version_filename)
        else:
            print(f"{self.dataset_path} Numeric Features v{numeric_versions}가 캐시에서 로드되었습니다.")
            if self.features_names is None:
                self.features_names = self.cached_feature["features_names"]
    # ---------- 서브클래스가 구현 ----------

    def get_metadata_path(self) -> str:
        """
        메타데이터 파일 경로를 반환하는 메소드.
        """
        raise NotImplementedError

    def load_metadata(self) -> List[Dict[str, Any]]:
        """
        메타데이터를 List 형태로 로드하는 메소드.
        """
        raise NotImplementedError
    
    def serialize_metadata_for_save(self) -> Dict:
        """
        서브클래스에서 포맷에 맞춰 구현.
        메타데이터를 직렬화하여 저장할 때 사용됩니다.
        """
        raise NotImplementedError
    
    # def get_piece_path(self, index):
    #     """
    #     Piece의 경로를 반환하는 메소드. Midi 파일이 하나만 있을 때 사용.
    #     """
    #     raise NotImplementedError
    
    # def get_piece_paths(self, index):
    #     """
    #     Piece의 경로들을 반환하는 메소드. Midi 파일이 2개 이상 있을 때 사용.
    #     """
    #     raise NotImplementedError

    def get_label(self, index):
        """
        Piece의 Label을 반환하는 메소드.
        """
        raise NotImplementedError
    
    # ---------- 내부 유틸 ----------
    
    def _load_cached_features(self, filename: str) -> Dict:
        os.makedirs(os.path.join(self.dataset_path, 'features'), exist_ok=True)
        cached_features = os.path.join(self.dataset_path, 'features', f"{filename}.json")
        if os.path.exists(cached_features):
            with open(cached_features, "r", encoding="utf-8") as f:
                print("Loading cached features from", cached_features)
                return json.load(f)
        return {}
    
    def _save_cache_features(self, filename: str):
        os.makedirs(os.path.join(self.dataset_path, 'features'), exist_ok=True)
        cached_features_path = os.path.join(self.dataset_path, 'features', f"{filename}.json")
        _atomic_write_json(cached_features_path, self.cached_feature, indent=2)
        print(f"Features saved to {cached_features_path}")

    def compute_features_for_index(self, index: int) -> np.ndarray:
        """
        단일 파일 또는 다중 파일(CIPI) 평균.
        CIPI의 경우 개별 파일 실패는 스킵하고, 성공한 것만 평균.
        모든 경로가 실패하면 예외 발생.
        """
        if hasattr(self, "get_piece_paths"):
            feats = []
            last_error = None
            for p in self.get_piece_paths(index):
                try:
                    midiFeatures = MidiFeatures(midi_path=p)
                    if self.features_names is None:
                        self.features_names = midiFeatures.get_numeric_features_names()
                        print(f"Features names: {self.features_names}")
                    feats.append(midiFeatures.get_numeric_features())
                except Exception as e:
                    last_error = e
                    continue
            if not feats:
                print(f"All paths failed for index {index}")
                return last_error
            return np.mean(np.asarray(feats, dtype=np.float32), axis=0)
        elif hasattr(self, "get_piece_path"):
            try:
                midiFeatures = MidiFeatures(midi_path=self.get_piece_path(index))
                if self.features_names is None:
                    self.features_names = midiFeatures.get_numeric_features_names()
                    print(f"Features names: {self.features_names}")
                return np.asarray(midiFeatures.get_numeric_features(), dtype=np.float32)
            except Exception as e:
                print(f"Failed to compute features for index {index}: {e}")
                return e

    def _save_metadata(self):
        """현재 self.metadata를 파일 포맷에 맞춰 저장(서브클래스별 직렬화)."""
        out = self.serialize_metadata_for_save()
        _atomic_write_json(self.metadata_path, out, indent=2)

    # ---------- 표준 Dataset API ----------
    def __len__(self):
        return self.cached_feature["valid_size"]

    def __getitem__(self, index):
        if index >= self.cached_feature["valid_size"]:
            raise IndexError("Index out of bounds for dataset size.")

        feats = self.cached_feature["features_mem"][self.cached_feature["valid_indices"][index]["key"]]
        label = self.metadata[self.cached_feature["valid_indices"][index]["index"]]["henle"]
        return torch.tensor(feats), torch.tensor(label)


class MikrokosmosDataset(MidiFeaturesPerformanceDataset):
    def get_metadata_path(self) -> str:
        return os.path.join(self.dataset_path, 'metadata', 'henle_mikrokosmos.json')

    def load_metadata(self):
        path = self.get_metadata_path()
        with open(path, 'r', encoding='utf-8') as f:
            raw = json.load(f)  # {"1": 3, "2": 4, ...} 또는 {"1":{"henle":3,self.features_name:[...]}, ...}

        items = []
        # 키를 1..N 정렬하여 리스트화
        for k in sorted(raw.keys(), key=lambda x: int(x)):
            v = raw[k]
            items.append({
                "filename": f"{k}.xml",
                "henle": v,
                "key": k,
            })     # 구버전 → dict 업그레이드
        return items

    def serialize_metadata_for_save(self) -> Any:
        out = {}
        for i, entry in enumerate(self.metadata, start=1):
            out[str(i)] = {
                "henle": entry["henle"],
                self.feature_version_filename: entry.get(self.feature_version_filename)
            }
            # 실패 로그가 있다면 유지(선택)
            if "error" in entry:
                out[str(i)]["error"] = entry["error"]
        return out

    def get_label(self, index):
        return self.metadata[index]["henle"]
    
    def get_piece_path(self, index):
        filename = self.metadata[index]["filename"]
        return os.path.join(self.dataset_path, 'musicxml', filename)


class CipiDataset(MidiFeaturesPerformanceDataset):
    def get_metadata_path(self) -> str:
        return os.path.join(self.dataset_path, 'index.json')

    def load_metadata(self):
        path = self.get_metadata_path()
        with open(path, 'r', encoding='utf-8') as f:
            raw = json.load(f)  # 일반적으로 dict
        items = []
        # 키를 1..N 정렬하여 리스트화
        for k in sorted(raw.keys(), key=lambda x: str(x)):
            raw[k]['key'] = k  # 키를 추가하여 원본 키 유지
            items.append(raw[k])
        # values()를 사용하던 기존 코드와 호환: 리스트로 변환
        return items

    def serialize_metadata_for_save(self) -> Any:
        # 간단히 0..N-1 키로 저장(기존 코드가 values()만 사용하므로 호환)
        out = {}
        for i, entry in enumerate(self.metadata):
            out[str(i)] = entry
        return out

    def get_label(self, index):
        return self.metadata[index]['henle']

    def get_piece_paths(self, index):
        paths = []
        # 경로 키 순서를 고정(재현성)
        for _, path in sorted(self.metadata[index]['path'].items()):
            paths.append(os.path.join(self.dataset_path, 'scores', path))
        return paths

class AudioTranscriptionDataset(MidiFeaturesPerformanceDataset):
    def __init__(self, dataset_path='/Users/simhyeongju/AVAPT/EDA/_transcribed_MIDI/OnsetsAndFrames_2_5sec', numeric_versions=1):
        super().__init__(dataset_path, numeric_versions)

    def get_metadata_path(self) -> str:
        return os.path.join(self.dataset_path, 'metadata.json')

    def load_metadata(self):
        path = self.get_metadata_path()
        with open(path, 'r', encoding='utf-8') as f:
            raw_metadata = json.load(f)  # 일반적으로 dict
        raw_metadata = raw_metadata['items']
        items = []
        # 키를 1..N 정렬하여 리스트화
        for k in sorted(raw_metadata.keys(), key=lambda x: str(x)):
            with open(os.path.join(self.dataset_path, k, "evaluation.json"), 'r', encoding='utf-8') as f:
                seg_evaluation_metadata = json.load(f)
                # print(seg_evaluation_metadata["segments"])
            for i in range(1, raw_metadata[k]['num_midi_segments_created'] + 1):
                if seg_evaluation_metadata["segments"].get(str(i)) is None:
                    continue
                new_dict = {}
                new_dict['key'] = f"{k}_{i}"
                new_dict['path'] = os.path.join(k, f"{i}.mid")
                new_dict['henle'] = min(1 + int(100 - seg_evaluation_metadata["segments"][str(i)]["F1-Score"] * 100), 9) # 반올림 해서 1부터 시작, 최대 9
                items.append(new_dict)
        # values()를 사용하던 기존 코드와 호환: 리스트로 변환
        return items

    def get_piece_path(self, index):
        return os.path.join(self.dataset_path, self.metadata[index]['path'])
    
    def get_label(self, index):
        return self.metadata[index]['henle']

    def serialize_metadata_for_save(self) -> Any:
        # 간단히 0..N-1 키로 저장(기존 코드가 values()만 사용하므로 호환)
        out = {}
        for i, entry in enumerate(self.metadata):
            out[str(i)] = entry
        return out

    def get_max_label(self):
        max_label = 0
        for i, entry in enumerate(self.metadata):
            max_label = max(max_label, entry['henle'])
        return max_label