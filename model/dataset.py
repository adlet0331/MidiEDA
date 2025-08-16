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
        self.features_name = f"features_v{numeric_versions}"  # Numeric Features 버전

        # 서브클래스가 제공해야 하는 메타데이터 파일 경로
        self.metadata_path = self.get_metadata_path()

        # 메타데이터 로드(서브클래스 구현)
        self.metadata = self.load_metadata()  # dictionary 형태로 로드
        self.size = len(self.metadata)

        # 레이블 캐시
        self._labels_mem: List[Any] = [self.get_label(i) for i in range(self.size)]
        # feature 메모리 캐시
        self._features_mem: List[np.ndarray] = [None] * self.size

        # 이미 저장된 features가 있으면 메모리에 올림
        have = 0
        for i, entry in enumerate(self.metadata):
            if isinstance(entry, dict) and entry.get(self.features_name) is not None:
                self._features_mem[i] = np.asarray(entry[self.features_name], dtype=np.float32)
                have += 1
        self._features_ready = (have == self.size)

        # __init__ 시점에 전량 준비(최초 1회)
        if not self._features_ready:
            print(f"Preparing features for {self.size} items in {self.features_name}...")
            self._prepare_all_features()

    # ---------- 서브클래스가 구현 ----------
    def get_metadata_path(self) -> str:
        raise NotImplementedError

    def load_metadata(self):
        raise NotImplementedError

    def get_piece_path(self, index):
        raise NotImplementedError

    def get_label(self, index):
        raise NotImplementedError

    # ---------- 내부 유틸 ----------
    def _compute_features_for_index(self, index: int) -> np.ndarray:
        """
        단일 파일 또는 다중 파일(CIPI) 평균.
        CIPI의 경우 개별 파일 실패는 스킵하고, 성공한 것만 평균.
        모든 경로가 실패하면 예외 발생.
        """
        if hasattr(self, "get_piece_paths"):
            feats = []
            last_exc = None
            for p in self.get_piece_paths(index):
                try:
                    feats.append(MidiFeatures(midi_path=p).get_numeric_features())
                except Exception as e:
                    last_exc = e
                    continue
            if not feats:
                raise last_exc or RuntimeError(f"All paths failed for index {index}")
            return np.mean(np.asarray(feats, dtype=np.float32), axis=0)
        else:
            return MidiFeatures(midi_path=self.get_piece_path(index)).get_numeric_features()

    def _set_features_in_metadata(self, index: int, feats_list: List[float]):
        """
        self.metadata[index]에 features 필드 주입.
        Mikrokosmos 구버전(정수만)도 dict로 업그레이드.
        """
        entry = self.metadata[index]
        if isinstance(entry, dict):
            entry[self.features_name] = feats_list
        else:
            # 구버전: int 레이블만 존재 → dict로 업그레이드
            self.metadata[index] = {"henle": entry, self.features_name: feats_list}

    def _save_metadata(self):
        """현재 self.metadata를 파일 포맷에 맞춰 저장(서브클래스별 직렬화)."""
        out = self._serialize_metadata_for_save()
        _atomic_write_json(self.metadata_path, out, indent=2)

    def _serialize_metadata_for_save(self) -> Any:
        """서브클래스에서 포맷에 맞춰 구현."""
        raise NotImplementedError

    def _prepare_all_features(self):
        """
        features가 비어있는 항목을 계산하여 메타데이터에 저장하고,
        학습 시 사용할 유효 인덱스를 구성.
        """
        updated = False
        invalid = []
        for i in range(self.size):
            if self._features_mem[i] is not None:
                continue
            try:
                feats = self._compute_features_for_index(i)
                feats = np.asarray(feats, dtype=np.float32)
                self._features_mem[i] = feats
                self._set_features_in_metadata(i, feats.tolist())
                updated = True
            except Exception as e:
                # 실패 샘플은 features=None로 표기, error 메시지 기록
                entry = self.metadata[i]
                if not isinstance(entry, dict):
                    entry = {"henle": entry}
                entry[self.features_name] = None
                entry["error"] = str(e)
                self.metadata[i] = entry
                invalid.append(i)
                updated = True
                continue

        if updated:
            self._save_metadata()

        # 학습에는 features가 있는 인덱스만 사용
        self.valid_indices = [i for i in range(self.size) if self.metadata[i].get(self.features_name) is not None]
        self._features_ready = True

    # ---------- 표준 Dataset API ----------
    def __len__(self):
        if hasattr(self, "valid_indices"):
            return len(self.valid_indices)
        return self.size

    def __getitem__(self, index):
        # (방어) 혹시 __init__에서 못했으면 첫 호출 때 보장
        if not self._features_ready:
            self._prepare_all_features()

        if hasattr(self, "valid_indices"):
            if index >= len(self.valid_indices):
                raise IndexError("Index out of range")
            real_idx = self.valid_indices[index]
        else:
            if index >= self.size:
                raise IndexError("Index out of range")
            real_idx = index

        feats = self._features_mem[real_idx]
        if feats is None:
            # 유효 인덱스인데도 비어있으면 재계산 + 저장
            feats = np.asarray(self._compute_features_for_index(real_idx), dtype=np.float32)
            self._features_mem[real_idx] = feats
            self._set_features_in_metadata(real_idx, feats.tolist())
            self._save_metadata()

        label = self._labels_mem[real_idx]
        return torch.from_numpy(feats), torch.tensor(label)


class MikrokosmosDataset(MidiFeaturesPerformanceDataset):
    def __init__(self, dataset_path=MIKROKOSMOS_PATH, numeric_versions=1):
        super().__init__(dataset_path=dataset_path, numeric_versions=numeric_versions)

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
            if isinstance(v, dict):
                v["filename"] = f"{k}.xml"  # Mikrokosmos는 파일 이름이 1.xml, 2.xml, ...
                items.append(v)                # {"henle":..., self.features_name:...}
            else:
                items.append({
                    "filename": f"{k}.xml",
                    "henle": v,
                })     # 구버전 → dict 업그레이드
        return items

    def _serialize_metadata_for_save(self) -> Any:
        # 파일에는 다시 "1","2",... 키로 저장
        out = {}
        for i, entry in enumerate(self.metadata, start=1):
            out[str(i)] = {
                "henle": entry["henle"],
                self.features_name: entry.get(self.features_name)
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
    def __init__(self, dataset_path=CIPI_PATH, numeric_versions=1):
        super().__init__(dataset_path=dataset_path, numeric_versions=numeric_versions)

    def get_metadata_path(self) -> str:
        return os.path.join(self.dataset_path, 'index.json')

    def load_metadata(self):
        path = self.get_metadata_path()
        with open(path, 'r', encoding='utf-8') as f:
            raw = json.load(f)  # 일반적으로 dict
        # values()를 사용하던 기존 코드와 호환: 리스트로 변환
        items = list(raw.values()) if isinstance(raw, dict) else list(raw)
        return items

    def _serialize_metadata_for_save(self) -> Any:
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

    def __getitem__(self, index):
        # CIPI는 다중 경로 평균이므로 오버라이드 유지
        if not self._features_ready:
            self._prepare_all_features()

        if hasattr(self, "valid_indices"):
            if index >= len(self.valid_indices):
                raise IndexError("Index out of range")
            real_idx = self.valid_indices[index]
        else:
            if index >= self.size:
                raise IndexError("Index out of range")
            real_idx = index

        feats = self._features_mem[real_idx]
        if feats is None:
            feats = np.asarray(self._compute_features_for_index(real_idx), dtype=np.float32)
            self._features_mem[real_idx] = feats
            self._set_features_in_metadata(real_idx, feats.tolist())
            self._save_metadata()

        return torch.from_numpy(feats), torch.tensor(self.get_label(real_idx))
