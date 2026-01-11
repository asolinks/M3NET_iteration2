"""
Unified Dataloader for Multimodal Emotion Recognition
======================================================
Supports: IEMOCAP, MELD, CMU-MOSEI

All datasets output the same tuple format:
    (r1, r2, r3, r4, visual, audio, speakers, ones, labels, vid)
"""

from typing import List, Tuple, Any, Optional
import torch
from torch import Tensor
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import pickle
import numpy as np
import os

Sample = Tuple[
    Tensor, Tensor, Tensor, Tensor,  # r1,r2,r3,r4 (T, D)
    Tensor, Tensor,                   # visual, audio (T, V/A)
    Tensor, Tensor, Tensor,           # speakers, ones, labels
    str                               # vid
]


class IEMOCAPDataset(Dataset[Sample]):
    """IEMOCAP Dataset - preserved from working implementation."""

    def __init__(self, train: bool = True, root: str = "./IEMOCAP_features"):
        (
            self.videoIDs, self.videoSpeakers, self.videoLabels,
            self.videoText, self.videoAudio, self.videoVisual,
            self.videoSentence, self.trainVid, self.testVid
        ) = pickle.load(open(f"{root}/IEMOCAP_features.pkl", "rb"), encoding="latin1")

        (
            _, _, self.roberta1, self.roberta2, self.roberta3, self.roberta4,
            _, _, _, _,
        ) = pickle.load(open(f"{root}/iemocap_features_roberta.pkl", "rb"), encoding='latin1')

        self.keys: List[str] = list(self.trainVid if train else self.testVid)
        assert len(self.keys) > 0, "No videos found for this split."
        self.len = len(self.keys)

    def __len__(self) -> int:
        return len(self.keys)

    def __getitem__(self, index: int) -> Sample:
        vid = self.keys[index]

        r1 = torch.as_tensor(np.asarray(self.roberta1[vid]), dtype=torch.float32)
        r2 = torch.as_tensor(np.asarray(self.roberta2[vid]), dtype=torch.float32)
        r3 = torch.as_tensor(np.asarray(self.roberta3[vid]), dtype=torch.float32)
        r4 = torch.as_tensor(np.asarray(self.roberta4[vid]), dtype=torch.float32)
        vv = torch.as_tensor(np.asarray(self.videoVisual[vid]), dtype=torch.float32)
        va = torch.as_tensor(np.asarray(self.videoAudio[vid]), dtype=torch.float32)

        spk_arr = np.asarray(self.videoSpeakers[vid], dtype=object)
        m_mask = (spk_arr == "M").astype(np.float32)
        f_mask = 1.0 - m_mask
        spk = torch.as_tensor(np.stack([m_mask, f_mask], axis=1), dtype=torch.float32)

        T = len(self.videoLabels[vid])
        ones = torch.ones(T, dtype=torch.float32)
        y = torch.as_tensor(np.asarray(self.videoLabels[vid]), dtype=torch.int64)

        for name, x in [("r1", r1), ("r2", r2), ("r3", r3), ("r4", r4),
                        ("visual", vv), ("audio", va), ("speakers", spk), ("ones", ones)]:
            assert x.shape[0] == T, f"{name} length {x.shape[0]} != labels {T} for video {vid}"

        return r1, r2, r3, r4, vv, va, spk, ones, y, vid

    @staticmethod
    def collate_fn(batch: List[Sample]) -> Tuple[Any, ...]:
        r1_list, r2_list, r3_list, r4_list, vv_list, va_list, spk_list, ones_list, y_list, vids = zip(*batch)

        def pad_time_first(lst):
            return pad_sequence(lst, batch_first=False)

        r1 = pad_time_first(r1_list)
        r2 = pad_time_first(r2_list)
        r3 = pad_time_first(r3_list)
        r4 = pad_time_first(r4_list)
        vv = pad_time_first(vv_list)
        va = pad_time_first(va_list)
        spk = pad_time_first(spk_list)

        ones = pad_sequence(ones_list, batch_first=True)
        y = pad_sequence(y_list, batch_first=True, padding_value=-100)

        return r1, r2, r3, r4, vv, va, spk, ones, y, list(vids)


class MELDDataset(Dataset[Sample]):
    """MELD Dataset - preserved from working implementation."""

    def __init__(self, train: bool = True, root: str = "./MELD_features") -> None:
        main_path = f"{root}/MELD_features_raw1.pkl"
        try:
            (
                self.videoIDs, self.videoSpeakers, self.videoLabels,
                self.videoText, self.videoAudio, self.videoVisual,
                self.videoSentence, self.trainVid, self.testVid, _
            ) = pickle.load(open(main_path, "rb"))
        except Exception:
            (
                self.videoIDs, self.videoSpeakers, self.videoLabels,
                self.videoText, self.videoAudio, self.videoVisual,
                self.videoSentence, self.trainVid, self.testVid, _
            ) = pickle.load(open(main_path, "rb"), encoding="latin1")

        rob_path = f"{root}/meld_features_roberta.pkl"
        try:
            (
                _, _, _, self.roberta1, self.roberta2, self.roberta3, self.roberta4,
                _, self.trainIds, self.testIds, self.validIds
            ) = pickle.load(open(rob_path, "rb"))
        except Exception:
            (
                _, _, _, self.roberta1, self.roberta2, self.roberta3, self.roberta4,
                _, self.trainIds, self.testIds, self.validIds
            ) = pickle.load(open(rob_path, "rb"), encoding="latin1")

        self.keys: List[str] = list(self.trainVid if train else self.testVid)
        assert len(self.keys) > 0, "No videos found for this split."
        self.len = len(self.keys)

    def __len__(self) -> int:
        return len(self.keys)

    def __getitem__(self, index: int) -> Sample:
        vid = self.keys[index]

        r1 = torch.as_tensor(np.asarray(self.roberta1[vid]), dtype=torch.float32)
        r2 = torch.as_tensor(np.asarray(self.roberta2[vid]), dtype=torch.float32)
        r3 = torch.as_tensor(np.asarray(self.roberta3[vid]), dtype=torch.float32)
        r4 = torch.as_tensor(np.asarray(self.roberta4[vid]), dtype=torch.float32)
        vv = torch.as_tensor(np.asarray(self.videoVisual[vid]), dtype=torch.float32)
        va = torch.as_tensor(np.asarray(self.videoAudio[vid]), dtype=torch.float32)
        spk = torch.as_tensor(np.asarray(self.videoSpeakers[vid]), dtype=torch.float32)

        T = len(self.videoLabels[vid])
        ones = torch.ones(T, dtype=torch.float32)
        y = torch.as_tensor(np.asarray(self.videoLabels[vid]), dtype=torch.int64)

        for name, x in [("r1", r1), ("r2", r2), ("r3", r3), ("r4", r4),
                        ("visual", vv), ("audio", va), ("speakers", spk), ("ones", ones)]:
            assert x.shape[0] == T, f"{name} length {x.shape[0]} != labels {T} for video {vid}"

        return r1, r2, r3, r4, vv, va, spk, ones, y, vid

    def return_labels(self) -> List[int]:
        out: List[int] = []
        for key in self.keys:
            out += list(self.videoLabels[key])
        return out

    @staticmethod
    def collate_fn(batch: List[Sample]) -> Tuple[Any, ...]:
        (r1_list, r2_list, r3_list, r4_list,
         vv_list, va_list, spk_list, ones_list, y_list, vids) = zip(*batch)

        def pad_time_first(lst):
            return pad_sequence(lst, batch_first=False)

        r1 = pad_time_first(r1_list)
        r2 = pad_time_first(r2_list)
        r3 = pad_time_first(r3_list)
        r4 = pad_time_first(r4_list)
        vv = pad_time_first(vv_list)
        va = pad_time_first(va_list)
        spk = pad_time_first(spk_list)

        ones = pad_sequence(ones_list, batch_first=True)
        y = pad_sequence(y_list, batch_first=True, padding_value=-100)

        return r1, r2, r3, r4, vv, va, spk, ones, y, list(vids)


class MOSEIDataset(Dataset[Sample]):
    """
    CMU-MOSEI Dataset - Online opinion video monologues
    6 emotion classes, single speaker, compatible with IEMOCAP/MELD pipeline
    """
    
    NUM_CLASSES = 6
    
    def __init__(self, split: str = 'train', root: str = './MOSEI_features', max_seq_len: int = 50):
        assert split in ('train', 'val', 'valid', 'test')
        self.split = split if split != 'valid' else 'val'
        self.root = root
        self.max_seq_len = max_seq_len
        
        self._load_features()
        
        if self.split == 'train':
            self.keys = list(self.trainVid)
        elif self.split == 'val':
            self.keys = list(self.validVid) if self.validVid else list(self.trainVid)[:100]
        else:
            self.keys = list(self.testVid)
        
        assert len(self.keys) > 0, f"No samples found for split={split}"
        print(f"[MOSEIDataset] Loaded {len(self.keys)} samples for split={split}")
    
    def _load_features(self):
        possible_files = ['MOSEI_features_full.pkl', 'MOSEI_features.pkl', 'mosei_features.pkl']
        
        data = None
        for fname in possible_files:
            path = os.path.join(self.root, fname)
            if os.path.exists(path):
                try:
                    with open(path, 'rb') as f:
                        data = pickle.load(f)
                    break
                except:
                    try:
                        with open(path, 'rb') as f:
                            data = pickle.load(f, encoding='latin1')
                        break
                    except:
                        continue
        
        if data is None:
            raise FileNotFoundError(f"Could not find MOSEI features in {self.root}")
        
        if isinstance(data, dict):
            self.videoLabels = data.get('videoLabels', {})
            self.videoText = data.get('videoText', {})
            self.videoAudio = data.get('videoAudio', {})
            self.videoVisual = data.get('videoVisual', {})
            self.trainVid = data.get('trainVid', set())
            self.testVid = data.get('testVid', set())
            self.validVid = data.get('validVid', set())
            self.roberta1 = data.get('roberta1', None)
            self.roberta2 = data.get('roberta2', None)
            self.roberta3 = data.get('roberta3', None)
            self.roberta4 = data.get('roberta4', None)
        else:
            (_, _, self.videoLabels, self.videoText, self.videoAudio, self.videoVisual,
             _, self.trainVid, self.testVid, *rest) = data
            self.validVid = rest[0] if rest else set()
            self.roberta1 = self.roberta2 = self.roberta3 = self.roberta4 = None
        
        sample_vid = next(iter(self.videoText.keys()), None)
        if sample_vid:
            t = np.array(self.videoText[sample_vid])
            a = np.array(self.videoAudio[sample_vid])
            v = np.array(self.videoVisual[sample_vid])
            self._text_dim = t.shape[-1] if t.ndim > 1 else 300
            self._audio_dim = a.shape[-1] if a.ndim > 1 else 74
            self._visual_dim = v.shape[-1] if v.ndim > 1 else 35
        else:
            self._text_dim, self._audio_dim, self._visual_dim = 300, 74, 35
        
        self._has_roberta = self.roberta1 is not None and len(self.roberta1) > 0

    def __len__(self) -> int:
        return len(self.keys)

    def __getitem__(self, index: int) -> Sample:
        vid = self.keys[index]
        
        if self._has_roberta:
            r1 = np.array(self.roberta1.get(vid, [[0]*1024]), dtype=np.float32)
            r2 = np.array(self.roberta2.get(vid, [[0]*1024]), dtype=np.float32)
            r3 = np.array(self.roberta3.get(vid, [[0]*1024]), dtype=np.float32)
            r4 = np.array(self.roberta4.get(vid, [[0]*1024]), dtype=np.float32)
        else:
            text = np.array(self.videoText.get(vid, [[0]*self._text_dim]), dtype=np.float32)
            if text.ndim == 1:
                text = text.reshape(1, -1)
            r1 = r2 = r3 = r4 = text.copy()
        
        va = np.array(self.videoAudio.get(vid, [[0]*self._audio_dim]), dtype=np.float32)
        vv = np.array(self.videoVisual.get(vid, [[0]*self._visual_dim]), dtype=np.float32)
        
        # Ensure 2D
        for arr in [r1, r2, r3, r4, va, vv]:
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
        
        r1 = np.nan_to_num(r1 if r1.ndim > 1 else r1.reshape(1,-1), nan=0.0)
        r2 = np.nan_to_num(r2 if r2.ndim > 1 else r2.reshape(1,-1), nan=0.0)
        r3 = np.nan_to_num(r3 if r3.ndim > 1 else r3.reshape(1,-1), nan=0.0)
        r4 = np.nan_to_num(r4 if r4.ndim > 1 else r4.reshape(1,-1), nan=0.0)
        va = np.nan_to_num(va if va.ndim > 1 else va.reshape(1,-1), nan=0.0)
        vv = np.nan_to_num(vv if vv.ndim > 1 else vv.reshape(1,-1), nan=0.0)
        
        labels = self.videoLabels.get(vid, [0])
        if isinstance(labels, np.ndarray):
            labels = labels.tolist()
        if len(labels) > 0 and isinstance(labels[0], (list, np.ndarray)):
            labels = [int(np.argmax(l[1:7])) if len(l) >= 7 else int(np.argmax(l)) for l in labels]
        labels = [min(max(int(l), 0), self.NUM_CLASSES - 1) for l in labels]
        
        T = max(1, r1.shape[0], va.shape[0], vv.shape[0], len(labels))
        T = min(T, self.max_seq_len)
        
        def pad_or_trunc(arr, target_len, feat_dim):
            if arr.shape[0] >= target_len:
                return arr[:target_len]
            return np.vstack([arr, np.zeros((target_len - arr.shape[0], feat_dim), dtype=np.float32)])
        
        r1 = pad_or_trunc(r1, T, r1.shape[1])
        r2 = pad_or_trunc(r2, T, r2.shape[1])
        r3 = pad_or_trunc(r3, T, r3.shape[1])
        r4 = pad_or_trunc(r4, T, r4.shape[1])
        va = pad_or_trunc(va, T, va.shape[1])
        vv = pad_or_trunc(vv, T, vv.shape[1])
        
        if len(labels) < T:
            labels = labels + [labels[-1]] * (T - len(labels))
        else:
            labels = labels[:T]
        
        r1 = torch.as_tensor(r1, dtype=torch.float32)
        r2 = torch.as_tensor(r2, dtype=torch.float32)
        r3 = torch.as_tensor(r3, dtype=torch.float32)
        r4 = torch.as_tensor(r4, dtype=torch.float32)
        vv = torch.as_tensor(vv, dtype=torch.float32)
        va = torch.as_tensor(va, dtype=torch.float32)
        
        # Single speaker one-hot (2-dim for compatibility)
        spk = torch.zeros(T, 2, dtype=torch.float32)
        spk[:, 0] = 1.0
        
        ones = torch.ones(T, dtype=torch.float32)
        y = torch.tensor(labels, dtype=torch.int64)
        
        return r1, r2, r3, r4, vv, va, spk, ones, y, vid

    def return_labels(self) -> List[int]:
        out = []
        for vid in self.keys:
            labels = self.videoLabels.get(vid, [0])
            if isinstance(labels[0], (list, np.ndarray)):
                out += [int(np.argmax(l[1:7])) if len(l) >= 7 else 0 for l in labels]
            else:
                out += [int(l) for l in labels]
        return out

    @staticmethod
    def collate_fn(batch: List[Sample]) -> Tuple[Any, ...]:
        (r1_list, r2_list, r3_list, r4_list,
         vv_list, va_list, spk_list, ones_list, y_list, vids) = zip(*batch)

        def pad_time_first(lst):
            return pad_sequence(lst, batch_first=False)

        r1 = pad_time_first(r1_list)
        r2 = pad_time_first(r2_list)
        r3 = pad_time_first(r3_list)
        r4 = pad_time_first(r4_list)
        vv = pad_time_first(vv_list)
        va = pad_time_first(va_list)
        spk = pad_time_first(spk_list)

        ones = pad_sequence(ones_list, batch_first=True)
        y = pad_sequence(y_list, batch_first=True, padding_value=-100)

        return r1, r2, r3, r4, vv, va, spk, ones, y, list(vids)


# Dataset info for train.py
DATASET_INFO = {
    'IEMOCAP': {'n_classes': 6, 'n_speakers': 2, 'D_audio': 1582, 'D_visual': 342, 'D_text': 1024},
    'MELD': {'n_classes': 7, 'n_speakers': 9, 'D_audio': 300, 'D_visual': 342, 'D_text': 1024},
    'MOSEI': {'n_classes': 6, 'n_speakers': 1, 'D_audio': 74, 'D_visual': 35, 'D_text': 300},
}
