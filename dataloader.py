from typing import List, Tuple, Any
import torch
from torch import Tensor
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import pickle, numpy as np

# sample tuple returned by __getitem__
Sample = Tuple[
    Tensor, # roberta1 -> (T, D1)
    Tensor, # roberta2 -> (T, D2)
    Tensor, # roberta3 -> (T, D3)
    Tensor, # roberta4 -> (T, D4)
    Tensor, # visual -> (T, V)
    Tensor, # audio -> (T, A)
    Tensor, # speakers -> (T, 2) one-hot
    Tensor, # ones -> (T,)
    Tensor, # labels -> (T,) int64
    str     # vid
]

class IEMOCAPDataset(Dataset[Sample]):

    def __init__(self, train=True, root: str = "./IEMOCAP_features"):
        # load main feature bundle
        (
            self.videoIDs, 
            self.videoSpeakers, 
            self.videoLabels, 
            self.videoText,
            self.videoAudio, 
            self.videoVisual, 
            self.videoSentence, 
            self.trainVid,
            self.testVid 
        ) = pickle.load(open(f"{root}/IEMOCAP_features.pkl", "rb"), encoding="latin1")

        # load Roberta pooled layers (we only keep 3rd..6th here)
        (
            _, 
            _, 
            self.roberta1, 
            self.roberta2, 
            self.roberta3, 
            self.roberta4,
            _, 
            _, 
            _, 
            _,
        ) = pickle.load(open(f"{root}/iemocap_features_roberta.pkl", "rb"), encoding='latin1')

        '''
        label index mapping = {'hap' :0, 'sad' :1, 'neu' :2, 'ang' :3, 'exc' :4, 'fru' :5}
        '''
        
        # split keys        
        self.keys: List[str] = list(self.trainVid if train else self.testVid)

        # quick sanity
        assert len(self.keys) > 0, "No videos found for this split."

        self.len = len(self.keys)
    
    def __len__(self) -> int:
        return len(self.keys)

    def __getitem__(self, index):
        vid = self.keys[index]

        # convert lists -> numpy -> torch with minimal copying and explicit dtypes 
        r1 = torch.as_tensor(np.asarray(self.roberta1[vid]), dtype=torch.float32) # (T, D1)
        r2 = torch.as_tensor(np.asarray(self.roberta2[vid]), dtype=torch.float32) # (T, D2)
        r3 = torch.as_tensor(np.asarray(self.roberta3[vid]), dtype=torch.float32) # (T, D3)
        r4 = torch.as_tensor(np.asarray(self.roberta4[vid]), dtype=torch.float32) # (T, D4)
        vv = torch.as_tensor(np.asarray(self.videoVisual[vid]), dtype=torch.float32) # (T, V)
        va = torch.as_tensor(np.asarray(self.videoAudio[vid]), dtype=torch.float32) # (T, A)

        # speakers -> one-hot (vectorised)
        spk_arr = np.asarray(self.videoSpeakers[vid], dtype=object)

        # M (male) -> [1,0], else -> [0,1], which is F (female)
        m_mask = (spk_arr == "M").astype(np.float32)
        f_mask = 1.0 - m_mask
        spk = torch.as_tensor(np.stack([m_mask, f_mask], axis=1), dtype=torch.float32) # (T, 2)

        # ones and labels
        T = len(self.videoLabels[vid])
        ones = torch.ones(T, dtype=torch.float32) # (T,)
        y = torch.as_tensor(np.asarray(self.videoLabels[vid]), dtype=torch.int64) # (T,)

        # sanity: all time lengths must match labels length
        for name , x in [
            ("r1", r1),
            ("r2", r2),
            ("r3", r3),
            ("r4", r4),
            ("visual", vv),
            ("audio", va),
            ("speakers", spk),
            ("ones", ones),
            ]:
            
            assert x.shape[0] == T, f"{name} length {x.shape[0]} !=labels{T} for video {vid}"
        
        return r1, r2, r3, r4, vv, va, spk, ones, y, vid

    @staticmethod
    def collate_fn(batch: List[Sample]) -> Tuple[Any, ...]:
        """
        A list of 10-tuples are collected into a batched 10-tuple.
            - items 0..6 are padded with batch_first=False -> shapes (T_max, B, ...)
            - items 7..8 are padded with batch_first=True -> shapes (B, T_max)
            - item 9 is a list[str] of video ids
        """

        # unzip list of tuples into 10 lists
        r1_list, r2_list, r3_list, r4_list, vv_list, va_list, spk_list, ones_list, y_list, vids = zip(*batch)

        # helper: pad (T, F) tensors to (T_max, B, F) [time-first to preserve original behaviour 0..6]
        def pad_time_first(lst: Tuple[Tensor, ...]) -> Tensor:
            # pad_sequence with default batch_first -> (T_max, B, F)
            return pad_sequence(lst, batch_first=False)
        
        # 0..6: features, speakers -> (T_max, B, F)
        r1 = pad_time_first(r1_list)
        r2 = pad_time_first(r2_list)
        r3 = pad_time_first(r3_list)
        r4 = pad_time_first(r4_list)
        vv = pad_time_first(vv_list)
        va = pad_time_first(va_list)
        spk = pad_time_first(spk_list)

        # 7: ones -> (B, T_max)
        ones = pad_sequence(ones_list, batch_first=True)

        # 8: labels -> (B, T_max) with padding value -100 (good for cross entropy(ignore_index = -100))
        y = pad_sequence(y_list, batch_first=True, padding_value=-100)

        # 9: vids -> keep as list[str]
        vids = list(vids)

        return r1, r2, r3, r4, vv, va, spk, ones, y, vids

class MELDDataset(Dataset[Sample]):
    """
    Refurbished MELD dataset in the same style as your new IEMOCAPDataset.
    Expects:
      - {root}/MELD_features.pkl
      - {root}/meld_features_roberta.pkl
    """

    def __init__(self, train: bool = True, root: str = "./MELD_features") -> None:
        # --- Load main feature bundle ---
        # Structure matches your old code:
        # (videoIDs, videoSpeakers, videoLabels, videoText, videoAudio, videoVisual,
        #  videoSentence, trainVid, testVid, _)
        main_path = f"{root}/MELD_features_raw1.pkl"
        try:
            (
                self.videoIDs,
                self.videoSpeakers,
                self.videoLabels,
                self.videoText,
                self.videoAudio,
                self.videoVisual,
                self.videoSentence,
                self.trainVid,
                self.testVid,
                _
            ) = pickle.load(open(main_path, "rb"))
        except Exception:
            (
                self.videoIDs,
                self.videoSpeakers,
                self.videoLabels,
                self.videoText,
                self.videoAudio,
                self.videoVisual,
                self.videoSentence,
                self.trainVid,
                self.testVid,
                _
            ) = pickle.load(open(main_path, "rb"), encoding="latin1")

        # --- Load RoBERTa pooled layers ---
        # Structure (per your old MELD loader):
        # (_, _, _, roberta1, roberta2, roberta3, roberta4, _, trainIds, testIds, validIds)
        rob_path = f"{root}/meld_features_roberta.pkl"
        try:
            (
                _,
                _,
                _,
                self.roberta1,
                self.roberta2,
                self.roberta3,
                self.roberta4,
                _,
                self.trainIds,
                self.testIds,
                self.validIds
            ) = pickle.load(open(rob_path, "rb"))
        except Exception:
            (
                _,
                _,
                _,
                self.roberta1,
                self.roberta2,
                self.roberta3,
                self.roberta4,
                _,
                self.trainIds,
                self.testIds,
                self.validIds
            ) = pickle.load(open(rob_path, "rb"), encoding="latin1")

        # Split keys (keep parity with your IEMOCAP new class)
        self.keys: List[str] = list(self.trainVid if train else self.testVid)
        assert len(self.keys) > 0, "No videos found for this split."
        self.len = len(self.keys)

    def __len__(self) -> int:
        return len(self.keys)

    def __getitem__(self, index: int) -> Sample:
        vid = self.keys[index]

        # Convert lists -> numpy -> torch with explicit dtypes (minimal copy)
        r1 = torch.as_tensor(np.asarray(self.roberta1[vid]), dtype=torch.float32)  # (T, D1)
        r2 = torch.as_tensor(np.asarray(self.roberta2[vid]), dtype=torch.float32)  # (T, D2)
        r3 = torch.as_tensor(np.asarray(self.roberta3[vid]), dtype=torch.float32)  # (T, D3)
        r4 = torch.as_tensor(np.asarray(self.roberta4[vid]), dtype=torch.float32)  # (T, D4)

        vv = torch.as_tensor(np.asarray(self.videoVisual[vid]), dtype=torch.float32)  # (T, V)
        va = torch.as_tensor(np.asarray(self.videoAudio[vid]), dtype=torch.float32)   # (T, A)

        # MELD speakers are already one-hot per your old implementation -> (T, 2)
        spk = torch.as_tensor(np.asarray(self.videoSpeakers[vid]), dtype=torch.float32)

        # ones and labels
        T = len(self.videoLabels[vid])
        ones = torch.ones(T, dtype=torch.float32)  # (T,)
        y = torch.as_tensor(np.asarray(self.videoLabels[vid]), dtype=torch.int64)  # (T,)

        # Sanity: all time dims must match label length
        for name, x in [
            ("r1", r1), ("r2", r2), ("r3", r3), ("r4", r4),
            ("visual", vv), ("audio", va), ("speakers", spk), ("ones", ones),
        ]:
            assert x.shape[0] == T, f"{name} length {x.shape[0]} != labels {T} for video {vid}"

        return r1, r2, r3, r4, vv, va, spk, ones, y, vid

    def return_labels(self) -> List[int]:
        """Optional helper preserved from the old class."""
        out: List[int] = []
        for key in self.keys:
            out += list(self.videoLabels[key])
        return out

    @staticmethod
    def collate_fn(batch: List[Sample]) -> Tuple[Any, ...]:
        """
        Collate a list of 10-tuples into a batched 10-tuple.
        - Items 0..6: pad to (T_max, B, F)   [time-first]
        - Item 7    : ones -> (B, T_max)
        - Item 8    : labels -> (B, T_max), padding_value = -100 (for CE ignore_index)
        - Item 9    : list[str] video ids
        """
        # Unzip list of tuples into 10 lists
        (r1_list, r2_list, r3_list, r4_list,
         vv_list, va_list, spk_list, ones_list, y_list, vids) = zip(*batch)

        # Helper: pad (T, F) -> (T_max, B, F) (time-first to preserve prior behavior)
        def pad_time_first(lst: Tuple[Tensor, ...]) -> Tensor:
            return pad_sequence(lst, batch_first=False)

        # 0..6 features/speakers
        r1  = pad_time_first(r1_list)
        r2  = pad_time_first(r2_list)
        r3  = pad_time_first(r3_list)
        r4  = pad_time_first(r4_list)
        vv  = pad_time_first(vv_list)
        va  = pad_time_first(va_list)
        spk = pad_time_first(spk_list)

        # 7: ones -> (B, T_max)
        ones = pad_sequence(ones_list, batch_first=True)

        # 8: labels -> (B, T_max), CE-friendly padding
        y = pad_sequence(y_list, batch_first=True, padding_value=-100)

        # 9: vids -> list[str]
        vids = list(vids)

        return r1, r2, r3, r4, vv, va, spk, ones, y, vids