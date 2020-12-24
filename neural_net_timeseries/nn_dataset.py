from typing import List, Union, Tuple, Optional
from copy import deepcopy

import torch
import numpy as np
import pandas as pd

from torch.utils.data import Dataset

def slice_and_pad(
    seq: np.ndarray,
    start_idx: int,
    seq_len: int,
    pad_value: Union[float, int],
    forward_padding: bool
):
    seq = seq[start_idx:start_idx+seq_len]
    
    pad_tuple = (0, pad_value) if forward_padding else (pad_value, 0)
    seq = np.pad(seq, ((0,seq_len-len(seq)),(0,0)), constant_values=pad_tuple)
    
    return seq

class BaseTimeseriesDataset(Dataset):
    
    def __init__(
        self,
        cont_cols: List[str],
        cat_cols: List[str],
        y_col: str,
        arrange_col: str,
        groupby_cols: List[str],
        pad_value_cont: float = 0,
        pad_value_target: float = 0,
    ):
        self.cont_cols = cont_cols
        self.cat_cols = cat_cols
        self.y_col = y_col
        self.arrange_col = arrange_col
        self.groupby_cols = groupby_cols
        self.pad_value_cont = pad_value_cont
        self.pad_value_target = pad_value_target
        
    def _sort_dataframe(self, df: pd.DataFrame):
        return df.sort_values(self.arrange_col).reset_index(drop=True)
    
    def _compute_lens(self, df: pd.DataFrame):
        return df.groupby(self.groupby_cols)[self.cont_cols[0]].apply(len)
    
    def _cast_df_dtypes_pp_cat(self, df: pd.DataFrame):
        # Convert category columns to int
        # Increase by one, cause 0 refers to padding 
        df[self.cat_cols] = df[self.cat_cols].astype(int) + 1
        df[self.cont_cols] = df[self.cont_cols].astype(float)
        df[self.y_col] = df[self.y_col].astype(float)

        return df
        
    def _create_sequances(self, df: pd.DataFrame, seq_cols: List[str], max_len: Optional[int] = None):
        if max_len is None:
            return df.groupby(self.groupby_cols).apply(lambda x: x[seq_cols].values).tolist()
        else:
            return df.groupby(self.groupby_cols).apply(lambda x: x[seq_cols].values[-max_len:,:]).tolist()
        
    
class TrainTimeseriesDataset(BaseTimeseriesDataset):
    
    def __init__(
        self,
        df: pd.DataFrame,
        cont_cols: List[str],
        cat_cols: List[str],
        y_col: str,
        arrange_col: str,
        groupby_cols: List[str],
        pad_value_cont: float = 0,
        pad_value_target: float = 0,
        seq_len: Union[str, int] = 'median',
    ):
        super().__init__(
            cont_cols=cont_cols,
            cat_cols=cat_cols,
            y_col=y_col,
            arrange_col=arrange_col,
            groupby_cols=groupby_cols,
            pad_value_cont=pad_value_cont,
            pad_value_target=pad_value_target
        )
        
        df = deepcopy(df)
        
        df = self._sort_dataframe(df)
        self.lens = self._compute_lens(df)
        self.seq_len = seq_len
        self._set_seq_len()
        
        df = self._cast_df_dtypes_pp_cat(df)
        
        self.cont_seqs = self._create_sequances(df, cont_cols)
        self.cat_seqs = self._create_sequances(df, cat_cols)
        self.y_seqs = self._create_sequances(df, [y_col])
        
        del df
            
    def _set_seq_len(self):
        if isinstance(self.seq_len, str):
            if self.seq_len == 'median':
                self.seq_len = int(self.lens.median())
            elif self.seq_len == 'mean':
                self.seq_len = int(self.lens.mean())
            elif self.seq_len == 'min':
                self.seq_len = int(self.lens.min())
            elif self.seq_len == 'max':
                self.seq_len = int(self.lens.max())
            else:
                raise RuntimeError(f"{self.seq_len} is invalid value for `seq_len`")
        elif isinstance(self.seq_len, int):
            pass
        else: 
            raise RuntimeError(f"{self.seq_len} is invalid value for `seq_len`") 
                
        print(f"Seuqance length set to {self.seq_len}")
        
    def __getitem__(self, idx):
        features = []
        features.append((0, self.cat_seqs[idx]))
        features.append((self.pad_value_cont, self.cont_seqs[idx]))
        features.append((self.pad_value_target, self.y_seqs[idx]))
        
        features, initial_len = self._slice_pad_seqs(
            features
        )
        cat_features, cont_features, target = features
        
        tf = target[:-1]
        cat_features, cont_features, target = cat_features[1:], cont_features[1:], target[1:]
                
        return dict(
            cat_features=torch.from_numpy(cat_features).long(),
            cont_features=torch.from_numpy(cont_features).float(),
            tf=torch.from_numpy(tf).float(),
            target=torch.from_numpy(target).float(),
            initial_len=initial_len
        )
    
    def _slice_pad_seqs(self, seqs: Tuple[Union[float, int], np.array]):
        seq_len = seqs[0][1].shape[0]
        if seq_len > self.seq_len:
            start_idx = np.random.randint(
                low=0, 
                high=(seq_len - self.seq_len) + 1
            )
            seq_len = self.seq_len
        else:
            start_idx = 0
            
        seqs = [slice_and_pad(
            seq=el,
            start_idx=start_idx,
            seq_len=self.seq_len,
            pad_value=pad_v,
            forward_padding=True
        ) for pad_v, el in seqs]
        
        return seqs, seq_len
    
    def __len__(self):
        return len(self.cont_seqs)
    
    
class TestTimeseriesDataset(BaseTimeseriesDataset):
    
    def __init__(
        self,
        df: pd.DataFrame,
        warmup_df: pd.DataFrame,
        cont_cols: List[str],
        cat_cols: List[str],
        y_col: str,
        arrange_col: str,
        groupby_cols: List[str],
        pad_value_cont: float = 0,
        pad_value_target: float = 0,
        warmup_len: int = 3,
    ):
        super().__init__(
            cont_cols=cont_cols,
            cat_cols=cat_cols,
            y_col=y_col,
            arrange_col=arrange_col,
            groupby_cols=groupby_cols,
            pad_value_cont=pad_value_cont,
            pad_value_target=pad_value_target
        )
        
        df = deepcopy(df)
        warmup_df = deepcopy(warmup_df)
        
        warmup_df = self._select_sub_warmup(df, warmup_df)
        
        df = self._sort_dataframe(df)
        self.lens = self._compute_lens(df)
        
        self.seq_len = self.lens.max()
        self.warmup_len = warmup_len
        print(f"Seuqance length set to {self.seq_len}")
        
        warmup_df = self._sort_dataframe(warmup_df)
        
        df = self._cast_df_dtypes_pp_cat(df)
        warmup_df = self._cast_df_dtypes_pp_cat(warmup_df)
        
        self.cont_seqs = self._create_sequances(df, cont_cols)
        self.cat_seqs = self._create_sequances(df, cat_cols)
        self.y_seqs = self._create_sequances(df, [y_col])
        
        self.warmup_cont_seqs = self._create_sequances(warmup_df, cont_cols, max_len=warmup_len)
        self.warmup_cat_seqs = self._create_sequances(warmup_df, cat_cols, max_len=warmup_len)
        self.warmup_y_seqs = self._create_sequances(warmup_df, [y_col], max_len=warmup_len)
        
        del df
        del warmup_df
        
    def _select_sub_warmup(self, df: pd.DataFrame, warmup_df: pd.DataFrame):
        df['temp_col'] = ''
        warmup_df['temp_col'] = ''
        for col in self.groupby_cols:
            df['temp_col'] += '_' + df[col].astype(str)
            warmup_df['temp_col'] += '_' + warmup_df[col].astype(str)
                        
        warmup_df = warmup_df[warmup_df['temp_col'].isin(df['temp_col'])].reset_index(drop=True)
        warmup_df = warmup_df.drop(columns='temp_col')
        df = df.drop(columns='temp_col')
        
        return warmup_df
        
    def __getitem__(self, idx):
        features = []
        features.append((0, self.cat_seqs[idx]))
        features.append((self.pad_value_cont, self.cont_seqs[idx]))
        features.append((self.pad_value_target, self.y_seqs[idx]))
        features, initial_len = self._slice_pad_seqs(
            features, seq_len=self.seq_len
        )
        cat_features, cont_features, target = features
        
        warmup_features = []
        warmup_features.append((0, self.warmup_cat_seqs[idx]))
        warmup_features.append((self.pad_value_cont, self.warmup_cont_seqs[idx]))
        warmup_features.append((self.pad_value_target, self.warmup_y_seqs[idx]))
        warmup_features, _ = self._slice_pad_seqs(
            warmup_features, seq_len=self.warmup_len, forward_padding=False
        )
        warmup_cat_features, warmup_cont_features, warmup_target = warmup_features
                
        return dict(
            cat_features=torch.from_numpy(cat_features).long(),
            cont_features=torch.from_numpy(cont_features).float(),
            warmup_cat_features=torch.from_numpy(warmup_cat_features).long(),
            warmup_cont_features=torch.from_numpy(warmup_cont_features).float(),
            warmup_target=torch.from_numpy(warmup_target).float(),
            target=torch.from_numpy(target).float(),
            initial_len=initial_len
        )
    
    def _slice_pad_seqs(self, seqs: Tuple[Union[float, int], np.array], seq_len: int, forward_padding: bool = True): 
        initial_len = seqs[0][1].shape[0]
        
        seqs = [slice_and_pad(
            seq=el,
            start_idx=0,
            seq_len=seq_len,
            pad_value=pad_v,
            forward_padding=forward_padding
        ) for pad_v, el in seqs]
        
        return seqs, initial_len
    
    def __len__(self):
        return len(self.cont_seqs)