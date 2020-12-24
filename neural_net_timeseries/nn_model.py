from typing import List, Union, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

class Mish(nn.Module):
    def forward(self, x):
        return x * torch.tanh(F.softplus(x))
    
ACT_DICT = {
    "RELU":nn.ReLU,
    "ELU":nn.ELU,
    "MISH":Mish,
}

class MultiEmbedding(torch.nn.Module):
    def __init__(
        self, 
        embedding_sizes_list: List[Tuple[int,int]],
        padding_idx: Optional[int] = None
    ):
        super().__init__()
        self.embeddings = nn.ModuleList([
            nn.Embedding(size[0],size[1], padding_idx=padding_idx) for size in embedding_sizes_list
        ])

    def forward(self, x):
        x = torch.cat([
            self.embeddings[i](x[:,:,i]) for i in range(x.shape[-1])
        ], axis=-1)
        return x
    
class SequanceEncoderLayer(nn.Module):
    
    def __init__(
        self,
        input_size: int,
        recurrent_units: int,
        dropout: float
    ):
        super().__init__()
        
        self.recurrent_part = nn.LSTM(input_size, recurrent_units, bidirectional=True, batch_first=True)
        self.linear_part = nn.Sequential(
            nn.Conv1d(recurrent_units*2, recurrent_units, kernel_size=1),
            nn.BatchNorm1d(recurrent_units),
            nn.ELU(),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        x, _ = self.recurrent_part(x)
        x = self.linear_part(x.permute(0,2,1)).permute(0,2,1)
        return x

class Prenet(nn.Module):
    
    def __init__(
        self,
        input_size: int,
        output_size: int
    ):
        super().__init__()
        
        self.linear_part = nn.Sequential(
            nn.Conv1d(input_size, output_size, kernel_size=1),
            nn.BatchNorm1d(output_size)
        )
        
    def forward(self, x):
        if len(x.shape) < 3:
            x = torch.unsqueeze(x, dim=1)
            do_squeeze = True
        else:
            do_squeeze = False

        x = self.linear_part(x.permute(0,2,1)).permute(0,2,1)

        if do_squeeze:
            x = torch.squeeze(x, dim=1)

        return x
    
class MLP(nn.Module):
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        dropout: float,
        n_layers: int,
        activation_name: str = 'ELU',
        layer_specifications: Optional[List[int]] = None
    ):
        super().__init__()
        
        if n_layers > 0:
            self.hidden_layers = []
            for i in range(n_layers):
                if layer_specifications is None:
                    output_dim = hidden_size
                else:
                    output_dim = layer_specifications[i]

                if i == 0:
                    self.hidden_layers.append(nn.Sequential(
                        nn.Linear(input_size, output_dim),
                        ACT_DICT[activation_name](),
                        nn.Dropout(dropout)
                    ))
                else:
                    self.hidden_layers.append(nn.Sequential(
                        nn.Linear(input_dim, output_dim),
                        ACT_DICT[activation_name](),
                        nn.Dropout(dropout)
                    ))

                input_dim = output_dim

            self.hidden_layers = nn.ModuleList(self.hidden_layers)
            self.final_linear = nn.Linear(hidden_size, output_size)
        else:
            self.hidden_layers = None
            self.final_linear = nn.Linear(input_size, output_size)
            
    def forward(self, x):
        if self.hidden_layers is not None:
            for h_layer in self.hidden_layers:
                x = h_layer(x)  
        x = self.final_linear(x)
        return x
            

class SequanceEncoder(nn.Module):
    
    def __init__(
        self,
        cont_input_size: int,
        cat_input_size_list: List[Tuple[int,int]],
        
        prenet_out: int,
        
        num_enc_layers: int,
        enc_units: int,
        enc_dropout: float,
        
        use_mlp: bool = False,
        mlp_hidden_size: Optional[int] = None,
        mlp_output_size: Optional[int] = None,
        mlp_dropout: Optional[float] = None,
        mlp_n_layers: Optional[float] = None,
    ):
        super().__init__()
        
        self.embedding_layer = MultiEmbedding(cat_input_size_list, padding_idx=0)
        size_propogation = cont_input_size + sum(el[1] for el in cat_input_size_list)
        
        self.prenet = Prenet(size_propogation, prenet_out)
        
        self.encoder = []
        for i in range(num_enc_layers):
            if i == 0:
                self.encoder.append(SequanceEncoderLayer(
                    input_size=prenet_out,
                    recurrent_units=enc_units,
                    dropout=enc_dropout
                ))
            else:
                self.encoder.append(SequanceEncoderLayer(
                    input_size=enc_units,
                    recurrent_units=enc_units,
                    dropout=enc_dropout
                ))
        self.encoder = nn.ModuleList(self.encoder)
        
        if use_mlp:
            self.mlp = MLP(
                input_size=enc_units,
                hidden_size=mlp_hidden_size,
                output_size=mlp_output_size,
                dropout=mlp_dropout,
                n_layers=mlp_n_layers
            )
        else:
            self.mlp = None
            
    def forward(self, cont_x, cat_x):
        cat_x = self.embedding_layer(cat_x)
        x = torch.cat((cat_x, cont_x), axis=-1)
        x = self.prenet(x)
                
        for enc_layer in self.encoder:
            x = enc_layer(x)
            
        if self.mlp is not None:
            x = self.mlp(x)
            
        return x, cat_x

class ReccurentDecoder(nn.Module):
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        
        mlp_hidden_size: int,
        mlp_dropout: float,
        mlp_n_layers: int,
        mlp_output_size: int,
        
        use_rec_prenet: bool = False,
        rec_prenet_out: Optional[int] = None,        
    ):
        super().__init__()
        
        if use_rec_prenet:
            self.prenet = Prenet(1, rec_prenet_out)
            size_propogation = rec_prenet_out
        else:
            self.prenet = None
            size_propogation = 1
        
        self.reccurent_layer = nn.GRU(input_size+size_propogation, hidden_size, batch_first=True)
        self.mlp = MLP(
            input_size=hidden_size,
            hidden_size=mlp_hidden_size,
            output_size=mlp_output_size,
            dropout=mlp_dropout,
            n_layers=mlp_n_layers
        )
                
    def _get_rnn_cell(self, rnn):
        cell = nn.GRUCell(rnn.input_size, rnn.hidden_size)

        cell.weight_hh.data = rnn.weight_hh_l0.data
        cell.weight_ih.data = rnn.weight_ih_l0.data
        cell.bias_hh.data = rnn.bias_hh_l0.data
        cell.bias_ih.data = rnn.bias_ih_l0.data

        return cell
    
    def _straight_forward(self, x, y_tf):
        if self.prenet is not None:
            y_tf = self.prenet(y_tf)
                        
        x = torch.cat((x, y_tf), dim=-1)
        x, hidden = self.reccurent_layer(x)
        x = self.mlp(x)
        
        return x, hidden
    
    def _rec_forward(self, x, hidden, previous_pred):
        reccurent_cell = self._get_rnn_cell(self.reccurent_layer)
        
        previous_pred = previous_pred[:,-1,:]
        hidden = hidden[0,:,:]
        
        sequence = []
        for i in range(x.shape[1]):
            if self.prenet is not None:
                previous_pred = self.prenet(previous_pred)
            seq_input = torch.cat((x[:,i,:], previous_pred), dim=-1)
            hidden = reccurent_cell(seq_input, hidden)
            pred = self.mlp(hidden)
            sequence.append(pred)
            previous_pred = pred
        sequence = torch.stack(sequence, dim=1)
        
        return sequence

    def _train_mode(
        self, 
        x: torch.Tensor, 
        y_tf: torch.Tensor
    ):
        return self._straight_forward(x, y_tf)[0]
    
    def _eval_mode(
        self, 
        x_warmup: torch.Tensor, 
        y_warmup: torch.Tensor, 
        x: torch.Tensor
    ):
        y_hat_warmup, hidden = self._straight_forward(x_warmup, y_warmup)
        y_hat = self._rec_forward(
            x, 
            hidden, 
            y_hat_warmup,
        )
        
        y_hat = torch.cat((y_hat_warmup[:,-1:,:], y_hat), dim=1)
        return y_hat
    
    def forward(
        self,
        x: torch.Tensor,
        x_warmup: Optional[torch.Tensor] = None,
        y_warmup: Optional[torch.Tensor] = None,
        y_tf: Optional[torch.Tensor] = None,
    ):
        if self.training:
            if y_tf is None:
                raise RuntimeError("Provide `y_tf` for training phase")
            return self._train_mode(x, y_tf)
        else:
            if x_warmup is None or y_warmup is None:
                raise RuntimeError("Provide `x_warmup` and `y_warmup` for eval phase")
            # We shift to transform `y_warmup` into `y_tf_warmup`
            # The very first `x_warmup` do not have previous `y_warmup`
            # The very first `x` can be predicted in Teacher Forced mode
            x_warmup = torch.cat((x_warmup[:,1:,:],x[:,:1,:]), dim=1)
            x = x[:,1:,:]
            return self._eval_mode(x_warmup, y_warmup, x)

class SequanceEncoderDecoder(nn.Module):

    def __init__(
        self,

        cont_input_size: int,
        cat_input_size_list: List[Tuple[int,int]],
        
        prenet_inp_out: int,
        
        num_enc_layers: int,
        enc_units: int,
        enc_dropout: float,
        
        dec_hidden_size: int,
        
        mlp_hidden_size: int,
        mlp_dropout: float,
        mlp_n_layers: int,
        mlp_output_size: int,

        mean_estimator_hidden_size: int,
        mean_estimator_dropout: float,
        mean_estimator_n_layers: int,
        
        use_rec_prenet: bool = False,
        rec_prenet_out: Optional[int] = None,

        mean_estimator_layer_specifications: Optional[List[int]] = None
    ):
        super().__init__()

        self.encoder = SequanceEncoder(
            cont_input_size=cont_input_size,
            cat_input_size_list=cat_input_size_list,
            prenet_out=prenet_inp_out,
            num_enc_layers=num_enc_layers,
            enc_units=enc_units,
            enc_dropout=enc_dropout,
            use_mlp=False,
        )

        self.decoder = ReccurentDecoder(
            input_size=enc_units,
            hidden_size=dec_hidden_size,
            
            mlp_hidden_size=mlp_hidden_size,
            mlp_dropout=mlp_dropout,
            mlp_n_layers=mlp_n_layers,
            mlp_output_size=mlp_output_size,
            
            use_rec_prenet=use_rec_prenet,
            rec_prenet_out=rec_prenet_out
        )

        size_propogation = sum(el[1] for el in cat_input_size_list) 
        self.mean_estimator = MLP(
            input_size=size_propogation,
            hidden_size=mean_estimator_hidden_size,
            output_size=mlp_output_size,
            dropout=mean_estimator_dropout,
            n_layers=mean_estimator_n_layers,
            layer_specifications=mean_estimator_layer_specifications
        )

    def forward(
        self,
        x: Tuple[torch.Tensor,torch.Tensor],
        x_warmup: Optional[Tuple[torch.Tensor,torch.Tensor]] = None,
        y_warmup: Optional[torch.Tensor] = None,
        y_tf: Optional[torch.Tensor] = None,
        enable_all_training: bool = True,
        sum_parts: bool = False
    ):
        x, raw_features = self.encoder(*x)
        if x_warmup is not None:
            x_warmup, _ = self.encoder(*x_warmup)

        mean_estimation = self.mean_estimator(raw_features)
        if enable_all_training:
            variation_estimation = self.decoder(
                x=x,
                x_warmup=x_warmup,
                y_warmup=y_warmup,
                y_tf=y_tf
            )
        else:
            variation_estimation = None

        if sum_parts:
            return mean_estimation + variation_estimation
        else:
            return mean_estimation, variation_estimation
