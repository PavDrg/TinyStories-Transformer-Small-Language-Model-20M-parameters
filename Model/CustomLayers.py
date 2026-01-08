import torch
from torch import nn
from Model.MultiHeadAttention import FlashMultiHeadAttention


class SinusoidalPositionalEmbedding(nn.Module):
    
    def __init__(self, d_model: int, max_len: int = 5000, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * 
            (-torch.log(torch.tensor(10000.0,dtype=torch.float)) / d_model)
        )
      
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
     
        pe = pe.unsqueeze(0)
        
        self.register_buffer("pe", pe)
    
    def forward(self, x: torch.Tensor):
       
        seq_len = x.size(1)
      
        positional_encoding = self.pe[:, :seq_len].to(dtype=x.dtype, device=x.device)
        
        return x + positional_encoding
    


class TransformerBlock(nn.Module):
    def __init__(self,d_model,num_heads,d_qk,d_v, dropout_rate,feedforward_h_dim,feed_forward_dropout,*args, **kwargs):
        super().__init__(*args, **kwargs)

        self.mha = FlashMultiHeadAttention(d_model,num_heads,d_qk,d_v,dropout_rate)
        self.prenorm1 = nn.RMSNorm(d_model)
        self.prenorm2 = nn.RMSNorm(d_model)
        self.postnorm1 = nn.RMSNorm(d_model)
        self.postnorm2= nn.RMSNorm(d_model)
        self.feedforward = FeedForwardBlock(d_model,feedforward_h_dim,feed_forward_dropout)

    
    def forward(self,input,is_causal):
        x = self.prenorm1(input)
        x = self.mha(x, is_causal=is_causal)
        x = self.postnorm1(x)
        skip_x =x + input

        x = self.prenorm2(skip_x)
        x = self.feedforward(x)
        x = self.postnorm2(x)

        x = x +skip_x

        return x
    


class FeedForwardBlock(nn.Module):
    def __init__(self,d_model,hidden_dim,dropout=0):
        super().__init__()
        self.inp_linear = nn.Linear(d_model,hidden_dim)
        self.gate_linear = nn.Linear(d_model,hidden_dim)
        self.out_linear = nn.Linear(hidden_dim,d_model)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()


    def forward(self,x):
        inp_x = self.inp_linear(x)
        inp_x = torch.nn.functional.silu(inp_x)

        gate_x = self.gate_linear(x)

        x = inp_x * gate_x
        x = self.dropout(x)
        x = self.out_linear(x)

        return x
