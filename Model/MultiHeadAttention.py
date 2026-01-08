import torch
from torch import nn


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, d_qk, d_v,dropout=0.1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_heads = num_heads
        self.d_qk = d_qk
        self.d_v = d_v


        self.W_q = nn.Linear(d_model,num_heads* d_qk)
        self.W_k = nn.Linear(d_model,num_heads*d_qk)
        self.W_v = nn.Linear(d_model,num_heads* d_v)

        self.out_prog = nn.Linear(num_heads * d_v,d_model)
        
        self.dropout= nn.Dropout(dropout)



    def forward(self,input_q,input_k,input_v,attention_mask:torch.Tensor =None,return_attn_weights=False):
        
        if input_k is None:
            input_k = input_q
        if input_v is None:
            input_v = input_k

        batch_size, seq_len_q, _ = input_q.shape
        seq_len_k = input_k.shape[1] 
        seq_len_v = input_v.shape[1]

        q: torch.Tensor = self.W_q(input_q).reshape((batch_size,seq_len_q,self.num_heads,self.d_qk)).transpose(1,2)
        k: torch.Tensor = self.W_k(input_k).reshape((batch_size,seq_len_k,self.num_heads,self.d_qk)).transpose(1,2)
        v: torch.Tensor = self.W_v(input_v).reshape((batch_size,seq_len_v,self.num_heads,self.d_v)).transpose(1,2)

        

        scores:torch.Tensor = torch.matmul(q,k.transpose(-2,-1)) / (self.d_qk**0.5)

        if attention_mask is not None:
            if attention_mask.dim() == 3:
                attention_mask = attention_mask.unsqueeze(1)  
            scores = scores.masked_fill(attention_mask, -1e9)

        attn_weights = torch.softmax(scores,dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        context = torch.matmul(attn_weights,v)

        context = context.transpose(1,2).contiguous().reshape((batch_size,seq_len_q,-1))
        
        if return_attn_weights:
            return self.out_prog(context), attn_weights
        
        return self.out_prog(context)


class FlashMultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, d_qk, d_v, dropout=0.1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_heads = num_heads
        self.d_qk = d_qk
        self.d_v = d_v
        self.dropout_p = dropout

        self.W_q = nn.Linear(d_model, num_heads * d_qk)
        self.W_k = nn.Linear(d_model, num_heads * d_qk)
        self.W_v = nn.Linear(d_model, num_heads * d_v)

        self.out_proj = nn.Linear(num_heads * d_v, d_model)

    def forward(self, input_q, input_k=None, input_v=None, 
                attention_mask: torch.Tensor = None,is_causal=False):
        
        if input_k is None:
            input_k = input_q
        if input_v is None:
            input_v = input_k

        batch_size, seq_len_q, _ = input_q.shape
        seq_len_k = input_k.shape[1]
        seq_len_v = input_v.shape[1]

        q = self.W_q(input_q).reshape((batch_size, seq_len_q, self.num_heads, self.d_qk)).transpose(1, 2)
        k = self.W_k(input_k).reshape((batch_size, seq_len_k, self.num_heads, self.d_qk)).transpose(1, 2)
        v = self.W_v(input_v).reshape((batch_size, seq_len_v, self.num_heads, self.d_v)).transpose(1, 2)


        if attention_mask is not None:
            if attention_mask.dim() == 3:
                attention_mask = attention_mask.unsqueeze(1)
        
        
        attn_output = torch.nn.functional.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attention_mask,
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=is_causal
        )
        
        
        context = attn_output.transpose(1, 2).contiguous().reshape((batch_size, seq_len_q, -1))
        
        
        return self.out_proj(context)



        