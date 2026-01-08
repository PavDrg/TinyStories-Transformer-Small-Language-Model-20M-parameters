import torch
from torch import nn
from Model.CustomLayers import SinusoidalPositionalEmbedding,TransformerBlock
from Model import MultiHeadAttention
class SLMModel(nn.Module):
    def __init__(self,vocab_size,embedding_dim,num_transformer_blocks,num_heads,d_qk,d_v,transformer_dropout_rate,feedforward_h_dim,feedforward_dropout,*args, **kwargs):
        super().__init__(*args, **kwargs)
        self.embedding = nn.Embedding(vocab_size,embedding_dim,padding_idx=0)
        self.pos_embedding = SinusoidalPositionalEmbedding(embedding_dim)
        
        self.transformer_blocks = nn.ModuleList()

        for i in range(num_transformer_blocks):
            self.transformer_blocks.append(TransformerBlock(embedding_dim,num_heads,d_qk,d_v,transformer_dropout_rate,feedforward_h_dim,feedforward_dropout))
        
        self.finalnorm = nn.RMSNorm(embedding_dim)
        self.classificator = nn.Linear(embedding_dim,vocab_size)



    def forward(self,x):
        x = self.embedding(x)
        x = self.pos_embedding(x)

        for layer in self.transformer_blocks:
            x = layer(x,is_causal=True)
        
        x= self.finalnorm(x)
        x = self.classificator(x)

        return x

