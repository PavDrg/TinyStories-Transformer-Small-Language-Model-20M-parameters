from tokenizers import Tokenizer
import tokenizers
import torch
import pandas
from tqdm import tqdm
import os


def get_tokenizer(path_to_tokenizer,max_seq_len=None):

    tokenizer:Tokenizer = Tokenizer.from_file(path_to_tokenizer)
    if max_seq_len is not None:
        tokenizer.enable_truncation(max_seq_len)
        tokenizer.enable_padding(pad_id=0,pad_token="[PAD]",length=max_seq_len)

    return tokenizer



class TinyStoriesDataset(torch.utils.data.Dataset):
    def __init__(self,dataset:pandas.DataFrame,tokenizer:tokenizers.Tokenizer,cache_path=None):
        super().__init__()
        if cache_path is not None and os.path.exists(cache_path):
            self.dataset = torch.load(cache_path)
        else:
            self.dataset = []
        
            for i in tqdm(dataset.iloc,"Sequences",unit="sequence"):
                tokens = tokenizer.encode(f"[START] {i.text} [END]").ids
                self.dataset.append(tokens)
        
            self.dataset = torch.tensor(data=self.dataset,dtype=torch.int16)

            if cache_path is not None:
                torch.save(self.dataset,cache_path)

    
    def __getitem__(self, index):
        tokens_tensor = self.dataset[index].long()
        return(tokens_tensor[:-1],tokens_tensor[1:])


    def __len__(self):
        return self.dataset.shape[0]


def get_dataset_loader(dataset_df:pandas.DataFrame,tokenizer:Tokenizer,dataset_cache_path=None,batch_size=64,shuffle=True,num_workers=4, prefetch_factor=2):
    dataset = TinyStoriesDataset(dataset_df,tokenizer,dataset_cache_path)

    dataset_loader = torch.utils.data.DataLoader(dataset,batch_size,shuffle,num_workers=num_workers,prefetch_factor=prefetch_factor,drop_last=True)

    return dataset_loader
