import torch
from torch.utils.data import Dataset

class BilingualDataset(Dataset):
    
    def __init__(self, ds, tokenizer_src, tokenizer_tgt, src_lang, tgt_lang, seq_len) -> None:
        
        super().__init__()  
        self.ds= ds
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.seq_len = seq_len
        
        self.sos_token = torch.tensor([tokenizer_tgt.token_to_id('[SOS]')], dtype=torch.int64)
        self.eos_token = torch.tensor([tokenizer_tgt.token_to_id('[EOS]')], dtype=torch.int64)
        self.pad_token = torch.tensor([tokenizer_tgt.token_to_id('[PAD]')], dtype=torch.int64)
        
    
    def __len__(self):
        #gets len from huggingface dataset
        return len(self.ds)
    
    def __getitem__(self, index):
        src_target_pair = self.ds[index]
        src_text = src_target_pair['translation'][self.src_lang]
        tgt_text = src_target_pair['translation'][self.tgt_lang]
        #We convert each text into tokens and then into input id's,
        #Tokenizer fist splits sentence into single words and then map each word into its corresponding number in the vocabulary in one path only.
        
        enc_input_tokens = self.tokenizer_src.encode(src_text).ids # encode().ids give us the numbers corresponding to each word as an array
        dec_input_tokens = self.tokenizer_tgt.encode(tgt_text).ids
        
        # Now we also need to padd teh sentece to reach teh seq_len, the model always works iwth a fixed
        # seq_len so we use the padding tokens to fill the sentence until it reaches the seq_len
        enc_num_padding_tokens = self.seq_len - len(enc_input_tokens) - 2 # The -2 comes from adding the SOS and EOS tokens to the encoder side
        dec_num_padding_tokens = self.seq_len - len(dec_input_tokens) - 1 # Only -1 because when training, decoder side only has SOS token and then in the label we add the EOS token
        
        if enc_num_padding_tokens < 0 or dec_num_padding_tokens < 0:
            raise ValueError('Sentence is too long')
        
        #Now we build the 2 tensors, for the encoder and decoder input and also for the label.
        #One sentence will be sent to input to encoder and another to the deocder and another one is what we expect from the output of the decoder.
        
        # Add SOS and EOS to the source text
        encoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(enc_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * enc_num_padding_tokens, dtype=torch.int64)
            ]
        )
        
        #Add EOS to the decoder input
        decoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64)
            ]
        )
        
        #Add EOS to the label (output expected from the decoder, aka target)
        label = torch.cat(
                    [
                        torch.tensor(dec_input_tokens, dtype=torch.int64),
                        self.eos_token,
                        torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64)
                    ]
                )
        
        assert encoder_input.size(0) == self.seq_len
        assert decoder_input.size(0) == self.seq_len
        assert label.size(0) == self.seq_len
        
        
        #Encoder mask: Increasing the size of the eoncder input with padding which we dont want to use uin the self attentiom mechanism, so we use a mask to remove them
        #Decoder mask: Causal mask, each word can only look at previous words and each word can only look at non-padding works, we only want real words to participate. Only words that come before it
        return {
            "encoder_input": encoder_input, # (seq_len)
            "decoder_input": decoder_input, # (seq_len)
            "encoder_mask": (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(), # (1, 1, Seq_len)
            "decoder_mask": (decoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int() & causal_mask(decoder_input.size(0)), # (1, seq_len) & (1, seq_len, seq_len)
            "label": label, # (seq_len)
            "src_text": src_text,
            "tgt_text": tgt_text
        }
        
def causal_mask(size):
    # We want all the values in the upper diagonal of the matrix to be masked since we only want self attention with the words before them in the sentence
    mask = torch.triu(torch.ones(1, size, size), diagonal= 1).type(torch.int)
    return mask == 0