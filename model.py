import torch
import torch.nn as nn
import math

class InputEmbeddings(nn.Module):
    
    def __init__(self, d_model: int, vocab_size: int,):
         "d_model: dimension of the vector"
         "vocab_size: how many words in the vacabulary"
         super().__init__()
         self.d_model = d_model
         self.vocab_size = vocab_size
         self.embedding = nn.Embedding(vocab_size, d_model) #This is a pytorch layer that given a number provides the same vector every time. 
                                                            #Mapping between numbers and vector of size 512 (d_model)
                                                            #This is a vector that is learnt by the model
    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model) #Look atr paper
        

class PositionalEncoding(nn.Module):
    "Tells the model that this particular word ocuppies this position in the sentence"
    def __init__(self, d_model: int, seq_len: int, dropout: float):
        "d_model: size of the vector that the positional enconding should be"
        "seq_len: max length of sentence, we need to create one vector per postion"
        "dropout: makes model less overfit"
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)
        
        #Create a matrix of shape (seq_len, d_model)
        pe = torch.zeros(seq_len, d_model)
        
        #Create a vector of shape (seq_len, 1)
        "This formula comes from Attention is all you need paper with modifications in the div_term for numerical stability"
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0)/ d_model))
        
        #Apply the sin to even positons
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0) # (1, seq_len, d_model) Add one dimention to be able to apply it to the batch of sentences
        
        self.register_buffer('pe', pe) #When you want to keep a tensor not as a learnable parameter but you want it saved when saving the file of the model
    
    def forward(self, x):
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False) #Tell the model that this encodings don't have to be learned since they are fixed 
        return self.dropout(x)
    
class LayerNormalization(nn.Module):
    
    def __init__(self, eps: float = 10**-6):
        super().__init__()
        self.eps = eps #This exists so that the denominator of the normalization is never 0 and numerical stability
        
        "This 2 parameters are there in order to modify the distributions in case needed by the model"
        self.alpha = nn.Parameter(torch.ones(1)) #Multiplied nn.Parameter makes it learnable
        self.bias = nn.Parameter(torch.zeros(1)) #Added
        
    def forward(self, x):
        mean = x.mean(dim = -1, keepdim= True) #Everything after batchsize, Normally mean cancels dimensions to which its applied but we wsnt to keep it
        std = x.std(dime = -1, keepdim= True)
        return self.alpha * (x - mean) / (std - self.eps) + self.bias 
    
    
class FeedForwardBlock(nn.Module):
    
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.linear_1 = nn.Linear(d_model,d_ff) #W1 and B1 from paper
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model) #W2 and B2 (No Bias included because it's active by default)
        
    def forward(self, x):
        #(Batch, seq_len, d_model) --> (Batch, seq_len, d_ff) --> (Batch, seq_len, d_model)
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))
    
class MultiHeadAttentionBlock(nn.Module):
    
    def __init__(self, d_model: int, h: int, dropout: float):
        "We're going to divide or models into h heads so d_model ahs to be divisible by h to divide equally"
        super().__init__()
        self.d_model = d_model
        self.h = h
        assert d_model % h == 0, "d_model is not divisible by h" #d_k = d_model / h
        
        
        #We want output to be (seq_len, d_model):        
        #(seq_len, d_model) x (d_model, d_model) --> (seq, d_model)
        
        self.d_k = d_model // h

        self.w_q = nn.Linear(d_model, d_model) #Wq 
        self.w_k= nn.Linear(d_model, d_model) #Wk 
        self.w_v= nn.Linear(d_model, d_model) #Wv
        
        self.w_o = nn.Linear(d_model, d_model) #Wo
        self.dropout = nn.Dropout(dropout)
    
    @staticmethod #means you can call the function without having an instance of the class, you can do MultiHeadAttentionBlock.attention()
    def attention(query, key, value, mask, dropout: nn.Dropout):
        d_k = query.shape[-1]
        #(Batch, h , seq_len, d_k) --> (Batch, h ,seq_len, seq_len)
        attention_scores= (query @ key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            attention_scores.masked_fill_(mask == 0, -1e9)  #This is for example if we don't want words to be included in the attention
                                                            #For example padding words that are just filler words to get to the min seq_len
        attention_scores = attention_scores.softmax(div = -1) #(Batch, h, seq_len, seq_len)
        if dropout is not None:
            attention_scores = dropout(attention_scores)
        
        return (attention_scores @ value), attention_scores
    
    def forward(self, q, k, v, mask):
        "Mask: If we want words not to interact with other words, we can mask them"
        "Before doing the softmax(QK^T/sqrt(d_k)), we basically have a matrix of shape (seq_len, seq_len) or word by word"
        "So if we want words not to interact, we can mask this matrix by replacing the Attention values for very low values"
        "so that the softmax makes them 0"     
        
        query = self.w_q(q) #(Batch, seq_len, d_model) --> (Batch, seq_len, d_model)
        key = self.w_k(k) #(Batch, seq_len, d_model) --> (Batch, seq_len, d_model)
        value = self.w_v(v) #(Batch, seq_len, d_model) --> (Batch, seq_len, d_model)
        
        # (Batch, seq_len, d_model) --> (Batch, seq_len, h , d_k) --> (Batch, h, seq_len, d_k)
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2) 
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2) 
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2) 
        #We use view() to keep batch dimension, we don't want to split the sentence, we want to split the embedings into h parts
        #We want to transpose so that h is the 2nd dimension and not the 3rd, this way each head will see all the sentence (seq_len, d_k)
        #Each head watch's (seq_len, d_k), so they will see the whole sentence but only a smaller part of the embedding
        
        x, self.attention_scores = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout)
        
        #(Batch, h, seq_len, d_k) --> (Batch, seq_len, h, d_k) --> (Batch, seq_len, d_model)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k) # .contiguous() is needed to save the matrix momentarily to do the operation inplace
        #(Batch, seq_len, d_model) --> (Batch, seq_len, d_model)
        return self.w_o(x)


class ResidualConnection(nn.Module):
    
    def __init__(self, dropout: float):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization()
        
    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x))) #In the paper they apply first the sublayer and then the norm
    
    

class EncoderBlock(nn.Module):
    
    def __init__(self, self_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float):
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(2)])
        
    def forward(self, x, src_mask):
        "We need the src_mask to mask the padding words" #Look into it
        
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, src_mask)) #Each word of the sentence is interacting with other words of the same sentence, hence the name self-attention
        x = self.feed_forward_block[1](x, self.feed_forward_block) #In the case of the Decoder, we have cross-attention where the queries from the decoder are watching the keys and values coming from the encoder.  
        return x
    
class Encoder(nn.Module):
    
    def __init__(self, layers: nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()
        
    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x,mask)
        return self.norm(x)
    


class DecoderBlock(nn.Module):
    
    def __init__(self,self_attention_block: MultiHeadAttentionBlock, cross_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float):
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block= cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(3)]) 
        
    def forward(self, x, encoder_output, src_mask, tgt_mask):
        #In this case we're dealing with translation so one is the source language(ENG) and another the target language(ESP)
        x = self.residual_connections[0](x, lambda x:self.self_attention_block(x, x, x, tgt_mask))
        x = self.residual_connections[1](x, lambda x:self.cross_attention_block(x, encoder_output, encoder_output, src_mask))
        x = self.residual_connections[2](x, self.feed_forward_block)
        return x
    
class Decoder(nn.Module):
    
    def __init__(self, layers: nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()
        
    def forward(self, x, encoder_output, src_mask, tgt_mask):
        for layer in self.layers:
            x= layer(x, encoder_output, src_mask, tgt_mask)
        return self.norm(x)
    
    
class ProjectionLayer(nn.Module):
    "This Layer basically translates the embeddings back into the words from our 'dictionary'"
    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)
        
    def forward(self, x):
        #(Batch, seq_len, d_model) --> (Batch, seq_len, vocab_size)
        return torch.log_softmax(self.proj(x), dim = -1)
    
class Transformer(nn.Module):
    
    def __init__(self, encoder: Encoder, decoder: Decoder, src_embed: InputEmbeddings, tgt_embed: InputEmbeddings, src_pos: PositionalEncoding, tgt_pos: PositionalEncoding,  projection_layer: ProjectionLayer) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.projection_layer = projection_layer
        
    
    #We define 3 methods, one to encode, one to decode and one to project. We want 3 and not only one to use the output of the encoder for inference + it's good for visualizing attention
    def encode(self, src, src_mask):
        src = self.src_embed(src)
        src = self.src_pos(src)
        return self.encoder(src, src_mask)
    
    def decode(self, encoder_output, src_mask, tgt, tgt_mask):
        tgt = self.tgt_embed(tgt)
        tgt = self.tgt_pos(tgt)
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)
    
    def project(self, x):
        return self.projection_layer(x)
    
    
def build_transformer(src_vocab_size: int, tgt_vocab_size: int, src_seq_len: int, tgt_seq_len: int, d_model: int = 512, N: int = 6, h: int = 8, dropout: float = 0.1, d_ff: int = 2048) -> Transformer:
    # Create the embedding layers
    src_embed = InputEmbeddings(d_model, src_vocab_size)
    tgt_embed = InputEmbeddings(d_model, tgt_vocab_size)
    
    # Create the positional encoding layers
    src_pos = PositionalEncoding(d_model, src_seq_len, dropout)
    tgt_pos = PositionalEncoding(d_model, tgt_seq_len, dropout)
    
    # Create the Encoder Blocks
    encoder_blocks = []
    for _ in range(N):
        encoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(encoder_self_attention_block, feed_forward_block, dropout)
        encoder_blocks.append(encoder_block)
        
    # Create the Decoder Blocks
    decoder_blocks = []
    for _ in range(N):
        decoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        decoder_cross_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        decoder_block = DecoderBlock(decoder_self_attention_block, decoder_cross_attention_block, feed_forward_block, dropout)
        decoder_blocks.append(decoder_block)
        
    # Create the Encoder and the Decoder
    encoder = Encoder(nn.ModuleList(encoder_blocks))
    decoder = Decoder(nn.ModuleList(decoder_blocks))
    
    # Create the projection layer
    projection_layer = ProjectionLayer(d_model, tgt_vocab_size)
    
    # Create the Transformer
    transformer = Transformer(encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos, projection_layer)
    
    # Initialize the parameters
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
            
    return transformer
    
    
    
    
    
