from dataclasses import dataclass
import math
import torch
import torch.nn as nn
from torch.nn import functional as F

@dataclass
class GPTConfig:
    block_size: int = 1024 # max sequence length
    vocab_size: int = 50257 # 50,000 BPE merges, 256 byte tokens, 1 special token '<|endoftext|>'
    n_layer: int = 12 # number of layers
    n_head: int = 12  # number of attention heads
    n_embd: int = 768 # embedding/hidden dimension

class CausalSelfAttention(nn.Module):

    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        # key, query, value projections
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        # regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd

        T = config.block_size
        # This is actually the mask for casual self-attention, but we follow the OpenAI/HF naming.
        # 
        # The tensor is lower triangular matrix of 1s, then extended to 4D for broadcasting. 
        # tensor([[1., 0., 0.],  (T = 3)
        #         [1., 1., 0.],
        #         [1., 1., 1.]])
        self.register_buffer("bias", torch.tril(torch.ones(T, T)).view(1, 1, T, T))


    
    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimension (n_embd)

        qkv = self.c_attn(x) # (B, T, 3 * C)
        q, k, v = qkv.split(C, dim=2) # (B, T, C)

        # nh is "number of heads", hs is "head size", and C (number of channels) = nh * hs
        # e.g. in GPT-2 (124M), n_head=12, hs=64, so nh*hs=C=768 channels in the Transformer
        nh = self.n_head
        hs = C // nh


        k = k.view(B, T, nh, hs).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, nh, hs).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, nh, hs).transpose(1, 2) # (B, nh, T, hs)

        att = q @ k.transpose(-2, -1) * (1.0 / math.sqrt(hs)) # (B, nh, T, T)
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf')) # type: ignore
        att = F.softmax(att, dim=-1)
        y = att @ v # (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-arrange all head outputs of a single sequence together

        # output projection
        y = self.c_proj(y)
        return y

class MLP(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
    
    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

class Block(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)
        pass
    
    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

    @classmethod
    def from_pretrained(cls, model_type: str):
        """Loads pretrained GPT-2 model from HuggingFace."""
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        # vacabulary size and sequence length are same for GPT-2 models
        config_args['vocab_size'] = 50257
        config_args['block_size'] = 1024

        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        # keys of all tensors but ignore CausalSelfAttention's mask (register_buffer("bias", ...))
        sd_keys = [k for k in sd.keys() if not k.endswith('.attn.bias')] 

        # init a huggingface GPT-2 model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()
        sd_keys_hf = sd_hf.keys()
        # similarly, ignore mask from the huggingface model
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')]
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')]
        # The huggingface model uses Tensorflow and some of weights are transposed from PyTorch format.
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']

        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys:
            # The openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
            # thus we need to transpose their weights
            if any(k.endswith(w) for w in transposed):
                assert sd_hf[k].shape[::-1] == sd[k].shape, f"mismatched shape for {k}: {sd_hf[k].shape} vs {sd[k].shape}"
                with torch.no_grad():  # disable gradient tracking for copy operation
                    sd[k].copy_(sd_hf[k].t())
            else:
                assert sd_hf[k].shape == sd[k].shape, f"mismatched shape for {k}: {sd_hf[k].shape} vs {sd[k].shape}"
                with torch.no_grad():  # disable gradient tracking for copy operation
                    sd[k].copy_(sd_hf[k])
        return model

    def forward(self, idx):
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is {self.config.block_size}."

        # construct tokend and positional embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device) # shape (T)
        pos_emb = self.transformer['wpe'](pos) # position embeddings of shape (T, n_embd)
        token_emb = self.transformer['wte'](idx) # token embeddings of shape (B, T, n_embd)
        x = pos_emb + token_emb

        # forward to blocks of transformer
        for block in self.transformer['h']:  # type: ignore
            x = block(x)

        # forward to the final layernorm and classifier
        x = self.transformer['ln_f'](x)
        logits = self.lm_head(x) # (B, T, vocab_size)

        return logits

def generate(model, x, max_length):
    """
    Generate tokens from the model given a starting sequence x.
    Args:
        model: The GPT model.
        x: Input tensor of shape (B, T) containing the starting token indices.
        max_length: The maximum length of the generated sequence.
    Returns:
        The generated sequence tensor of shape (B, max_length).
    """
    while x.size(1) < max_length:
        with torch.no_grad():
            logits = model(x) # (B, T, vocab_size)
        
        # take the logits at the last time step
        logits = logits[:, -1, :] # (B, vocab_size)
        # get probabilities from softmax
        probs = F.softmax(logits, dim=-1) # (B, vocab_size)
        topk_probs, topk_indices  = torch.topk(probs, 50, dim=-1) # (B, 50)
        # select a token from the top-k probabilities (aka. sample from the distribution)
        ix = torch.multinomial(topk_probs, num_samples=1) # (B, 1)
        # gather the corresponding token indices 
        xcol = torch.gather(topk_indices, 1, ix) # (B, 1)
        # append to the sequence
        x = torch.cat((x, xcol), dim=1) # append to the sequence 
    return x

def sample(model, num_samples, max_length):
    """Sample generated sequences from the model and print them."""
    import tiktoken
    enc = tiktoken.get_encoding("gpt2")
    tokens = enc.encode("Hello, I'm a language model,")
    tokens = torch.tensor(tokens, dtype=torch.long) # (8,)
    tokens = tokens.unsqueeze(0).repeat(num_samples, 1) # (1, 8)
    x = tokens.to('cuda')

    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    x = generate(model, x, max_length=max_length) # (B, max_length)

    for i in range(num_samples):
        tokens = x[i, :max_length].tolist()
        decoded = enc.decode(tokens)
        print(">", decoded)



if __name__ == "__main__":
    model = GPT.from_pretrained('gpt2')
    model.eval()
    model.to('cuda' if torch.cuda.is_available() else 'cpu')

    print("Model loaded successfully. Did not crash yay!") 
    sample(model, num_samples=5, max_length=30)