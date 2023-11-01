"""
taken from: https://github.com/karpathy/minGPT/
GPT model:
- the initial stem consists of a combination of token encoding and a positional encoding
- the meat of it is a uniform sequence of Transformer blocks
    - each Transformer is a sequential combination of a 1-hidden-layer MLP block and a self-attention block
    - all blocks feed into a central residual pathway similar to resnets
- the final decoder is a linear projection into a vanilla Softmax classifier
"""

import math
import copy

import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import top_k_top_p_filtering
from cmota.modules.transformer.pos_embedding import AxialPositionalEmbedding

class GPTConfig:
    """ base GPT config, params common to all GPT versions """
    embd_pdrop = 0.1
    resid_pdrop = 0.1
    attn_pdrop = 0.1

    def __init__(self, vocab_size, block_size, cond_vocab_size = 0, cond_seq_len = 0, image_seq_len=1024, **kwargs):
        self.cond_vocab_size = cond_vocab_size
        self.cond_seq_len = cond_seq_len
        self.image_seq_len = image_seq_len
        self.vocab_size = vocab_size
        self.block_size = block_size
        for k,v in kwargs.items():
            setattr(self, k, v)


class GPT1Config(GPTConfig):
    """ GPT-1 like network roughly 125M params """
    n_layer = 12
    n_head = 12
    n_embd = 768


class BertLayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(BertLayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias

def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT"s gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

class MemoryInitializer(nn.Module):
    def __init__(self, config=None):
        super(MemoryInitializer, self).__init__()
        self.n_memory_cells = config.n_memory_cells
        self.init_memory_bias = nn.Parameter(torch.randn(1, self.n_memory_cells, 1))  # (1, M, D)
        self.init_memory_fc = nn.Sequential(
                            nn.Linear(config.n_embd, config.n_embd),
                            BertLayerNorm(config.n_embd),
                            nn.Dropout(0.1)
                        )

    def forward(self, input_states, attention_mask):
        """ initialize the model with the first input states
            input_states: (N, L, D)
            attention_mask: (N, L)
        """

        t = input_states.size()[1]
        pooled_input_states = torch.sum(input_states * attention_mask[:,:t].unsqueeze(-1), dim=1)  # (B, T, D) --> (B, D)
        pooled_input_states = pooled_input_states / attention_mask.sum(1, keepdim=True)            # (B, D) no zero here
        pooled_input_states = pooled_input_states.unsqueeze(1).repeat(1, self.n_memory_cells, 1)   # (B, M, D)
        pooled_input_states = pooled_input_states + self.init_memory_bias  # (B, M, D)
        init_memory = self.init_memory_fc(pooled_input_states)  # (B, M, D)
        return init_memory

class BertSelfAttention(nn.Module):
    def __init__(self, config):
        super(BertSelfAttention, self).__init__()

        self.num_attention_heads = config.n_head
        self.attention_head_size = int(config.n_embd / config.n_head)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        hidden_size = config.n_embd

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size,   self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(0.1) #config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)  # (N, L, nh, dh)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)  # (N, nh, L, dh)

    def forward(self, query_states, key_states, value_states, attention_mask=None):
        """
        Args:
            query_states: (N, Lq, D)
            key_states: (N, L, D)
            value_states: (N, L, D)
            attention_mask: (N, Lq, L)

        Returns:

        """
        # only need to mask the dimension where the softmax (last dim) is applied, as another dim (second last)
        # will be ignored in future computation anyway
        
        # if attention_mask != None:
        #     attention_mask = (1 - attention_mask.unsqueeze(1)) * -10000.0  # (N, 1, Lq, L)

        mixed_query_layer = self.query(query_states)
        mixed_key_layer = self.key(key_states)
        mixed_value_layer = self.value(value_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)                          # (B, nh, L_m, dh)
        key_layer   = self.transpose_for_scores(mixed_key_layer)                            # (B, nh, L,   dh)
        value_layer = self.transpose_for_scores(mixed_value_layer)                          # (B, nh, L,   dh)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))           # (B, nh, L_m, L)
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        
        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        t = attention_scores.shape[-1]
        attention_mask = attention_mask.unsqueeze(1)                                        # (B, 1, L_m, L)
        attention_scores = attention_scores.masked_fill(attention_mask[:,:,:,:t] == 0, float('-inf'))

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer


class MemoryUpdater(nn.Module):
    def __init__(self, config):
        super(MemoryUpdater, self).__init__()
        self.memory_update_attention = BertSelfAttention(config)

        hidden_size = config.n_embd

        self.mc = nn.Linear(hidden_size, hidden_size, bias=False)
        self.sc = nn.Linear(hidden_size, hidden_size, bias=True)

        self.sz = nn.Linear(hidden_size, hidden_size, bias=True)
        self.mz = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, prev_m, input_states, attention_mask=None):
        """ This module should have access to all the text at this step,
        since its state will not be used for generation at current step
        Args:
            prev_m: (N, M, D), M is memory size
            input_states: (N, L, D)
            attention_mask: (N, L)
        Returns:

        """
        # memory attended inputs
        n_memory_cells = prev_m.shape[1]
        update_mask = attention_mask.unsqueeze(1).repeat(1, n_memory_cells, 1)  # (N, M, L)
        s_t = self.memory_update_attention(prev_m, input_states, input_states, update_mask)  # (N, M, D),

        c_t = torch.tanh(self.mc(prev_m) + self.sc(s_t))  # (N, M, D)
        z_t = torch.sigmoid(self.mz(prev_m) + self.sz(s_t))  # (N, M, D)

        updated_memory = (1 - z_t) * c_t + z_t * prev_m  # (N, M, D)
        return updated_memory


class MemoryUpdaterGRU(nn.Module):
    def __init__(self, config):
        super(MemoryUpdaterGRU, self).__init__()

        hidden_size = config.n_embd

        self.xr = nn.Linear(hidden_size, hidden_size, bias=True)
        self.hr = nn.Linear(hidden_size, hidden_size, bias=True)

        self.xz = nn.Linear(hidden_size, hidden_size, bias=True)
        self.hz = nn.Linear(hidden_size, hidden_size, bias=True)

        self.xx = nn.Linear(hidden_size, hidden_size, bias=True)
        self.hh = nn.Linear(hidden_size, hidden_size, bias=True)


    def forward(self, prev_m, input_states, attention_mask=None):
        """ This module should have access to all the text at this step,
        since its state will not be used for generation at current step
        Args:
            prev_m: (N, M, D), M is memory size
            input_states: (N, L, D)
            attention_mask: (N, L)
        Returns:
        """
        
        r_t = torch.sigmoid(self.xr(input_states) + self.hr(prev_m))     # [B, T, d] 
        z_t = torch.sigmoid(self.xz(input_states) + self.hz(prev_m))     # [B, T, d]
        h_  = torch.tanh(self.xx(input_states) + self.hh(prev_m) * r_t)  # [B, T, d]

        updated_memory = (1 - z_t) * h_ + z_t * prev_m                   # [B, T, d]

        # simple memory aggregation
        updated_memory = updated_memory.mean(dim=1, keepdim=True)

        return updated_memory


class ReCausalSelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        # key, query, value projections for all heads
        self.key   = nn.Linear(config.n_embd, config.n_embd)
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)

        # regularization
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)

        # output projection
        self.proj = nn.Linear(config.n_embd, config.n_embd)

        # Auto-regressive mask
        # causal mask to ensure that attention is only applied to the left in the input sequence
        #mask = torch.tril(torch.ones(config.block_size, config.block_size)) # [195, 195] or [387, 387]
        #self.register_buffer("mask", mask.view(1, 1, config.block_size, config.block_size))

        # For hybrid-mask all layers
        mask = torch.tril(torch.ones(config.block_size, config.block_size)) # [195, 195] or [387, 387]
        if hasattr(config, "n_unmasked"):
            mask[:config.n_unmasked, :config.n_unmasked] = 1
        self.register_buffer("mask", mask.view(1, 1, config.block_size, config.block_size))

        self.n_head = config.n_head

    def forward(self, x, gen_type='t2i'):
        B, T, C = x.size() # [B, 386, 512]

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)   # (B, nh, T, hs)    # [B, 8, 386, 64]
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)    # [B, 8, 386, 64]
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)    # [B, 8, 386, 64]

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.mask[:,:,:T,:T] == 0, float('-inf'))

        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        return y


class ReCausalSelfAttentionL2(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads
        self.key   = nn.Linear(config.n_embd, config.n_embd)
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)

        # regularization
        self.attn_drop  = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)

        # output projection
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        
        # Hybrid mask with bidirectional mask over condition and unidirectional mask over target
        # causal mask to ensure that attention is only applied to the left in the input sequence

        mask_t2i = torch.tril(torch.ones(config.block_size, config.block_size)) # [195, 195] or [387, 387]
        if hasattr(config, "n_unmasked"):
            mask_t2i[:config.n_unmasked_text, :config.n_unmasked_text] = 1
        self.register_buffer("mask_t2i", mask_t2i.view(1, 1, config.block_size, config.block_size))

        mask_i2t = torch.tril(torch.ones(config.block_size, config.block_size)) # [195, 195] or [387, 387]
        if hasattr(config, "n_unmasked"):
            mask_i2t[:config.n_unmasked_img, :config.n_unmasked_img] = 1
        self.register_buffer("mask_i2t", mask_i2t.view(1, 1, config.block_size, config.block_size))


        self.n_head = config.n_head

    def forward(self, x, prev_m=None, gen_type='t2i'):
        B, T, C = x.size() # [B, 386, 512]

        if prev_m == None:
            # calculate query, key, values for all heads in batch and move head forward to be the batch dim
            k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)       # (B, nh, T+1, hs)      # [B, 8, 387, 64] or [B, 8, 195, 64]
            q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)     # (B, nh, T, hs)        # [B, 8, 386, 64] or [B, 8, 194, 64]
            v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)     # (B, nh, T+1, hs)      # [B, 8, 387, 64] or [B, 8, 195, 64]
            
            # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, (T+1)) -> (B, nh, T, (T+1))
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            if gen_type == 't2i':
                att = att.masked_fill(self.mask_t2i[:,:,:T,:T] == 0, float('-inf'))
            else:
                att = att.masked_fill(self.mask_i2t[:,:,:T,:T] == 0, float('-inf'))

        else:
            concat_mh = torch.cat([prev_m, x], dim=1) # [(N,Mn,Di); (N,L,Di)] --> [N, M+L, Di]
            # calculate query, key, values for all heads in batch and move head forward to be the batch dim
            k = self.key(concat_mh).view(B, (T+1), self.n_head, C // self.n_head).transpose(1, 2)       # (B, nh, T+1, hs)      # [B, 8, 387, 64] or [B, 8, 195, 64]
            q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)                 # (B, nh, T, hs)        # [B, 8, 386, 64] or [B, 8, 194, 64]
            v = self.value(concat_mh).view(B, (T+1), self.n_head, C // self.n_head).transpose(1, 2)     # (B, nh, T+1, hs)      # [B, 8, 387, 64] or [B, 8, 195, 64]

            # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, (T+1)) -> (B, nh, T, (T+1))
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            if gen_type == 't2i':
                att = att.masked_fill(self.mask_t2i[:,:,:T,:T+1] == 0, float('-inf'))
            else:
                att = att.masked_fill(self.mask_i2t[:,:,:T,:T+1] == 0, float('-inf'))

        att = F.softmax(att, dim=-1)
        #att = self.attn_drop(att)
        y = att @ v # (B, nh, (T+1), T) x (B, nh, (T+1), hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        return y

    def make_shifted_mask(self, input_mask, max_v_len, max_t_len, memory_len=0):
        """
        Args:
            input_mask: (N, L) with `1` indicates valid bits, `0` indicates pad
            max_v_len: int, the first `max_v_len` is for video and its padding, the length
                of the rest of the bits is `max_t_len`. We have L = `max_v_len` + `max_t_len`.
                Note max_v_len may also include the memory len (M), thus max_v_len += M
            max_t_len: int
            memory_len: int, M
        Returns:

        >>> max_v_len = 2; max_t_len=3; input_mask = torch.randn(2, 5)
        >>> make_pad_shifted_mask(input_mask, max_v_len, max_t_len)[0]
        tensor([[1., 1., 0., 0., 0.],
                [1., 1., 0., 0., 0.],
                [1., 1., 1., 0., 0.],
                [1., 1., 1., 1., 0.],
                [1., 1., 1., 1., 1.]])
        """
        bsz, seq_len = input_mask.shape
        assert max_v_len + max_t_len + memory_len == seq_len, (max_v_len, max_t_len, memory_len, seq_len)
        shifted_mask = input_mask.new_zeros(bsz, max_v_len + max_t_len, seq_len)  # (N, L, M+L)
        shifted_mask[:, :, :memory_len + max_v_len] = 1
        shifted_mask[:, max_v_len:, memory_len + max_v_len:] = torch.tril(input_mask.new_ones(max_t_len, max_t_len), diagonal=0)
        return shifted_mask

class ReCausalSelfAttentionL3(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads
        self.key   = nn.Linear(config.n_embd, config.n_embd)
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)

        # regularization
        self.attn_drop  = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)

        # output projection
        self.proj = nn.Linear(config.n_embd, config.n_embd)

        self.fc_q_mem = nn.Linear(config.n_embd, config.n_embd)
        self.fc_k_mem = nn.Linear(config.n_embd, config.n_embd)
        self.fc_v_mem = nn.Linear(config.n_embd, config.n_embd)
        
        # Hybrid mask with bidirectional mask over condition and unidirectional mask over target
        # causal mask to ensure that attention is only applied to the left in the input sequence
        mask_t2i = torch.tril(torch.ones(config.block_size, config.block_size)) # [195, 195] or [387, 387]
        if hasattr(config, "n_unmasked"):
            mask_t2i[:config.n_unmasked_text, :config.n_unmasked_text] = 1
        self.register_buffer("mask_t2i", mask_t2i.view(1, 1, config.block_size, config.block_size))

        mask_i2t = torch.tril(torch.ones(config.block_size, config.block_size)) # [195, 195] or [387, 387]
        if hasattr(config, "n_unmasked"):
            mask_i2t[:config.n_unmasked_img, :config.n_unmasked_img] = 1
        self.register_buffer("mask_i2t", mask_i2t.view(1, 1, config.block_size, config.block_size))

        # For new memory
        tmp_memory_size = 2
        tmp_stacked_memory = torch.ones(config.block_size, tmp_memory_size)
        # Text to Image mask
        mask_t2i_new_memory = torch.tril(torch.ones(config.block_size, config.block_size)) # [195, 195] or [387, 387]
        mask_t2i_new_memory[:config.n_unmasked_text, :config.n_unmasked_text] = 1
        mask_t2i_new_memory = torch.cat([tmp_stacked_memory, mask_t2i_new_memory], dim=1)
        self.register_buffer("mask_t2i_new_memory", mask_t2i_new_memory.view(1, 1, config.block_size, config.block_size+2))
        # Image to Text mask
        mask_i2t_new_memory = torch.tril(torch.ones(config.block_size, config.block_size)) # [195, 195] or [387, 387]
        mask_i2t_new_memory[:config.n_unmasked_img, :config.n_unmasked_img] = 1
        mask_i2t_new_memory = torch.cat([tmp_stacked_memory, mask_i2t_new_memory], dim=1)
        self.register_buffer("mask_i2t_new_memory", mask_i2t_new_memory.view(1, 1, config.block_size, config.block_size+2))

        self.n_head = config.n_head

    def forward(self, x, prev_m=None, cached_memory=None, gen_type='t2i'):
        B, T, C = x.size() # [B, 386, 512]

        if prev_m == None:
            # calculate query, key, values for all heads in batch and move head forward to be the batch dim
            k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)       # (B, nh, T+1, hs)      # [B, 8, 387, 64] or [B, 8, 195, 64]
            q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)     # (B, nh, T, hs)        # [B, 8, 386, 64] or [B, 8, 194, 64]
            v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)     # (B, nh, T+1, hs)      # [B, 8, 387, 64] or [B, 8, 195, 64]
            
            # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, (T+1)) -> (B, nh, T, (T+1))
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            if gen_type == 't2i':
                att = att.masked_fill(self.mask_t2i[:,:,:T,:T] == 0, float('-inf'))
            else:
                att = att.masked_fill(self.mask_i2t[:,:,:T,:T] == 0, float('-inf'))

        elif cached_memory != None:

            # check wheter or not cached_memory[0][0] list?
            if cached_memory[0] != None:
                ## DO SOMETHING
                
                # last_layer_memory
                cached_memory = [x[-1] for x in cached_memory]
                cached_memory = torch.cat(cached_memory, dim=1) # shape: (n, d) n --> 2, 3, 4

                cache_B, cache_T, cache_C = cached_memory.size() 
                
                q_mem = self.fc_q_mem(prev_m).view(B, 1, self.n_head, C // self.n_head).transpose(1,2)
                k_mem = self.fc_k_mem(cached_memory).view(cache_B, cache_T, self.n_head, cache_C // self.n_head).transpose(1,2)
                v_mem = self.fc_k_mem(cached_memory).view(cache_B, cache_T, self.n_head, cache_C // self.n_head).transpose(1,2)

                att = (q_mem @ k_mem.transpose(-2,-1)) * (1.0 / math.sqrt(k_mem.size(-1)))
                att = F.softmax(att, dim=-1)
                cached_memory = att @ v_mem
                cached_memory = cached_memory.transpose(1,2).contiguous().view(cache_B, 1, cache_C)

            else:
                cached_memory = cached_memory[-1]

            prev_m = torch.cat([cached_memory, prev_m], dim=1)
            concat_mh = torch.cat([prev_m, x], dim=1) # [(N,Mn,Di); (N,L,Di)] --> [N, M+L, Di]
            # calculate query, key, values for all heads in batch and move head forward to be the batch dim
            k = self.key(concat_mh).view(B, (T+2), self.n_head, C // self.n_head).transpose(1, 2)       # (B, nh, T+1, hs)      # [B, 8, 387, 64] or [B, 8, 195, 64]
            q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)                 # (B, nh, T, hs)        # [B, 8, 386, 64] or [B, 8, 194, 64]
            v = self.value(concat_mh).view(B, (T+2), self.n_head, C // self.n_head).transpose(1, 2)     # (B, nh, T+1, hs)      # [B, 8, 387, 64] or [B, 8, 195, 64]

            # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, (T+1)) -> (B, nh, T, (T+1))
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            if gen_type == 't2i':
                att = att.masked_fill(self.mask_t2i_new_memory[:,:,:T,:T+2] == 0, float('-inf'))
            else:
                att = att.masked_fill(self.mask_i2t_new_memory[:,:,:T,:T+2] == 0, float('-inf'))

        else:
            concat_mh = torch.cat([prev_m, x], dim=1) # [(N,Mn,Di); (N,L,Di)] --> [N, M+L, Di]
            # calculate query, key, values for all heads in batch and move head forward to be the batch dim
            k = self.key(concat_mh).view(B, (T+1), self.n_head, C // self.n_head).transpose(1, 2)       # (B, nh, T+1, hs)      # [B, 8, 387, 64] or [B, 8, 195, 64]
            q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)                 # (B, nh, T, hs)        # [B, 8, 386, 64] or [B, 8, 194, 64]
            v = self.value(concat_mh).view(B, (T+1), self.n_head, C // self.n_head).transpose(1, 2)     # (B, nh, T+1, hs)      # [B, 8, 387, 64] or [B, 8, 195, 64]

            # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, (T+1)) -> (B, nh, T, (T+1))
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            if gen_type == 't2i':
                att = att.masked_fill(self.mask_t2i[:,:,:T,:T+1] == 0, float('-inf'))
            else:
                att = att.masked_fill(self.mask_i2t[:,:,:T,:T+1] == 0, float('-inf'))

        att = F.softmax(att, dim=-1)
        y = att @ v # (B, nh, (T+1), T) x (B, nh, (T+1), hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        return y

    def make_shifted_mask(self, input_mask, max_v_len, max_t_len, memory_len=0):
        """
        Args:
            input_mask: (N, L) with `1` indicates valid bits, `0` indicates pad
            max_v_len: int, the first `max_v_len` is for video and its padding, the length
                of the rest of the bits is `max_t_len`. We have L = `max_v_len` + `max_t_len`.
                Note max_v_len may also include the memory len (M), thus max_v_len += M
            max_t_len: int
            memory_len: int, M
        Returns:

        >>> max_v_len = 2; max_t_len=3; input_mask = torch.randn(2, 5)
        >>> make_pad_shifted_mask(input_mask, max_v_len, max_t_len)[0]
        tensor([[1., 1., 0., 0., 0.],
                [1., 1., 0., 0., 0.],
                [1., 1., 1., 0., 0.],
                [1., 1., 1., 1., 0.],
                [1., 1., 1., 1., 1.]])
        """
        bsz, seq_len = input_mask.shape
        assert max_v_len + max_t_len + memory_len == seq_len, (max_v_len, max_t_len, memory_len, seq_len)
        shifted_mask = input_mask.new_zeros(bsz, max_v_len + max_t_len, seq_len)  # (N, L, M+L)
        shifted_mask[:, :, :memory_len + max_v_len] = 1
        shifted_mask[:, max_v_len:, memory_len + max_v_len:] = torch.tril(input_mask.new_ones(max_t_len, max_t_len), diagonal=0)
        return shifted_mask


class ReBlockWOMeM(nn.Module):
    """ Transformer block """
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.ln3 = nn.LayerNorm(config.n_embd)
        self.ln4 = nn.LayerNorm(config.n_embd)

        self.attn1 = ReCausalSelfAttention(config)
        self.attn2 = ReCausalSelfAttentionL2(config)
        
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),  # nice
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.resid_pdrop),
        )

        self.mlp2 = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),  # nice
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.resid_pdrop),
        )

    def forward(self, x, prev_m=None, gen_type='t2i'):
        # x.shape: [B, 386, 512]
        attn = self.attn1(self.ln1(x), gen_type=gen_type)

        x = x + attn
        x = x + self.mlp(self.ln2(x))

        prev_m = None
        updated_m = None
        
        attn2 = self.attn2(self.ln3(x), prev_m=prev_m, gen_type=gen_type)

        x = x + attn2
        x = x + self.mlp2(self.ln4(x))

        return updated_m, x


class ReBlockWMeMs(nn.Module):
    """ Transformer block """
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.ln3 = nn.LayerNorm(config.n_embd)
        self.ln4 = nn.LayerNorm(config.n_embd)

        self.attn1 = ReCausalSelfAttention(config)
        self.attn2 = ReCausalSelfAttentionL3(config)
        
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),  # nice
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.resid_pdrop),
        )

        self.mlp2 = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),  # nice
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.resid_pdrop),
        )
        
        # Memory network
        self.memory_initializer  = MemoryInitializer(config)
        self.memory_updater      = MemoryUpdater(config)
    
        self.mask = None
        #self.second_order_mask = None

    def make_text_only_mask(self, input_mask, max_t_len):
        text_only_mask = copy.deepcopy(input_mask)
        text_only_mask[:, max_t_len:] = 0
        return text_only_mask

    def forward(self, x, prev_m=None, gen_type='t2i', cached_memory=None):
        # x.shape: [B, 386, 512]
        attn = self.attn1(self.ln1(x), gen_type=gen_type)

        x = x + attn
        x = x + self.mlp(self.ln2(x))

        if prev_m is None:
            # only allow the initializer to see text part, not image token part as it will be used for generation at current step
            tmp_batch_size = x.size()[0]

            if gen_type == 't2i':
                tmp_seq = self.config.text_seq_length
            else: # gen_type == 'i2t'
                tmp_seq = self.config.img_tok_seq_length

            self.mask = torch.ones(tmp_batch_size, self.config.text_seq_length+self.config.img_tok_seq_length+2) # Batchsize is 8 # 338 = 80 + 256 + 2
            #self.mask = self.mask.to(x.device)
            self.mask = self.make_text_only_mask(self.mask, tmp_seq+1).to(x.device)

            prev_m = self.memory_initializer(x, self.mask)  # (B, M, Di)

        updated_m = self.memory_updater(prev_m, x, attention_mask=self.mask)

        attn2 = self.attn2(self.ln3(x), prev_m=prev_m, cached_memory=cached_memory, gen_type=gen_type)

        x = x + attn2
        x = x + self.mlp2(self.ln4(x))

        return updated_m, x


class RecurrentGPT(nn.Module):
    """  the full GPT language model, with a context size of block_size """
    def __init__(self, vocab_size, block_size, n_layer=12, n_head=8, n_embd=256, batch_size=8, hybrid_mask=False,
                 text_seq_length=80, img_tok_seq_length=256, n_memory_cells=1, memory=True,
                 embd_pdrop=0., resid_pdrop=0., attn_pdrop=0., n_unmasked=0):
        super().__init__()

        if hybrid_mask:
            n_unmasked = text_seq_length + 1
            n_unmasked_text = text_seq_length + 1
            n_unmasked_img  = img_tok_seq_length + 1 

        config = GPTConfig(vocab_size=vocab_size, block_size=block_size,
                           embd_pdrop=embd_pdrop, resid_pdrop=resid_pdrop, attn_pdrop=attn_pdrop,
                           n_layer=n_layer, n_head=n_head, n_embd=n_embd,
                           n_unmasked=n_unmasked, batch_size=batch_size, n_unmasked_text=n_unmasked_text, n_unmasked_img=n_unmasked_img,
                           text_seq_length=text_seq_length, img_tok_seq_length=img_tok_seq_length,
                           n_memory_cells=n_memory_cells)

        # input embedding stem
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)                   # [50435 (49408+1+1024+2), 512]
        self.seg_emb = nn.Embedding(2, config.n_embd)                                   # [2, 512]
        self.pos_emb = nn.Parameter(torch.zeros(1, config.block_size, config.n_embd))   # [387 (256+2+128+1), 512] # 16 x 16 codebook
                                                                                        # [195 (64 +2+128+1), 512] # 8 x 8 codebook
        self.drop = nn.Dropout(config.embd_pdrop)                                       # 0.0 (default)


        blocks = []
        for _ in range(config.n_layer - 1):
            blocks.append(ReBlockWOMeM(config))  # w/o  memory 
        blocks.append(ReBlockWMeMs(config))      # w    memory for last layer
        self.blocks = nn.Sequential(*blocks)

            
        # decoder head
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.block_size = config.block_size

        self.apply(self._init_weights)
        self.config = config

    def get_block_size(self):
        return self.block_size

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


    def forward(self, idx, prev_ms, seg, cached_memory=None, gen_type='t2i'):
        # forward the GPT model
        # HERE "idx" --> cz_indices[:, :-1] # [B, 386] --> [SOS] + 128 + [SOI] + 256
        # vocab size: 50435 = 1024 (size of codebook) + 2 (?) + 49408 + 1 (?, unknown token?) for dalle training

        token_embeddings = self.tok_emb(idx) # each index maps to a (learnable) vector, token_embeddings shape: # [B, 386, 512] at Training # self.tok_emb: [50435, 512]      
        segment_embeddings = self.seg_emb(seg)

        t = token_embeddings.shape[1] # [B, 386, 512]
        assert t <= self.block_size, "Cannot forward, model block size is exhausted." # self.block_size 387: 130 + 257
        position_embeddings = self.pos_emb[:, :t, :]                                  # each position maps to a (learnable) vector
                                                                                      # position_embeddings: [1, 386, 512], self.pos_emb: [1, 387, 512] 
        x = self.drop(token_embeddings + segment_embeddings + position_embeddings)


        if cached_memory != None and cached_memory[0][-1] != None:
            for layer_idx, block in enumerate(self.blocks):
                if layer_idx == len(self.blocks)-1:
                    prev_ms[layer_idx], x = block(x, prev_m=prev_ms[layer_idx], gen_type=gen_type, cached_memory=cached_memory)
                else:
                    prev_ms[layer_idx], x = block(x, prev_m=prev_ms[layer_idx], gen_type=gen_type)
                
        else:
            for layer_idx, block in enumerate(self.blocks):
                prev_ms[layer_idx], x = block(x, prev_m=prev_ms[layer_idx], gen_type=gen_type)


        x = self.ln_f(x)                             # x shape: [B, 1 (text cls) + 128 (text) + 1 (img cls) + 256 (img), d]
        logits = self.head(x)                        # self.head: 512 -> 50435, x.shape [B, 386, 512], 386 = [CLS,1] + 128 + [SEP,1] + 256 

        return prev_ms, logits




################################################################################################################################################################################################
################################################################################################################################################################################################
################################################################################################################################################################################################
################################################################################################################################################################################################
#### sampling utils

def top_k_logits(logits, k):
    v, ix = torch.topk(logits, k)
    out = logits.clone()
    out[out < v[:, [-1]]] = -float('Inf')
    return out

@torch.no_grad()
def sample(model, x, steps, temperature=1.0, sample=False, top_k=None):
    """
    take a conditioning sequence of indices in x (of shape (b,t)) and predict the next token in
    the sequence, feeding the predictions back into the model each time. Clearly the sampling
    has quadratic complexity unlike an RNN that is only linear, and has a finite context window
    of block_size, unlike an RNN that has an infinite context window.
    """
    block_size = model.get_block_size()
    model.eval()
    for k in range(steps):
        x_cond = x if x.size(1) <= block_size else x[:, -block_size:]  # crop context if needed
        logits, _ = model(x_cond)
        # pluck the logits at the final step and scale by temperature
        logits = logits[:, -1, :] / temperature
        # optionally crop probabilities to only the top k options
        if top_k is not None:
            logits = top_k_logits(logits, top_k)
        # apply softmax to convert to probabilities
        probs = F.softmax(logits, dim=-1)
        # sample from the distribution or take the most likely
        if sample:
            ix = torch.multinomial(probs, num_samples=1)
        else:
            _, ix = torch.topk(probs, k=1, dim=-1)
        # append to the sequence and continue
        x = torch.cat((x, ix), dim=1)

    return x


@torch.no_grad()
def sample_with_past(x, model, steps, temperature=1., sample_logits=True,
                     top_k=None, top_p=None, callback=None):
    # x is conditioning
    sample = x
    cond_len = x.shape[1]
    past = None
    for n in range(steps):
        if callback is not None:
            callback(n)
        logits, _, present = model.forward_with_past(x, past=past, past_length=(n+cond_len-1))
        if past is None:
            past = [present]
        else:
            past.append(present)
        logits = logits[:, -1, :] / temperature
        if top_k is not None:
            logits = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)

        probs = F.softmax(logits, dim=-1)
        if not sample_logits:
            _, x = torch.topk(probs, k=1, dim=-1)
        else:
            x = torch.multinomial(probs, num_samples=1)
        # append to the sequence and continue
        sample = torch.cat((sample, x), dim=1)
    del past
    sample = sample[:, cond_len:]  # cut conditioning off
    return sample


#### clustering utils

# class KMeans(nn.Module):
#     def __init__(self, ncluster=512, nc=3, niter=10):
#         super().__init__()
#         self.ncluster = ncluster
#         self.nc = nc
#         self.niter = niter
#         self.shape = (3,32,32)
#         self.register_buffer("C", torch.zeros(self.ncluster,nc))
#         self.register_buffer('initialized', torch.tensor(0, dtype=torch.uint8))

#     def is_initialized(self):
#         return self.initialized.item() == 1

#     @torch.no_grad()
#     def initialize(self, x):
#         N, D = x.shape
#         assert D == self.nc, D
#         c = x[torch.randperm(N)[:self.ncluster]] # init clusters at random
#         for i in range(self.niter):
#             # assign all pixels to the closest codebook element
#             a = ((x[:, None, :] - c[None, :, :])**2).sum(-1).argmin(1)
#             # move each codebook element to be the mean of the pixels that assigned to it
#             c = torch.stack([x[a==k].mean(0) for k in range(self.ncluster)])
#             # re-assign any poorly positioned codebook elements
#             nanix = torch.any(torch.isnan(c), dim=1)
#             ndead = nanix.sum().item()
#             print('done step %d/%d, re-initialized %d dead clusters' % (i+1, self.niter, ndead))
#             c[nanix] = x[torch.randperm(N)[:ndead]] # re-init dead clusters

#         self.C.copy_(c)
#         self.initialized.fill_(1)


#     def forward(self, x, reverse=False, shape=None):
#         if not reverse:
#             # flatten
#             bs,c,h,w = x.shape
#             assert c == self.nc
#             x = x.reshape(bs,c,h*w,1)
#             C = self.C.permute(1,0)
#             C = C.reshape(1,c,1,self.ncluster)
#             a = ((x-C)**2).sum(1).argmin(-1) # bs, h*w indices
#             return a
#         else:
#             # flatten
#             bs, HW = x.shape
#             """
#             c = self.C.reshape( 1, self.nc,  1, self.ncluster)
#             c = c[bs*[0],:,:,:]
#             c = c[:,:,HW*[0],:]
#             x =      x.reshape(bs,       1, HW,             1)
#             x = x[:,3*[0],:,:]
#             x = torch.gather(c, dim=3, index=x)
#             """
#             x = self.C[x]
#             x = x.permute(0,2,1)
#             shape = shape if shape is not None else self.shape
#             x = x.reshape(bs, *shape)

#             return x