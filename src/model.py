import math
import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F


class LayerNorm(nn.Module):
    """LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False"""

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


class RelSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.mem_len = config.mem_len
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout

        # Query, key, value projections
        self.queries = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.keys = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.values = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

        self.head_size = config.n_embd // config.n_head

        # Learned relative position embeddings
        self.u = nn.Parameter(torch.zeros(config.n_head, self.head_size))
        self.v = nn.Parameter(torch.zeros(config.n_head, self.head_size))

        # Project relative position embeddings to embedding dimension
        self.r_proj = nn.Linear(self.head_size, config.n_embd, bias=False)

        # Sinusoidal position embeddings (0 .. block_size+mem_len-1)
        max_len = config.block_size + config.mem_len
        pos_table = self.sinusoidal_table(max_len, self.head_size)  # (M+T, hs)
        self.register_buffer("pos_emb_table", pos_table, persistent=False)

        # Combine joined heads into final output
        self.out_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

        # Regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

    @staticmethod
    def sinusoidal_table(
        max_len: int,
        d_h: int,
    ) -> torch.Tensor:
        """Out: table (max_len, d_h)"""
        pos = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)  # (max_len, 1)
        i = torch.arange(d_h // 2, dtype=torch.float32).unsqueeze(0)  # (1, d_h/2)
        denom = torch.pow(10000, 2 * i / d_h)  # (1, d_h/2)

        angles = pos / denom  # (max_len, d_h/2)
        table = torch.zeros(max_len, d_h)
        table[:, 0::2] = torch.sin(angles)
        table[:, 1::2] = torch.cos(angles)
        return table  # (max_len, d_h)

    def forward(self, h, mem=None):
        """
        In: h: (B, T, n_embd), mem: (B, M, n_embd)
        Out: y: (B, T, n_embd), new_mem: (B, M + T, n_embd)
        """
        B, T, n_embd = h.size()

        if mem is not None:
            M = mem.size(1)
            x_ext = torch.cat([mem, h], dim=1)  # (B,L,n_embd)
        else:
            M = 0
            x_ext = h  # (B,T,n_embd)
        L = M + T

        pos_seq = torch.arange(L - 1, -1, -1, device=h.device)  # (L), reversed
        R = self.r_proj(self.pos_emb_table[pos_seq])  # (L, C)
        R = R.view(1, L, self.n_head, n_embd // self.n_head).transpose(
            1, 2
        )  # (1, nh, L, hs)

        u = self.u.unsqueeze(0).unsqueeze(2)  # (1, nh, 1, hs)
        v = self.v.unsqueeze(0).unsqueeze(2)  # (1, nh, 1, hs)

        # Query, key, value projections
        # (B,T,n_embd) -> (B,T,nh,hs) -> (B,nh,T,hs)
        q = self.queries(h)
        q = q.view(B, T, self.n_head, n_embd // self.n_head).transpose(1, 2)

        k = self.keys(x_ext)
        k = k.view(B, L, self.n_head, n_embd // self.n_head).transpose(1, 2)

        v = self.values(x_ext)
        v = v.view(B, L, self.n_head, n_embd // self.n_head).transpose(1, 2)

        key_idx = torch.arange(L, device=h.device)
        query_idx = torch.arange(T, device=h.device) + M
        causal = key_idx.unsqueeze(0) <= query_idx.unsqueeze(1)
        attn_mask = ~causal
        attn_mask = attn_mask.view(1, 1, T, L)

        # Compute attention scores
        # score = (Q + u)K^T + (Q + v)R^T
        q_u = q + u  # (B, nh, T, hs)
        score_u = torch.matmul(q_u, k.transpose(-2, -1))  # (B,nh,T,L)

        q_v = q + v  # (B, nh, T, hs)
        score_v = torch.matmul(q_v, R.transpose(-2, -1))  # (B,nh,T,L)
        score_v = self.rel_shift(score_v)  # (B,nh,T,L)
        scores = (score_u + score_v) / math.sqrt(self.head_size)
        scores = scores.masked_fill(attn_mask, float("-inf"))

        # Calculate output from scores
        attn = F.softmax(scores, dim=-1)
        attn = self.attn_dropout(attn)
        y = attn @ v  # (B,nh,T,L) x (B,nh,L,hs) -> (B,nh,T,hs)
        y = y.transpose(1, 2).contiguous().view(B, T, n_embd)
        y = self.resid_dropout(self.out_proj(y))

        # Update memory
        if self.mem_len > 0:
            if mem is None:
                new_mem_full = h.detach()  # (B, T, C)
            else:
                new_mem_full = torch.cat([mem, h.detach()], dim=1)  # (B, M+T, C)
            new_mem = new_mem_full[:, -self.mem_len :, :]  # (B, mem_len, C)
        else:
            new_mem = None

        return y, new_mem

    def rel_shift(self, x):
        """
        Maps relative position embeddings to absolute position embeddings.
        In: x: (B, H, T, L)
        Out: x: (B, H, T, L)
        """
        zero_pad = torch.zeros(
            (x.size(0), x.size(1), x.size(2), 1), device=x.device, dtype=x.dtype
        )
        x_padded = torch.cat([zero_pad, x], dim=3)
        x_padded = x_padded.view(x.size(0), x.size(1), x.size(3) + 1, x.size(2))
        x = x_padded[:, :, 1:, :].view_as(x)  # (B, H, T, L)
        return x


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = RelSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x, mem=None):
        attn_out, new_mem = self.attn(self.ln_1(x), mem)
        x = x + attn_out
        x = x + self.mlp(self.ln_2(x))
        return x, new_mem


@dataclass
class TransformerXLConfig:
    block_size: int = 1024
    vocab_size: int = 50304  # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True  # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    mem_len: int = 2048


class TransformerXL(nn.Module):
    """Implements [Transformer-XL](https://arxiv.org/abs/1901.02860)"""

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),
                drop=nn.Dropout(config.dropout),
                h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                ln_f=LayerNorm(config.n_embd, bias=config.bias),
            )
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight

        # Initialize weights, per GPT-2
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(
                    p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer)
                )

        print("Number of parameters: %.2fM" % (self.get_num_params() / 1e6,))

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters())

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, mem=None, targets=None):
        """
        In: idx: (B,T), mem: list[(M, n_embd)], targets: (B,T)
        Out: logits: (B,T,C), loss: np.float32
        """
        assert idx.size(1) <= self.config.block_size, (
            f"Cannot forward sequence of length {idx.size(1)}, block size is only {self.config.block_size}"
        )

        tok_emb = self.transformer.wte(idx)  # (B,T,n_embd)
        x = self.transformer.drop(tok_emb)

        if mem is None:
            mem = [None] * self.config.n_layer
        new_mem = []

        # For each block, forward the model and update the memory
        # This way further blocks can attend to memory
        for block, mem_i in zip(self.transformer.h, mem):
            x, mem_i = block(x, mem_i)
            new_mem.append(mem_i)
        x = self.transformer.ln_f(x)

        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-1
            )
        else:
            # nanoGPT optimization: Only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :])
            loss = None

        return logits, new_mem, loss

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(
            f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters"
        )
        print(
            f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters"
        )
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(
            optim_groups, lr=learning_rate, betas=betas, **extra_args
        )
        print(f"using fused AdamW: {use_fused}")

        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """Model flops utilization (MFU) from nanoGPT implementation"""
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd // cfg.n_head, cfg.block_size
        flops_per_token = 6 * N + 12 * L * H * Q * T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        flops_achieved = flops_per_iter * (1.0 / dt)  # per second
        flops_promised = 312e12  # A100 GPU bfloat16 peak flops is 312 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        mem = None
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = (
                idx
                if idx.size(1) <= self.config.block_size
                else idx[:, -self.config.block_size :]
            )
            # forward the model to get the logits for the index in the sequence
            logits, mem, _ = self(idx_cond, mem)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx
