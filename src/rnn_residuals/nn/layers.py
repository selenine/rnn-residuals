import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from rnn_residuals.config import TransformerConfig


class RoPE(nn.Module):
    def __init__(
        self,
        cfg: TransformerConfig,
        base: float = 10000.0,
    ) -> None:
        super().__init__()

        freqs = base ** (-torch.arange(0, cfg.d_head, 2).float() / cfg.d_head)
        angles = torch.outer(torch.arange(cfg.n_ctx).float(), freqs)

        self.register_buffer("cos", angles.cos(), persistent=False)
        self.register_buffer("sin", angles.sin(), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = x.shape[-2]
        cos, sin = self.cos[:s].to(x.dtype), self.sin[:s].to(x.dtype)
        x1, x2 = x.chunk(2, dim=-1)

        return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


class MHSA(nn.Module):
    def __init__(
        self,
        cfg: TransformerConfig,
    ) -> None:
        super().__init__()

        self.cfg = cfg

        self.norm = nn.RMSNorm(cfg.d_model)
        self.rope = RoPE(cfg)

        self.WQ = nn.Linear(cfg.d_model, cfg.n_heads * cfg.d_head, bias=False)
        self.WK = nn.Linear(cfg.d_model, cfg.n_heads * cfg.d_head, bias=False)
        self.WV = nn.Linear(cfg.d_model, cfg.n_heads * cfg.d_head, bias=False)
        self.WO = nn.Linear(cfg.n_heads * cfg.d_head, cfg.d_model, bias=False)

        for layer in [self.WQ, self.WK, self.WV, self.WO]:
            nn.init.kaiming_normal_(layer.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        (b, s, _), n, d = x.shape, self.cfg.n_heads, self.cfg.d_head
        h = self.norm(x)
        q = self.rope(self.WQ(h).view(b, s, n, d).transpose(1, 2))
        k = self.rope(self.WK(h).view(b, s, n, d).transpose(1, 2))
        v = self.WV(h).view(b, s, n, d).transpose(1, 2)

        out = self.WO(
            F.scaled_dot_product_attention(q, k, v, is_causal=self.cfg.causal)
            .transpose(1, 2)
            .reshape(b, s, n * d)
        )

        return out


class MLP(nn.Module):
    def __init__(
        self,
        cfg: TransformerConfig,
    ) -> None:
        super().__init__()

        self.cfg = cfg

        self.norm = nn.RMSNorm(cfg.d_model)

        self.Wup = nn.Linear(cfg.d_model, cfg.d_mlp, bias=False)
        self.Wgate = nn.Linear(cfg.d_model, cfg.d_mlp, bias=False)
        self.Wdown = nn.Linear(cfg.d_mlp, cfg.d_model, bias=False)

        for layer in [self.Wup, self.Wgate, self.Wdown]:
            nn.init.kaiming_normal_(layer.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        out, gate = self.Wup(h), self.Wgate(h)
        out = self.Wdown(out * F.silu(gate))

        return out


class GDN2(nn.Module):
    def __init__(
        self,
        cfg: TransformerConfig,
        first: bool = False,
    ) -> None:
        super().__init__()

        assert cfg.n_heads * cfg.d_head == cfg.d_model
        self.cfg = cfg
        n, d = cfg.n_heads, cfg.d_head
        self.use_alpha = cfg.use_alpha and not first
        self.use_beta = cfg.use_beta and not first
        self.use_gamma = cfg.use_gamma

        self.Wqkv = nn.Linear(cfg.d_model, 3 * n * d, bias=False)
        self.qk_bias = nn.Parameter(torch.randn(2 * n * d))

        nn.init.kaiming_normal_(self.Wqkv.weight)
        nn.init.zeros_(self.Wqkv.weight[: 2 * n * d])

        if self.use_alpha:
            self.Wa = nn.Linear(cfg.d_model, n * d, bias=False)
            nn.init.zeros_(self.Wa.weight)
            self.A_log = nn.Parameter(torch.zeros(n * d))
            self.dt_bias = nn.Parameter(torch.full((n * d,), -18.0))

        if self.use_beta:
            self.Wb = nn.Linear(cfg.d_model, n * d, bias=True)
            nn.init.zeros_(self.Wb.bias)

        if self.use_gamma:
            self.Wc = nn.Linear(cfg.d_model, n * d, bias=True)
            nn.init.zeros_(self.Wc.bias)

    def forward(
        self, S: torch.Tensor | None, z: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        (b, s, _), n, d = z.shape, self.cfg.n_heads, self.cfg.d_head

        qk, v = self.Wqkv(z).float().split([2 * n * d, n * d], dim=-1)
        q, k = (qk + self.qk_bias).view(b, s, 2, n, d).unbind(2)
        v = v.view(b, s, n, d)
        q, k = F.normalize(q, dim=-1), F.normalize(k, dim=-1)
        if self.use_gamma:
            v = torch.sigmoid(self.Wc(z).float()).view(b, s, n, d) * v
        if self.use_beta:
            bk = torch.sigmoid(self.Wb(z).float()).view(b, s, n, d) * k

        if self.use_alpha:
            dt = F.softplus(self.Wa(z).float() + self.dt_bias)
            a = torch.exp(-self.A_log.exp() * dt).view(b, s, n, d)

        with torch.autocast(z.device.type, enabled=False):
            # S <- (I - k (beta * k)^T) Diag(a) S + k (gamma * v)^T
            write = k.unsqueeze(-1) * v.unsqueeze(-2)
            if S is None:
                S = write
            else:
                if self.use_alpha:
                    S = a.unsqueeze(-1) * S
                if self.use_beta:
                    S = S - k.unsqueeze(-1) * torch.einsum(
                        "...i,...ij->...j", bk, S
                    ).unsqueeze(-2)
                S = S + write

            h = torch.einsum("...i,...ij->...j", q, S).reshape(b, s, n * d)

        return S, h


class AttnRes(nn.Module):
    def __init__(
        self,
        cfg: TransformerConfig,
        first: bool = False,
    ) -> None:
        super().__init__()

        if not first:
            self.query = nn.Parameter(torch.zeros(cfg.d_model))

    def forward(
        self, S: tuple[torch.Tensor, ...] | None, z: torch.Tensor
    ) -> tuple[tuple[torch.Tensor, ...], torch.Tensor]:
        if S is None:
            return (z,), z.float()

        S = S + (z,)
        V = torch.stack(S, dim=-2)
        logits = F.rms_norm(V, (V.shape[-1],)) @ self.query.to(V.dtype)
        h = torch.einsum("...t,...td->...d", logits.float().softmax(-1).to(V.dtype), V)

        return S, h.float()


RESIDUALS = {"gdn2": GDN2, "attnres": AttnRes}


class Block(nn.Module):
    def __init__(
        self,
        cfg: TransformerConfig,
    ) -> None:
        super().__init__()

        self.attn = MHSA(cfg)
        self.mlp = MLP(cfg)
        self.attn_res = RESIDUALS[cfg.residual](cfg)
        self.mlp_res = RESIDUALS[cfg.residual](cfg)

    def forward(self, S, h: torch.Tensor):
        S, h = self.attn_res(S, self.attn(h))
        S, h = self.mlp_res(S, self.mlp(h))

        return S, h


class Transformer(nn.Module):
    def __init__(
        self,
        cfg: TransformerConfig,
    ) -> None:
        super().__init__()

        self.cfg = cfg

        self.embed = nn.Embedding(cfg.n_vocab, cfg.d_model)
        self.embed_res = RESIDUALS[cfg.residual](cfg, first=True)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.norm = nn.RMSNorm(cfg.d_model)
        self.unembed = nn.Linear(cfg.d_model, cfg.n_vocab, bias=False)

        nn.init.normal_(self.embed.weight, std=1.0)
        nn.init.normal_(self.unembed.weight, std=cfg.d_model**-0.5)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.embed(tokens)

        S, h = self.embed_res(None, x)
        for block in self.blocks:
            if self.cfg.grad_ckpt and torch.is_grad_enabled():
                S, h = checkpoint(block, S, h, use_reentrant=False)
            else:
                S, h = block(S, h)

        return self.unembed(self.norm(h))
