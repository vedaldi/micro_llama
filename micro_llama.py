import torch
import tiktoken
import tiktoken.load


def rope(x, theta):
    B, N, H, D = x.shape
    freq = theta ** -torch.arange(0, 1, 2 / D, device=x.device)
    time = torch.arange(N, device=x.device)
    phase = freq.reshape(1, 1, 1, D // 2) * time.reshape(1, N, 1, 1)  # (1, N, 1, D//2)
    c = torch.cos(phase)
    s = torch.sin(phase)
    rot = torch.stack([c, s, -s, c], dim=-1).reshape(1, N, 1, D // 2, 2, 2)
    x = x.reshape(B, N, H, D // 2, 1, 2) @ rot
    return x.reshape(B, N, H, D)


def make_tokenizer(path):
    tokenizer_model = tiktoken.load.load_tiktoken_bpe(path)
    special_tokens = [
        "<|begin_of_text|>",  # Marks the beginning of a text sequence.
        "<|end_of_text|>",  # Marks the end of a text sequence.
        "<|reserved_special_token_0|>",  # Reserved for future use.
        "<|reserved_special_token_1|>",  # Reserved for future use.
        "<|reserved_special_token_2|>",  # Reserved for future use.
        "<|reserved_special_token_3|>",  # Reserved for future use.
        "<|start_header_id|>",  # Indicates the start of a header ID.
        "<|end_header_id|>",  # Indicates the end of a header ID.
        "<|reserved_special_token_4|>",  # Reserved for future use.
        "<|eot_id|>",  # Marks the end of a turn (in a conversational context).
    ] + [f"<|reserved_special_token_{i}|>" for i in range(5, 256 - 5)]
    tokenize_breaker = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
    return tiktoken.Encoding(
        name=path,
        pat_str=tokenize_breaker,
        mergeable_ranks=tokenizer_model,
        special_tokens={
            token: len(tokenizer_model) + i for i, token in enumerate(special_tokens)
        },
    )


class RMSNorm(torch.nn.Module):
    def __init__(self, dim, epsilon):
        super().__init__()
        self.epsilon = epsilon
        self.weight = torch.nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return (x * self.weight) * torch.rsqrt(
            (x * x).mean(-1, keepdim=True) + self.epsilon
        )


class Attention(torch.nn.Module):
    def __init__(self, dim, n_heads, n_kv_heads, rope_theta):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.rope_theta = rope_theta
        self.wq = torch.nn.Linear(dim, dim, bias=False)
        self.wk = torch.nn.Linear(dim, dim // (n_heads // n_kv_heads), bias=False)
        self.wv = torch.nn.Linear(dim, dim // (n_heads // n_kv_heads), bias=False)
        self.wo = torch.nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        B, N, D = x.shape
        H = self.n_heads
        J = self.n_kv_heads
        q = self.wq(x)
        k = self.wk(x)
        v = self.wv(x)

        q = q.reshape((B, N, H, D // H))
        k = k.reshape((B, N, J, D // H))
        v = v.reshape((B, N, J, D // H))

        q = rope(q, theta=self.rope_theta)
        k = rope(k, theta=self.rope_theta)

        k = (
            k.reshape((B, N, J, 1, D // H))
            .expand((B, N, J, H // J, D // H))
            .reshape((B, N, H, D // H))
        )
        v = (
            v.reshape((B, N, J, 1, D // H))
            .expand((B, N, J, H // J, D // H))
            .reshape((B, N, H, D // H))
        )

        q = q.transpose(1, 2)  # (B, H, N, D//H)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        dot = q @ k.transpose(2, 3) / (D // H) ** 0.5
        mask = torch.full((N, N), float("-inf"), device=x.device)
        mask = torch.triu(mask, diagonal=1)
        dot = dot + mask  # (B, H, N, N)

        weight = torch.nn.functional.softmax(dot, dim=-1)
        x = weight @ v  # (B, H, N, D//H)
        x = x.transpose(1, 2).reshape((B, N, D))
        x = self.wo(x)
        return x


class FeedForward(torch.nn.Module):
    def __init__(self, dim, latent_dim):
        super().__init__()
        self.w1 = torch.nn.Linear(dim, latent_dim, bias=False)
        self.w2 = torch.nn.Linear(latent_dim, dim, bias=False)
        self.w3 = torch.nn.Linear(dim, latent_dim, bias=False)

    def forward(self, x):
        a = torch.nn.functional.silu(self.w1(x))
        b = self.w3(x)
        return self.w2(a * b)


class Layer(torch.nn.Module):
    def __init__(self, dim, n_heads, n_kv_heads, norm_eps, latent_dim, rope_theta):
        super().__init__()
        self.attention_norm = RMSNorm(dim, norm_eps)
        self.ffn_norm = RMSNorm(dim, norm_eps)
        self.attention = Attention(dim, n_heads, n_kv_heads, rope_theta)
        self.feed_forward = FeedForward(dim, latent_dim)

    def forward(self, x):
        y = self.attention_norm(x)
        y = self.attention(y)
        x = x + y

        y = self.ffn_norm(x)
        y = self.feed_forward(y)
        x = x + y
        return x


class Llama(torch.nn.Module):
    def __init__(
        self,
        dim=4096,
        n_layers=32,
        n_heads=32,
        n_kv_heads=8,
        vocab_size=128256,
        norm_eps=1e-5,
        latent_dim=14336,
        rope_theta=500000.0,
    ):
        super().__init__()
        self.dim = dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.vocab_size = vocab_size
        self.norm_eps = norm_eps
        self.latent_dim = latent_dim
        self.rope_theta = rope_theta

        self.tok_embeddings = torch.nn.Embedding(self.vocab_size, self.dim)
        self.layers = torch.nn.ModuleList(
            [
                Layer(
                    dim=self.dim,
                    n_heads=self.n_heads,
                    n_kv_heads=self.n_kv_heads,
                    norm_eps=self.norm_eps,
                    latent_dim=self.latent_dim,
                    rope_theta=self.rope_theta,
                )
                for _ in range(self.n_layers)
            ]
        )
        self.norm = RMSNorm(self.dim, self.norm_eps)
        self.output = torch.nn.Linear(self.dim, self.vocab_size, bias=False)

    def forward(self, x):
        x = self.tok_embeddings(x)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        x = self.output(x)
        return x
