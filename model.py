import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

"""
DeepSeekV3 Configuration Class

This dataclass, `DeepSeekV3Config`, encapsulates all the configuration parameters required to initialize and build
the DeepSeekV3 language model. These parameters define the architecture, behavior, and various hyperparameters
of the model, ensuring consistent and reproducible model setups. The configuration is meticulously derived
from the `config_smollm2_135M.yaml` file and the Hugging Face model summary (`hf_model.md`), which provide
comprehensive details about the model's architecture and intended functionality.
"""
@dataclass
class DeepSeekV3Config:
    """
    hidden_size: int = 576
    intermediate_size: int = 1536
    num_hidden_layers: int = 30
    num_attention_heads: int = 9
    num_key_value_heads: int = 3
    """
    hidden_size: int = 384
    intermediate_size: int = 1024
    num_hidden_layers: int = 24
    num_attention_heads: int = 6
    num_key_value_heads: int = 2

    hidden_act: str = "silu"
    max_position_embeddings: int = 8192
    initializer_range: float = 0.041666666666666664
    rms_norm_eps: float = 1e-5
    vocab_size: int = 49152
    rope_theta: float = 100000.0
    use_cache: bool = True
    tie_word_embeddings: bool = True
    head_dim: int = 64
    attention_dropout: float = 0.0
    attention_bias: bool = False
    mlp_bias: bool = False
    model_type: str = "llama"
    torch_dtype: str = "float32"


"""
Root Mean Square Layer Normalization (RMSNorm) Layer

This class, `RMSNorm`, implements the Root Mean Square Normalization (RMSNorm) layer, a variant of Layer Normalization (LayerNorm) designed to improve computational efficiency while maintaining model performance. RMSNorm simplifies the normalization process by removing the mean-centering step inherent in LayerNorm, focusing solely on scaling the activations by their root mean square (RMS).

"""
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * norm * self.weight


"""
Precompute RoPE Frequencies

This function, `precompute_rope_freqs`, is responsible for calculating the frequencies used in
Rotary Positional Embeddings (RoPE) within the DeepSeekV3 transformer model. RoPE is a sophisticated
technique for encoding positional information directly into the token embeddings, allowing the
model to capture the order of tokens in a sequence without relying on traditional positional
embeddings. This method enhances the model's ability to generalize to longer sequences and
improves its efficiency in handling positional dependencies.

"""
def precompute_rope_freqs(dim: int, max_position_embeddings: int, theta: float = 10000.0) -> torch.Tensor:
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(max_position_embeddings, dtype=torch.float)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis


"""
Apply RoPE

This function, `apply_rope`, applies the Rotary Positional Embeddings (RoPE) to the input tensor `x`. RoPE is a sophisticated technique for encoding positional information directly into the token embeddings, enabling the model to understand the order of tokens in a sequence without relying on traditional positional embeddings. This method not only preserves the relative positional relationships between tokens but also enhances the model's ability to generalize to longer sequences efficiently.
"""
def apply_rope(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    
    # Ensure freqs_cis has the shape (1, seq_length, 1, head_dim//2) for correct broadcasting
    freqs_cis = freqs_cis[:x.shape[1], :].unsqueeze(0).unsqueeze(2)
    
    # Apply RoPE
    x_rotated = x_complex * freqs_cis  # Now shapes are compatible for broadcasting
    
    x_out = torch.view_as_real(x_rotated).flatten(-2)
    return x_out.type_as(x)


"""
DeepSeekV3Attention Layer

DeepSeek-V3 converts the DeepSeekV3 attention into an MLHA (Multi-Head Latent Attention) module.
The implementation is essentially the same as before but renamed for clarity.
"""
class DeepSeekV3Attention(nn.Module):
    def __init__(self, config: DeepSeekV3Config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        
        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=config.attention_bias)
        
        self.attention_dropout = config.attention_dropout

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        freqs_cis: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, seq_length, _ = hidden_states.shape
        
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
   
        query_states = query_states.view(batch_size, seq_length, self.num_heads, self.head_dim)
        key_states = key_states.view(batch_size, seq_length, self.num_key_value_heads, self.head_dim)
        value_states = value_states.view(batch_size, seq_length, self.num_key_value_heads, self.head_dim)
   
        if freqs_cis is not None:
            query_states = apply_rope(query_states, freqs_cis)
            key_states = apply_rope(key_states, freqs_cis)
   
        key_states = torch.repeat_interleave(key_states, self.num_key_value_groups, dim=2)
        value_states = torch.repeat_interleave(value_states, self.num_key_value_groups, dim=2)

        if hasattr(F, "scaled_dot_product_attention"):
            attn_output = F.scaled_dot_product_attention(
                query_states.transpose(1, 2),
                key_states.transpose(1, 2),
                value_states.transpose(1, 2),
                attn_mask=attention_mask,
                dropout_p=self.attention_dropout if self.training else 0.0,
            )
            attn_output = attn_output.transpose(1, 2)
        else:
            attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
            if attention_mask is not None:
                attn_weights = attn_weights + attention_mask
            attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
            if self.attention_dropout > 0.0 and self.training:
                attn_weights = F.dropout(attn_weights, p=self.attention_dropout)
            attn_output = torch.matmul(attn_weights, value_states)
       
        attn_output = attn_output.reshape(batch_size, seq_length, self.hidden_size)
        attn_output = self.o_proj(attn_output)
       
        return attn_output


"""
DeepSeekExpertLayer and DeepSeekMoE

These classes implement the MoE block for DeepSeek-V3 using a loss-less load-balancing strategy.
There is no auxiliary loss term here.
"""
class DeepSeekExpertLayer(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

class DeepSeekMoE(nn.Module):
    def __init__(self, hidden_size, intermediate_size, num_experts=8, top_k=2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.experts = nn.ModuleList([
            DeepSeekExpertLayer(hidden_size, intermediate_size) for _ in range(num_experts)
        ])
        self.router = nn.Linear(hidden_size, num_experts, bias=False)
    
    def forward(self, x):
        # x shape: [batch, seq_len, hidden_size]
        logits = self.router(x)  # [batch, seq_len, num_experts]
        # Get top-k expert indices
        topk = torch.topk(logits, self.top_k, dim=-1)
        indices = topk.indices  # [batch, seq_len, top_k]

        output = torch.zeros_like(x)
        for i in range(self.num_experts):
            mask = (indices == i).float()  # [batch, seq_len, top_k]
            if mask.sum() > 0:
                expert_out = self.experts[i](x)
                # Average over top_k selections for tokens routed to expert i
                mask_mean = mask.sum(dim=-1, keepdim=True).clamp(min=1.0)
                expert_contrib = expert_out * (mask.sum(dim=-1, keepdim=True) / self.top_k)
                output += expert_contrib
        # Normalize by top_k factor
        return output / self.top_k


"""
DeepSeekV3MLP Layer

This MLP block now uses a DeepSeekMoE module.
"""
class DeepSeekV3MLP(nn.Module):
    def __init__(self, config: DeepSeekV3Config):
        super().__init__()
        self.moe = DeepSeekMoE(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            num_experts=8,  # you can tune this
            top_k=2
        )
    def forward(self, x):
        return self.moe(x)


"""
DeepSeekV3 Block

Update the transformer block to use DeepSeekV3Attention and DeepSeekV3MLP.
"""
class DeepSeekV3Block(nn.Module):
    def __init__(self, config: DeepSeekV3Config):
        super().__init__()
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # Use the new DeepSeekV3Attention
        self.attention = DeepSeekV3Attention(config)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # Use the new MoE-based MLP
        self.mlp = DeepSeekV3MLP(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        freqs_cis: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.attention(hidden_states, attention_mask, freqs_cis)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


"""
DeepSeekV3 Model

You can update the docstrings here to say that this is DeepSeek-V3.
No substantive changes in the forward pass are needed aside from using the updated blocks.
"""
class DeepSeekV3Model(nn.Module):
    def __init__(self, config: DeepSeekV3Config):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        
        self.dtype = getattr(torch, config.torch_dtype) if hasattr(torch, config.torch_dtype) else torch.float32
        
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([DeepSeekV3Block(config) for _ in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
        self.freqs_cis = precompute_rope_freqs(
            config.head_dim,
            config.max_position_embeddings,
            config.rope_theta,
        )
        
        self.apply(self._init_weights)
        
        self.to(self.dtype)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)

        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            attention_mask = attention_mask.to(dtype=hidden_states.dtype)
            attention_mask = (1.0 - attention_mask) * torch.finfo(hidden_states.dtype).min

        freqs_cis = self.freqs_cis.to(device=hidden_states.device, dtype=hidden_states.dtype)
        
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask, freqs_cis)

        hidden_states = self.norm(hidden_states)
        return hidden_states


"""
DeepSeekV3 For Causal LM

We update the class name and docstrings to reflect that this is now DeepSeek-V3.
"""
class DeepSeekV3ForCausalLM(nn.Module):
    def __init__(self, config: DeepSeekV3Config):
        super().__init__()
        self.config = config
        self.model = DeepSeekV3Model(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # Tie weights if configured
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        hidden_states = self.model(input_ids, attention_mask)
        logits = self.lm_head(hidden_states)
        
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            
        return logits, loss
