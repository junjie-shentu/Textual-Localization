"""
This file is modified from the original file of the HuggingFace Diffuser library.
https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py#L1367
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from diffusers.models.attention_processor import Attention

from textual_localization.utils.CrossAttnMap import AttentionStore

class CustomDiffusionAttnProcessor2_0(nn.Module):
    r"""
    Processor for implementing attention for the Custom Diffusion method using PyTorch 2.0's memory-efficient scaled
    dot-product attention.

    Args:
        train_kv (`bool`, defaults to `True`):
            Whether to newly train the key and value matrices corresponding to the text features.
        train_q_out (`bool`, defaults to `True`):
            Whether to newly train query matrices corresponding to the latent image features.
        hidden_size (`int`, *optional*, defaults to `None`):
            The hidden size of the attention layer.
        cross_attention_dim (`int`, *optional*, defaults to `None`):
            The number of channels in the `encoder_hidden_states`.
        out_bias (`bool`, defaults to `True`):
            Whether to include the bias parameter in `train_q_out`.
        dropout (`float`, *optional*, defaults to 0.0):
            The dropout probability to use.
    """

    def __init__(
        self,
        train_k: bool = True,
        train_v: bool = True,
        train_q: bool = True,
        train_out: bool = True,
        hidden_size: Optional[int] = None,
        cross_attention_dim: Optional[int] = None,
        out_bias: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.train_k = train_k
        self.train_v = train_v
        self.train_q = train_q
        self.train_out = train_out

        self.hidden_size = hidden_size
        self.cross_attention_dim = cross_attention_dim

        # `_custom_diffusion` id for easy serialization and loading.
        if self.train_k:
            self.to_k_custom_diffusion = nn.Linear(cross_attention_dim or hidden_size, hidden_size, bias=False)
        if self.train_v:
            self.to_v_custom_diffusion = nn.Linear(cross_attention_dim or hidden_size, hidden_size, bias=False)
        if self.train_q:
            self.to_q_custom_diffusion = nn.Linear(hidden_size, hidden_size, bias=False)
        if self.train_out:
            self.to_out_custom_diffusion = nn.ModuleList([])
            self.to_out_custom_diffusion.append(nn.Linear(hidden_size, hidden_size, bias=out_bias))
            self.to_out_custom_diffusion.append(nn.Dropout(dropout))

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
    ) -> torch.FloatTensor:
        batch_size, sequence_length, _ = hidden_states.shape
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
        if self.train_q:
            query = self.to_q_custom_diffusion(hidden_states)
        else:
            query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            crossattn = False
            encoder_hidden_states = hidden_states
        else:
            crossattn = True
            if attn.norm_cross:
                encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        if self.train_k:
            key = self.to_k_custom_diffusion(encoder_hidden_states)
        else:
            key = attn.to_k(encoder_hidden_states)

        if self.train_v:
            value = self.to_v_custom_diffusion(encoder_hidden_states)
        else:
            value = attn.to_v(encoder_hidden_states)

        if crossattn:
            detach = torch.ones_like(key)
            detach[:, :1, :] = detach[:, :1, :] * 0.0
            key = detach * key + (1 - detach) * key.detach()
            value = detach * value + (1 - detach) * value.detach()

        inner_dim = hidden_states.shape[-1]

        head_dim = inner_dim // attn.heads
        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        # TODO: add support for attn.scale when we move to Torch 2.1
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        if self.train_out:
            # linear proj
            hidden_states = self.to_out_custom_diffusion[0](hidden_states)
            # dropout
            hidden_states = self.to_out_custom_diffusion[1](hidden_states)
        else:
            # linear proj
            hidden_states = attn.to_out[0](hidden_states)
            # dropout
            hidden_states = attn.to_out[1](hidden_states)

        return hidden_states




############################################################################################################################################################

class CustomDiffusionAttnProcessor(nn.Module):
    r"""
    Processor for implementing attention for the Custom Diffusion method.

    This version add a controller to get the cross attention map.

    Args:
        train_kv (`bool`, defaults to `True`):
            Whether to newly train the key and value matrices corresponding to the text features.
        train_q_out (`bool`, defaults to `True`):
            Whether to newly train query matrices corresponding to the latent image features.
        hidden_size (`int`, *optional*, defaults to `None`):
            The hidden size of the attention layer.
        cross_attention_dim (`int`, *optional*, defaults to `None`):
            The number of channels in the `encoder_hidden_states`.
        out_bias (`bool`, defaults to `True`):
            Whether to include the bias parameter in `train_q_out`.
        dropout (`float`, *optional*, defaults to 0.0):
            The dropout probability to use.
    """

    def __init__(
        self,
        train_k: bool = True,
        train_v: bool = True,
        train_q: bool = True,
        train_out: bool = True,
        hidden_size: Optional[int] = None,
        cross_attention_dim: Optional[int] = None,
        out_bias: bool = True,
        dropout: float = 0.0,
        controller: AttentionStore = None,
        place_in_unet: str = None,
    ):
        super().__init__()
        self.train_k = train_k
        self.train_v = train_v
        self.train_q = train_q
        self.train_out = train_out

        self.hidden_size = hidden_size
        self.cross_attention_dim = cross_attention_dim

        self.controller = controller
        self.place_in_unet = place_in_unet

        if self.train_k:
            self.to_k_custom_diffusion = nn.Linear(cross_attention_dim or hidden_size, hidden_size, bias=False)
        if self.train_v:
            self.to_v_custom_diffusion = nn.Linear(cross_attention_dim or hidden_size, hidden_size, bias=False)
        if self.train_q:
            self.to_q_custom_diffusion = nn.Linear(hidden_size, hidden_size, bias=False)
        if self.train_out:
            self.to_out_custom_diffusion = nn.ModuleList([])
            self.to_out_custom_diffusion.append(nn.Linear(hidden_size, hidden_size, bias=out_bias))
            self.to_out_custom_diffusion.append(nn.Dropout(dropout))

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
    ) -> torch.Tensor:
        batch_size, sequence_length, _ = hidden_states.shape
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
        if self.train_q:
            query = self.to_q_custom_diffusion(hidden_states).to(attn.to_q.weight.dtype)
        else:
            query = attn.to_q(hidden_states.to(attn.to_q.weight.dtype))

        if encoder_hidden_states is None:
            crossattn = False
            encoder_hidden_states = hidden_states
        else:
            crossattn = True
            if attn.norm_cross:
                encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        if self.train_k:
            key = self.to_k_custom_diffusion(encoder_hidden_states.to(self.to_k_custom_diffusion.weight.dtype))
            key = key.to(attn.to_q.weight.dtype)
        else:
            key = attn.to_k(encoder_hidden_states)

        if self.train_v:
            value = self.to_v_custom_diffusion(encoder_hidden_states.to(self.to_v_custom_diffusion.weight.dtype))
            value = value.to(attn.to_q.weight.dtype)
        else:
            value = attn.to_v(encoder_hidden_states)

        if crossattn:
            detach = torch.ones_like(key)
            detach[:, :1, :] = detach[:, :1, :] * 0.0
            key = detach * key + (1 - detach) * key.detach()
            value = detach * value + (1 - detach) * value.detach()

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask) # the shape should be: (batch_size * head_size, image_sequence_length, text_sequence_length)
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        #get the attention map
        if attention_probs.shape[-1] == 77: # this attention processor processes both self attention layer and cross attention layer, but we only need the cross attention map. so we filter out the self attention map.
            # print("place_in_unet: ", self.place_in_unet)
            # print("attention_probs_dimension: ", attention_probs.shape)
            self.controller(attn=attention_probs, is_cross=True, place_in_unet=self.place_in_unet)

        if self.train_out:
            # linear proj
            hidden_states = self.to_out_custom_diffusion[0](hidden_states)
            # dropout
            hidden_states = self.to_out_custom_diffusion[1](hidden_states)
        else:
            # linear proj
            hidden_states = attn.to_out[0](hidden_states)
            # dropout
            hidden_states = attn.to_out[1](hidden_states)

        return hidden_states

