# Adapted from https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention.py

from dataclasses import dataclass
from typing import Optional
from unittest import skip

import torch
import torch.nn.functional as F
from torch import nn
import math
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.modeling_utils import ModelMixin
from diffusers.utils import BaseOutput
from diffusers.utils.import_utils import is_xformers_available
from diffusers.models.attention import CrossAttention, FeedForward, AdaLayerNorm

from einops import rearrange, repeat


@dataclass
class Transformer3DModelOutput(BaseOutput):
    sample: torch.FloatTensor


if is_xformers_available():
    import xformers
    import xformers.ops
else:
    xformers = None


class Transformer3DModel(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(
        self,
        num_attention_heads: int = 16,
        attention_head_dim: int = 88,
        in_channels: Optional[int] = None,
        num_layers: int = 1,
        dropout: float = 0.0,
        norm_num_groups: int = 32,
        cross_attention_dim: Optional[int] = None,
        attention_bias: bool = False,
        activation_fn: str = "geglu",
        num_embeds_ada_norm: Optional[int] = None,
        use_linear_projection: bool = False,
        only_cross_attention: bool = False,
        upcast_attention: bool = False,
    ):
        super().__init__()
        self.use_linear_projection = use_linear_projection
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim
        inner_dim = num_attention_heads * attention_head_dim

        # Define input layers
        self.in_channels = in_channels

        self.norm = torch.nn.GroupNorm(num_groups=norm_num_groups, num_channels=in_channels, eps=1e-6, affine=True)
        if use_linear_projection:
            self.proj_in = nn.Linear(in_channels, inner_dim)
        else:
            self.proj_in = nn.Conv2d(in_channels, inner_dim, kernel_size=1, stride=1, padding=0)

        # Define transformers blocks
        self.transformer_blocks = nn.ModuleList(
            [
                BasicTransformerBlock(
                    inner_dim,
                    num_attention_heads,
                    attention_head_dim,
                    dropout=dropout,
                    cross_attention_dim=cross_attention_dim,
                    activation_fn=activation_fn,
                    num_embeds_ada_norm=num_embeds_ada_norm,
                    attention_bias=attention_bias,
                    only_cross_attention=only_cross_attention,
                    upcast_attention=upcast_attention,
                )
                for d in range(num_layers)
            ]
        )

        # 4. Define output layers
        if use_linear_projection:
            self.proj_out = nn.Linear(in_channels, inner_dim)
        else:
            self.proj_out = nn.Conv2d(inner_dim, in_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, hidden_states, encoder_hidden_states=None, timestep=None, return_dict: bool = True):
        # Input
        assert hidden_states.dim() == 5, f"Expected hidden_states to have ndim=5, but got ndim={hidden_states.dim()}."
        video_length = hidden_states.shape[2]
        hidden_states = rearrange(hidden_states, "b c f h w -> (b f) c h w")
        encoder_hidden_states = repeat(encoder_hidden_states, 'b n c -> (b f) n c', f=video_length)

        batch, channel, height, weight = hidden_states.shape
        residual = hidden_states

        hidden_states = self.norm(hidden_states)
        if not self.use_linear_projection:
            hidden_states = self.proj_in(hidden_states)
            inner_dim = hidden_states.shape[1]
            hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(batch, height * weight, inner_dim)
        else:
            inner_dim = hidden_states.shape[1]
            hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(batch, height * weight, inner_dim)
            hidden_states = self.proj_in(hidden_states)

        # Blocks
        for block in self.transformer_blocks:
            hidden_states = block(
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                timestep=timestep,
                video_length=video_length
            )

        # Output
        if not self.use_linear_projection:
            hidden_states = (
                hidden_states.reshape(batch, height, weight, inner_dim).permute(0, 3, 1, 2).contiguous()
            )
            hidden_states = self.proj_out(hidden_states)
        else:
            hidden_states = self.proj_out(hidden_states)
            hidden_states = (
                hidden_states.reshape(batch, height, weight, inner_dim).permute(0, 3, 1, 2).contiguous()
            )

        output = hidden_states + residual

        output = rearrange(output, "(b f) c h w -> b c f h w", f=video_length)
        if not return_dict:
            return (output,)

        return Transformer3DModelOutput(sample=output)

class Adapter(torch.nn.Module):
    def __init__(self, D_features, mlp_ratio=0.25, act_layer=torch.nn.GELU, skip_connect=True):
        super().__init__()
        self.skip_connect = skip_connect
        D_hidden_features = int(D_features * mlp_ratio)
        self.act = act_layer()
        self.D_fc1 = torch.nn.Linear(D_features, D_hidden_features)
        self.D_fc2 = torch.nn.Linear(D_hidden_features, D_features)
        
    def forward(self, x):
        # x is (BT, HW+1, D)
        xs = self.D_fc1(x)
        xs = self.act(xs)
        xs = self.D_fc2(xs)
        if self.skip_connect:
            x = x + xs
        else:
            x = xs
        return x

class BasicTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        dropout=0.0,
        cross_attention_dim: Optional[int] = None,
        activation_fn: str = "geglu",
        num_embeds_ada_norm: Optional[int] = None,
        attention_bias: bool = False,
        only_cross_attention: bool = False,
        upcast_attention: bool = False,
    ):
        super().__init__()
        self.only_cross_attention = only_cross_attention
        self.use_ada_layer_norm = num_embeds_ada_norm is not None

        # SC-Attn
        self.attn1 = SparseCausalAttention(
            query_dim=dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            dropout=dropout,
            bias=attention_bias,
            cross_attention_dim=cross_attention_dim if only_cross_attention else None,
            upcast_attention=upcast_attention,
        )
        self.norm1 = AdaLayerNorm(dim, num_embeds_ada_norm) if self.use_ada_layer_norm else nn.LayerNorm(dim)

        # Cross-Attn
        if cross_attention_dim is not None:
            self.attn2 = CrossAttention(
                query_dim=dim,
                cross_attention_dim=cross_attention_dim,
                heads=num_attention_heads,
                dim_head=attention_head_dim,
                dropout=dropout,
                bias=attention_bias,
                upcast_attention=upcast_attention,
            )
        else:
            self.attn2 = None

        if cross_attention_dim is not None:
            self.norm2 = AdaLayerNorm(dim, num_embeds_ada_norm) if self.use_ada_layer_norm else nn.LayerNorm(dim)
        else:
            self.norm2 = None

        # Feed-forward
        self.ff = FeedForward(dim, dropout=dropout, activation_fn=activation_fn)
        self.norm3 = nn.LayerNorm(dim)

        self.adapter_s = Adapter(dim, skip_connect=True)
        self.adapter_ffn = Adapter(dim, skip_connect=False)


        # nn.init.constant_(self.adapter_s.D_fc1.weight, 0)
        # nn.init.constant_(self.adapter_s.D_fc1.bias, 0)
        nn.init.constant_(self.adapter_s.D_fc2.weight, 0)
        nn.init.constant_(self.adapter_s.D_fc2.bias, 0)

        # nn.init.constant_(self.adapter_t.D_fc1.weight, 0)
        # nn.init.constant_(self.adapter_t.D_fc1.bias, 0)
        nn.init.constant_(self.adapter_ffn.D_fc2.weight, 0)
        nn.init.constant_(self.adapter_ffn.D_fc2.bias, 0)

        # Temp-Attn
        # self.attn_temp = CrossAttention(
        #     query_dim=dim,
        #     heads=num_attention_heads,
        #     dim_head=attention_head_dim,
        #     dropout=dropout,
        #     bias=attention_bias,
        #     upcast_attention=upcast_attention,
        # )
        # nn.init.zeros_(self.attn_temp.to_out[0].weight.data)
        # self.norm_temp = AdaLayerNorm(dim, num_embeds_ada_norm) if self.use_ada_layer_norm else nn.LayerNorm(dim)

    def set_use_memory_efficient_attention_xformers(self, use_memory_efficient_attention_xformers: bool):
        if not is_xformers_available():
            print("Here is how to install it")
            raise ModuleNotFoundError(
                "Refer to https://github.com/facebookresearch/xformers for more information on how to install"
                " xformers",
                name="xformers",
            )
        elif not torch.cuda.is_available():
            raise ValueError(
                "torch.cuda.is_available() should be True but is False. xformers' memory efficient attention is only"
                " available for GPU "
            )
        else:
            try:
                # Make sure we can run the memory efficient attention
                _ = xformers.ops.memory_efficient_attention(
                    torch.randn((1, 2, 40), device="cuda"),
                    torch.randn((1, 2, 40), device="cuda"),
                    torch.randn((1, 2, 40), device="cuda"),
                )
            except Exception as e:
                raise e
            self.attn1._use_memory_efficient_attention_xformers = use_memory_efficient_attention_xformers
            if self.attn2 is not None:
                self.attn2._use_memory_efficient_attention_xformers = use_memory_efficient_attention_xformers
            # self.attn_temp._use_memory_efficient_attention_xformers = use_memory_efficient_attention_xformers

    def forward(self, hidden_states, encoder_hidden_states=None, timestep=None, attention_mask=None, video_length=None):
        # SparseCausal-Attention
        # print('SC', hidden_states.shape)

        norm_hidden_states = (
            self.norm1(hidden_states, timestep) if self.use_ada_layer_norm else self.norm1(hidden_states)
        )

        if self.only_cross_attention:
            hidden_states = (
                self.attn1(norm_hidden_states, encoder_hidden_states, attention_mask=attention_mask) + hidden_states
            )
        else:
            hidden_states = self.adapter_s(self.attn1(norm_hidden_states, attention_mask=attention_mask, video_length=video_length)) + hidden_states

        # hidden_states = self.adapter_s(hidden_states)

        if self.attn2 is not None:
            # Cross-Attention
            norm_hidden_states = (
                self.norm2(hidden_states, timestep) if self.use_ada_layer_norm else self.norm2(hidden_states)
            )
            # print('attn2', norm_hidden_states.shape, encoder_hidden_states.shape)
            hidden_states = (
                self.attn2(
                    norm_hidden_states, encoder_hidden_states=encoder_hidden_states, attention_mask=attention_mask
                )
                + hidden_states
            )

        # Feed-forward
        hidden_states = self.ff(self.norm3(hidden_states)) + hidden_states + self.adapter_ffn(hidden_states)


        #reused temporal attention
        # d = hidden_states.shape[1]
        # hidden_states = rearrange(hidden_states, "(b f) d c -> (b d) f c", f=video_length)
        # norm_hidden_states = (
        #     self.norm1(hidden_states, timestep) if self.use_ada_layer_norm else self.norm1(hidden_states)
        # )
        # hidden_states = self.attn1(norm_hidden_states, is_temp = True) + hidden_states
        # hidden_states = self.adapter_t(hidden_states)
        # hidden_states = rearrange(hidden_states, "(b d) f c -> (b f) d c", d=d)

        # Temporal-Attention
        # d = hidden_states.shape[1]
        # hidden_states = rearrange(hidden_states, "(b f) d c -> (b d) f c", f=video_length)
        # norm_hidden_states = (
        #     self.norm_temp(hidden_states, timestep) if self.use_ada_layer_norm else self.norm_temp(hidden_states)
        # )
        # hidden_states = self.attn_temp(norm_hidden_states) + hidden_states
        # hidden_states = rearrange(hidden_states, "(b d) f c -> (b f) d c", d=d)

        return hidden_states


class SparseCausalAttention(CrossAttention):
    def forward(self, hidden_states, encoder_hidden_states=None, attention_mask=None, video_length=None, is_temp=False):
        batch_size, sequence_length, _ = hidden_states.shape

        encoder_hidden_states = encoder_hidden_states

        if self.group_norm is not None:
            hidden_states = self.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)
        # print('q',hidden_states.shape)

        query = self.to_q(hidden_states)
        # print('query', query.shape)
        dim = query.shape[-1]
        # print('before reshape', query.shape)
        query = self.reshape_heads_to_batch_dim(query)
        # print('after reshape', query.shape)

        if self.added_kv_proj_dim is not None:
            raise NotImplementedError

        encoder_hidden_states = encoder_hidden_states if encoder_hidden_states is not None else hidden_states

        if True:
            x = encoder_hidden_states.reshape(encoder_hidden_states.size(0), int(math.sqrt(encoder_hidden_states.size(1))), int(math.sqrt(encoder_hidden_states.size(1))), encoder_hidden_states.size(-1))
            # print(x.shape)
            x = rearrange(x,'(b f) h w c -> b f h w c', f = video_length)
            out = x.clone()
            out[:,4:,0::4, 0::4,:] = torch.roll(x[:,:,0::4, 0::4,: ], shifts=4, dims=1)[:,4:,:, : ]
            out[:,3:,0::4, 1::4,:] = torch.roll(x[:,:,0::4, 1::4,: ], shifts=3, dims=1)[:,3:,:, : ]
            out[:,2:,0::4, 2::4,:] = torch.roll(x[:,:,0::4, 2::4,: ], shifts=2, dims=1)[:,2:,:, : ]
            out[:,1:,0::4, 3::4,:] = torch.roll(x[:,:,0::4, 3::4,: ], shifts=1, dims=1)[:,1:,:, : ]

            out[:,3:,1::4, 0::4,:] = torch.roll(x[:,:,1::4, 0::4,: ], shifts=3, dims=1)[:,3:,:, : ]
            out[:,2:,1::4, 1::4,:] = torch.roll(x[:,:,1::4, 1::4,: ], shifts=2, dims=1)[:,2:,:, : ]
            out[:,1:,1::4, 2::4,:] = torch.roll(x[:,:,1::4, 2::4,: ], shifts=1, dims=1)[:,1:,:, : ]
            out[:,0:,1::4, 3::4,:] = torch.roll(x[:,:,1::4, 3::4,: ], shifts=0, dims=1)[:,0:,:, : ]

            out[:,2:,2::4, 0::4,:] = torch.roll(x[:,:,2::4, 0::4,: ], shifts=2, dims=1)[:,2:,:, : ]
            out[:,1:,2::4, 1::4,:] = torch.roll(x[:,:,2::4, 1::4,: ], shifts=1, dims=1)[:,1:,:, : ]
            out[:,0:,2::4, 2::4,:] = torch.roll(x[:,:,2::4, 2::4,: ], shifts=0, dims=1)[:,0:,:, : ]
            out[:,0:,2::4, 3::4,:] = torch.roll(x[:,:,2::4, 3::4,: ], shifts=0, dims=1)[:,0:,:, : ]

            out[:,1:,3::4, 0::4,:] = torch.roll(x[:,:,3::4, 0::4,: ], shifts=1, dims=1)[:,1:,:, : ]
            out[:,2:,3::4, 1::4,:] = torch.roll(x[:,:,3::4, 1::4,: ], shifts=2, dims=1)[:,2:,:, : ]
            out[:,3:,3::4, 2::4,:] = torch.roll(x[:,:,3::4, 2::4,: ], shifts=3, dims=1)[:,3:,:, : ]
            out[:,4:,3::4, 3::4,:] = torch.roll(x[:,:,3::4, 3::4,: ], shifts=4, dims=1)[:,4:,:, : ]
            out = rearrange(out,'b f h w c -> (b f) h w c', f = video_length)
            out = out.reshape(out.size(0), out.size(1)*out.size(2), out.size(-1))
            encoder_hidden_states = out

        key = self.to_k(encoder_hidden_states)
        value = self.to_v(encoder_hidden_states)
        # print('key', key.shape)
        # print('value', value.shape)

        former_frame_index = torch.arange(video_length)  -1
        former_frame_index[0] = 0

        now_frame_index = torch.arange(video_length)

        key = rearrange(key, "(b f) d c -> b f d c", f=video_length)
        # # print('key_re', key.shape)
        key = torch.cat([key[:, former_frame_index], key[:, now_frame_index]], dim=2)
        # # print('key_cat', key.shape)
        key = rearrange(key, "b f d c -> (b f) d c")
        # # print('key_af', key.shape)


        value = rearrange(value, "(b f) d c -> b f d c", f=video_length)
        value = torch.cat([value[:, former_frame_index], value[:, now_frame_index]], dim=2)
        value = rearrange(value, "b f d c -> (b f) d c")

        key = self.reshape_heads_to_batch_dim(key)
        value = self.reshape_heads_to_batch_dim(value)

        if attention_mask is not None:
            if attention_mask.shape[-1] != query.shape[1]:
                target_length = query.shape[1]
                attention_mask = F.pad(attention_mask, (0, target_length), value=0.0)
                attention_mask = attention_mask.repeat_interleave(self.heads, dim=0)

        # attention, what we cannot get enough of
        if self._use_memory_efficient_attention_xformers:
            hidden_states = self._memory_efficient_attention_xformers(query, key, value, attention_mask)
            # Some versions of xformers return output in fp32, cast it back to the dtype of the input
            hidden_states = hidden_states.to(query.dtype)
        else:
            if self._slice_size is None or query.shape[0] // self._slice_size == 1:
                hidden_states = self._attention(query, key, value, attention_mask)
            else:
                hidden_states = self._sliced_attention(query, key, value, sequence_length, dim, attention_mask)

        # linear proj
        hidden_states = self.to_out[0](hidden_states)

        # dropout
        hidden_states = self.to_out[1](hidden_states)
        return hidden_states
