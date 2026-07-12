"""
MTA-Swin with Component-level Control
Complete implementation with all fixes applied
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from timm.layers import DropPath, to_2tuple, trunc_normal_
import math

# Import fused window process (if available)
try:
    import os, sys
    kernel_path = os.path.abspath(os.path.join('..'))
    sys.path.append(kernel_path)
    from kernels.window_process.window_process import WindowProcess, WindowProcessReverse
except:
    WindowProcess = None
    WindowProcessReverse = None
    print("[Warning] Fused window process have not been installed. Please refer to get_started.md for installation.")

# Common Components
def get_valid_groups(num_features, max_groups=8):
    """Get maximum number of groups that can evenly divide num_features"""
    g = min(max_groups, num_features)
    while g > 1 and (num_features % g != 0):
        g -= 1
    return max(1, g)

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

def window_partition(x, window_size):
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows

def window_reverse(windows, window_size, H, W):
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

# MTA Components
class DepthScaledGroupNorm(nn.Module):
    def __init__(self, num_features, layer_depth=1, max_depth=24, num_groups=None):
        super().__init__()
        self.num_features = num_features
        self.layer_depth = layer_depth
        self.max_depth = max_depth
        
        # Ensure GroupNorm group count can evenly divide num_features
        if num_groups is None:
            num_groups = get_valid_groups(num_features, max_groups=8)
        
        self.group_norm = nn.GroupNorm(
            num_groups=num_groups, 
            num_channels=num_features
        )
        
        base_scale = layer_depth / max_depth
        # Store parameter in log space to ensure positivity after exp()
        self.log_depth_scale = nn.Parameter(
            torch.log(torch.tensor(base_scale, dtype=torch.float32))
        )
    
    def forward(self, x):
        B, N, C = x.shape
        x_reshaped = x.transpose(1, 2)
        normed = self.group_norm(x_reshaped)
        normed = normed.transpose(1, 2)
        # Use torch.exp() to recover parameter from log space, ensuring scale > 0
        scaled = normed * torch.exp(self.log_depth_scale)
        return scaled

class WindowMultiTokenAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0.,
                 cq=5, ck=11, ch=2, layer_depth=1, max_depth=24,
                 use_qk_conv=True, use_head_mixing=True, use_group_norm=True):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.cq = cq
        self.ck = ck
        self.ch = ch
        
        # Component switches
        self.use_qk_conv = use_qk_conv
        self.use_head_mixing = use_head_mixing
        self.use_group_norm = use_group_norm
        
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # Relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))

        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        # QKV projection
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        if self.use_qk_conv:
            pad_q = (cq - 1) // 2
            pad_k = (ck - 1) // 2
            
            # Define grouped conv layer for vectorized computation
            self.key_query_conv = nn.Conv2d(
                in_channels=num_heads, out_channels=num_heads,
                kernel_size=(cq, ck), padding=(pad_q, pad_k),
                groups=num_heads, bias=False
            )
            # Initialize weights equivalently
            with torch.no_grad():
                self.key_query_conv.weight.zero_()
                # Shape: (out_channels, in_channels // groups, kH, kW) -> (num_heads, 1, cq, ck)
                center_h, center_w = self.key_query_conv.weight.shape[2]//2, self.key_query_conv.weight.shape[3]//2
                for h in range(num_heads):
                    self.key_query_conv.weight[h, 0, center_h, center_w] = 1.0
        else:
            self.key_query_conv = None

        # MTA Component 2: Head Mixing
        if self.use_head_mixing and num_heads % ch == 0:
            self.num_groups = num_heads // ch
            self.head_mixing = nn.Conv2d(
                in_channels=num_heads, out_channels=num_heads,
                kernel_size=1, groups=self.num_groups, bias=False
            )
            self._init_head_mixing()
        else:
            self.head_mixing = None

        # MTA Component 3: Group Normalization
        if self.use_group_norm:
            self.head_group_norms = nn.ModuleList([
                DepthScaledGroupNorm(
                    num_features=head_dim, layer_depth=layer_depth,
                    max_depth=max_depth, num_groups=get_valid_groups(head_dim)
                ) for _ in range(num_heads)
            ])
        else:
            self.head_group_norms = None

        trunc_normal_(self.relative_position_bias_table, std=.02)

    def _init_head_mixing(self):
        if self.head_mixing is None:
            return
        with torch.no_grad():
            weight = self.head_mixing.weight
            weight.zero_()
            for group_idx in range(self.num_groups):
                start_head = group_idx * self.ch
                for local_out in range(self.ch):
                    for local_in in range(self.ch):
                        global_out = start_head + local_out
                        if local_out == local_in:
                            weight[global_out, local_in, 0, 0] = 1.0

    def forward(self, x, mask=None):
        B_, N, C = x.shape
        head_dim = C // self.num_heads

        # QKV projection
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Basic attention
        q = q * self.scale
        attn_logits = (q @ k.transpose(-2, -1))

        # Relative position bias
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn_logits = attn_logits + relative_position_bias.unsqueeze(0)

        if self.use_qk_conv and self.key_query_conv is not None:
            # attn_logits shape (B_, num_heads, N, N) directly feed into grouped conv
            enhanced_attn_logits = self.key_query_conv(attn_logits)
        else:
            enhanced_attn_logits = attn_logits

        # Handle mask
        if mask is not None:
            nW = mask.shape[0]
            B_batch = B_ // nW
            try:
                enhanced_attn_logits = enhanced_attn_logits.view(B_batch, nW, self.num_heads, N, N)
                enhanced_attn_logits = enhanced_attn_logits + mask.unsqueeze(1).unsqueeze(0)
                enhanced_attn_logits = enhanced_attn_logits.view(-1, self.num_heads, N, N)
            except RuntimeError:
                print(f"Error applying mask. Logits shape: {enhanced_attn_logits.shape}, Mask shape: {mask.shape}")

        # (If enabled) Apply Head Mixing to Logits first
        if self.use_head_mixing and self.head_mixing is not None:
            mixed_logits = self.head_mixing(enhanced_attn_logits)
        else:
            mixed_logits = enhanced_attn_logits  # If disabled, use original logits directly

        # Apply Softmax to mixed logits
        attn_weights = torch.nn.functional.softmax(mixed_logits, dim=-1)
        
        # Dropout
        attn_weights = self.attn_drop(attn_weights)

        # Complete attention computation for all heads at once
        # (B_, H, N, N) @ (B_, H, N, D) -> (B_, H, N, D)
        x = torch.matmul(attn_weights, v)

        B_, H, N, D = x.shape

        # If GroupNorm is enabled, use functional interface for vectorized processing
        if self.use_group_norm and self.head_group_norms is not None:
            # Reshape tensor to match GroupNorm (B, C, L) input format
            # (B_, H, N, D) -> (B_, H, D, N) -> (B_, H*D, N)
            x = x.permute(0, 1, 3, 2).reshape(B_, H * D, N)
            
            # Concatenate independent parameters from all heads in ModuleList
            groups_each = self.head_group_norms[0].group_norm.num_groups
            # Concatenate gamma and beta
            w = torch.cat([m.group_norm.weight for m in self.head_group_norms], dim=0)
            b = torch.cat([m.group_norm.bias for m in self.head_group_norms], dim=0)

            # Call functional group_norm, expand groups by H times
            x = F.group_norm(x, num_groups=H * groups_each, weight=w, bias=b, eps=1e-5)
            
            # Concatenate and apply independent depth scaling factors for each head
            # (B_, H*D, N) -> (B_, H, D, N) -> (B_, H, N, D)
            x = x.view(B_, H, D, N).permute(0, 1, 3, 2)
            scales = torch.exp(torch.stack(
                [m.log_depth_scale for m in self.head_group_norms]
            )).view(1, H, 1, 1)
            x = x * scales

        # Concatenate features from all heads
        # (B_, H, N, D) -> (B_, N, H, D) -> (B_, N, H*D)
        output = x.transpose(1, 2).reshape(B_, N, H * D)

        # Final projection and dropout same as original
        output = self.proj(output)
        output = self.proj_drop(output)
        return output

class WindowMTABlock(nn.Module):
    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, cq=5, ck=11, ch=2,
                 layer_depth=1, max_depth=24, fused_window_process=False,
                 use_qk_conv=True, use_head_mixing=True, use_group_norm=True):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.fused_window_process = fused_window_process

        if min(self.input_resolution) <= self.window_size:
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"
            
        self.norm1 = norm_layer(dim)
        
        self.attn = WindowMultiTokenAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop,
            cq=cq, ck=ck, ch=ch, layer_depth=layer_depth, max_depth=max_depth,
            use_qk_conv=use_qk_conv, use_head_mixing=use_head_mixing, use_group_norm=use_group_norm)
        
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        
        # Shifted window masking
        if self.shift_size > 0:
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))
            h_slices = (slice(0, -self.window_size),
                       slice(-self.window_size, -self.shift_size),
                       slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                       slice(-self.window_size, -self.shift_size),
                       slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1
            mask_windows = window_partition(img_mask, self.window_size)
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-1e4)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None
        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        
        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)
        
        # Cyclic shift
        if self.shift_size > 0:
            if not self.fused_window_process:
                shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
                x_windows = window_partition(shifted_x, self.window_size)
            else:
                x_windows = WindowProcess.apply(x, B, H, W, C, -self.shift_size, self.window_size)
        else:
            shifted_x = x
            x_windows = window_partition(shifted_x, self.window_size)
            
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)
        
        # Window attention
        attn_windows = self.attn(x_windows, mask=self.attn_mask)
        
        # Merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        
        # Reverse cyclic shift
        if self.shift_size > 0:
            if not self.fused_window_process:
                shifted_x = window_reverse(attn_windows, self.window_size, H, W)
                x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
            else:
                x = WindowProcessReverse.apply(attn_windows, B, H, W, C, self.shift_size, self.window_size)
        else:
            shifted_x = window_reverse(attn_windows, self.window_size, H, W)
            x = shifted_x
            
        x = x.view(B, H * W, C)
        x = shortcut + self.drop_path(x)
        
        # FFN
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class MTABasicLayer(nn.Module):
    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False,
                 stage_idx=0, cq=5, ck=11, ch=2, max_depth=24, fused_window_process=False,
                 depth_offset=0,  # New depth_offset parameter
                 use_qk_conv=True, use_head_mixing=True, use_group_norm=True):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.stage_idx = stage_idx
        self.depth_offset = depth_offset

        # Check if Head Mixing is truly enabled
        hm_enabled = use_head_mixing and (num_heads % ch == 0)
        
        components = []
        if use_qk_conv: components.append("QK")
        if hm_enabled: 
            components.append("HM")
        elif use_head_mixing:
            components.append("HM*off")  # Want to use but conditions not met
        if use_group_norm: components.append("GN")
        
        comp_str = "+".join(components) if components else "Vanilla"
        print(f"Stage {stage_idx}: {comp_str}")

        # Build blocks
        self.blocks = nn.ModuleList()
        for i in range(depth):
            shift_size = 0 if (i % 2 == 0) else window_size // 2
            block_depth = self.depth_offset + i + 1  # Use correct depth calculation
            
            block = WindowMTABlock(
                dim=dim, input_resolution=input_resolution, num_heads=num_heads,
                window_size=window_size, shift_size=shift_size, mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop, attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer, cq=cq, ck=ck, ch=ch,
                layer_depth=block_depth, max_depth=max_depth,
                fused_window_process=fused_window_process,
                use_qk_conv=use_qk_conv, use_head_mixing=use_head_mixing, use_group_norm=use_group_norm
            )
            self.blocks.append(block)

        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x

class PatchMerging(nn.Module):
    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], -1)
        x = x.view(B, -1, 4 * C)
        x = self.norm(x)
        x = self.reduction(x)
        return x

class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]
        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        if self.norm is not None:
            x = self.norm(x)
        return x

# Main Model
class MTASwin(nn.Module):
    def __init__(self, img_size=224, patch_size=4, in_chans=3, num_classes=1000,
                 embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
                 window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False, fused_window_process=False, cq=5, ck=11, ch=2,
                 # Default switches aligned with target configuration
                 stage_qk_conv=[True, True, False, False],
                 stage_head_mixing=[False, False, True, True], 
                 stage_group_norm=[True, True, False, False],
                 **kwargs):
        super().__init__()

        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio

        # Validate configuration lengths
        assert len(stage_qk_conv) == 4, "stage_qk_conv must have 4 elements"
        assert len(stage_head_mixing) == 4, "stage_head_mixing must have 4 elements"
        assert len(stage_group_norm) == 4, "stage_group_norm must have 4 elements"

        print(f"MTA Configuration:")
        print(f"  QK Conv:      {stage_qk_conv}")
        print(f"  Head Mixing:  {stage_head_mixing}")
        print(f"  Group Norm:   {stage_group_norm}")

        # Patch embedding
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # Absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # Stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        max_depth = sum(depths)

        # Build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            # Calculate correct depth_offset
            depth_offset = sum(depths[:i_layer])
            
            layer = MTABasicLayer(
                dim=int(embed_dim * 2 ** i_layer),
                input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                patches_resolution[1] // (2 ** i_layer)),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                use_checkpoint=use_checkpoint,
                stage_idx=i_layer,
                cq=cq, ck=ck, ch=ch,
                max_depth=max_depth,
                fused_window_process=fused_window_process,
                depth_offset=depth_offset,  # Pass correct depth offset
                use_qk_conv=stage_qk_conv[i_layer],
                use_head_mixing=stage_head_mixing[i_layer],
                use_group_norm=stage_group_norm[i_layer]
            )
            self.layers.append(layer)

        self.norm = norm_layer(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def forward_features(self, x):
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)
        x = self.avgpool(x.transpose(1, 2))
        x = torch.flatten(x, 1)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x

# Factory Functions
def create_mta_swin(num_classes=1000, model_size='tiny', **kwargs):
    """Create MTA-Swin model"""
    model_configs = {
        'tiny': dict(embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24]),
        'small': dict(embed_dim=96, depths=[2, 2, 18, 2], num_heads=[3, 6, 12, 24]),
        'base': dict(embed_dim=128, depths=[2, 2, 18, 2], num_heads=[4, 8, 16, 32]),
        'large': dict(embed_dim=192, depths=[2, 2, 18, 2], num_heads=[6, 12, 24, 48])
    }
    
    if model_size not in model_configs:
        raise ValueError(f"Unknown model_size: {model_size}")
    
    config = model_configs[model_size]
    params = {
        'num_classes': num_classes,
        'img_size': 224,
        'patch_size': 4,
        'in_chans': 3,
        'window_size': 7,
        'mlp_ratio': 4.0,
        'cq': 6, 'ck': 11, 'ch': 2,
        **config,
        **kwargs
    }
    
    return MTASwin(**params)

def create_mta_swin_tiny(num_classes=1000, **kwargs):
    return create_mta_swin(num_classes, model_size='tiny', **kwargs)

def create_mta_swin_small(num_classes=1000, **kwargs):
    return create_mta_swin(num_classes, model_size='small', **kwargs)

def create_mta_swin_base(num_classes=1000, **kwargs):
    return create_mta_swin(num_classes, model_size='base', **kwargs)

def create_mta_swin_large(num_classes=1000, **kwargs):
    return create_mta_swin(num_classes, model_size='large', **kwargs)