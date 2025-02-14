import torch
import torch.nn as nn
from einops import rearrange


class FourierPositionalEncoding(nn.Module):
    def __init__(self, embed_dim):
        super(FourierPositionalEncoding, self).__init__()
        self.embed_dim = embed_dim
        freqs = torch.pow(10000, -torch.arange(0, embed_dim, 2).float() / embed_dim)
        self.register_buffer('freqs', freqs)

    def forward(self, x):
        batch_size, time_steps, _ = x.size()
        pos = torch.arange(0, time_steps, dtype=torch.float32).unsqueeze(1).to(x.device)
        sinusoid_in = pos * self.freqs.unsqueeze(0)
        pe = torch.cat([torch.sin(sinusoid_in), torch.cos(sinusoid_in)], dim=-1)
        return x + pe.unsqueeze(0).repeat(batch_size, 1, 1)
    

class ViTBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_dim, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, embed_dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        # x: (batch_size*time_steps, num_patches, embed_dim)
        x = self.norm1(x + self.attn(x, x, x, need_weights=False)[0])
        return self.norm2(x + self.mlp(x))
    
class TemporalConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        self.conv = nn.Conv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(kernel_size, 1, 1),
            padding=(kernel_size // 2, 0, 0)
        )
    
    def forward(self, x):
        # x: (batch, time_steps, channels, height, width)
        # Rearrange to match Conv3D input format: (batch, channels, time_steps, height, width)
        x = rearrange(x, 'b t c h w -> b c t h w')
        x = self.conv(x)
        return rearrange(x, 'b c t h w -> b t c h w')  # Revert to original shape


class CrossAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
    
    def forward(self, temporal, spatial):
        # temporal: (batch, time_steps, embed_dim)
        # spatial: (batch, num_patches, embed_dim)
        return self.attn(temporal, spatial, spatial)[0]

class TemporalPositionalEncoding(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim

    def forward(self, x):
        # x: (batch, time_steps, num_patches, embed_dim)
        batch, time_steps, num_patches, embed_dim = x.shape
        assert embed_dim == self.embed_dim, "Embedding dimension mismatch"
        temporal_pos = nn.Parameter(torch.randn(time_steps, embed_dim, requires_grad=True)).to(x.device)
        return x + temporal_pos.unsqueeze(0).unsqueeze(2)


class VisionTransformer(nn.Module):
    def __init__(self, in_channels, embed_dim, num_heads, depth, mlp_dim, num_classes=1):
        super().__init__()
        
        # Spatial and Temporal Components
        self.spatial_projection = nn.Linear(in_channels, embed_dim)  # Project spatial features to embed_dim
        self.spatial_transformer = nn.ModuleList([
            ViTBlock(embed_dim, num_heads, mlp_dim) for _ in range(depth)
        ])
        self.temporal_pos = FourierPositionalEncoding(embed_dim)  # Fourier positional encoding
        self.temporal_transformer = nn.ModuleList([
            ViTBlock(embed_dim, num_heads, mlp_dim) for _ in range(depth)
        ])
        
        # Reconstruction to spatial dimensions
        self.reconstruct = nn.Sequential(
            nn.Conv2d(embed_dim, num_classes, kernel_size=1),  # Restore spatial structure
            nn.Sigmoid()  # Optional normalization
        )

    def forward(self, dynamic, static, categorical):
        x = torch.cat((dynamic, static, categorical), dim=2)
        batch, time_steps, channels, height, width = x.shape

        # Project spatial dimensions and transform spatially
        x = rearrange(x, 'b t c h w -> (b t) (h w) c')  # Flatten spatial dimensions
        x = self.spatial_projection(x)  # Project to embed_dim
        for block in self.spatial_transformer:
            x = block(x)
        x = rearrange(x, '(b t) p e -> b t p e', b=batch, t=time_steps)  # Reshape to (batch, time_steps, patches, embed_dim)

        # Add Fourier positional encoding for temporal modeling
        x = rearrange(x, 'b t p e -> (b p) t e')  # Flatten patches into batch dimension
        x = self.temporal_pos(x)  # Add temporal positional encoding
        x = rearrange(x, '(b p) t e -> b t p e', b=batch)  # Reshape back

        # Transform temporally
        for block in self.temporal_transformer:
            x = rearrange(x, 'b t p e -> (b p) t e')  # Flatten patches for temporal transformer
            x = block(x)
            x = rearrange(x, '(b p) t e -> b t p e', b=batch)  # Reshape back

        # Restore spatial dimensions for reconstruction
        spatial_dim = int((x.size(2)) ** 0.5)  # Calculate spatial height and width
        x = rearrange(x, 'b t (h w) e -> (b t) e h w', h=spatial_dim, w=spatial_dim)

        # Reconstruct output
        x = self.reconstruct(x)  # Shape: (batch * time_steps, num_classes, height, width)

        # Restore batch and time dimensions
        x = rearrange(x, '(b t) c h w -> b t c h w', b=batch, t=time_steps)

        return x



def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# Example usage
if __name__ == "__main__":

    vit_config = {
        "in_channels": 28,
        "embed_dim": 512,
        "num_heads": 8,
        "depth": 6,
        "mlp_dim": 128,
        "num_classes": 1,
    }

    # Setup GPU device
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True  # Optimize performance for fixed input sizes

    # Input tensor
    x = torch.rand(2, 24, vit_config["in_channels"], 64, 64).to(device)

    # Initialize model
    model = VisionTransformer(**vit_config).to(device)

    # Use mixed precision for inference
    with torch.amp.autocast('cuda'):
        output = model(x)  # Output tensor of shape (2, 24, 1, 128, 128)

    print(output.shape)
    print(f"Total parameters: {count_parameters(model):,}")
