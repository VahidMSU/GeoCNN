import torch
import torch.nn as nn
class PatchEmbedding(nn.Module):
    def __init__(self, in_channels, embed_dim, patch_size):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        # Replace BatchNorm2d with GroupNorm
        self.norm = nn.GroupNorm(num_groups=32, num_channels=embed_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        batch_size, time_steps, channels, height, width = x.size()
        x = x.reshape(batch_size * time_steps, channels, height, width)
        x = self.proj(x)
        x = self.norm(x)
        x = x.flatten(2).transpose(1, 2)
        x = x.reshape(batch_size, time_steps, -1, x.size(-1))
        x = self.dropout(x)
        return x


class TransformerLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_dim, dropout):
        super().__init__()
        self.layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=mlp_dim,
            dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.norm(x + self.dropout(self.layer(x)))
        return x
import torch
import torch.nn as nn
import math

class RelativePositionalEmbedding(nn.Module):
    def __init__(self, num_heads, head_dim, max_seq_len=512):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.rel_pos_bias = nn.Parameter(torch.zeros(num_heads, 2*max_seq_len-1, 2*max_seq_len-1))
        
    def forward(self, x):
        return x + self.rel_pos_bias

class CrossAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query, key, value):
        batch_size, seq_len, embed_dim = query.size()
        
        # Split into multiple heads
        query = self.query(query).view(batch_size, seq_len, self.num_heads, self.head_dim)
        key = self.key(key).view(batch_size, seq_len, self.num_heads, self.head_dim)
        value = self.value(value).view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # Transpose for multi-head attention
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        
        # Compute attention scores
        attention_scores = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(self.head_dim)
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.dropout(attention_probs)
        
        # Apply attention
        context = torch.matmul(attention_probs, value)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)
        
        return self.out_proj(context)

class VisionTransformerForRegression(nn.Module):
    def __init__(self, in_channels=20, embed_dim=1024, num_heads=8, depth=4, 
                 mlp_dim=3072, patch_size=16, dropout=0.1, batch_window=64, num_classes=1):
        super().__init__()
        
        # Existing patch embedding and positional embedding
        self.patch_embed = PatchEmbedding(in_channels, embed_dim, patch_size)
        
        # Enhanced cross-attention mechanisms
        self.spatial_cross_attention = CrossAttention(embed_dim, num_heads, dropout)
        self.temporal_cross_attention = CrossAttention(embed_dim, num_heads, dropout)
        
        # Relative positional embedding
        self.relative_pos_embedding = RelativePositionalEmbedding(num_heads, embed_dim // num_heads)
        
        # Existing transformer layers with cross-attention integration
        self.spatial_transformer = nn.ModuleList([
            nn.Sequential(
                TransformerLayer(embed_dim, num_heads, mlp_dim, dropout),
                CrossAttentionBlock(embed_dim, num_heads, dropout)
            ) for _ in range(depth)
        ])
        
        self.temporal_transformer = nn.ModuleList([
            nn.Sequential(
                TransformerLayer(embed_dim, num_heads, mlp_dim, dropout),
                CrossAttentionBlock(embed_dim, num_heads, dropout)
            ) for _ in range(depth)
        ])
        
        # Rest of the existing implementation remains the same
        # Regression head
        
        # Regression head with GroupNorm
        self.regression_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.ConvTranspose2d(1, 128, kernel_size=2, stride=1, padding=1),
            # Replace BatchNorm2d with GroupNorm
            nn.GroupNorm(num_groups=32, num_channels=128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=7, stride=2, padding=1),
            # Replace BatchNorm2d with GroupNorm
            nn.GroupNorm(num_groups=32, num_channels=64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 1, kernel_size=2, stride=1, padding=1),
            nn.Sigmoid()
        )      
    def forward(self, dynamic, static, categorical):
        x = torch.cat((dynamic, static, categorical), dim=2)
        
        batch_size, time_steps, channels, height, width = x.size()
        
        # Patch embedding
        x = self.patch_embed(x)
        
        # Spatial processing with cross-attention
        spatial_outputs = []
        for t in range(time_steps):
            spatial_x = x[:, t, :, :]
            
            # Apply spatial transformer with cross-attention
            for layer in self.spatial_transformer:
                spatial_x = layer[0](spatial_x)  # Standard transformer layer
                spatial_x = layer[1](spatial_x, spatial_x, spatial_x)  # Cross-attention
            
            spatial_outputs.append(spatial_x)
        
        # Stack and process temporally
        x = torch.stack(spatial_outputs, dim=1)
        x = x.reshape(batch_size * time_steps, -1, x.size(-1))
        
        # Temporal processing with cross-attention
        for layer in self.temporal_transformer:
            x = layer[0](x)  # Standard transformer layer
            x = layer[1](x, x, x)  # Cross-attention
        
        # Regression head (existing implementation)
        x = self.regression_head[:4](x)
        x = x.reshape(batch_size * time_steps, 1, width//2, height//2)
        x = self.regression_head[4:](x)
        x = x.reshape(batch_size, time_steps, 1, width, height)
        
        return x

class CrossAttentionBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout):
        super().__init__()
        self.cross_attention = CrossAttention(embed_dim, num_heads, dropout)
        self.norm = nn.LayerNorm(embed_dim)
        
    def forward(self, query, key, value):
        # Cross-attention with residual connection and layer normalization
        x = self.cross_attention(query, key, value)
        return self.norm(x + query)

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
        "dropout": 0.1,

    }

    # Setup GPU device
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True  # Optimize performance for fixed input sizes

    # Input tensor
    x = torch.rand(2, 24, vit_config["in_channels"], 64, 64).to(device)

    # Initialize model
    model = VisionTransformerForRegression(**vit_config).to(device)

    # Use mixed precision for inference
    with torch.amp.autocast('cuda'):
        output = model(x)  # Output tensor of shape (2, 24, 1, 128, 128)

    print(output.shape)
    print(f"Total parameters: {count_parameters(model):,}")
