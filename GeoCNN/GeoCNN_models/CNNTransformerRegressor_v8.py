import torch
import torch.nn as nn
import math
from torchsummary import summary
import torch.autograd.profiler as autograd_profiler
import torch.profiler as profiler
import os
import torch.nn.functional as F 


class UpBlockWithDeformableConv(nn.Module):
    def __init__(self, in_channels, out_channels, dropout, upscale_factor=2):
        super(UpBlockWithDeformableConv, self).__init__()
        self.up = SubPixelConv(in_channels, out_channels, upscale_factor)
        self.deformable_conv = nn.Conv2d(out_channels + out_channels, out_channels, kernel_size=3, padding="same")
        self.relu = nn.ReLU()
        self.attention = SqueezeExcitation(out_channels * 2, reduction=16, dropout=dropout)  # Attention for skip

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)  # Concatenate along the channel axis
        x = self.attention(x)  # Apply attention to combined features

        # Apply deformable convolution
        x = self.deformable_conv(x)
        x = self.relu(x)


        return x

class SubPixelConv(nn.Module):
    def __init__(self, in_channels, out_channels, upscale_factor):
        super(SubPixelConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels * (upscale_factor ** 2), kernel_size=3, padding='same')
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)
        self.conv_after_shuffle = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding='same')


    def forward(self, x):
        x = self.pixel_shuffle(self.conv(x))
        x = self.conv_after_shuffle(x)
        return x


class TemporalMultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super(TemporalMultiHeadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim)
        self.attention_dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.out_dropout = nn.Dropout(dropout)

    def forward(self, x):
        batch_size, seq_len, embed_dim = x.size()
        qkv = self.qkv_proj(x)
        qkv = qkv.reshape(batch_size, seq_len, self.num_heads, 3 * self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1)  # [num_heads, batch_size, 3*head_dim, seq_len]
        q, k, v = torch.chunk(qkv, 3, dim=2)

        attn_scores = torch.matmul(q.transpose(-2, -1), k) / (self.head_dim ** 0.5)
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = self.attention_dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, v.transpose(-2, -1))
        attn_output = attn_output.permute(1, 3, 0, 2).reshape(batch_size, seq_len, embed_dim)
        output = self.out_proj(attn_output)
        return self.out_dropout(output)

class TransformerEncoderWithTemporalAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, forward_expansion, dropout, num_layers):
        super(TransformerEncoderWithTemporalAttention, self).__init__()
        self.pos_encoder = FourierPositionalEncoding(embed_dim)

        self.layers = nn.ModuleList([
            nn.ModuleDict({
                
                'attention': TemporalMultiHeadAttention(embed_dim, num_heads, dropout),
                'feedforward': nn.Sequential(
                    nn.Linear(embed_dim, forward_expansion * embed_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(forward_expansion * embed_dim, embed_dim),
                    nn.Dropout(dropout)
                ),

                'norm1': nn.LayerNorm(embed_dim),
                'norm2': nn.LayerNorm(embed_dim)
            })
            for _ in range(num_layers)
        ])

    def forward(self, x):
        x = self.pos_encoder(x)
        for layer in self.layers:
            attn_out = layer['attention'](x)
            x = layer['norm1'](x + attn_out)
            ff_out = layer['feedforward'](x)
            x = layer['norm2'](x + ff_out)
        return x


# Optimized Positional Encoding with Fourier Features
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
    

class ConvBlockWithSE(nn.Module):
    def __init__(self, in_channels, out_channels, dropout):
        super(ConvBlockWithSE, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding="same")
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding="same")
        self.relu = nn.ReLU()
        self.spatial_attention = nn.Conv2d(out_channels, 1, kernel_size=1, padding="same")
        self.channel_attention = SqueezeExcitation(out_channels, out_channels // 16, dropout)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)

        # Spatial attention
        spatial_mask = torch.sigmoid(self.spatial_attention(x))
        x = x * spatial_mask

        # Channel attention
        x = self.channel_attention(x)

        return x


# Squeeze-and-Excitation Block for Channel Attention
class SqueezeExcitation(nn.Module):
    def __init__(self, in_channels, reduction, dropout):
        super(SqueezeExcitation, self).__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1)
        self.relu = nn.ReLU()
        self.droupout = nn.Dropout(dropout)  
        self.fc2 = nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        scale = self.global_avg_pool(x)
        scale = self.fc1(scale)
        scale = self.relu(scale)
        scale = self.droupout(scale)
        scale = self.fc2(scale)
        scale = self.sigmoid(scale)

        return x * scale

class DownBlockWithSE(nn.Module):
    def __init__(self, in_channels, out_channels, dropout):
        super(DownBlockWithSE, self).__init__()
        self.conv1 = ConvBlockWithSE(in_channels, out_channels, dropout)
        self.conv2 = ConvBlockWithSE(out_channels, out_channels, dropout)
        self.coord_attention = CoordinateAttention(out_channels, out_channels)
        self.downsample = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1)  # Strided convolution

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.coord_attention(x)  # Apply Coordinate Attention
        p = self.downsample(x)  # Strided convolution for downsampling
        return x, p  # Return conv features and pooled output for skip conn

class ConvBlockWithDeformable(nn.Module):
    def __init__(self, in_channels, out_channels, dropout):
        super(ConvBlockWithDeformable, self).__init__()
        self.deformable_conv = nn.Conv2d(in_channels, out_channels, kernel_size=3,  padding="same")
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.deformable_conv(x)
        x = self.relu(x)
        return x


class CoordinateAttention(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CoordinateAttention, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        n, c, h, w = x.size()
        x_h = self.pool_h(x).permute(0, 1, 3, 2)
        x_w = self.pool_w(x)
        x_h = self.conv1(x_h)
        x_w = self.conv2(x_w)
        out = self.sigmoid(x_h + x_w)
        return x * out

class CNNTransformerRegressor_v8(nn.Module):
    def __init__(self, num_channels, num_categorical_channels, num_static_channels, num_dynamic_channels, embed_dim=1024, num_heads=8, forward_expansion=4, dropout=0.3, num_layers=6, num_classes=1):
        super(CNNTransformerRegressor_v8, self).__init__()
        # Encoder (Downsampling)
        self.down1 = DownBlockWithSE(num_channels, 64, dropout)
        self.down2 = DownBlockWithSE(64, 128, dropout)
        self.down3 = DownBlockWithSE(128, 256, dropout)
        self.down4 = DownBlockWithSE(256, 512, dropout)

        self.bottleneck = ConvBlockWithDeformable(512, embed_dim, dropout)

        # Transformer Encoder with positional encoding
        self.transformer = TransformerEncoderWithTemporalAttention(embed_dim=embed_dim,num_heads=num_heads,forward_expansion=forward_expansion,dropout=dropout,num_layers=num_layers)     

        self.up1 = UpBlockWithDeformableConv(embed_dim, 512, dropout)
        self.up2 = UpBlockWithDeformableConv(512, 256, dropout)
        self.up3 = UpBlockWithDeformableConv(256, 128, dropout)
        self.up4 = UpBlockWithDeformableConv(128, 64, dropout)

        self.num_classes = num_classes
        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)
        
    def forward(self, dynamic_tensor=None, static_tensor=None, categorical_tensor=None, x=None):
        if x is None:
            x = torch.cat([dynamic_tensor, static_tensor, categorical_tensor], dim=2)
        batch_size, time_steps, channels, height, width = x.shape  # Expect 5D input: [batch_size, time_steps, channels, height, width]
        
        cnn_features = []
        skip_connections = []

        # Apply CNNs to each time step independently
        for t in range(time_steps):
            x_t = x[:, t, :, :, :]  # Extract input for time step t
            skip1, x_t = self.down1(x_t)
            skip2, x_t = self.down2(x_t)
            skip3, x_t = self.down3(x_t)
            skip4, x_t = self.down4(x_t)
            # Bottleneck
            x_t = self.bottleneck(x_t)
            cnn_features.append(x_t)
            skip_connections.append((skip1, skip2, skip3, skip4))

        # Stack the features along the time dimension
        x = torch.stack(cnn_features, dim=1)

        # Apply spatial attention
        batch_size, time_steps, embed_dim, height, width = x.shape
        spatial_weights = torch.sigmoid(nn.Conv2d(embed_dim, 1, kernel_size=1).to(x.device)(x.view(-1, embed_dim, height, width)))
        spatial_weights = spatial_weights.view(batch_size, time_steps, 1, height, width)  # Reshape to match time dimension
        x = x * spatial_weights  # Weight spatial features
        x = x.sum(dim=(-2, -1))  # Sum across spatial dimensions

        # Apply Transformer for temporal modeling0
        x = self.transformer(x)  # Shape: [batch_size, time_steps, embed_dim]

        # Upsampling path (time dimension treated independently)
        up_features = []
        for t in range(time_steps):
            x_t = x[:, t, :].reshape(batch_size, embed_dim, 1, 1).expand(batch_size, embed_dim, height, width)
            # Get skip connections for the current time step
            skip1, skip2, skip3, skip4 = skip_connections[t]
            # Apply upsampling with skip connections
            x_t = self.up1(x_t, skip4)
            x_t = self.up2(x_t, skip3)
            x_t = self.up3(x_t, skip2)
            x_t = self.up4(x_t, skip1)
            up_features.append(x_t)

        # Stack the upsampled features back into the time dimension
        x = torch.stack(up_features, dim=1)  # Shape: [batch_size, time_steps, channels, height, width]

        # Apply final convolution to each time step independently
        outputs = []
        for t in range(time_steps):
            x_t = x[:, t, :, :, :]  # Shape: [batch_size, channels, height, width]
            x_t = self.final_conv(x_t)  # Apply final 1x1 convolution
            
            #if self.num_classes == 1:
            #    x_t = torch.sigmoid(x_t)
            #else:
            #    x_t = F.softmax(x_t, dim=1)

            outputs.append(x_t)

        # Stack along the time dimension to maintain 5D output
        x = torch.stack(outputs, dim=1)  # Shape: [batch_size, time_steps, num_classes, height, width]
        ### ReLU activation
        #x = F.relu(x)
        return x


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Set the default device to 'cuda:1'
    device = torch.device('cuda:2')
    torch.cuda.set_device(device)
    # Instantiate the model and move it to 'cuda:1'
    dynamic_tensor = torch.randn(3, 24, 3, 64, 64).to(device)
    static_tensor = torch.randn(3, 24, 10, 64, 64).to(device)
    categorical_tensor = torch.randn(3, 24, 12, 64, 64).to(device)

    model = CNNTransformerRegressor_v8(
        num_channels=categorical_tensor.size()[2]+static_tensor.size()[2]+dynamic_tensor.size()[2],
        num_categorical_channels=categorical_tensor.size()[2],
        num_static_channels=static_tensor.size()[2],
        num_dynamic_channels=dynamic_tensor.size()[2],
        embed_dim=1024,
        num_heads=8,
        forward_expansion=4,
        num_layers=6,
        dropout=0.3,
        num_classes=1,
    ).to(device)
    
    # Create a random input tensor and move it to 'cuda:1'
    
    # Forward pass
    output = model(dynamic_tensor, static_tensor, categorical_tensor)


    ## assert no negative
    print("Output shape:", output.shape)  ## torch.Size([4, 80, 1, 64, 64])
    test = True
    if test:    
        ## assert no nan in output
        assert not torch.isnan(output).any(), "NaN values in output tensor"
        # Print the model summary (if desired)
        input_tensor = torch.concat([dynamic_tensor, static_tensor, categorical_tensor], dim=2)
        #summary(model, input_size=(input_tensor.size()[2], input_tensor.size()[3], input_tensor.size()[4]))
        # profile the model 
        with autograd_profiler.profile(use_device="cuda") as prof:
            with autograd_profiler.record_function("model_inference"):
                output = model(dynamic_tensor, static_tensor, categorical_tensor)

        # Print profiling results sorted by CUDA time
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

        # Use torch.profiler to profile CPU and CUDA activities
        with profiler.profile(
            activities=[
                profiler.ProfilerActivity.CPU,
                profiler.ProfilerActivity.CUDA
            ],
            record_shapes=True,
            profile_memory=True,  # Enables memory profiling
            with_stack=True
        ) as prof:
            with profiler.record_function("model_inference"):
                output = model(dynamic_tensor, static_tensor, categorical_tensor)

        # Display profiling results sorted by CUDA memory usage
        print(prof.key_averages().table(sort_by="self_cuda_memory_usage", row_limit=10))

    print(f"Total parameters: {count_parameters(model):,}")