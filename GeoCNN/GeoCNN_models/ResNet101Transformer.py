import torch
import torch.nn as nn
import torchvision.models as models
import math
from torchsummary import summary
import torch.autograd.profiler as autograd_profiler
import torch.profiler as profiler
import os
import torch.nn.functional as F 

class UpBlockWithDeformableEdgePreserving(nn.Module):
    def __init__(self, in_channels, out_channels, skip_channels, dropout, upscale_factor=2):
        super(UpBlockWithDeformableEdgePreserving, self).__init__()
        self.up = SubPixelConv(in_channels, out_channels, upscale_factor)
        self.deformable_conv = nn.Conv2d(out_channels + skip_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.edge_preserving_filter = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x, skip):
        #print(f"upblock input shape: {x.shape}")
        x = self.up(x)
        #print(f"upblock upsampled shape: {x.shape}")
        x = torch.cat([x, skip], dim=1)  # Concatenate along the channel axis
        #print(f"upblock concatenated shape: {x.shape}")
        # Apply deformable convolution
        x = self.deformable_conv(x)
        #print(f"upblock deformable conv shape: {x.shape}")
        x = self.relu(x)
        #print(f"upblock relu shape: {x.shape}")

        # Apply edge-preserving filter
        x = self.edge_preserving_filter(x)
        #print(f"upblock edge-preserving shape: {x.shape}")
        x = self.relu(x)
        #print(f"upblock relu shape: {x.shape}")

        return x



class SubPixelConv(nn.Module):
    def __init__(self, in_channels, out_channels, upscale_factor):
        super(SubPixelConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels * (upscale_factor ** 2), kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)

    def forward(self, x):
        return self.pixel_shuffle(self.conv(x))

    

# Transformer Encoder Module with Positional Encoding
class TransformerEncoderWithPositionalEncoding(nn.Module):
    def __init__(self, embed_dim, num_heads, forward_expansion, dropout, num_layers):
        super(TransformerEncoderWithPositionalEncoding, self).__init__()

        self.pos_encoder = FourierPositionalEncoding(embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=forward_expansion * embed_dim, dropout=dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(embed_dim)  # Add LayerNorm for stability
    
    def forward(self, x):
        x = self.pos_encoder(x)
        x = self.encoder(x)
        x = self.norm(x)  # Apply normalization
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


class EdgeRefinement(nn.Module):
    def __init__(self, in_channels):
        super(EdgeRefinement, self).__init__()
        self.refine = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.refine(x)


class ResNet101Transformer(nn.Module):
    def __init__(self, num_channels=4, embed_dim=1024, num_heads=8, forward_expansion=4, dropout=0.3, num_layers=6, num_classes=1, pretrained=True):    
        super(ResNet101Transformer, self).__init__()
        # ResNet101 as Encoder
        self.encoder = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V1 if pretrained else None)
        self.encoder.conv1 = nn.Conv2d(num_channels, 64, kernel_size=7, stride=1, padding=3, bias=False)
        self.encoder.fc = nn.Identity()  # Remove the final fully connected layer

        # Bottleneck
        self.bottleneck = nn.Conv2d(2048, embed_dim, kernel_size=1)

        # Transformer Encoder with positional encoding
        self.transformer = TransformerEncoderWithPositionalEncoding(embed_dim=embed_dim, num_heads=num_heads, forward_expansion=forward_expansion, dropout=dropout, num_layers=num_layers)

        # Decoder (Upsampling)
        # Decoder (Upsampling)
        self.up1 = UpBlockWithDeformableEdgePreserving(embed_dim, 512, 1024, dropout)  # Skip4 has 1024 channels
        self.up2 = UpBlockWithDeformableEdgePreserving(512, 256, 512, dropout)        # Skip3 has 512 channels
        self.up3 = UpBlockWithDeformableEdgePreserving(256, 128, 256, dropout)        # Skip2 has 256 channels
        self.up4 = UpBlockWithDeformableEdgePreserving(128, 64, 64, dropout)          # Skip1 has 64 channels


        # Final convolution to get the desired output channels (1 for regression)
        self.edge_refinement = EdgeRefinement(64)
        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)
       
    def forward(self, dynamic, static, categorical):
        x = torch.cat([dynamic, static, categorical], dim=2)  # (batch, time, feature, height, width)
        batch_size, time_steps, channels, height, width = x.shape  # Expect 5D input: [batch_size, time_steps, channels, height, width]
        #print(f"Input shape: {x.shape}")
        cnn_features = []
        skip_connections = []

        # Apply ResNet101 to each time step independently
        for t in range(time_steps):
            x_t = x[:, t, :, :, :]  # Extract input for time step t
            x_t = self.encoder.conv1(x_t)
            #print(f"Conv1 shape: {x_t.shape}")
            x_t = self.encoder.bn1(x_t)
            #print(f"BN1 shape: {x_t.shape}")
            x_t = self.encoder.relu(x_t)
            #print(f"ReLU shape: {x_t.shape}")
            skip1 = x_t
            #print(f"Skip1 shape: {skip1.shape}")
            x_t = self.encoder.maxpool(x_t)
            #print(f"MaxPool shape: {x_t.shape}")
            x_t = self.encoder.layer1(x_t)
            #print(f"Layer1 shape: {x_t.shape}")
            skip2 = x_t
            #print(f"Skip2 shape: {skip2.shape}")
            x_t = self.encoder.layer2(x_t)
            #print(f"Layer2 shape: {x_t.shape}")
            skip3 = x_t
            #print(f"Skip3 shape: {skip3.shape}")
            x_t = self.encoder.layer3(x_t)
            #print(f"Layer3 shape: {x_t.shape}")
            skip4 = x_t
            #print(f"Skip4 shape: {skip4.shape}")
            x_t = self.encoder.layer4(x_t)
            #print(f"Layer4 shape: {x_t.shape}")

            # Bottleneck

            x_t = self.bottleneck(x_t)
            #print(f"Bottleneck shape: {x_t.shape}")
            cnn_features.append(x_t)
            #print(f"Appended CNN features shape: {len(cnn_features)}")
            skip_connections.append((skip1, skip2, skip3, skip4))
            #print(f"Appended skip connections shape: {len(skip_connections)}")

        x = torch.stack(cnn_features, dim=1)
        #print(f"Stacked CNN features shape: {x.shape}")

        # Apply Transformer for temporal modeling
        batch_size, time_steps, embed_dim, height, width = x.shape

        
        x = x.sum(dim=(-2, -1))  # Sum across spatial dimensions
        #print(f"Summed CNN features shape: {x.shape}")  
        x = self.transformer(x)  # Shape: [batch_size, time_steps, embed_dim]
        #print(f"Transformer output shape: {x.shape}")
        # Upsampling path (time dimension treated independently)
        up_features = []
        for t in range(time_steps):

            x_t = x[:, t, :].reshape(batch_size, embed_dim, 1, 1).expand(batch_size, embed_dim, height, width)
            #print(f"Expanded transformer output shape: {x_t.shape}")
            # Get skip connections for the current time step
            skip1, skip2, skip3, skip4 = skip_connections[t]
            #print(f"Skip1 shape: {skip1.shape}")    
            # Apply upsampling with skip connections
            x_t = self.up1(x_t, skip4)
            #print(f"Up1 shape: {x_t.shape}")
            x_t = self.up2(x_t, skip3)
            #print(f"Up2 shape: {x_t.shape}")
            x_t = self.up3(x_t, skip2)
            #print(f"Up3 shape: {x_t.shape}")
            x_t = self.up4(x_t, skip1)
            #print(f"Up4 shape: {x_t.shape}")
            # Apply Edge Refinement
            x_t = self.edge_refinement(x_t)
            #print(f"Edge Refinement shape: {x_t.shape}")
            up_features.append(x_t)
            #print(f"Appended upsampled features shape: {len(up_features)}")

        # Stack the upsampled features back into the time dimension
        x = torch.stack(up_features, dim=1)
        #print(f"Stacked upsampled features shape: {x.shape}")
        x = x.reshape(batch_size * time_steps, x.size(2), x.size(3), x.size(4))
        #print(f"Reshaped upsampled features shape: {x.shape}")
        # Final convolution to produce the output
        x = self.final_conv(x)  # Shape: [batch_size * time_steps, num_classes, height, width]
        #print(f"Final Conv shape: {x.shape}")
        # Reshape back to [batch_size, time_steps, num_classes, height, width]
        x = x.reshape(batch_size, time_steps, x.size(1), x.size(2), x.size(3))

        return x

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Set the default device to 'cuda:1'
    device = torch.device('cuda:1')
    torch.cuda.set_device(device)
    #input_tensor = torch.randn(3, 80, 29, 64, 64).to(device)  # (batch, time, channels, height, width)
    dynamic = torch.randn(3, 80, 9, 64, 64).to(device)
    static = torch.randn(3, 80, 10, 64, 64).to(device)
    categorical = torch.randn(3, 80, 16, 64, 64).to(device)
    # Instantiate the model and move it to 'cuda:1'
    model = ResNet101Transformer(
        num_channels=dynamic.shape[2] + static.shape[2] + categorical.shape[2],
        embed_dim=1024,
        num_heads=8,
        forward_expansion=4,
        num_layers=6,
        dropout=0.3,
        num_classes=1,
        pretrained=False
    ).to(device)

    # Perform a forward pass
    output = model(dynamic, static, categorical)

    # Output shape
    print("Output shape:", output.shape)
    
    print(f"Total parameters: {count_parameters(model):,}")