import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet101

class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(SEBlock, self).__init__()
        self.fc1 = nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1)
        self.fc2 = nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1)

    def forward(self, x):
        squeeze = F.adaptive_avg_pool2d(x, 1)
        excitation = F.relu(self.fc1(squeeze))
        excitation = torch.sigmoid(self.fc2(excitation))
        return x * excitation

class ResNetLSTMUnet(nn.Module):
    def __init__(self, num_channels=9, hidden_dim=1024, lstm_layers=1, pretrained=True):
        super(ResNetLSTMUnet, self).__init__()

        # Load a ResNet101 model for spatial feature extraction
        self.encoder = resnet101(weights='IMAGENET1K_V2' if pretrained else None)

        # Modify the first convolutional layer to accept num_channels
        self.encoder.conv1 = nn.Conv2d(num_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # Replace max-pooling with a SEBlock
        self.encoder.maxpool = nn.Sequential(
            SEBlock(64),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        # LSTM for temporal modeling
        self.lstm = nn.LSTM(
            input_size=1024 * 8 * 8,  # Flattened spatial dimensions from ResNet layer3
            hidden_size=hidden_dim,
            num_layers=lstm_layers,
            batch_first=True
        )

        # Decoder (UNet-style)
        self.deconv1 = self.make_decoder_block(hidden_dim, 512)
        self.deconv2 = self.make_decoder_block(512, 256)
        self.deconv3 = self.make_decoder_block(256, 64)

        # Final convolution to get the desired output size (1 channel)
        self.final_conv = nn.Conv2d(64, 1, kernel_size=1)

    def make_decoder_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            SEBlock(out_channels)
        )

    def forward(self, dynamic, static, categorical):
        # Concatenate inputs
        x = torch.cat([dynamic, static, categorical], dim=2)  # (batch, time, feature, height, width)

        batch_size, time_steps, features, height, width = x.size()

        # Reshape to (batch * time, features, height, width) for ResNet
        x = x.view(batch_size * time_steps, features, height, width)

        # Encoder
        x1 = self.encoder.conv1(x)
        x1 = self.encoder.bn1(x1)
        x1 = self.encoder.relu(x1)
        x1 = self.encoder.maxpool(x1)

        x2 = self.encoder.layer1(x1)
        x3 = self.encoder.layer2(x2)
        x4 = self.encoder.layer3(x3)  # Final ResNet encoder output

        # Reshape for LSTM: (batch, time, features)
        lstm_input = x4.view(batch_size, time_steps, -1)  # Flatten spatial dimensions
        lstm_output, _ = self.lstm(lstm_input)  # Temporal modeling

        # Decode only the last timestep output
        decoder_input = lstm_output[:, -1, :].view(batch_size, -1, 8, 8)  # Reshape back to (batch, features, height, width)

        # Decoder with skip connections
        d3 = self.deconv1(decoder_input)
        d2 = self.deconv2(d3)
        d1 = self.deconv3(d2)

        # Final output layer to produce 1-channel output
        output = self.final_conv(d1)

        # Reshape back to (batch, time, channels, height, width)
        return output.unsqueeze(1)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == "__main__":
    # Instantiate the model with extra blocks for more depth
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # Input tensors
    batch_size, time_steps, height, width = 1, 24, 32, 32
    dynamic = torch.randn(batch_size, time_steps, 9, height, width).to(device)
    static = torch.randn(batch_size, time_steps, 10, height, width).to(device)
    categorical = torch.randn(batch_size, time_steps, 16, height, width).to(device)

    # Model instantiation and forward pass
    model = ResNetLSTMUnet(
        num_channels=dynamic.shape[2] + static.shape[2] + categorical.shape[2],
        hidden_dim=1024,
        lstm_layers=1,
        pretrained=True
    ).to(device)
    
    output = model(dynamic, static, categorical)
    print("Output shape:", output.shape)
    print("Number of parameters:", count_parameters(model))
