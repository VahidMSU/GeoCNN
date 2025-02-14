import torch
import torch.nn as nn
import torch.nn.functional as F



class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False):
        super(SeparableConv2d, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=in_channels, bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
    
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

def UpsampleBlock(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU()
    )

class XceptionBlock(nn.Module):
    def __init__(self, in_channels, out_channels, reps, strides=1, grow_first=True, dropout_rate=0.3):
        super(XceptionBlock, self).__init__()
        self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=strides, bias=False) if strides != 1 or in_channels != out_channels else None
        self.skipbn = nn.BatchNorm2d(out_channels) if self.skip else None
        self.dropout = nn.Dropout(dropout_rate)

        rep = []
        filters = in_channels
        if grow_first:
            rep.extend(
                (
                    nn.ReLU(),
                    SeparableConv2d(
                        in_channels, out_channels, 3, stride=1, padding=1
                    ),
                    nn.BatchNorm2d(out_channels),
                    self.dropout,
                )
            )
            filters = out_channels

        for _ in range(reps - 1):
            rep.extend(
                (
                    nn.ReLU(),
                    SeparableConv2d(filters, filters, 3, stride=1, padding=1),
                    nn.BatchNorm2d(filters),
                    self.dropout,
                )
            )
        if not grow_first:
            rep.extend(
                (
                    nn.ReLU(),
                    SeparableConv2d(
                        in_channels, out_channels, 3, stride=1, padding=1
                    ),
                    nn.BatchNorm2d(out_channels),
                    self.dropout,
                )
            )
        if strides != 1:
            rep.append(nn.MaxPool2d(3, strides, 1))

        self.rep = nn.Sequential(*rep)

    def forward(self, x):
        res = self.skipbn(self.skip(x)) if self.skip else x
        x = self.rep(x)
        x += res
        return x


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

def DecoderBlockMaker(in_channels, out_channels, kernel_size1=3, stride1=2, padding1=0, kernel_size2=3, stride2=1, padding2=1):
    """A decoder block with residual skip connections within the block."""
    return nn.Sequential(
        nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size1, stride=stride1, padding=padding1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        SEBlock(out_channels),  # Squeeze-and-Excitation block

        # Residual part: additional Conv layer for residual connection
        nn.Conv2d(out_channels, out_channels, kernel_size=1),  # This acts as the residual path

        # Second layer in the block
        nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size2, stride=stride2, padding=padding2),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.3),
    )

class MultiScaleXception(nn.Module):
    def __init__(self, num_channels=20, output_channels=1, hidden_dim=2048, dropout_rate=0.3, batch_window=128):
        super(MultiScaleXception, self).__init__()

        # Encoder layers as before
        self.conv1 = nn.Conv2d(num_channels, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.dropout = nn.Dropout(dropout_rate)

        # Xception blocks
        self.block1 = XceptionBlock(64, 128, 2, strides=2, dropout_rate=dropout_rate)
        self.block2 = XceptionBlock(128, 256, 2, strides=2, dropout_rate=dropout_rate)
        self.block3 = XceptionBlock(256, 728, 2, strides=2, dropout_rate=dropout_rate)

        # Temporal LSTM
        self.temporal_lstm = nn.LSTM(input_size=728, hidden_size=hidden_dim, num_layers=2, batch_first=True, dropout=dropout_rate)
        self.lstm_norm = nn.LayerNorm(hidden_dim)  # Apply layer normalization after LSTM

        if batch_window == 128:
            # Decoder with skip connections
            self.deconv0 = nn.ConvTranspose2d(hidden_dim, 1280, kernel_size=10, stride=1, padding=1)
        elif batch_window == 64:
            # Decoder with skip connections
            self.deconv0 = nn.ConvTranspose2d(hidden_dim, 1280, kernel_size=6, stride=1, padding=1)
        elif batch_window == 32:
            # Decoder with skip connections
            self.deconv0 = nn.ConvTranspose2d(hidden_dim, 1280, kernel_size=2, stride=2)  # Upsample from 1x1 to 2x2


        self.deconv1 = DecoderBlockMaker(1536, 768, kernel_size1=2, stride1=2, kernel_size2=3, stride2=1)  # From 2x2 to 4x4
        self.deconv2 = DecoderBlockMaker(896, 512, kernel_size1=2, stride1=2, kernel_size2=3, stride2=1)  # From 4x4 to 8x8
        self.deconv3 = DecoderBlockMaker(576, 64, kernel_size1=2, stride1=2, kernel_size2=3, stride2=1)  # From 8x8 to 16x16
        self.deconv4 = DecoderBlockMaker(96, 32, kernel_size1=2, stride1=2, kernel_size2=3, stride2=1)  # From 16x16 to 32x32
        self.final_deconv = nn.ConvTranspose2d(32, output_channels, kernel_size=1)  # Final adjustment to output channels

        
    def forward(self, dynamic_features, static_features, categorical_features):
        x = torch.cat([dynamic_features, static_features, categorical_features], dim=2)
        batch_size, time_steps, channels, height, width = x.size()
        #print(f"batch_size: {batch_size}, time_steps: {time_steps}, channels: {channels}, height: {height}, width: {width}")
        x = x.reshape(batch_size * time_steps, channels, height, width)

        # Encoder
        x1 = F.relu(self.bn1(self.conv1(x)))
        #print(f"x1 after conv1: {x1.shape}")
        x1 = self.dropout(x1)
        #print(f"x1 after dropout: {x1.shape}")
        x2 = F.relu(self.bn2(self.conv2(x1)))
        #print(f"x2 after conv2: {x2.shape}")
        x2 = self.dropout(x2)
        #print(f"x2 after dropout: {x2.shape}")
        x3 = self.block1(x2)
        #print(f"x3 after block1: {x3.shape}")
        x4 = self.block2(x3)
        #print(f"x4 after block2: {x4.shape}")
        x5 = self.block3(x4)
        #print(f"x5 after block3: {x5.shape}")

        # Temporal LSTM
        x5 = F.adaptive_avg_pool2d(x5, (1, 1)).reshape(batch_size, time_steps, -1)
        #print(f"x5 after adaptive_avg_pool2d: {x5.shape}")
        lstm_out, _ = self.temporal_lstm(x5)
        #print(f"lstm_out: {lstm_out.shape}")
        lstm_out = self.lstm_norm(lstm_out)  # Apply layer normalization after LSTM
        #print(f"lstm_out after lstm_norm: {lstm_out.shape}")
        # Skip connections prepared for each time step
        x4 = x4.reshape(batch_size, time_steps, *x4.shape[1:])
        x3 = x3.reshape(batch_size, time_steps, *x3.shape[1:])
        x2 = x2.reshape(batch_size, time_steps, *x2.shape[1:])
        x1 = x1.reshape(batch_size, time_steps, *x1.shape[1:])

        # Decoder with skip connections from the encoder
        decoded_outputs = []
        for t in range(time_steps):
            xt = lstm_out[:, t].reshape(batch_size, -1, 1, 1)
            xt = F.relu(self.deconv0(xt))
            #print(f"xt after deconv0: {xt.shape}")
            # Add skip connections from the encoder layers
            xt = F.relu(self.deconv1(torch.cat((xt, x4[:, t]), dim=1)))
            #print(f"xt after deconv1: {xt.shape}")
            xt = F.relu(self.deconv2(torch.cat((xt, x3[:, t]), dim=1)))
            xt = F.relu(self.deconv3(torch.cat((xt, x2[:, t]), dim=1)))
            xt = F.relu(self.deconv4(torch.cat((xt, x1[:, t]), dim=1)))
            xt = self.final_deconv(xt)

            # Collect output for this time step
            decoded_outputs.append(xt)

        return torch.stack(decoded_outputs, dim=1)



if __name__ == "__main__":

    for batch_window in [128, 64, 32]:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")   
        #input_tensor = torch.randn(4, 80, 34, batch_window, batch_window).to(device)  # Example input (batch_size, time_steps, channels, height, width)
        dynamic_features = torch.randn(4, 80, 20, batch_window, batch_window).to(device)
        static_features = torch.randn(4, 80, 10, batch_window, batch_window).to(device)
        categorical_features = torch.randn(4, 80, 4, batch_window, batch_window).to(device)
        #print("Input shape:", input_tensor.shape)
        model = MultiScaleXception(num_channels=dynamic_features.shape[2] + static_features.shape[2] + categorical_features.shape[2],
                                output_channels=1, 
                                hidden_dim=1024, 
                                dropout_rate=0.3,
                                batch_window = batch_window).to(device)
        
        output = model(dynamic_features, static_features, categorical_features)
        print("Output shape:", output.shape)
