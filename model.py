import torch
import torch.nn as nn
import torch.optim as optim


class DualModalUNet(nn.Module):
    def __init__(
        self,
        thermal_out_channels=64,
        visible_out_channels=64,
        bottleneck_channels=128,
        decoder_in_channels=64,
    ):
        super(DualModalUNet, self).__init__()
        self.thermal_out_channels = thermal_out_channels
        self.visible_out_channels = visible_out_channels

        if thermal_out_channels > 0:
            self.encoder_thermal = nn.Sequential(
                nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(32, thermal_out_channels, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
            )
        else:
            self.encoder_thermal = None

        if visible_out_channels > 0:
            self.encoder_visible = nn.Sequential(
                nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(32, visible_out_channels, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
            )
        else:
            self.encoder_visible = None

        fusion_channels = thermal_out_channels + visible_out_channels
        if fusion_channels <= 0:
            raise ValueError("At least one modality must have channels > 0.")

        self.bottleneck = nn.Sequential(
            nn.Conv2d(fusion_channels, bottleneck_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(bottleneck_channels, decoder_in_channels, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(decoder_in_channels, 32, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 4, kernel_size=1),
            nn.ReLU(),
        )

        # Global average pooling and a fully connected layer
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(4, 4)

    def forward(self, x_thermal, x_visible):
        x_thermal = self.encoder_thermal(x_thermal) if self.encoder_thermal else None
        x_visible = self.encoder_visible(x_visible) if self.encoder_visible else None

        if x_thermal is None and x_visible is None:
            raise RuntimeError("Both modality encoders are disabled.")

        if x_thermal is None:
            x_thermal = x_visible[:, :0, ...]
        if x_visible is None:
            x_visible = x_thermal[:, :0, ...]

        x = torch.cat((x_thermal, x_visible), dim=1)

        x = self.bottleneck(x)
        x = self.decoder(x)

        x = self.global_avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


class DualModalMLPNet(nn.Module):
    def __init__(self, input1_dim, input2_dim, hidden1_dim, hidden2_dim, decoder_hidden_dim, output_dim):
        super(DualModalMLPNet, self).__init__()

        # Encoder for input1 with specific hidden dimension
        self.encoder1 = nn.Sequential(
            nn.Linear(input1_dim, hidden1_dim),
            nn.ReLU(),
            nn.Linear(hidden1_dim, hidden1_dim),
        )

        # Encoder for input2 with specific hidden dimension
        self.encoder2 = nn.Sequential(
            nn.Linear(input2_dim, hidden2_dim),
            nn.ReLU(),
            nn.Linear(hidden2_dim, hidden2_dim),
        )

        # Decoder after concatenation
        self.decoder = nn.Sequential(
            nn.Linear(hidden1_dim + hidden2_dim, decoder_hidden_dim),
            nn.ReLU(),
            nn.Linear(decoder_hidden_dim, output_dim),
        )

    def forward(self, x1, x2):
        encoded_x1 = self.encoder1(x1)
        encoded_x2 = self.encoder2(x2)

        combined = torch.cat([encoded_x1, encoded_x2], dim=1)
        output = self.decoder(combined)

        return output
