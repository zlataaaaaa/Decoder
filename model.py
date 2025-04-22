import torch
import torch.nn as nn


class SqueezeExcitation(nn.Module):
    """
    Squeeze-and-Excitation block for channel attention
    """
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class ResidualSEBlock(nn.Module):
    """
    Convolutional block with residual connection and SE attention
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.se = SqueezeExcitation(out_channels)
        if in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.skip = nn.Identity()

    def forward(self, x):
        identity = self.skip(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)
        out = out + identity
        return self.relu(out)


class CRNN(nn.Module):
    def __init__(self, num_classes, input_features=64):
        super().__init__()
        # --- CNN backbone with residual + SE blocks ---
        self.cnn = nn.Sequential(
            ResidualSEBlock(1, 32),
            nn.MaxPool2d((2, 2)),

            ResidualSEBlock(32, 64),
            nn.MaxPool2d((2, 2)),

            ResidualSEBlock(64, 128),
            nn.MaxPool2d((2, 2)),
        )

        self.dropout = nn.Dropout(0.3)

        # --- Recurrent block: bidirectional LSTM ---
        rnn_input_size = (input_features // 8) * 128
        self.rnn = nn.LSTM(
            input_size=rnn_input_size,
            hidden_size=384,
            num_layers=3,
            batch_first=True,
            bidirectional=True
        )
        self.norm_rnn = nn.LayerNorm(384 * 2)

        # --- Classifier ---
        self.classifier = nn.Sequential(
            nn.Linear(384 * 2, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        # x: (B, T, F)
        x = x.unsqueeze(1)      # (B, 1, T, F)
        x = self.cnn(x)         # (B, C, T', F')
        b, c, t, f = x.size()
        x = x.permute(0, 2, 1, 3)
        x = x.contiguous().view(b, t, c * f)  # (B, T', C*F')
        x = self.dropout(x)

        # RNN
        x, _ = self.rnn(x)            # (B, T', 2*H)
        x = self.norm_rnn(x)          # LayerNorm over feature dim

        # Classifier
        x = self.classifier(x)        # (B, T', num_classes)
        return x.permute(1, 0, 2)      # (T', B, num_classes)
