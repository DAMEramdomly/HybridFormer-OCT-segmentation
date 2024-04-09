import torch
import torch.nn as nn
import torch.nn.functional as F

class Stem(nn.Module):

    def __init__(self, num_in_channels, num_out_channels, alpha=1.67):
        super(Stem, self).__init__()
        self.alpha = alpha
        W = num_out_channels * alpha

        num_filters_3x3 = int(W * 0.167)
        num_filters_5x5 = int(W * 0.333)
        num_filters_7x7 = int(W * 0.5)

        # Convolution layers
        self.conv_3x3 = nn.Conv2d(num_in_channels, num_filters_3x3, kernel_size=3, padding=1, stride=2)
        self.conv_5x5 = nn.Conv2d(num_in_channels, num_filters_5x5, kernel_size=5, padding=2, stride=2)
        self.conv_7x7 = nn.Conv2d(num_in_channels, num_filters_7x7, kernel_size=7, padding=3, stride=2)

        # Batch normalization layers
        self.bn_3x3 = nn.BatchNorm2d(num_filters_3x3)
        self.bn_5x5 = nn.BatchNorm2d(num_filters_5x5)
        self.bn_7x7 = nn.BatchNorm2d(num_filters_7x7)

        # Ensure the output has the expected number of channels
        self.adjust_channels = nn.Conv2d(num_filters_3x3 + num_filters_5x5 + num_filters_7x7,
                                          num_out_channels, kernel_size=1, stride=1)

    def forward(self, x):
        x_3x3 = F.relu(self.bn_3x3(self.conv_3x3(x)))
        x_5x5 = F.relu(self.bn_5x5(self.conv_5x5(x)))
        x_7x7 = F.relu(self.bn_7x7(self.conv_7x7(x)))

        x_concat = torch.cat([x_3x3, x_5x5, x_7x7], dim=1)
        x_out = self.adjust_channels(x_concat)
        return x_out


if __name__ == "__main__":
    model = Stem(num_in_channels=3, num_out_channels=64)
    dummy_input = torch.randn(4, 3, 256, 512)
    output = model(dummy_input)
    print(output.size())