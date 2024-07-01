# Gym Equipment Detection using ResNet

This project demonstrates the use of a ResNet model to detect various types of gym equipment in images.

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Results](#results)
- [Requirements](#requirements)
- [Usage](#usage)

## Overview
![image](https://github.com/tyl-99/gym-equipment-detector/assets/71328888/153cc544-8262-44cc-879c-9118b01da8c8)

This project aims to detect different types of gym equipment in images using a ResNet model. The model is trained to accurately identify and locate various gym equipment in different settings.

## Dataset

The dataset used for this project consists of images containing different types of gym equipment. The images are annotated with bounding boxes around the equipment. The dataset is split into training and test sets for model evaluation.

## Model Architecture

The model used in this project is based on ResNet-34, which is a popular convolutional neural network architecture known for its effectiveness in image recognition tasks. The specific model architecture used in this project is as follows:

```python
class EquipmentModelRes34(nn.Module):
    def __init__(self, num_classes=22):
        super(EquipmentModelRes34, self).__init__()
        self.in_channels = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(64, 3, stride=1)
        self.layer2 = self._make_layer(128, 4, stride=2)
        self.layer3 = self._make_layer(256, 6, stride=2)
        self.layer4 = self._make_layer(512, 3, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(self._residual_block(self.in_channels, channels, stride))
            self.in_channels = channels
        return nn.Sequential(*layers)

    def _residual_block(self, in_channels, out_channels, stride):
        return ResidualBlock(in_channels, out_channels, stride)

    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += self.shortcut(identity)
        out = self.relu(out)

        return out
```

## Training

The ResNet model is trained on the annotated dataset for several epochs. The training process involves optimizing the model to minimize detection loss, improving its ability to detect gym equipment accurately.

## Results

The model achieves high accuracy in detecting various types of gym equipment in images. 

## Requirements

- Python 3.x
- PyTorch
- torchvision
- OpenCV
- numpy

You can install the required packages using the following command:

```bash
pip install torch torchvision opencv-python numpy
```

## Usage

To run the notebook and perform gym equipment detection, use the following command:

```bash
jupyter notebook GymDetector.ipynb
```

You can also use the ResNet model to detect gym equipment in your own images by following the steps in the notebook.

