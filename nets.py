import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os

"""
Given an image, we initially resize it to different scales to build
an image pyramid, which is the input of the following
three-stage cascaded framework:
Stage 1: We exploit a fully convolutional network, called
Proposal Network (P-Net), to obtain the candidate windows
and their bounding box regression vectors.
Then we use the estimated bounding box regression
vectors to calibrate the candidates. After that, we employ
non-maximum suppression (NMS) to merge highly overlapped
candidates.

Stage 2: all candidates are fed to another CNN, called Refine
Network (R-Net), which further rejects a large number of false
candidates, performs calibration with bounding box regression,
and NMS candidate merge.

Stage 3: This stage is similar to the second stage, but in this
stage we aim to describe the face in more details. In particular,
the network will output five facial landmarks’ positions.
"""


class PNet(nn.Module):
    def __init__(self):
        super(PNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=10, kernel_size=3)
        self.prelu1 = nn.PReLU(num_parameters=10)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        self.conv2 = nn.Conv2d(in_channels=10, out_channels=16, kernel_size=3)
        self.prelu2 = nn.PReLU(num_parameters=16)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3)
        self.prelu3 = nn.PReLU(num_parameters=32)

        self.conv_face_classification = nn.Conv2d(in_channels=32, out_channels=2, kernel_size=1)
        self.conv_bound_box_regression = nn.Conv2d(in_channels=32, out_channels=4, kernel_size=1)
        # self.conv_facial_landmark_localization = nn.Conv2d(in_channels=32, out_channels=10, kernel_size=1)

    def forward(self, x):
        x = self.pool(self.prelu1(self.conv1(x)))
        x = self.prelu2(self.conv2(x))
        x = self.prelu3(self.conv3(x))

        face_classification = F.softmax(self.conv_face_classification(x), dim=1)
        bound_box_regression = self.conv_bound_box_regression(x)
        # facial_landmark_localization = self.conv_facial_landmark_localization(x)

        return bound_box_regression, face_classification  # , facial_landmark_localization


class RNet(nn.Module):
    def __init__(self):
        super(RNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=28, kernel_size=3)
        self.prelu1 = nn.PReLU(num_parameters=28)
        self.conv2 = nn.Conv2d(in_channels=28, out_channels=48, kernel_size=3)
        self.prelu2 = nn.PReLU(num_parameters=48)
        self.conv3 = nn.Conv2d(in_channels=48, out_channels=64, kernel_size=2)
        self.prelu3 = nn.PReLU(num_parameters=64)
        self.fc1 = nn.Linear(in_features=576, out_features=128)
        self.prelu4 = nn.PReLU(num_parameters=128)

        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)

        self.fc_face_classification = nn.Linear(in_features=128, out_features=2)
        self.fc_bound_box_regression = nn.Linear(in_features=128, out_features=4)
        # self.fc_facial_landmark_localization = nn.Linear(in_features=128, out_features=10)

    def forward(self, x):
        x = self.pool(self.prelu1(self.conv1(x)))
        x = self.pool(self.prelu2(self.conv2(x)))
        x = self.prelu3(self.conv3(x))
        x = x.transpose(3, 2).contiguous()
        x = x.view(x.size(0), -1)
        x = self.prelu4(self.fc1(x))

        face_classification = F.softmax(self.fc_face_classification(x), dim=1)
        bound_box_regression = self.fc_bound_box_regression(x)
        # facial_landmark_localization = self.fc_facial_landmark_localization(x)

        return bound_box_regression, face_classification  # , facial_landmark_localization


class ONet(nn.Module):
    def __init__(self):
        super(ONet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3)
        self.prelu1 = nn.PReLU(num_parameters=32)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
        self.prelu2 = nn.PReLU(num_parameters=64)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3)
        self.prelu3 = nn.PReLU(num_parameters=64)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=2)
        self.prelu4 = nn.PReLU(num_parameters=128)

        self.fc = nn.Linear(in_features=1152, out_features=256)
        self.dropout = nn.Dropout(p=0.25)
        self.prelu5 = nn.PReLU(num_parameters=256)

        self.fc_face_classification = nn.Linear(in_features=256, out_features=2)
        self.fc_bound_box_regression = nn.Linear(in_features=256, out_features=4)
        self.fc_facial_landmark_localization = nn.Linear(in_features=256, out_features=10)

    def forward(self, x):
        x = self.pool1(self.prelu1(self.conv1(x)))
        x = self.pool2(self.prelu2(self.conv2(x)))
        x = self.pool3(self.prelu3(self.conv3(x)))
        x = self.prelu4(self.conv4(x))
        x = x.transpose(3, 2).contiguous()
        x = x.view(x.size(0), -1)
        x = self.prelu5(self.dropout(self.fc(x)))

        face_classification = F.softmax(self.fc_face_classification(x))
        bound_box_regression = self.fc_bound_box_regression(x)
        facial_landmark_localization = self.fc_facial_landmark_localization(x)

        return facial_landmark_localization, bound_box_regression, face_classification
