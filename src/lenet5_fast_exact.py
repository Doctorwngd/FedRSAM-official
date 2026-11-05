
import torch.nn.functional as F
from torch import nn


class LeNet5(nn.Module):

    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=5)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=5)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 5 * 5, 384)
        self.fc2 = nn.Linear(384, 192)
        self.fc3 = nn.Linear(192, num_classes)

    def forward(self, inputs):
        outputs = self.pool(F.relu(self.conv1(inputs)))
        outputs = self.pool(F.relu(self.conv2(outputs)))
        outputs = outputs.view(-1, 64 * 5 * 5)
        outputs = F.relu(self.fc1(outputs))
        outputs = F.relu(self.fc2(outputs))
        outputs = self.fc3(outputs)
        return F.log_softmax(outputs, dim=1)
