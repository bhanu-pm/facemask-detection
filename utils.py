import torch
import torchvision
import torch.nn as nn

# GPU Usage
# if torch.cuda.is_available():
#     device = 'cuda'
# else:
device = 'cpu'

# Defining image transformations
transformations = torchvision.transforms.Compose([torchvision.transforms.Resize((32, 32)),
                                                  torchvision.transforms.ToTensor(),
                                                  torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                                   std=[0.229, 0.224, 0.225])])


# Defining the Model
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),

            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # 16

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),

            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)  # 8
        )

        self.fc = nn.Sequential(
            nn.Linear(in_features=4096, out_features=1024),
            nn.ReLU(),
            nn.Dropout(p=0.5),

            nn.Linear(in_features=1024, out_features=2)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(-1, 4096)
        out = self.fc(x)

        return out

def classifier(x):
    if x < 0.5:
        return 0
    else:
        return 1
