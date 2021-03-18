import torch
from torch.nn import Module
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

class SimpleCnn(Module):
    """Simple Convolutional Net"""
    
    def __init__(self, num_classes, in_channels,):
        super().__init__()
        
        self.num_classes = num_classes
        self.in_channels = in_channels
        
        self.conv1 = nn.Sequential(
            
            nn.Conv2d(in_channels=self.in_channels,
                      out_channels=6*self.in_channels,
                      kernel_size=5,
                      padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            
            nn.Conv2d(in_channels=6*self.in_channels,
                      out_channels=16*self.in_channels,
                      kernel_size=5,
                      padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            
            nn.Conv2d(in_channels=16*self.in_channels,
                     out_channels=32*self.in_channels,
                     kernel_size=5,
                     padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(32*28*28*self.in_channels, 120*self.in_channels),
            nn.ReLU(),
            nn.Linear(120*self.in_channels, 60*self.in_channels),
            nn.ReLU(),
            nn.Linear(60*self.in_channels, self.num_classes))
        
    def forward(self, x):
        x = self.conv1(x)
        x = torch.flatten(x, 1)
        logits = self.classifier(x)
        probas = F.softmax(logits, dim=1)
        return logits, probas

# creating a string to integer mapping 
label_to_int = {
    'sunrise': 0,
    'cloudy day' : 1,
    'sunshine'  : 2,
      'rainy day' : 3
}

# creating an integer to string mapping

int_to_label = dict({(val, key) for key, val in label_to_int.items()})

def weather_predictor(image, model):
    """Takes in an image 
    and returns weather prediction"""
    
    image = image.unsqueeze_(0)
    _, probas = model(image)
    probas, indices = torch.max(probas, dim=1)
    probas = np.int(probas.item() * 100)
    pred_string = int_to_label[indices.item()]
    
    plt.imshow(image[0].permute(1,2,0))
    plt.title("Your image")
#     print("Pred values: ", pred_string)
#     print("Probabilities: ", str(probas) + "%")
    return f"The classifier is {probas}% confident that the image represents {pred_string}."


