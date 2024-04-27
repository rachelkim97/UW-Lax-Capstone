import torch
import torch.nn as nn
import pytorch_lightning as pl
import torchvision.models as models
import torchvision.transforms.v2 as v2

# Define the classifier model
class ClassifierModel(nn.Module):
    def __init__(self, num_classes, model_choice='Resnet18', pretrained=True):
        super(ClassifierModel, self).__init__()

        if model_choice not in ['Resnet18','EfficientNetB0','EfficientNetB3']:
            raise ValueError("Invalid model choice")
        
        if pretrained:
            match model_choice: # python >= 3.10
                case 'Resnet18':
                    weights = models.ResNet18_Weights.DEFAULT
                case 'EfficientNetB0':
                    weights = models.EfficientNet_B0_Weights.DEFAULT
                case 'EfficientNetB3':
                    weights = models.EfficientNet_B3_Weights.DEFAULT
        else:
            weights = None
            
        if model_choice=='Resnet18':
            self.model = models.resnet18(weights=weights)
            num_features = self.model.fc.in_features
            self.model.fc = nn.Linear(num_features, num_classes)
        elif model_choice=='EfficientNetB0':
            self.model = models.efficientnet_b0(weights=weights)
            num_features = self.model.classifier[1].in_features
            self.model.classifier[1] = nn.Linear(num_features, num_classes)
        elif model_choice=='EfficientNetB3':
            self.model = models.efficientnet_b3(weights=weights)
            num_features = self.model.classifier[1].in_features
            self.model.classifier[1] = nn.Linear(num_features, num_classes)            

    def forward(self, x):
        return self.model(x)