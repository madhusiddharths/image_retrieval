import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class ResNetTransferModel(nn.Module):
    def __init__(self, num_classes=101, embedding_size=128, pretrained=True):
        super(ResNetTransferModel, self).__init__()
        # Use ResNet50 for better feature extraction
        self.resnet = models.resnet50(pretrained=pretrained)
        
        # Freeze all layers initially
        for param in self.resnet.parameters():
            param.requires_grad = False
        
        # Only unfreeze layer4 initially (will progressively unfreeze more)
        for param in self.resnet.layer4.parameters():
            param.requires_grad = True
        
        self.resnet.fc = nn.Identity()
        
        # Larger embedding with dropout
        self.embedding = nn.Sequential(
            nn.Linear(2048, 512),  # ResNet50 outputs 2048 features
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, embedding_size),
            nn.BatchNorm1d(embedding_size),
            nn.ReLU(inplace=True)
        )
        
        # Classification layer with dropout
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(embedding_size, num_classes)
        )
    
    def forward(self, x):
        features = self.resnet(x)
        embedding = self.embedding(features)
        logits = self.classifier(embedding)
        return logits
    
    def extract_features(self, x):
        """Extract feature embeddings for image retrieval"""
        features = self.resnet(x)
        embedding = self.embedding(features)
        normalized_embedding = F.normalize(embedding, p=2, dim=1)
        return normalized_embedding