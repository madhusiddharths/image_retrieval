import os
import random
import json
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from datetime import datetime
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import torchvision.models as models
from sklearn.model_selection import train_test_split

# Path to the Caltech101 dataset
data_dir = 'caltech101/101_ObjectCategories'

# Check if the path exists
if not os.path.exists(data_dir):
    print(f"Error: Dataset directory {data_dir} not found.")
else:
    print(f"Dataset directory found at {data_dir}")

# Create dataset class for Caltech101
class Caltech101Dataset(Dataset):
    def __init__(self, data_dir, categories=None, transform=None, max_per_class=None, seed=42):
        self.data_dir = data_dir
        self.transform = transform
        self.max_per_class = max_per_class

        # Get available categories
        all_categories = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
        # Filter BACKGROUND_Google to avoid confusing the model with non-object images
        all_categories = [c for c in all_categories if c != 'BACKGROUND_Google']

        # Use all categories if none specified
        if categories is None:
            categories = all_categories

        # Sort categories for reproducibility
        categories.sort()

        # Get all image paths and labels
        self.image_paths = []
        self.labels = []
        self.categories = categories

        # FIXED: Use seed parameter
        random.seed(seed)

        for i, category in enumerate(categories):
            category_dir = os.path.join(data_dir, category)
            if os.path.isdir(category_dir):
                image_files = sorted([f for f in os.listdir(category_dir) 
                                     if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
                
                if max_per_class and len(image_files) > max_per_class:
                    image_files = random.sample(image_files, max_per_class)
                
                for image_file in image_files:
                    self.image_paths.append(os.path.join(category_dir, image_file))
                    self.labels.append(i)

        self.num_classes = len(categories)
        print(f"Loaded {len(self.image_paths)} images from {len(categories)} categories")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            # Return a dummy image if loading fails
            image = Image.new('RGB', (224, 224), color='gray')

        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label, image_path


# ADDED: Helper class to apply transforms to subsets
class TransformSubset(Dataset):
    """Wrapper to apply transforms to a Subset"""
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform
    
    def __getitem__(self, idx):
        img, label, path = self.subset[idx]
        if self.transform:
            img = self.transform(img)
        return img, label, path
    
    def __len__(self):
        return len(self.subset)


# ============================================================================
# LABEL SMOOTHING LOSS
# ============================================================================
class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing
    
    def forward(self, pred, target):
        confidence = 1.0 - self.smoothing
        log_probs = F.log_softmax(pred, dim=-1)
        nll_loss = -log_probs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -log_probs.mean(dim=-1)
        loss = confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


# ============================================================================
# IMPROVED MODEL ARCHITECTURE (ResNet50 with Dropout)
# ============================================================================
class ResNetTransferModel(nn.Module):
    def __init__(self, num_classes, embedding_size=128, pretrained=True):
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


# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================
def train_epoch(model, train_loader, criterion, optimizer, epoch, writer):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    # Progress bar
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]")

    for i, (inputs, labels, _) in enumerate(pbar):
        inputs, labels = inputs.to(device), labels.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # Track statistics
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        # Update progress bar
        accuracy = 100. * correct / total
        pbar.set_postfix({"loss": running_loss/(i+1), "acc": f"{accuracy:.2f}%"})

        # Log batch metrics to TensorBoard (every 10 batches)
        if i % 10 == 9:
            step = epoch * len(train_loader) + i
            writer.add_scalar('Loss/train_batch', loss.item(), step)
            writer.add_scalar('Accuracy/train_batch', accuracy, step)

    # Calculate epoch statistics
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total

    # Log epoch metrics to TensorBoard
    writer.add_scalar('Loss/train', epoch_loss, epoch)
    writer.add_scalar('Accuracy/train', epoch_acc, epoch)

    return epoch_loss, epoch_acc


def validate(model, val_loader, criterion, epoch, writer):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    # Disable gradient computation for validation
    with torch.no_grad():
        pbar = tqdm(val_loader, desc=f"Epoch {epoch+1} [Valid]")

        for i, (inputs, labels, _) in enumerate(pbar):
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Track statistics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            # Update progress bar
            accuracy = 100. * correct / total
            pbar.set_postfix({"loss": running_loss/(i+1), "acc": f"{accuracy:.2f}%"})

    # Calculate epoch statistics
    epoch_loss = running_loss / len(val_loader)
    epoch_acc = 100. * correct / total

    # Log epoch metrics to TensorBoard
    writer.add_scalar('Loss/val', epoch_loss, epoch)
    writer.add_scalar('Accuracy/val', epoch_acc, epoch)

    return epoch_loss, epoch_acc

    
def train_model_progressive_complete(model, train_loader, val_loader, writer, num_epochs=30, log_dir='./logs'):
    """
    Complete training function using ALL improvements:
    - Label smoothing
    - Progressive unfreezing
    - AdamW optimizer with different LRs
    - Cosine annealing scheduler
    - Best model saving
    """
    best_val_acc = 0.0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    # Use Label Smoothing Loss
    criterion = LabelSmoothingCrossEntropy(smoothing=0.1)
    
    # Phase 1: Only layer4 + embedding + classifier (epochs 0-9)
    print("\n" + "="*70)
    print("PHASE 1: Training layer4 + embedding + classifier")
    print("="*70 + "\n")
    
    optimizer = optim.AdamW([
        {'params': model.resnet.layer4.parameters(), 'lr': 0.0001},
        {'params': model.embedding.parameters(), 'lr': 0.001},
        {'params': model.classifier.parameters(), 'lr': 0.001}
    ], weight_decay=0.01)
    
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6
    )
    
    for epoch in range(num_epochs):
        # Progressive unfreezing
        if epoch == 10:
            print("\n" + "="*70)
            print("PHASE 2: Unfreezing layer3")
            print("="*70 + "\n")
            for param in model.resnet.layer3.parameters():
                param.requires_grad = True
            
            optimizer = optim.AdamW([
                {'params': model.resnet.layer3.parameters(), 'lr': 0.00005},
                {'params': model.resnet.layer4.parameters(), 'lr': 0.0001},
                {'params': model.embedding.parameters(), 'lr': 0.0005},
                {'params': model.classifier.parameters(), 'lr': 0.0005}
            ], weight_decay=0.01)
            
            scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, T_0=10, T_mult=2, eta_min=1e-6
            )
            
        elif epoch == 20:
            print("\n" + "="*70)
            print("PHASE 3: Unfreezing layer2")
            print("="*70 + "\n")
            for param in model.resnet.layer2.parameters():
                param.requires_grad = True
            
            optimizer = optim.AdamW([
                {'params': model.resnet.layer2.parameters(), 'lr': 0.00002},
                {'params': model.resnet.layer3.parameters(), 'lr': 0.00005},
                {'params': model.resnet.layer4.parameters(), 'lr': 0.0001},
                {'params': model.embedding.parameters(), 'lr': 0.0002},
                {'params': model.classifier.parameters(), 'lr': 0.0002}
            ], weight_decay=0.01)
            
            scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, T_0=10, T_mult=2, eta_min=1e-6
            )
        
        # Train for one epoch
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, epoch, writer)
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, epoch, writer)
        
        # Step scheduler (for CosineAnnealing, step after each epoch)
        scheduler.step()
        
        # Record metrics
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # Print epoch summary with all learning rates
        print(f"\nEpoch {epoch+1}/{num_epochs} completed:")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        print(f"  Learning Rates:")
        for i, param_group in enumerate(optimizer.param_groups):
            print(f"    Group {i}: {param_group['lr']:.6f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
                'num_classes': model.classifier[-1].out_features,
                'embedding_size': model.embedding[0].out_features,
            }, f"{log_dir}/best_classification_model.pth")
            print(f"  ✓ New best model saved with validation accuracy: {val_acc:.2f}%")
        
        print("-" * 70)
    
    # Log model graph to TensorBoard
    dummy_input = torch.randn(1, 3, 224, 224).to(next(model.parameters()).device)
    writer.add_graph(model, dummy_input)
    
    print("\n" + "="*70)
    print(f"Training completed! Best validation accuracy: {best_val_acc:.2f}%")
    print("="*70 + "\n")
    
    return model, history


# ============================================================================
# COMPLETE SETUP AND TRAINING SCRIPT
# ============================================================================

# Device configuration
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using device:", torch.cuda.get_device_name(0))
elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using device: Apple Silicon MPS GPU")
else:
    device = torch.device("cpu")
    print("Using device: CPU")

# Better data augmentation
train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.3)
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# FIXED: Create single dataset and split it
print("\n" + "="*70)
print("Loading Caltech101 Dataset...")
print("="*70)
full_dataset = Caltech101Dataset(data_dir, transform=None, max_per_class=None, seed=42)

# Split dataset
train_indices, val_indices = train_test_split(
    range(len(full_dataset)), 
    test_size=0.2, 
    random_state=42, 
    stratify=full_dataset.labels
)

# Create subsets
train_subset = Subset(full_dataset, train_indices)
val_subset = Subset(full_dataset, val_indices)

# Apply transforms
train_dataset = TransformSubset(train_subset, transform=train_transform)
val_dataset = TransformSubset(val_subset, transform=val_transform)

print(f"\nDataset Split:")
print(f"  Train set size: {len(train_dataset)}")
print(f"  Validation set size: {len(val_dataset)}")
print(f"  Number of classes: {full_dataset.num_classes}")
print("="*70 + "\n")

# FIXED: Create DataLoaders
batch_size = 16
train_loader = DataLoader(
    train_dataset, 
    batch_size=batch_size, 
    shuffle=True,  # IMPORTANT: Shuffle for training!
    num_workers=0,  # Use 4 if not on Mac, 0 for Mac to avoid issues
    pin_memory=True if torch.cuda.is_available() else False
)
val_loader = DataLoader(
    val_dataset, 
    batch_size=batch_size, 
    shuffle=False, 
    num_workers=0,
    pin_memory=True if torch.cuda.is_available() else False
)

# Initialize the improved model
print("Initializing ResNet50 Transfer Learning Model...")
model = ResNetTransferModel(num_classes=101, embedding_size=128, pretrained=True).to(device)
print(f"Model loaded on {device}")
print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}\n")

# Setup TensorBoard
log_dir = f"./logs/classification_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
writer = SummaryWriter(log_dir)

# Train the model with ALL improvements
num_epochs = 30
model, history = train_model_progressive_complete(
    model, 
    train_loader, 
    val_loader, 
    writer, 
    num_epochs=num_epochs,
    log_dir=log_dir
)

# Close TensorBoard writer
writer.close()

# Plot training history
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history['train_loss'], label='Train Loss', linewidth=2)
plt.plot(history['val_loss'], label='Val Loss', linewidth=2)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.legend(fontsize=10)
plt.title('Training and Validation Loss', fontsize=14)
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(history['train_acc'], label='Train Acc', linewidth=2)
plt.plot(history['val_acc'], label='Val Acc', linewidth=2)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Accuracy (%)', fontsize=12)
plt.legend(fontsize=10)
plt.title('Training and Validation Accuracy', fontsize=14)
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{log_dir}/training_history.png', dpi=150)
plt.show()

print(f"\nTraining history saved to: {log_dir}/training_history.png")
print(f"Best model saved to: {log_dir}/best_classification_model.pth")
print(f"TensorBoard logs: {log_dir}")
print(f"\nTo view TensorBoard: tensorboard --logdir={log_dir}")

# ============================================================================
# SAVE SIMPLE MODEL COPY AS model.pth
# ============================================================================
print("\n" + "="*70)
print("Saving simplified model copy as 'model.pth'...")
print("="*70)

# Load the best model
best_checkpoint = torch.load(f"{log_dir}/best_classification_model.pth")
model.load_state_dict(best_checkpoint['model_state_dict'])

# Save just the model state dict (smaller file, easier to load)
torch.save(model.state_dict(), 'model.pth')
print("✓ Model saved as 'model.pth'")

# Also save with categories for easier loading later
torch.save({
    'model_state_dict': model.state_dict(),
    'categories': full_dataset.categories,
    'num_classes': full_dataset.num_classes,
    'embedding_size': 128,
    'val_acc': best_checkpoint['val_acc'],
}, 'model_with_info.pth')
print("✓ Model with metadata saved as 'model_with_info.pth'")
print("="*70 + "\n")