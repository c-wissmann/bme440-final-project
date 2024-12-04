import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import random
from tqdm import tqdm
import seaborn as sns
import os
import shutil

class ImageDataset(Dataset):
    def __init__(self, normal_dir, diseased_dir, transform=None, diseased_sample_size=None):
        self.transform = transform
        normal_paths = list(Path(normal_dir).glob('*.*'))
        diseased_paths = list(Path(diseased_dir).glob('*.*'))

        print(f"Found {len(normal_paths)} normal images")
        print(f"Found {len(diseased_paths)} diseased images")

        if diseased_sample_size is not None:
            if diseased_sample_size > len(diseased_paths):
                print(f"Warning: requested sample sixe ({diseased_sample_size}) is larger than available diseased images ({len(diseased_paths)})")
                print("Using all available diseased images instead")
            else:
                diseased_paths = random.sample(diseased_paths, diseased_sample_size)

        self.image_paths = normal_paths + diseased_paths
        self.labels = ([0] * len(normal_paths) + ([1] * len(diseased_paths)))

        print(f"Dataset created with {len(normal_paths)} normal images and {len(diseased_paths)} diseased images")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, index):
        image = Image.open(self.image_paths[index])
        label = self.labels[index]

        if self.transform: image = self.transform(image)

        return image, label
    
class ImprovedCNN(nn.Module):
    def __init__(self, input_channels=1, dropout_rate=0.3):
        super(ImprovedCNN, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.1)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.15)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.2)
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.25)
        )

        self.adaptive_pool = nn.AdaptiveAvgPool2d((1,1))

        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout_rate/2),
            nn.Linear(128,2)
        )

        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                    nn.init_constant_(m.bias, 0)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_acc = None
        self.early_stop = False
        
    def __call__(self, val_acc):
        if self.best_acc is None:
            self.best_acc = val_acc
        elif val_acc < self.best_acc - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        elif val_acc > self.best_acc:
            self.best_acc = val_acc
            self.counter = 0
        return self.early_stop

def save_checkpoint(model, epoch, val_acc, path='checkpoints'):
    os.makedirs(path, exist_ok=True)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'val_acc': val_acc
    }, f'{path}/checkpoint_epoch_{epoch}_acc_{val_acc:.2f}.pth')

def average_checkpoints(checkpoint_paths):
    averaged_state = torch.load(checkpoint_paths[0])['model_state_dict']

    for k in averaged_state.keys():
        averaged_state[k] = averaged_state[k].float()
    
    for path in checkpoint_paths[1:]:
        state = torch.load(path)['model_state_dict']
        for k in averaged_state.keys():
            averaged_state[k] += state[k].float()
    
    for k in averaged_state.keys():
        averaged_state[k] = averaged_state[k] / len(checkpoint_paths)
    
    return averaged_state

def ensemble_predict(models, input_tensor, device):
    predictions = []

    with torch.no_grad():
        for model in models:
            model.eval()
            output = model(input_tensor)
            pred = output.max(1)[1]
            predictions.append(pred)
    
    stacked_preds = torch.stack(predictions)
    final_pred = torch.mode(stacked_preds, dim=0)[0]

    return final_pred

def train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=50):
    best_val_acc = 0.0
    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    early_stopping = EarlyStopping(patience=15)
    checkpoint_accs = []

    total_steps = len(train_loader) * num_epochs

    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=0.001,
        total_steps=total_steps,
        pct_start=0.2,
        div_factor=10,
        final_div_factor=1e3,
        anneal_strategy='cos'
    )

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}'):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            
            # Add gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            scheduler.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        train_loss = running_loss / len(train_loader)
        train_acc = 100. * correct / total

        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        val_loss = val_loss / len(val_loader)
        val_acc = 100. * correct / total

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')

        if val_acc > 90:
            save_checkpoint(model, epoch, val_acc)
            checkpoint_accs.append((epoch, val_acc))

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pth')
        
        if early_stopping(val_acc):
            print(f'Early stopping triggered at epoch {epoch+1}')
            break
    
    if len(checkpoint_accs) >= 3:
        top_n = min(5, len(checkpoint_accs))
        best_checkpoints = sorted(checkpoint_accs, key=lambda x: x[1], reverse=True)[:top_n]
        checkpoint_paths = [f'checkpoints/checkpoint_epoch_{epoch}_acc_{acc:.2f}.pth'
                            for epoch, acc in best_checkpoints]
        
        ensemble_models = []
        for path in checkpoint_paths:
            model_state = torch.load(path)['model_state_dict']
            model_copy = ImprovedCNN(input_channels=1).to(device)
            model_copy.load_state_dict(model_state)
            ensemble_models.append(model_copy)
        
        averaged_state = average_checkpoints(checkpoint_paths)
        averaged_model = ImprovedCNN(input_channels=1).to(device)
        averaged_model.load_state_dict(averaged_state)

        ensemble_models.append(averaged_model)

        return (train_losses, val_losses, train_accs, val_accs), ensemble_models

    return (train_losses, val_losses, train_accs, val_accs), [model]

def evaluate_ensemble(models, data_loader, device):
    correct = 0
    total = 0
    all_preds, all_labels = [], []

    for inputs, labels in data_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        predictions = ensemble_predict(models, inputs, device)
        total += labels.size(0)
        correct += (predictions == labels).sum().item()

        all_preds.extend(predictions.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    accuracy = 100. * correct / total
    return accuracy, np.array(all_preds), np.array(all_labels)

def plot_training_metrics(train_losses, val_losses, train_accs, val_accs):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15,5))

    ax1.plot(train_losses, label='training loss')
    ax1.plot(val_losses, label='validation loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training + Validation Loss')
    ax1.legend()
    ax1.grid(True)

    ax2.plot(train_accs, label='training accuracy')
    ax2.plot(val_accs, label='validation accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Training + Validation Accuracy')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    return fig

if __name__ == "__main__":
    if os.path.exists('checkpoints'):
        shutil.rmtree('checkpoints')
    os.makedirs('checkpoints')

    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)

    normal_dir = 'og-augmented/normal'
    diseased_dir = 'og-augmented/scoliosis'

    # Enhanced data augmentation
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    dataset = ImageDataset(
        normal_dir=normal_dir,
        diseased_dir=diseased_dir,
        transform=transform,
        diseased_sample_size=1000
    )

    train_indices, val_indices = train_test_split(
        range(len(dataset)),
        test_size=0.2,
        random_state=42,
        stratify=[dataset.labels[i] for i in range(len(dataset))]
    )

    train_loader = DataLoader(
        dataset,
        batch_size=32,
        sampler=torch.utils.data.SubsetRandomSampler(train_indices),
        num_workers=4
    )

    val_loader = DataLoader(
        dataset,
        batch_size=32,
        sampler=torch.utils.data.SubsetRandomSampler(val_indices),
        num_workers=4
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ImprovedCNN(input_channels=1).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)

    # Train the model
    metrics, ensemble_models = train_model(
        model, train_loader, val_loader, criterion, optimizer, device, num_epochs=100
    )
    train_losses, val_losses, train_accs, val_accs = metrics

    # Plot and save training metrics
    fig = plot_training_metrics(train_losses, val_losses, train_accs, val_accs)
    fig.savefig('training_metrics_improved.png')
    plt.close(fig)

    # Evaluate ensemble performance
    ensemble_acc, ensemble_preds, ensemble_labels = evaluate_ensemble(
        ensemble_models, val_loader, device
    )

    print(f"\nEnsemble Accuracy: {ensemble_acc:.2f}%")

    # Plot and save confusion matrix
    cm_fig = plt.figure(figsize=(8,6))
    sns.heatmap(confusion_matrix(ensemble_labels, ensemble_preds),
                annot=True, fmt='d', cmap='Blues',
                xticklabels=['Normal', 'Diseased'],
                yticklabels=['Normal', 'Diseased'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    cm_fig.savefig('confusion_matrix_improved.png')
    plt.close(cm_fig)

    print("\nClassification Report:")
    print(classification_report(ensemble_labels, ensemble_preds,
                              target_names=['Normal', 'Diseased']))