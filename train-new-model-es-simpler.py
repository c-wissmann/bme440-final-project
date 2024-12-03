import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import confusion_matrix, classification_report
import random
from tqdm import tqdm
import seaborn as sns

class ImageDataset(Dataset):
    def __init__(self, normal_dir, diseased_dir, transform=None, diseased_sample_size=None):
        self.transform = transform
        normal_paths = list(Path(normal_dir).glob('*.*'))
        diseased_paths = list(Path(diseased_dir).glob('*.*'))

        print(f"Found {len(normal_paths)} normal images")
        print(f"Found {len(diseased_paths)} diseased images")
        
        if diseased_sample_size is not None:
            if diseased_sample_size > len(diseased_paths):
                print(f"Warning: Requested sample size ({diseased_sample_size}) is larger than available diseased images ({len(diseased_paths)})")
                print("Using all available diseased images instead")
            else:
                diseased_paths = random.sample(diseased_paths, diseased_sample_size)
        
        self.image_paths = normal_paths + diseased_paths
        self.labels = ([0] * len(normal_paths)) + ([1] * len(diseased_paths))

        print(f"Dataset created with {len(normal_paths)} normal images and {len(diseased_paths)} diseased images")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, index):
        image = Image.open(self.image_paths[index])
        label = self.labels[index]

        if self.transform:
            image = self.transform(image)

        return image, label

class SimplerCNN(nn.Module):
    def __init__(self, input_channels=1):
        super(SimplerCNN, self).__init__()
        
        self.features = nn.Sequential(
            # First conv block
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Second conv block
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Third conv block
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.adaptive_pool = nn.AdaptiveAvgPool2d((1,1))
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        x = self.features(x)
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
        elif val_acc < self.best_acc + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_acc = val_acc
            self.counter = 0
        return self.early_stop

def train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=50):
    best_val_acc = 0.0
    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    early_stopping = EarlyStopping(patience=5)

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
            optimizer.step()

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

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pth')
        
        if early_stopping(val_acc):
            print(f"Early stopping triggered at epoch {epoch+1}")
            break
    
    return train_losses, val_losses, train_accs, val_accs

def plot_training_metrics(train_losses, val_losses, train_accs, val_accs, fold=None):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    ax1.plot(train_losses, label='training loss')
    ax1.plot(val_losses, label='validation loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title(f'Training and Validation Loss{"" if fold is None else f" (Fold {fold})"}')
    ax1.legend()

    ax2.plot(train_accs, label='training accuracy')
    ax2.plot(val_accs, label='validation accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title(f'Training and Validation Accuracy{"" if fold is None else f" (Fold {fold})"}')

    plt.tight_layout()
    return fig

def plot_confusion_matrix(y_true, y_pred, classes=['Normal', 'Diseased']):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    return plt.gcf()

def evaluate_model(model, data_loader, device):
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = outputs.max(1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return np.array(all_preds), np.array(all_labels)

def train_with_kfold(dataset, k_folds=5, num_epochs=50, batch_size=128):  # Increased batch size
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)

    fold_results = []

    for fold, (train_index, val_index) in enumerate(kfold.split(dataset)):
        print(f'\nFold {fold+1}/{k_folds}')
        print('-' * 50)

        train_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=SubsetRandomSampler(train_index),
            num_workers=4
        )
        val_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=SubsetRandomSampler(val_index),
            num_workers=4
        )

        model = SimplerCNN(input_channels=1).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        train_losses, val_losses, train_accs, val_accs = train_model(
            model, train_loader, val_loader, criterion, optimizer, device, num_epochs=num_epochs
        )

        fig = plot_training_metrics(train_losses, val_losses, train_accs, val_accs, fold+1)
        fig.savefig(f'og_fold_{fold+1}_training_metrics.png')
        plt.close(fig)

        val_preds, val_labels = evaluate_model(model, val_loader, device)

        cm_fig = plot_confusion_matrix(val_labels, val_preds)
        cm_fig.savefig(f'og_fold_{fold+1}_confusion_matrix.png')
        plt.close(cm_fig)

        report = classification_report(val_labels, val_preds,
                                    target_names=['Normal', 'Diseased'], output_dict=True)
        
        fold_results.append({
            'val_accuracy': report['accuracy'],
            'val_precision': report['weighted avg']['precision'],
            'val_recall': report['weighted avg']['recall'],
            'val_f1': report['weighted avg']['f1-score'],
            'confusion_matrix': confusion_matrix(val_labels, val_preds),
            'classification_report': report
        })

        torch.save(model.state_dict(), f'model_fold_{fold+1}.pth')
    
    print('\nAverage Results Across All Folds:')
    metrics = ['val_accuracy', 'val_precision', 'val_recall', 'val_f1']
    for metric in metrics:
        values = [fold[metric] for fold in fold_results]
        print(f'Average {metric}: {np.mean(values):.4f} +/- {np.std(values):.4f}')
    
    return fold_results

if __name__ == "__main__":
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)

    normal_dir = 'og-augmented/normal'
    diseased_dir = 'og-augmented/scoliosis'

    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    dataset = ImageDataset(
        normal_dir=normal_dir,
        diseased_dir=diseased_dir,
        transform=transform,
        diseased_sample_size=800
    )

    results = train_with_kfold(
        dataset,
        k_folds=5,
        num_epochs=50,  # Will likely stop earlier due to early stopping
        batch_size=128  # Increased from 32
    )