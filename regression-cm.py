import os
import numpy as np
from PIL import Image
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

class ImageClassificationComparison:
    def __init__(self, folder_path, test_size=0.2, random_state=42):
        self.folder_path = folder_path
        self.X, self.y = self.load_images_from_folder(folder_path)
        
        self.test_size = test_size
        self.random_state = random_state
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=random_state)
        
        self.regular_model = None
        self.logistic_model = None
        self.results = {}
        self.class_names = ['normal', 'scoliosis']

    def load_images_from_folder(self, folder_path):
        images = []
        labels = []
        class_names = ['normal', 'scoliosis']
        label_encoder = LabelEncoder()
        label_encoder.fit(class_names)
        
        for class_name in class_names:
            class_folder = os.path.join(folder_path, class_name)
            if os.path.isdir(class_folder):
                for file_name in os.listdir(class_folder):
                    if file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                        img_path = os.path.join(class_folder, file_name)
                        img = Image.open(img_path)
                        img_array = np.array(img)
                        img_array = img_array.flatten()
                        images.append(img_array)
                        labels.append(class_name)
        
        X = np.array(images)
        y = label_encoder.transform(labels)
        
        return X, y
    
    def train_regular(self):
        self.regular_model = LinearRegression()
        y_train_float = self.y_train.astype(float)
        
        self.regular_model.fit(self.X_train, y_train_float)
        y_pred = self.regular_model.predict(self.X_test)
        y_pred_rounded = np.round(y_pred).clip(0, 1).astype(int)
        
        accuracy = accuracy_score(self.y_test, y_pred_rounded)
        conf_matrix = confusion_matrix(self.y_test, y_pred_rounded)
        
        self.results['regular'] = {
            'predictions': y_pred_rounded,
            'accuracy': accuracy,
            'confusion_matrix': conf_matrix
        }
    
    def train_logistic(self):
        self.logistic_model = LogisticRegression(solver='lbfgs', max_iter=1000)
        self.logistic_model.fit(self.X_train, self.y_train)
        
        y_pred = self.logistic_model.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        conf_matrix = confusion_matrix(self.y_test, y_pred)
        
        self.results['logistic'] = {
            'predictions': y_pred,
            'accuracy': accuracy,
            'confusion_matrix': conf_matrix
        }
    
    def plot_confusion_matrices(self):
        fig, axes = plt.subplots(1, 2, figsize=(15, 7.5))
        
        methods = ['regular', 'logistic']
        titles = ['Regular Linear Regression', 'Logistic Regression']
        
        for ax, method, title in zip(axes, methods, titles):
            cm = self.results[method]['confusion_matrix']
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                       xticklabels=self.class_names, 
                       yticklabels=self.class_names)
            ax.set_title(f'{title}\nAccuracy: {self.results[method]["accuracy"]:.3f}')
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
        
        plt.tight_layout()
        return fig
    
    def print_comparison_report(self):
        print("\nModel Comparison Report:")
        print("=" * 50)
        
        methods = ['regular', 'logistic']
        names = ['Regular Linear Regression', 'Logistic Regression']
        
        for method, name in zip(methods, names):
            print(f"\n{name}:")
            print("-" * 30)
            print(f"Accuracy: {self.results[method]['accuracy']:.4f}")
            print("\nConfusion Matrix:")
            print(self.results[method]['confusion_matrix'])

# Usage
comparison = ImageClassificationComparison('og-augmented')
comparison.train_regular()
comparison.train_logistic()
comparison.print_comparison_report()
plot = comparison.plot_confusion_matrices()
plot.savefig('regression_confusion_matrices-balanced.png')
plt.close(plot)