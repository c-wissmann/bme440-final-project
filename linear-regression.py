### Imports ###
import os
import numpy as np
import matplotlib.pyplot as plt
import time
from PIL import Image
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder


class ImageClassificationComparison:
    def __init__(self, folder_path, test_size=0.2, random_state=42):
        self.load_start = time.time()
        self.folder_path = folder_path
        self.X, self.y = self.load_images_from_folder(folder_path)
        self.load_time = time.time() - self.load_start

        self.test_size = test_size
        self.random_state = random_state

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=random_state)
        
        self.regular_model = None
        self.onehot_model = None
        self.logistic_model = None
        self.results = {}

    def load_images_from_folder(self, folder_path):
        images = []
        labels = []
        class_names = ['normal', 'scoliosis']  # Define the two classes
        label_encoder = LabelEncoder()
        label_encoder.fit(class_names)  # Encode the class labels as integers
        
        for class_name in class_names:
            class_folder = os.path.join(folder_path, class_name)
            if os.path.isdir(class_folder):
                for file_name in os.listdir(class_folder):
                    if file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                        img_path = os.path.join(class_folder, file_name)
                        img = Image.open(img_path)
                        img_array = np.array(img)  # Convert to numpy array
                        img_array = img_array.flatten()  # Flatten the image
                        images.append(img_array)
                        labels.append(class_name)
        
        X = np.array(images)
        y = label_encoder.transform(labels)  # Encode the labels as integers
        
        return X, y
    
    def train_regular(self):
        self.regular_model = LinearRegression()
        y_train_float = self.y_train.astype(float)

        train_start = time.time()
        self.regular_model.fit(self.X_train, y_train_float)
        train_time = time.time() - train_start

        predict_start = time.time()
        y_pred = self.regular_model.predict(self.X_test)
        predict_time = time.time() - predict_start

        postprocess_start = time.time()
        y_pred_rounded = np.round(y_pred).clip(0,39)
        postprocess_time = time.time() - postprocess_start

        mse = mean_squared_error(self.y_test, y_pred_rounded)
        r2 = r2_score(self.y_test, y_pred_rounded)
        accuracy = accuracy_score(self.y_test, y_pred_rounded)

        self.results['regular'] = {
            'raw_predictions': y_pred,
            'predictions': y_pred_rounded,
            'rounded_predictions': y_pred_rounded,
            'mse': mse,
            'r2': r2,
            'accuracy': accuracy,
            'train_time': train_time,
            'predict_time': predict_time,
            'postprocess_time': postprocess_time,
            'total_time': train_time + predict_time + postprocess_time
        }
    
    def train_onehot(self):
        preprocess_start = time.time()
        oh = OneHotEncoder(sparse_output=False)
        y_train_oh = oh.fit_transform(self.y_train.reshape(-1,1))
        preprocess_time = time.time() - preprocess_start

        train_start = time.time()
        self.onehot_model = LinearRegression()
        self.onehot_model.fit(self.X_train, y_train_oh)
        train_time = time.time() - train_start

        predict_start = time.time()
        y_pred_oh = self.onehot_model.predict(self.X_test)
        predict_time = time.time() - predict_start

        postprocess_start = time.time()
        y_pred_labels = np.argmax(y_pred_oh, axis=1)
        confidences = np.max(y_pred_oh, axis=1)
        postprocess_time = time.time() - postprocess_start

        accuracy = accuracy_score(self.y_test, y_pred_labels)

        self.results['onehot'] = {
            'raw_predictions': y_pred_oh,
            'predictions': y_pred_labels,
            'confidences': confidences,
            'accuracy': accuracy,
            'preprocess_time': preprocess_time,
            'train_time': train_time,
            'predict_time': predict_time,
            'postprocess_time': postprocess_time,
            'total_time': preprocess_time + train_time + predict_time + postprocess_time
        }
    
    def train_logistic(self):
        self.logistic_model = LogisticRegression(solver='lbfgs', max_iter=1000)

        train_start = time.time()
        self.logistic_model.fit(self.X_train, self.y_train)
        train_time = time.time() - train_start

        predict_start = time.time()
        y_pred = self.logistic_model.predict(self.X_test)
        predict_time = time.time() - predict_start

        prob_start = time.time()
        y_pred_proba = self.logistic_model.predict_proba(self.X_test)
        confidences = np.max(y_pred_proba, axis=1)
        prob_time = time.time() - prob_start
        
        accuracy = accuracy_score(self.y_test, y_pred)

        self.results['logistic'] = {
            'predictions': y_pred,
            'probabilities': y_pred_proba,
            'confidences': confidences,
            'accuracy': accuracy,
            'train_time': train_time,
            'predict_time': predict_time,
            'prob_time': prob_time,
            'total_time': train_time + predict_time + prob_time
        }
    
    def plot_comparisons(self):
        fig, axs = plt.subplots(3,2,figsize=(15,18))

        # regular
        axs[0,0].scatter(self.y_test, self.results['regular']['predictions'], alpha=0.5)
        axs[0,0].plot([0,1], [0,1], 'r--', lw=2)
        axs[0,0].set_xlabel('Actual class labels')
        axs[0,0].set_ylabel('Predicted labels (rounded)')
        axs[0,0].set_title('Regular LR: actual vs predicted (rounded)')
        axs[0,0].grid(True)

        # jitter to show density
        jitter = np.random.normal(0, 0.1, size=len(self.y_test))

        # onehot
        axs[0,1].scatter(self.y_test + jitter, self.results['onehot']['predictions'] + jitter, alpha=0.5)
        axs[0,1].plot([0,1], [0,1], 'r--', lw=2)
        axs[0,1].set_xlabel('Actual class labels')
        axs[0,1].set_ylabel('Predicted labels')
        axs[0,1].set_title('OneHot LR: actual vs predicted')
        axs[0,1].grid(True)

        # logistic
        axs[1,0].scatter(self.y_test + jitter, self.results['logistic']['predictions'] + jitter, alpha=0.5)
        axs[1,0].plot([0,1], [0,1], 'r--', lw=2)
        axs[1,0].set_xlabel('Actual class labels')
        axs[1,0].set_ylabel('Predicted labels')
        axs[1,0].set_title('Logistic Regression: actual vs predicted')
        axs[1,0].grid(True)

        # confidences
        axs[1,1].hist(self.results['onehot']['confidences'],
                      alpha=0.5, label='OneHot', bins=30)
        axs[1,1].hist(self.results['logistic']['confidences'],
                      alpha=0.5, label='Logistic', bins=30)
        axs[1,1].set_xlabel('Confidence score')
        axs[1,1].set_ylabel('Count')
        axs[1,1].set_title('Confidence score distributions')
        axs[1,1].legend()
        axs[1,1].grid(True)

        # timing
        methods = ['regular','onehot','logistic']
        times = [self.results['regular']['total_time'],
                 self.results['onehot']['total_time'],
                 self.results['logistic']['total_time']]
        accuracies = [self.results['regular']['accuracy'],
                      self.results['onehot']['accuracy'],
                      self.results['logistic']['accuracy']]
        
        ax2 = axs[2,0].twinx()
        axs[2,0].bar(methods, times, alpha=0.7, color='b', label='Total time')
        ax2.plot(methods, accuracies, 'r-o', label='Accuracy')
        axs[2,0].set_ylabel('time (s)', color='b')
        ax2.set_ylabel('accuracy', color='r')
        axs[2,0].set_title('Time vs Accuracy')

        # detailed timing
        phases = ['preprocess/train', 'predict', 'postprocess']
        regular_times = [self.results['regular']['train_time'],
                        self.results['regular']['predict_time'],
                        self.results['regular']['postprocess_time']]
        
        onehot_times = [self.results['onehot']['train_time'] + self.results['onehot']['preprocess_time'],
                       self.results['onehot']['predict_time'],
                       self.results['onehot']['postprocess_time']]
        
        logistic_times = [self.results['logistic']['train_time'],
                         self.results['logistic']['predict_time'],
                         self.results['logistic']['prob_time']]
        
        x = np.arange(len(phases))
        width = 0.25
        axs[2,1].bar(x - width, regular_times, width, label='regular', alpha=0.7)
        axs[2,1].bar(x, onehot_times, width, label='onehot', alpha=0.7)
        axs[2,1].bar(x + width, logistic_times, width, label='logistic', alpha=0.7)
        axs[2,1].set_ylabel('time (s)')
        axs[2,1].set_title('Computation time by phase')
        axs[2,1].set_xticks(x)
        axs[2,1].set_xticklabels(phases)
        axs[2,1].legend()
        axs[2,1].grid(True)

        plt.tight_layout()
        return fig
    
    def print_comparison_report(self):
        print("\nModel Comparison Report:")
        print("=" * 50)

        methods = ['regular', 'onehot', 'logistic']
        names = ['Regular Linear Regression','OneHot Linear Regression','Logistic Regression']

        for method, name in zip(methods, names):
            print(f"\n{name}:")
            print("-" * 30)
            print(f"Accuracy: {self.results[method]['accuracy']:.4f}")
            print(f"Total Time: {self.results[method]['total_time']:.4f} seconds")
            if method == 'regular':
                print(f"MSE (rounded): {self.results[method]['mse']:.4f}")
                print(f"R^2 score (rounded): {self.results[method]['r2']:.4f}")
            if method in ['onehot', 'logistic']:
                print(f"Mean confidence: {np.mean(self.results[method]['confidences']):.4f}")
        
        print("\nTime-Accuracy Tradeoff:")
        for method, name in zip(methods, names):
            time_per_acc = self.results[method]['total_time'] / self.results[method]['accuracy']
            print(f"{name}: {time_per_acc:.4f} seconds per accuracy point")


# run everything
comparison = ImageClassificationComparison('dataset-processed')
comparison.train_regular()
comparison.train_onehot()
comparison.train_logistic()
comparison.print_comparison_report()
plot = comparison.plot_comparisons()
plot.savefig('regression_results.png')
plt.close(plot)