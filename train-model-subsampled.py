import tensorflow as tf
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from collections import Counter
import matplotlib.pyplot as plt

def load_data(data_dir, target_size=(384,128)):
    images, labels = [], []
    categories = ['normal', 'scoliosis', 'spondylolisthesis']

    class_sizes = []
    for category in categories:
        size = len(list(Path(data_dir).glob(f'{category}/*.jpg')))
        class_sizes.append(size)
    min_size = min(class_sizes)

    for i, c in enumerate(categories):
        paths = list(Path(data_dir).glob(f'{c}/*.jpg'))
        np.random.shuffle(paths)
        for img_path in paths[:min_size]:
            img = tf.keras.preprocessing.image.load_img(
                img_path,
                color_mode='grayscale',
                target_size=target_size
            )
            img_array = tf.keras.preprocessing.image.img_to_array(img)
            images.append(img_array)
            labels.append(i)

    print(min_size)
    return np.array(images), np.array(labels)

def plot_loss(history):
    ## Plot training and validation loss
    plt.figure(figsize=(8, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss', linestyle='--')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Model Loss')
    plt.legend()
    plt.grid()
    plt.show()

def create_residual_block(x, filters, kernel_size=3):
   shortcut = x
   x = tf.keras.layers.Conv2D(filters, kernel_size, padding='same')(x)
   x = tf.keras.layers.BatchNormalization()(x)
   x = tf.keras.layers.ReLU()(x)
   x = tf.keras.layers.Conv2D(filters, kernel_size, padding='same')(x)
   x = tf.keras.layers.BatchNormalization()(x)
   
   if shortcut.shape[-1] != filters:
       shortcut = tf.keras.layers.Conv2D(filters, 1, padding='same')(shortcut)
   
   x = tf.keras.layers.Add()([shortcut, x])
   x = tf.keras.layers.ReLU()(x)
   return x

def create_model():
   inputs = tf.keras.layers.Input(shape=(384, 128, 1))
   
   # Data augmentation
   x = tf.keras.Sequential([
       tf.keras.layers.RandomRotation(0.2),
       tf.keras.layers.RandomZoom(0.2),
       tf.keras.layers.RandomTranslation(0.1, 0.1),
       tf.keras.layers.RandomBrightness(0.3),
       tf.keras.layers.RandomContrast(0.3),
       tf.keras.layers.GaussianNoise(0.1)
   ])(inputs)
   
   # Initial convolution
   x = tf.keras.layers.Conv2D(32, 7, strides=2, padding='same')(x)
   x = tf.keras.layers.BatchNormalization()(x)
   x = tf.keras.layers.ReLU()(x)
   x = tf.keras.layers.MaxPooling2D()(x)
   
   # Residual blocks
   x = create_residual_block(x, 32)
   x = tf.keras.layers.MaxPooling2D()(x)
   x = create_residual_block(x, 64)
   x = tf.keras.layers.MaxPooling2D()(x)
   x = create_residual_block(x, 128)
   
   # Dense layers
   x = tf.keras.layers.GlobalAveragePooling2D()(x)
   x = tf.keras.layers.Dense(128, kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
   x = tf.keras.layers.BatchNormalization()(x)
   x = tf.keras.layers.ReLU()(x)
   x = tf.keras.layers.Dropout(0.5)(x)
   outputs = tf.keras.layers.Dense(3, activation='softmax')(x)
   
   return tf.keras.Model(inputs, outputs)

# Load and preprocess data
X, y = load_data('dataset-processed')
X = X / 255.0
y = tf.keras.utils.to_categorical(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and compile model
model = create_model()
model.compile(
   optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
   loss='categorical_crossentropy',
   metrics=['accuracy']
)

# Train with warmup and cycling learning rate
initial_learning_rate = 0.001
decay_steps = 1000
decay_rate = 0.9

lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
   initial_learning_rate=initial_learning_rate,
   decay_steps=decay_steps,
   decay_rate=decay_rate
)

callbacks = [
   tf.keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True),
   tf.keras.callbacks.ReduceLROnPlateau(
       monitor='val_loss',
       factor=0.5,
       patience=5,
       min_lr=1e-6
   )
]

history = model.fit(
   X_train, y_train,
   epochs=100,
   batch_size=16,
   validation_split=0.2,
   callbacks=callbacks,
   class_weight={0: 1.0, 1: 1.2, 2: 1.2}
)

# Evaluate and show detailed results
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_test_classes = np.argmax(y_test, axis=1)

print("\nPrediction Probabilities:")
for i, (true, pred_probs) in enumerate(zip(y_test_classes, y_pred)):
   true_class = ['normal', 'scoliosis', 'spondylolisthesis'][true]
   pred_class = ['normal', 'scoliosis', 'spondylolisthesis'][np.argmax(pred_probs)]
   print(f"\nSample {i}:")
   print(f"True class: {true_class}")
   print(f"Predicted class: {pred_class}")
   print("Class probabilities:")
   for j, prob in enumerate(['normal', 'scoliosis', 'spondylolisthesis']):
       print(f"{prob}: {pred_probs[j]:.3f}")

from sklearn.metrics import confusion_matrix, classification_report

cm = confusion_matrix(y_test_classes, y_pred_classes)
categories = ['normal', 'scoliosis', 'spondylolisthesis']

print("\nClassification Report:")
print(classification_report(y_test_classes, y_pred_classes, target_names=categories))

print("\nConfusion Matrix")
print(cm)

print("\nDetailed Misclassifications:")
for i, (true, pred) in enumerate(zip(y_test_classes, y_pred_classes)):
    if true != pred:
        print(f"Sample {i}: True={categories[true]}, Predicted={categories[pred]}")

plot_loss(history)

