import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, regularizers
from sklearn.model_selection import train_test_split
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

def load_dataset(data_dir: str):
    '''
    load dataset from processed dataset directory
    '''
    images, labels = [], []
    categories = ['normal', 'scoliosis']

    for i, c in enumerate(categories):
        path = Path(data_dir) / c
        for img_path in path.glob('*.jpg'):
            img = tf.keras.preprocessing.image.load_img(
                img_path,
                color_mode='grayscale',
                target_size=(384, 128)
            )
            img_array = tf.keras.preprocessing.image.img_to_array(img)
            images.append(img_array)
            labels.append(i)
    
    return np.array(images), np.array(labels)

def create_augmentation_layers():
    '''
    create data augmentation pipeline
    '''
    return tf.keras.Sequential([
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
        layers.RandomTranslation(0.1, 0.1),
        layers.RandomBrightness(0.2),
        layers.RandomContrast(0.2),
    ])

def create_model():
    '''
    construct CNN model architecture
    '''
    inputs = layers.Input(shape=(384, 128, 1))
    
    x = create_augmentation_layers()(inputs, training=True)

    # first convolutional block
    x = layers.Conv2D(16, (3, 3), padding='same',
                      kernel_regularizer=regularizers.l2(0.01))(x)
    x = layers.LeakyReLU(0.1)(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    # second convolutional block
    x = layers.Conv2D(32, (3, 3), padding='same',
                      kernel_regularizer=regularizers.l2(0.01))(x)
    x = layers.LeakyReLU(0.1)(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    # third convolutional block
    x = layers.Conv2D(64, (3, 3), padding='same',
                      kernel_regularizer=regularizers.l2(0.01))(x)
    x = layers.LeakyReLU(0.1)(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    # dense layers
    x = layers.Flatten()(x)
    x = layers.Dropout(0.6)(x)
    x = layers.Dense(64, kernel_regularizer=regularizers.l2(0.01))(x)
    x = layers.LeakyReLU(0.1)(x)
    x = layers.BatchNormalization()(x)

    outputs = layers.Dense(2, activation='softmax')(x)

    return models.Model(inputs=inputs, outputs=outputs)

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

def train_model():
    X, y = load_dataset('dataset-processed')
    X = X / 255.0
    y = tf.keras.utils.to_categorical(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = create_model()
    model.compile(
        optimizer=optimizers.Adam(learning_rate=0.0001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=5,
            min_lr=1e-6
        )
    ]

    history = model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=32,
        validation_split=0.2,
        callbacks=callbacks,
        class_weight={
            0: 1,
            1: 1,
            2: 1,
        }
    )

    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    print(f"\nTest accuracy: {test_accuracy:.4f}")

    plot_loss(history)

    return model, history

if __name__ == "__main__":
    print("Training model:")
    
    tf.keras.mixed_precision.set_global_policy('mixed_float16')

    model, history = train_model()
    model.save('spine_classifier-no-spondy.keras')

