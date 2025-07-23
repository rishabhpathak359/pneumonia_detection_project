
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Flatten, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import warnings
warnings.filterwarnings('ignore')

class PneumoniaDetectionModel:
    def __init__(self, img_size=(224, 224), batch_size=32):
        self.img_size = img_size
        self.batch_size = batch_size
        self.model = None

    def create_data_generators(self, train_dir, val_dir, test_dir):
        """Create data generators for training, validation, and testing"""

        # Data augmentation for training
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
        )

        # Only rescaling for validation and test
        val_test_datagen = ImageDataGenerator(rescale=1./255)

        # Create generators
        train_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='binary',
            classes=['NORMAL', 'PNEUMONIA']
        )

        val_generator = val_test_datagen.flow_from_directory(
            val_dir,
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='binary',
            classes=['NORMAL', 'PNEUMONIA']
        )

        test_generator = val_test_datagen.flow_from_directory(
            test_dir,
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='binary',
            classes=['NORMAL', 'PNEUMONIA'],
            shuffle=False
        )

        return train_generator, val_generator, test_generator

    def build_cnn_model(self):
        """Build CNN model from scratch"""
        model = Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=(*self.img_size, 3)),
            layers.MaxPooling2D(2, 2),
            layers.BatchNormalization(),

            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D(2, 2),
            layers.BatchNormalization(),

            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.MaxPooling2D(2, 2),
            layers.BatchNormalization(),

            layers.Conv2D(256, (3, 3), activation='relu'),
            layers.MaxPooling2D(2, 2),
            layers.BatchNormalization(),

            layers.Flatten(),
            layers.Dropout(0.5),
            layers.Dense(512, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(1, activation='sigmoid')
        ])

        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )

        return model

    def build_transfer_learning_model(self):
        """Build model using VGG16 transfer learning"""
        # Load pre-trained VGG16 model
        base_model = VGG16(
            weights='imagenet',
            include_top=False,
            input_shape=(*self.img_size, 3)
        )

        # Freeze base model layers
        base_model.trainable = False

        # Add custom classifier
        model = Sequential([
            base_model,
            GlobalAveragePooling2D(),
            Dropout(0.2),
            Dense(1, activation='sigmoid')
        ])

        model.compile(
            optimizer=Adam(learning_rate=0.0001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )

        return model

    def train_model(self, model, train_gen, val_gen, epochs=20, model_name='pneumonia_model'):
        """Train the model"""

        # Callbacks
        callbacks = [
            ModelCheckpoint(
                f'models/{model_name}.h5',
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            ),
            EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=3,
                min_lr=0.0001
            )
        ]

        # Train model
        history = model.fit(
            train_gen,
            epochs=epochs,
            validation_data=val_gen,
            callbacks=callbacks,
            verbose=1
        )

        return history

    def evaluate_model(self, model, test_gen):
        """Evaluate model performance"""
        test_loss, test_acc = model.evaluate(test_gen, verbose=0)
        print(f"Test Accuracy: {test_acc:.4f}")
        print(f"Test Loss: {test_loss:.4f}")

        return test_loss, test_acc

    def plot_training_history(self, history):
        """Plot training history"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        # Plot accuracy
        ax1.plot(history.history['accuracy'], label='Training Accuracy')
        ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()

        # Plot loss
        ax2.plot(history.history['loss'], label='Training Loss')
        ax2.plot(history.history['val_loss'], label='Validation Loss')
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()

        plt.tight_layout()
        plt.savefig('models/training_history.png')
        plt.show()

def main():
    """Main function to train the model"""

    # Create models directory
    os.makedirs('models', exist_ok=True)

    # Note: You need to download the dataset from Kaggle
    # Dataset: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

    print("Pneumonia Detection Model Training")
    print("==================================")
    print()
    print("IMPORTANT: Please download the chest X-ray dataset from:")
    print("https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia")
    print()
    print("Extract the dataset and organize it as follows:")
    print("chest_xray/")
    print("├── train/")
    print("│   ├── NORMAL/")
    print("│   └── PNEUMONIA/")
    print("├── val/")
    print("│   ├── NORMAL/")
    print("│   └── PNEUMONIA/")
    print("└── test/")
    print("    ├── NORMAL/")
    print("    └── PNEUMONIA/")
    print()

    # Check if dataset exists
    if not os.path.exists('chest_xray'):
        print("Dataset not found! Please download and extract the dataset first.")
        return

    # Initialize model
    detector = PneumoniaDetectionModel()

    # Create data generators
    train_gen, val_gen, test_gen = detector.create_data_generators(
        'chest_xray/train',
        'chest_xray/val', 
        'chest_xray/test'
    )

    print(f"Training samples: {train_gen.samples}")
    print(f"Validation samples: {val_gen.samples}")
    print(f"Test samples: {test_gen.samples}")

    # Build and train CNN model
    print("\nTraining CNN Model...")
    cnn_model = detector.build_cnn_model()
    cnn_model.summary()

    cnn_history = detector.train_model(
        cnn_model, train_gen, val_gen, 
        epochs=25, model_name='pneumonia_cnn_model'
    )

    # Evaluate CNN model
    print("\nEvaluating CNN Model...")
    detector.evaluate_model(cnn_model, test_gen)
    detector.plot_training_history(cnn_history)

    # Build and train Transfer Learning model
    print("\nTraining Transfer Learning Model (VGG16)...")
    tl_model = detector.build_transfer_learning_model()
    tl_model.summary()

    tl_history = detector.train_model(
        tl_model, train_gen, val_gen,
        epochs=15, model_name='pneumonia_vgg16_model'
    )

    # Evaluate Transfer Learning model
    print("\nEvaluating Transfer Learning Model...")
    detector.evaluate_model(tl_model, test_gen)
    detector.plot_training_history(tl_history)

    print("\nTraining completed! Models saved in the 'models' directory.")

if __name__ == "__main__":
    main()
