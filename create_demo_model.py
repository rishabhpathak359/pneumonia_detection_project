
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout, BatchNormalization
from tensorflow.keras.applications import VGG16
import numpy as np

def create_demo_model():
    """Create a demo model for testing purposes (not trained on real data)"""

    # Create models directory
    os.makedirs('models', exist_ok=True)

    print("Creating demo pneumonia detection model...")

    # Build a simple CNN architecture (same as the real model)
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
        MaxPooling2D(2, 2),
        BatchNormalization(),

        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        BatchNormalization(),

        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        BatchNormalization(),

        Flatten(),
        Dropout(0.5),
        Dense(512, activation='relu'),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])

    # Compile the model
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    # Initialize with random weights (for demo purposes)
    # In real training, this would be trained on actual data
    dummy_input = np.random.random((1, 224, 224, 3))
    _ = model.predict(dummy_input)

    # Save the demo model
    model.save('models/pneumonia_demo_model.h5')
    print("Demo model saved as 'models/pneumonia_demo_model.h5'")

    # Also create a VGG16-based model for comparison
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = False

    transfer_model = Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])

    transfer_model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    # Initialize the transfer learning model
    _ = transfer_model.predict(dummy_input)

    # Save the transfer learning model
    transfer_model.save('models/pneumonia_vgg16_model.h5')
    print("Transfer learning model saved as 'models/pneumonia_vgg16_model.h5'")

    print("\nDemo models created successfully!")
    print("\nNOTE: These are untrained demo models for testing the web interface.")
    print("To get accurate results, please:")
    print("1. Download the chest X-ray dataset from Kaggle")
    print("2. Run 'python train_model.py' to train the models properly")

    return model, transfer_model

if __name__ == "__main__":
    create_demo_model()
