from tensorflow.keras import models, layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def create_and_save_demo():
    model = models.Sequential([
        layers.InputLayer(input_shape=(224,224,3)),
        layers.Conv2D(16, 3, activation='relu'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    # skip training, save untrained model for demo
    model.save('model.h5')

if __name__ == '__main__':
    create_and_save_demo()
