import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models

# Data directories
train_dir = 'data/train'
val_dir = 'data/val'

def build_model():
    base_model = tf.keras.applications.VGG16(include_top=False, input_shape=(224,224,3), weights='imagenet')
    base_model.trainable = False
    model = models.Sequential([
        base_model,
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

if __name__ == '__main__':
    train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=20, width_shift_range=0.2,
                                       height_shift_range=0.2, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
    val_datagen = ImageDataGenerator(rescale=1./255)
    train_generator = train_datagen.flow_from_directory(train_dir, target_size=(224,224), batch_size=32, class_mode='binary')
    val_generator = val_datagen.flow_from_directory(val_dir, target_size=(224,224), batch_size=32, class_mode='binary')

    model = build_model()
    model.fit(train_generator, epochs=10, validation_data=val_generator)
    model.save('model.h5')
