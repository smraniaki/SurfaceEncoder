
from PIL import Image
import os
from tensorflow.keras import layers, models
import numpy as np

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Conv2DTranspose, Dense, Reshape
from tensorflow.keras.models import Model
import numpy as np
import matplotlib.pyplot as plt

original_image = Image.open('Subss.jpg')

# tile_size = (64, 64)
tile_size = (128, 128)

tiles_x = original_image.size[0] // tile_size[0]
tiles_y = original_image.size[1] // tile_size[1]

images_dir = 'images/all'
os.makedirs(images_dir, exist_ok=True)

for i in range(tiles_x):
    for j in range(tiles_y):
        left = i * tile_size[0]
        upper = j * tile_size[1]
        right = left + tile_size[0]
        lower = upper + tile_size[1]

        tile = original_image.crop((left, upper, right, lower))
        tile_path = os.path.join(images_dir, f"tile_{i}_{j}.jpg")
        tile.save(tile_path)

latent_dim = 10

encoder_input = layers.Input(shape=(128, 128, 1))
x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(encoder_input)
x = layers.MaxPooling2D((2, 2), padding='same')(x)
x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
x = layers.MaxPooling2D((2, 2), padding='same')(x)
# x = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(x) # New layer
# x = layers.MaxPooling2D((2, 2), padding='same')(x) # New layer
shape_before_flattening = x.shape[1:]
x = layers.Flatten()(x)
latent_vector = layers.Dense(latent_dim, name='latent_vector')(x)

encoder = models.Model(encoder_input, latent_vector, name='encoder')

decoder_input = layers.Input(shape=(latent_dim,))
x = layers.Dense(np.prod(shape_before_flattening))(decoder_input)
x = layers.Reshape(shape_before_flattening)(x)
# x = layers.Conv2DTranspose(512, (3, 3), activation='relu', padding='same')(x) # New layer
# x = layers.UpSampling2D((2, 2))(x) # New layer
x = layers.Conv2DTranspose(256, (3, 3), activation='relu', padding='same')(x)
x = layers.UpSampling2D((2, 2))(x)
x = layers.Conv2DTranspose(128, (3, 3), activation='relu', padding='same')(x)
x = layers.UpSampling2D((2, 2))(x)
decoder_output = layers.Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)

decoder = models.Model(decoder_input, decoder_output, name='decoder')

autoencoder_input = layers.Input(shape=(128, 128, 1))
encoded_img = encoder(autoencoder_input)
decoded_img = decoder(encoded_img)
autoencoder = models.Model(autoencoder_input, decoded_img, name='autoencoder')

autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

encoder.summary()
decoder.summary()
autoencoder.summary()

train_data_dir = 'images/all'

train_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    './images',
    classes=['all'],
    target_size=(128, 128),
    color_mode='grayscale',
    batch_size=12,
    class_mode='input',
    shuffle=True
)

if train_generator.samples == 0:
    raise ValueError("The data generator has found 0 images. Please check the directory structure and paths.")

autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

history = autoencoder.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    epochs=10
)

autoencoder.save('my_autoencoder')





