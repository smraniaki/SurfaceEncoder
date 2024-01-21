from tensorflow.keras.models import load_model

autoencoder = load_model('my_autoencoder70k')

autoencoder.save('my_autoencoder70k.h5')

