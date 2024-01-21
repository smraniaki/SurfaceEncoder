from PIL import Image
import os
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.models import load_model
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from skimage.metrics import structural_similarity as ssim

def calculate_psnr(original, reconstructed):
    mse = np.mean((original - reconstructed) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 1.0
    psnr = 20 * math.log10(max_pixel / math.sqrt(mse))
    return psnr



def convert_to_grayscale(image):
    if image.shape[-1] == 3:
        # Convert RGB to grayscale
        return np.dot(image[...,:3], [0.2989, 0.5870, 0.1140])
    return image


def calculate_metrics(original_images, reconstructed_images):
    mse_values = []
    ssim_values = []
    psnr_values = []

    for original, reconstructed in zip(original_images, reconstructed_images):
        # Convert to grayscale if necessary
        reconstructed_gray = convert_to_grayscale(reconstructed)

        original_flat = original.flatten()
        reconstructed_flat = reconstructed_gray.flatten()

        mse = mean_squared_error(original_flat, reconstructed_flat)
        mse_values.append(mse)

        psnr = calculate_psnr(original_flat, reconstructed_flat)
        psnr_values.append(psnr)

        original_reshaped = original.reshape(128, 128)
        reconstructed_reshaped = reconstructed_gray.reshape(128, 128)
        s = ssim(original_reshaped, reconstructed_reshaped, data_range=original_reshaped.max() - original_reshaped.min(), multichannel=False)
        ssim_values.append(s)

    return mse_values, psnr_values, ssim_values

# Load test images
data_dir = 'Data/images/all'
output_dir = 'Data/output'
os.makedirs(output_dir, exist_ok=True)

original_images = []
file_names = os.listdir(data_dir)
for file in file_names:
    img_path = os.path.join(data_dir, file)
    img = load_img(img_path, color_mode='grayscale', target_size=(128, 128))
    img_array = img_to_array(img) / 255.0
    original_images.append(img_array)

original_images = np.array(original_images)

# Load the autoencoder model and predict
autoencoder = load_model('preTrainedModels/my_autoencoder30k')
reconstructed_images = autoencoder.predict(original_images)

# Save reconstructed images
saved_reconstructed_images = []
for i, img_array in enumerate(reconstructed_images):
    img = Image.fromarray((img_array * 255).astype('uint8'))
    output_path = os.path.join(output_dir, file_names[i])
    img.save(output_path)
    saved_reconstructed_images.append(img_to_array(img) / 255.0)

saved_reconstructed_images = np.array(saved_reconstructed_images)

mse_values, psnr_values, ssim_values = calculate_metrics(original_images, reconstructed_images)
avg_mse = np.mean(mse_values)
avg_psnr = np.mean(psnr_values)
avg_ssim = np.mean(ssim_values)

print(f"Average Mean Squared Error: {avg_mse}")
print(f"Average Peak Signal-to-Noise Ratio: {avg_psnr}")
print(f"Average Structural Similarity Index: {avg_ssim}")

# Visual inspection
n = 5  # Number of images to showww
plt.figure(figsize=(20, 4))
for i in range(n):
    #original images
    ax = plt.subplot(2, n, i + 1)
    original_img = original_images[i].reshape(128, 128) if original_images[i].shape[-1] == 1 else original_images[i]
    plt.imshow(original_img, cmap='gray' if original_images[i].shape[-1] == 1 else None)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # reconstructed images
    ax = plt.subplot(2, n, i + 1 + n)
    reconstructed_img = reconstructed_images[i].reshape(128, 128) if reconstructed_images[i].shape[-1] == 1 else reconstructed_images[i]
    plt.imshow(reconstructed_img, cmap='gray' if reconstructed_images[i].shape[-1] == 1 else None)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
