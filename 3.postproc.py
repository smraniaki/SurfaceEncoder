import sys
import numpy as np
import os
import random
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QSlider, QLabel, QPushButton
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap
from tensorflow.keras.models import load_model

from PIL import Image
from io import BytesIO
import cv2

class AutoencoderDemo(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
    
    def initUI(self):
        # Determine the path to the model file
        if getattr(sys, 'frozen', False):
            # Running in a bundle
            bundle_dir = sys._MEIPASS
        else:
            # Running in a normal Python environment
            bundle_dir = os.path.dirname(os.path.abspath(__file__))

        model_path = os.path.join(bundle_dir, 'preTrainedModels/Surfencoder70k.h5')

        self.model = load_model(model_path)
        self.model.summary()
        self.decoder = self.model.get_layer('decoder')

        self.latent_dim = 10  

        mainLayout = QHBoxLayout()

        sliderLayout = QVBoxLayout()
        self.sliders = []
        for i in range(self.latent_dim):
            slider = QSlider(Qt.Horizontal)
            slider.setMinimum(-30)
            slider.setMaximum(30)
            slider.setValue(0)
            slider.valueChanged[int].connect(self.changeValue)
            self.sliders.append(slider)
            sliderLayout.addWidget(slider)

        mainLayout.addLayout(sliderLayout)

        self.imageLabel = QLabel(self)
        mainLayout.addWidget(self.imageLabel)

        self.setLayout(mainLayout)
        self.setWindowTitle('Surface Decoder')
        self.setGeometry(300, 300, 1200, 800) 
        self.updateImage()

        self.denoisingButton = QPushButton('Denoising', self)
        self.denoisingButton.clicked.connect(self.denoiseImage)
        sliderLayout.addWidget(self.denoisingButton)

        self.randomButton = QPushButton('Random Generation', self)
        self.randomButton.clicked.connect(self.randomizeSliders)
        sliderLayout.addWidget(self.randomButton)

    def randomizeSliders(self):
        for slider in self.sliders:
            random_value = random.randint(slider.minimum(), slider.maximum())
            slider.setValue(random_value)
                    
    def changeValue(self, value):
        self.updateImage()

    def updateImage(self):
        if hasattr(self, 'sliders'):
            z_sample = np.array([slider.value() / 10.0 for slider in self.sliders]).reshape((1, self.latent_dim))
            reconstructed_img = self.decoder.predict(z_sample)[0]
            reconstructed_img = (reconstructed_img * 255).astype(np.uint8)

            height, width, channel = reconstructed_img.shape
            bytesPerLine = 3 * width
            qImg = QImage(reconstructed_img.data, width, height, bytesPerLine, QImage.Format_RGB888)
            qImg = qImg.scaled(width * 3, height * 3, Qt.KeepAspectRatio)
            
            self.imageLabel.setPixmap(QPixmap.fromImage(qImg))
            self.imageLabel.setAlignment(Qt.AlignCenter)  

    def denoiseImage(self):
        try:
            # Get the current state of the sliders to generate the encoded image
            z_sample = np.array([slider.value() / 10.0 for slider in self.sliders]).reshape((1, self.latent_dim))
            generated_img = self.decoder.predict(z_sample)[0]

            # Convert the generated image from the decoder to a format suitable for the autoencoder
            generated_img = (generated_img * 255).astype(np.uint8)
            generated_img_pil = Image.fromarray(generated_img)
            generated_img_pil = generated_img_pil.convert('L')  # Convert to grayscale
            generated_img_pil = generated_img_pil.resize((128, 128))  # Resize to 128x128

            # Convert PIL image to numpy array expected by the model
            denoise_input = np.array(generated_img_pil).reshape((1, 128, 128, 1)) / 255.0

            # Denoise the image using the autoencoder
            denoised_img = self.model.predict(denoise_input)[0]

            # Display the denoised image
            denoised_img = (denoised_img * 255).astype(np.uint8)
            height, width, channel = denoised_img.shape
            bytesPerLine = channel * width
            qImg = QImage(denoised_img.data, width, height, bytesPerLine, QImage.Format_RGB888)
            qImg = qImg.scaled(width * 3, height * 3, Qt.KeepAspectRatio)
            self.imageLabel.setPixmap(QPixmap.fromImage(qImg))
            self.imageLabel.setAlignment(Qt.AlignCenter)
        except Exception as e:
            print("Error in denoising: ", str(e))

def main():
    app = QApplication(sys.argv)
    ex = AutoencoderDemo()
    ex.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
