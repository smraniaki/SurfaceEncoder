import sys
import numpy as np
import os
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QSlider, QLabel, QPushButton
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap
from tensorflow.keras.models import load_model
import random 

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

        model_path = os.path.join(bundle_dir, 'my_autoencoder70k')

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

        self.randomButton = QPushButton('Random Generation', self)
        self.randomButton.clicked.connect(self.randomizeSliders)
        
        topRightLayout = QVBoxLayout()
        topRightLayout.addWidget(self.randomButton)
        topRightLayout.addStretch(1)
        mainLayout.addLayout(topRightLayout)


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
            qImg = qImg.scaled(width * 4, height * 4, Qt.KeepAspectRatio)  # Scale the image to 4 times its size keeping the aspect ratio
            
            self.imageLabel.setPixmap(QPixmap.fromImage(qImg))
            self.imageLabel.setAlignment(Qt.AlignCenter)  

def main():
    app = QApplication(sys.argv)
    ex = AutoencoderDemo()
    ex.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
