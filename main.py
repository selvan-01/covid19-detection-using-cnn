# -*- coding: utf-8 -*-

from PyQt5 import QtCore, QtGui, QtWidgets
import numpy as np

from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import model_from_json
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, BatchNormalization, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator


class Ui_MainWindow(object):

    def setupUi(self, MainWindow):

        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(800, 600)

        self.centralwidget = QtWidgets.QWidget(MainWindow)

        self.BrowseImage = QtWidgets.QPushButton(self.centralwidget)
        self.BrowseImage.setGeometry(QtCore.QRect(160, 370, 151, 51))
        self.BrowseImage.setText("Browse Image")

        self.imageLbl = QtWidgets.QLabel(self.centralwidget)
        self.imageLbl.setGeometry(QtCore.QRect(200, 80, 361, 261))
        self.imageLbl.setFrameShape(QtWidgets.QFrame.Box)

        self.Classify = QtWidgets.QPushButton(self.centralwidget)
        self.Classify.setGeometry(QtCore.QRect(160, 450, 151, 51))
        self.Classify.setText("Classify")

        self.Training = QtWidgets.QPushButton(self.centralwidget)
        self.Training.setGeometry(QtCore.QRect(400, 450, 151, 51))
        self.Training.setText("Training")

        self.textEdit = QtWidgets.QTextEdit(self.centralwidget)
        self.textEdit.setGeometry(QtCore.QRect(400, 390, 211, 51))

        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(430, 370, 150, 20))
        self.label.setText("Recognized Class")

        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(200, 20, 400, 30))

        font = QtGui.QFont()
        font.setPointSize(14)
        font.setBold(True)

        self.label_2.setFont(font)
        self.label_2.setText("COVID-19 DETECTION")

        MainWindow.setCentralWidget(self.centralwidget)

        self.BrowseImage.clicked.connect(self.loadImage)
        self.Classify.clicked.connect(self.classifyFunction)
        self.Training.clicked.connect(self.trainingFunction)

    # ---------------- LOAD IMAGE ---------------- #

    def loadImage(self):

        fileName, _ = QtWidgets.QFileDialog.getOpenFileName(
            None,
            "Select Image",
            "",
            "Image Files (*.png *.jpg *.jpeg *.bmp)"
        )

        if fileName:

            self.file = fileName

            pixmap = QtGui.QPixmap(fileName)

            pixmap = pixmap.scaled(
                self.imageLbl.width(),
                self.imageLbl.height(),
                QtCore.Qt.KeepAspectRatio
            )

            self.imageLbl.setPixmap(pixmap)
            self.imageLbl.setAlignment(QtCore.Qt.AlignCenter)

    # ---------------- CLASSIFY ---------------- #

    def classifyFunction(self):

        json_file = open("model.json", "r")
        loaded_model_json = json_file.read()
        json_file.close()

        loaded_model = model_from_json(loaded_model_json)

        loaded_model.load_weights("model.h5")

        print("Model Loaded")

        labels = ["Covid", "Normal"]

        img_path = self.file

        test_image = image.load_img(img_path, target_size=(128, 128))
        test_image = image.img_to_array(test_image) / 255.0
        test_image = np.expand_dims(test_image, axis=0)

        result = loaded_model.predict(test_image)

        prediction = labels[result.argmax()]

        print("Prediction:", prediction)

        self.textEdit.setText(prediction)

    # ---------------- TRAIN MODEL ---------------- #

    def trainingFunction(self):

        self.textEdit.setText("Training Started...")

        model = Sequential()

        model.add(Conv2D(32, (3,3), activation='relu', input_shape=(128,128,3)))
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(BatchNormalization())

        model.add(Conv2D(64,(3,3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(BatchNormalization())

        model.add(Conv2D(64,(3,3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(BatchNormalization())

        model.add(Conv2D(96,(3,3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(BatchNormalization())

        model.add(Dropout(0.2))

        model.add(Flatten())

        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.3))

        model.add(Dense(2, activation='softmax'))

        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        train_datagen = ImageDataGenerator(
            rescale=1./255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True
        )

        test_datagen = ImageDataGenerator(rescale=1./255)

        training_set = train_datagen.flow_from_directory(
            r"E:\linkdin projects\Covid\TrainingDataset",
            target_size=(128,128),
            batch_size=8,
            class_mode='categorical'
        )

        test_set = test_datagen.flow_from_directory(
            r"E:\linkdin projects\Covid\TestingDataset",
            target_size=(128,128),
            batch_size=8,
            class_mode='categorical'
        )

        model.fit(
            training_set,
            steps_per_epoch=100,
            epochs=10,
            validation_data=test_set,
            validation_steps=125
        )

        # SAVE MODEL

        model_json = model.to_json()

        with open("model.json", "w") as json_file:
            json_file.write(model_json)

        model.save_weights("model.h5")

        print("Model Saved")

        self.textEdit.setText("Training Complete")


# ---------------- MAIN ---------------- #

if __name__ == "__main__":

    import sys

    app = QtWidgets.QApplication(sys.argv)

    MainWindow = QtWidgets.QMainWindow()

    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)

    MainWindow.show()

    sys.exit(app.exec_())