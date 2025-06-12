# 🧠 Image Classification with CNN (TensorFlow + Keras)

This is a simple Image Classification project built using **Convolutional Neural Networks (CNNs)** with **TensorFlow** and **Keras**. The model is trained to classify images into two categories — Cats and Dogs.

## 🚀 Project Overview

* **Frameworks**: TensorFlow, Keras
* **Techniques**: CNN, Image Augmentation
* **Dataset**: Custom ZIP dataset containing `train/` and `test/` directories
* **Goal**: Predict whether an image is of a Cat or a Dog

## 🗂️ Directory Structure

```
/dataset.zip
    └── /train
        ├── /cats
        └── /dogs
    └── /test
        ├── /cats
        └── /dogs
```

## 🔧 Installation & Setup

1. Clone the repository or upload the code to Google Colab.
2. Upload your `dataset.zip` to your Google Drive.
3. Mount Google Drive and unzip dataset:

   ```python
   !unzip "/content/drive/MyDrive/dataset.zip" -d "/usr/local/dataset"
   ```
4. Make sure you have the required libraries installed:

   ```bash
   pip install tensorflow keras matplotlib
   ```

## 📊 Data Preprocessing

We use `ImageDataGenerator` for:

* Rescaling
* Shearing
* Zooming
* Horizontal flipping (augmentation)

```python
from keras._tf_keras.keras.preprocessing.image import ImageDataGenerator
```

## 🏗️ CNN Model Architecture

* 2 Convolutional + MaxPooling Layers
* Flatten Layer
* Dense Hidden Layer (128 units)
* Output Layer (Sigmoid for binary classification)

```python
cnn1 = tf.keras.models.Sequential()
cnn1.add(tf.keras.layers.Conv2D(...))
...
cnn1.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))
```

## 🏋️ Model Training

```python
cnn1.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
cnn1.fit(x=training_set1, validation_data=test_set1, epochs=25)
```

## 🔍 Make Predictions

To predict a single image:

```python
test_image = image.load_img('path_to_image.jpg', target_size=(64,64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)

result = cnn1.predict(test_image)
prediction = 'Dog' if result[0][0] == 1 else 'Cat'
```

## 📈 Visualization

```python
plt.imshow(test_image[0])
plt.title(f"Predicted: {prediction}")
plt.axis('off')
plt.show()
```

## ✅ Output Example

* Input Image: Cat
* **Prediction**: Cat

---

## 📌 Future Improvements

* Add multi-class classification support.
* Use transfer learning (e.g., MobileNet, ResNet).
* Deploy as a web app using Flask or Streamlit.

## 📄 License

This project is for educational purposes only.

🚀 Happy Coding! 🎉
