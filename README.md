# Start-CNN-with-Cat-Dog-Classification
 

## 📜 Description  
This project aims to classify images of cats and dogs using a **Convolutional Neural Network (CNN)**. It utilizes **TensorFlow/Keras** for building and training the model, **Albumentations** for advanced data augmentation, and other Python libraries for preprocessing, visualization, and evaluation.  

The model achieves high accuracy by leveraging effective techniques such as data augmentation, regularization, and callbacks to prevent overfitting and enhance performance.  

---

## 🛠️ Features  
- **Binary Classification**: Differentiates between cats and dogs.  
- **Deep Learning**: Built using CNN with TensorFlow/Keras.  
- **Data Augmentation**: Applies multiple transformations for robust learning.  
- **Callbacks**: Uses EarlyStopping, ModelCheckpoint, and learning rate schedulers for optimized training.  

---

## 📂 Dataset  
The dataset used in this project consists of labeled images of cats and dogs.  
You can download the dataset from [Kaggle](https://www.kaggle.com/competitions/dogs-vs-cats-redux-kernels-edition/data).  

### Dataset Structure:  
data/
├── train/

│ ├── cat.0.jpg
│ ├── dog.0.jpg
│ ├── ...
├── test/
│ ├── cat.1.jpg
│ ├── dog.1.jpg
│ ├── ...



---

## 🚀 Technologies Used  

### Libraries  
- **TensorFlow/Keras**: Deep learning framework.  
- **Albumentations**: Data augmentation library.  
- **Pandas**: Data manipulation and analysis.  
- **Matplotlib & Seaborn**: Data visualization.  
- **Scikit-Learn**: Preprocessing and splitting the dataset.  
- **Pillow**: Image handling and manipulation.  

---

## 🧠 Model Architecture  
The CNN model is composed of:  
1. **Convolutional Layers**: Extract features from images.  
2. **Pooling Layers**: Reduce the spatial dimensions of feature maps.  
3. **Dropout Layers**: Prevent overfitting.  
4. **Dense Layers**: Fully connected layers for classification.  
5. **Sigmoid Activation**: Used in the output layer for binary classification.  

### Hyperparameters  
- Optimizer: `Adam`  
- Loss Function: `Binary Crossentropy`  
- Metrics: `Accuracy`  

---

## 🎨 Data Augmentation  
To improve model generalization, the following data augmentation techniques are applied using **Albumentations**:  
- Horizontal and Vertical Flips  
- Shift, Scale, and Rotate  
- Random Brightness/Contrast  
- Gaussian Blur  
- Elastic Transform  

---

## 🏋️‍♂️ Training  
### Steps:  
1. Preprocess and split the dataset into training, validation, and test sets.  
2. Apply data augmentation to the training set.  
3. Train the model using the training set and validate it with the validation set.  
4. Save the best model using **ModelCheckpoint**.  

### Callbacks Used:  
- **EarlyStopping**: Stops training when validation loss stops improving.  
- **ModelCheckpoint**: Saves the best-performing model.  
- **ReduceLROnPlateau**: Reduces the learning rate when a plateau in validation loss is detected.  

---

## 📊 Evaluation  
The trained model is evaluated on the test dataset using metrics such as:  
- **Accuracy**  

A confusion matrix and classification report are generated to analyze the results in detail.  

---

## 📈 Results  
The model achieved an accuracy of approximately **95%** on the validation set and performed well on unseen test data.  

| Image              | True Label | Predicted Label |  
|--------------------|------------|-----------------|  
| ![cat](data/train/cat.0.jpg) | Cat        | Cat             |  
| ![dog](data/train/dog.0.jpg) | Dog        | Dog             |  

---

## 🔧 Usage  

### 1️⃣ Train the Model  
To train the model, run:
```bash  
python train.py  
