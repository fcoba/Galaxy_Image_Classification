# Galaxy Classification

## Motivation
This project was created to offer astronomers help in classifying galaxy morphologies so they don't have to manually look at all the photographic plates. Most models out there offer high accuracies but may not be the easiest to implement or interpret.


## Data Retrieval

The data used in this project consists of semi low resolution images of galaxies taken from the Sloan Digital Sky Survey (SDSS) and made available on Kaggle as part of the Galaxy Zoo - The Galaxy Challenge:

https://www.kaggle.com/c/galaxy-zoo-the-galaxy-challenge/data 

Note, this project only uses the images_training folder which consists of JPG images of 61578 galaxies and the solutions_training folder which consists of probability distributions for the classifications for each of the training images. 

## Data Column Descriptions
training_predictions: Probability distributions for the classifications for each of the training images.

images_train: JPG images of 61578 galaxies. Files are named according to their GalaxyId.

## Data Processing

The images in the image-training folder was split 50/50 into training images and testing images. This was then further split into spiral and elliptical based on maximum probability in the solutions_training folder. The solutions_training folder had inside a csv file called solutions_training which was renamed to predictions.csv. 

```
data/test
    /spiral
    /elliptical
```

```
data/train
    /spiral
    /elliptical
```

The categories for predictions were elliptical, spiral, and irregular. The number of galaxies which were definitively classified as irregular made up less than 1% of the entire dataset. 

* num in train elliptical = 13237 
* num in train spiral = 17526 
* num in train irregular = 26 
* num in test elliptical = 13456 
* num in test spiral = 17300 
* num in test irregular = 33 

Thus, all irregular galaxies were removed,making the problem a binary classification and correcting a imbalanced dataset. 

Batches of images are grabbed:
```
data_tr = 30000 of images in data/train
data_te = 20000 of images in data/test
```
Then are defined as:
```python
x_tr, y_tr=  next(data_tr)
x_te, y_te =  next(data_te)
```
Finally, a train-test is performed:

```python
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(x_tr, y_tr, test_size=0.20, random_state=123)
```

## Displaying the Images

### Spiral Galaxy 
![Spiral Galaxy](data/train/spiral/177755.jpg 'Spiral Galaxy 177755')

### Elliptical Galaxy
![Elliptical Galaxy](data/train/elliptical/100078.jpg 'Elliptical Galaxy 100078')

## Models

### CNN
For the convolutional neural network, 3-layers are used to train the model, using an stochastic gradient descent optimizer, and a binary-cross entropy loss function. Addtionally, a learning rate, momentum, and augmentation were also used. The activation functions were all relu except for the very last single layer which used a sigmoid activation function. Dropout was also used to make sure the accuracy did not oscillate too much. 

```python
cnn1 = cnn.fit(X_train, y_train,
                epochs=50,
                validation_data=(X_val, y_val),
                batch_size=500)
```
Running the fit:

Epoch 50/50
24000/24000 [==============================] - 107s 4ms/step - loss: 0.4483 - acc: 0.8303 - val_loss: 0.5667 - val_acc: 0.7848

Peeking at some of the hidden layers:
![CNN Hidden Layers](figures/hidden_layers_cnn.png 'Hidden Layers CNN')

![CNN Confusion Matrix](figures/CNN_ConfusionMatrix.png 'CNN Confusion Matrix')

![CNN ROC Curve](figures/CNN_ROC.png 'CNN ROC Curve')

Accuracy on testing data:
``` python
cnn.evaluate(x_te, y_te)
20000/20000 [==============================] - 27s 1ms/step
[0.5454217989444733, 0.7854]
```

## Decision Tree
![Decision Tree CM](figures/DecisionTree_ConfusionMatrix.png 'Decision Tree Confusion Matrix') 

Accuracy Score on testing: 0.6523

![Decision Tree ROC](figures/DecisionTree_ROC.png 'Decision ROC')


## 1-hidden-layer Neural Network (Baseline Model)

This model contains one hidden layer with 128 neurons, between the input and output layers. It contains more than an factor of ten fewer neurons compared to the custom CNN.

![1-Layer Neural Network CM](figures/Baseline_MLP_ConfusionMatrix.png 'Baseline Neural Network CM') 

![1-layer Neural Network ROC](figures/BaselineMLP_ROC.png 'Baseline Neural Network ROC')

## Inception Model

A simple classifier was placed on top of the Inception_V3 model, trained on the imagenet dataset.

![Inception CM](figures/Inception_ConfusionMatrix.png 'Incpetion Confusion Matrix')

![Inception ROC](figures/Inception_ROC.png 'Inception ROC')

This model performs significantly worse than expected, and worse than our custom CNN model. This is likely because the imagenet dataset consists of objects photographed on the Earth and they are higher resolution images. Our data tend to be fuzzy, and look nothing like Earthly objects. 

## Combined Models ROC Curves:
![Combined ROC](figures/all_models_roc.png 'Combined ROC')

As we can see, the CNN model performed the best and this is without any real tweaking. The model could perform even better if it is allowed to run longer with more data augmentation using the `fit_generator`. 

