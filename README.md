# Galaxy Classification

## Motivation



## Data Retrieval

The data used in this project consists of semi low resolution images of galaxies taken from the Sloan Digital Sky Survey (SDSS) and made available on Kaggle as part of the Galaxy Zoo - The Galaxy Challenge:

https://www.kaggle.com/c/galaxy-zoo-the-galaxy-challenge/data 

Note, this project only uses the images_training folder which consists of JPG images of 61578 galaxies and the solutions_training folder which consists of probability distributions for the classifications for each of the training images. 

## Data Processing

The images in the image-training folder was split 50/50 into training images and testing images. This was then further split into spiral and elliptical based on maximum probability in the solutions_training folder. The solutions_training folder had inside a csv file called solutions_training which was renamed to predictions.csv. 

The categories for predictions were elliptical, spiral, and irregular. The number of galaxies which were definitively classified as irregular made up less than 1% of the entire dataset. Thus, all irregular galaxies were removed,making the problem a binary classification and correcting a imbalanced dataset. 

## Displaing the Images

### Spiral Galaxy 
![Spiral Galaxy](data/train/spiral/177755.jpg 'Spiral Galaxy 177755')

### Elliptical Galaxy
![Elliptical Galaxy](data/train/elliptical/100078.jpg 'Elliptical Galaxy 100078')
