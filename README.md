# 2023 Car Image Classification Project

The existing public car classification models are trained on old datasets (e.g. [Stanford Car Dataset](https://ai.stanford.edu/~jkrause/papers/fgvc13.pdf#cite.WahCUB_200_2011)). In this projects, I have developed and deployed an ML model that is fine-tuned to classify the latest 2023 car models. To achieve this, I have scraped more than 200,000 car images for +19,000 car models from the year 2011 to 2024 including their full specifications. Based on this dataset, I have developed an image classification model based on PyTorch framework specifically tailored for the 2023 car models. I have also developed a web application ([Link](http://portfolioarthur.pythonanywhere.com/)) using Flask framework where users can upload their 2023 car images to receive their full specification and MSRP.

<p align="center">
  <img width="600" src="https://github.com/r2rro/webapp-car-classification-2023/blob/main/image/Introduction.png">
</p>

## Table of Contents

1. [Files](#files)
2. [Dataset](#dataset)
3. [Model Architecture](#model-architecture)
4. [Training](#training)
5. [Evaluation](#evaluation)
6. [Web App](#web-app)
7. [Future Work](#future-work)
8. [Full Report](#full-report)

## Files
| FILES | DESCRIPTION |
| ---   | ---         |
| `scrape.py` |  Contains functions for web scraping car images and specs | 
| `data_extractor.py` | Turns all URLs in the input `df` to labeled pictures and cleans and saves car specs into a csv file |
| `img_preprocessing.py` | Filters out the interior car images using a trained NN model and splits the data into training and testing sets| 
| `crop.py` | Implements image cropping functionality|
| `load_data.py` | Loads train and test data using Torch DataLoader| 
| `seed_everything.py` | Seeds all random number generators for reproducibility| 
| `transform_image.py` | Transforms the input image for use in Torchvision model| 
| `interior_classifier.py` | Handles classification of interior images using a helper NN model|
| `classifier.py` |  Contains the `ResNet50` and `ResNet101` models with initialized weights|
| `model_trainer.py` | A class used for training and managing ResNet models for the car dataset| 
| `main.py` | The main file for training the car classifier model| 
| `app.py` | Implements the `Flask` web application where users can upload car images for classification| 

## Dataset

The dataset scraped for this project is extensive, comprising over 200,000 images scraped from the internet. These images cover car models ranging from 2011 to 2024, totaling more than 1,900 different car models. The dataset includes both interior and exterior photos. For this project, the focus is on the 175 car classes for the year 2023.
Exterior Image Example             |  Interior Image Example
:-------------------------:|:-------------------------:
 <img height="200" src="https://github.com/r2rro/webapp-car-classification-2023/blob/main/image/2023%20Tesla%20Model%20S_85.jpg"> |  <img height="200" src="https://github.com/r2rro/webapp-car-classification-2023/blob/main/image/2023%20Tesla%20Model%20S_82.jpg">

## Model Architecture

I have employed two powerful pre-trained models, [ResNet-50 and ResNet-101](https://arxiv.org/pdf/1512.03385.pdf), to fine-tuning with the image classification task. These architectures are renowned for their deep learning capabilities and are well-suited for this project's requirements.
<p align="center">
  <img width="600" src="https://github.com/r2rro/webapp-car-classification-2023/blob/main/image/ResNet50_arc.png">
  <img width="600" src="https://github.com/r2rro/webapp-car-classification-2023/blob/main/image/resnet_size.png">
</p>

## Training

To enhance the quality of the training data, I have developed a car exterior detection algorithm. This algorithm identifies and excludes interior photos, ensuring that the model is trained specifically on exterior images. To train the exterior detector model, a group of exterior and interior images are randomly selected from different car folders and manually labeled. The cleaned training and test data are then organized into folders to facilitate seamless integration with the torchvision library.

## Evaluation

Monitoring the training process involves tracking accuracy, while the evaluation process goes beyond with the inclusion of confusion matrices and other evaluation metrics (see full report below for more details). These metrics provide a comprehensive understanding of the model's behavior in diverse scenarios.

<p align="center">
  <img width="600" src="https://github.com/r2rro/webapp-car-classification-2023/blob/main/image/accuracy.jpg">
</p>

| Metric             | Score  |
|--------------------|--------|
| Train Accuracy     | 97.3%  |
| Test Accuracy      | 73.4%  |

Studying the misclassified images provide more insight into the areas of improvment for this model. The example below shows that the model is able to detect important features of the car exterior, however, due to having a large number of classes and the similarities between different car bodies, the model can be misguided in the process. This opens up the opportunity to further enhance the accuracy of the model by different methods. See future work section for more detail. In addition, the confusion matrix shows that most misclassifications occur with adjacent classes i.e. similar models of the same make. 

2023 Audi A3             |  Misclassified As 2023 Toyota Corolla
:-------------------------:|:-------------------------:
 <img height="200" src="https://github.com/r2rro/webapp-car-classification-2023/blob/main/image/2023%20Audi%20A3.png"> |  <img height="200" src="https://github.com/r2rro/webapp-car-classification-2023/blob/main/image/2023%20Toyota%20Corolla.png">

## Web App

To make the model more accessible to users, I have also created a Flask web application. This user-friendly interface allows anyone to upload an image of a 2023 car model and receive a classification result along with the full specifications of the classified car.
<p align="center">
  <img width="600" src="https://github.com/r2rro/webapp-car-classification-2023/blob/main/image/webapp.jpg">
</p>

## Future Work

Looking ahead, I plan to enhance the performance of the model using three methods:
1. More Data: Increasing the size of the dataset with more image variations helps the model generalize better to different instances of the same class.
2. Hierarchical labeling: Organizing the classes into a hierarchical structure. Each class in the hierarchy represents a more specific subset of a more general class. For example, a general class of car make with subclasses of car models.
3. Model Ensembling: Implementing soft voting by combining predictions from multiple models can enhance the overall accuracy and robustness of the classification system.

## Full Report

| Class                              | Precision | Recall | F1-Score | Support |
|------------------------------------|-----------|--------|----------|---------|
| 2023 Acura Integra                 | 0.5       | 0.333  | 0.4      | 6.0     |
| 2023 Acura MDX                     | 0.333     | 0.125  | 0.181    | 8.0     |
| 2023 Acura RDX                     | 0.8       | 0.571  | 0.667    | 7.0     |
| 2023 Acura TLX                     | 0.714     | 0.625  | 0.667    | 8.0     |
| 2023 Alfa Romeo Giulia             | 0.833     | 0.833  | 0.833    | 6.0     |
| 2023 Alfa Romeo Stelvio            | 0.8       | 0.8    | 0.8      | 5.0     |
| 2023 Audi A3                       | 0.571     | 0.571  | 0.571    | 7.0     |
| 2023 Audi A4                       | 1.0       | 1.0    | 1.0      | 7.0     |
| 2023 Audi A5                       | 0.455     | 0.625  | 0.526    | 8.0     |
| 2023 Audi A6                       | 0.833     | 0.556  | 0.667    | 9.0     |
| 2023 Audi A7                       | 0.5       | 0.667  | 0.571    | 9.0     |
| 2023 Audi A8                       | 0.526     | 0.909  | 0.667    | 11.0    |
| 2023 Audi E-Tron GT                | 0.8       | 0.8    | 0.8      | 5.0     |
| 2023 Audi Q3                       | 0.833     | 0.833  | 0.833    | 6.0     |
| 2023 Audi Q4 E-Tron                | 0.8       | 0.571  | 0.667    | 7.0     |
| 2023 Audi Q5                       | 0.667     | 0.5    | 0.571    | 8.0     |
| 2023 Audi Q7                       | 0.75      | 0.857  | 0.8      | 7.0     |
| 2023 Audi R8                       | 0.692     | 1.0    | 0.818    | 9.0     |
| 2023 Audi TT                       | 0.714     | 0.714  | 0.714    | 7.0     |
| 2023 BMW 2-Series                  | 0.0       | 0.0    | 0.0      | 3.0     |
| 2023 BMW 3-Series                  | 0.583     | 0.875  | 0.7      | 8.0     |
| 2023 BMW 4-Series                  | 0.5       | 0.375  | 0.429    | 8.0     |
| 2023 BMW 5-Series                  | 0.75      | 0.75   | 0.75     | 4.0     |
| 2023 BMW 7-Series                  | 0.7       | 0.636  | 0.667    | 11.0    |
| 2023 BMW 8-Series                  | 1.0       | 0.75   | 0.857    | 4.0     |
| 2023 BMW X1                        | 0.818     | 0.9    | 0.857    | 10.0    |
| 2023 BMW X3                        | 0.667     | 1.0    | 0.8      | 8.0     |
| 2023 BMW X4                        | 0.714     | 0.714  | 0.714    | 7.0     |
| 2023 BMW X5                        | 1.0       | 0.667  | 0.8      | 6.0     |
| 2023 BMW X6                        | 0.636     | 0.875  | 0.737    | 8.0     |
| 2023 BMW X7                        | 0.833     | 0.556  | 0.667    | 9.0     |
| 2023 BMW Z4                        | 0.833     | 1.0    | 0.909    | 10.0    |
| 2023 BMW i4                        | 1.0       | 0.5    | 0.667    | 4.0     |
| 2023 BMW iX                        | 0.75      | 0.75   | 0.75     | 8.0     |
| 2023 Buick Enclave                 | 0.778     | 0.875  | 0.824    | 8.0     |
| 2023 Buick Encore GX               | 0.6       | 0.5    | 0.545    | 6.0     |
| 2023 Buick Envision                | 0.571     | 0.571  | 0.571    | 7.0     |
| 2023 Cadillac CT4                  | 0.75      | 0.5    | 0.6      | 6.0     |
| 2023 Cadillac CT5                  | 0.75      | 0.857  | 0.8      | 7.0     |
| 2023 Cadillac Escalade             | 0.667     | 0.8    | 0.727    | 5.0     |
| 2023 Cadillac Lyriq                | 1.0       | 0.25   | 0.4      | 4.0     |
| 2023 Cadillac XT4                  | 0.625     | 0.833  | 0.714    | 6.0     |
| 2023 Cadillac XT5                  | 0.833     | 0.714  | 0.769    | 7.0     |
| 2023 Cadillac XT6                  | 0.75      | 0.75   | 0.75     | 4.0     |
| 2023 Chevrolet Blazer              | 0.7       | 0.7    | 0.7      | 10.0    |
| 2023 Chevrolet Camaro              | 0.545     | 0.545  | 0.545    | 11.0    |
| 2023 Chevrolet Colorado            | 0.857     | 0.857  | 0.857    | 7.0     |
| 2023 Chevrolet Equinox             | 0.615     | 0.615  | 0.615    | 13.0    |
| 2023 Chevrolet Silverado           | 0.778     | 0.778  | 0.778    | 9.0     |
| 2023 Chevrolet Suburban            | 0.5       | 0.286  | 0.364    | 7.0     |
| 2023 Chevrolet Tahoe               | 0.75      | 0.75   | 0.75     | 4.0     |
| 2023 Chevrolet Traverse            | 0.8       | 0.8    | 0.8      | 5.0     |
| 2023 Chevrolet Trax                | 0.571     | 0.5    | 0.533    | 6.0     |
| 2023 Chevrolet Silverado HD        | 0.714     | 0.714  | 0.714    | 7.0     |
| 2023 Chevrolet Silverado 2500HD    | 0.667     | 0.667  | 0.667    | 6.0     |
| 2023 Chevrolet Silverado 3500HD    | 0.75      | 0.75   | 0.75     | 4.0     |
| 2023 Chrysler 300                  | 0.615     | 0.875  | 0.722    | 8.0     |
| 2023 Chrysler Pacifica             | 0.6       | 0.429  | 0.5      | 7.0     |
| 2023 Chrysler Voyager              | 0.625     | 0.833  | 0.714    | 6.0     |
| 2023 Dodge Challenger              | 0.75      | 0.75   | 0.75     | 4.0     |
| 2023 Dodge Charger                 | 0.571     | 0.571  | 0.571    | 7.0     |
| 2023 Dodge Durango                 | 0.833     | 0.833  | 0.833    | 6.0     |
| 2023 Dodge Grand Caravan           | 0.571     | 0.5    | 0.533    | 6.0     |
| 2023 Fiat 500X                     | 0.857     | 0.857  | 0.857    | 7.0     |
| 2023 Ford Bronco                   | 0.833     | 0.833  | 0.833    | 6.0     |
| 2023 Ford EcoSport                 | 0.714     | 0.833  | 0.769    | 6.0     |
| 2023 Ford Edge                     | 0.6       | 0.5    | 0.545    | 6.0     |
| 2023 Ford Escape                   | 0.625     | 0.625  | 0.625    | 8.0     |
| 2023 Ford Expedition               | 0.625     | 0.833  | 0.714    | 6.0     |
| 2023 Ford Explorer                 | 0.857     | 0.857  | 0.857    | 7.0     |
| 2023 Ford F-150                    | 0.8       | 0.8    | 0.8      | 5.0     |
| 2023 Ford F-250 Super Duty         | 0.833     | 0.833  | 0.833    | 6.0     |
| 2023 Ford F-350 Super Duty         | 0.857     | 0.857  | 0.857    | 7.0     |
| 2023 Ford Maverick                 | 0.75      | 0.75   | 0.75     | 4.0     |
| 2023 Ford Mustang                  | 0.571     | 0.571  | 0.571    | 7.0     |
| 2023 Ford Ranger                   | 0.857     | 0.857  | 0.857    | 7.0     |
| 2023 Ford Transit Connect          | 0.5       | 0.333  | 0.4      | 6.0     |
| 2023 GMC Acadia                   | 0.5       | 0.5    | 0.5      | 4.0     |
| 2023 GMC Canyon                   | 0.75      | 0.75   | 0.75     | 4.0     |
| 2023 GMC Sierra                   | 0.5       | 0.333  | 0.4      | 6.0     |
| 2023 GMC Terrain                  | 0.8       | 0.571  | 0.667    | 7.0     |
| 2023 GMC Yukon                    | 0.667     | 0.8    | 0.727    | 5.0     |
| 2023 Genesis G70                  | 0.857     | 0.857  | 0.857    | 7.0     |
| 2023 Genesis G80                  | 0.571     | 0.571  | 0.571    | 7.0     |
| 2023 Genesis G90                  | 0.6       | 0.75   | 0.667    | 8.0     |
| 2023 GMC Hummer EV                | 0.571     | 0.571  | 0.571    | 7.0     |
| 2023 Honda Accord                 | 0.7       | 0.7    | 0.7      | 10.0    |
| 2023 Honda Civic                  | 0.833     | 0.833  | 0.833    | 6.0     |
| 2023 Honda Clarity Plug-In Hybrid | 0.6       | 0.75   | 0.667    | 8.0     |
| 2023 Honda CR-V                   | 0.6       | 0.75   | 0.667    | 8.0     |
| 2023 Honda CR-V                    | 0.5       | 0.286  | 0.364    | 7.0     |
| 2023 Honda Civic                   | 0.538     | 1.0    | 0.7      | 7.0     |
| 2023 Honda HR-V                    | 0.889     | 0.8    | 0.842    | 10.0    |
| 2023 Honda Odyssey                 | 0.786     | 1.0    | 0.88     | 11.0    |
| 2023 Honda Passport                | 0.769     | 1.0    | 0.87     | 10.0    |
| 2023 Honda Pilot                   | 1.0       | 0.6    | 0.75     | 5.0     |
| 2023 Honda Ridgeline               | 0.8       | 0.571  | 0.667    | 7.0     |
| 2023 Hyundai Elantra               | 0.625     | 1.0    | 0.769    | 15.0    |
| 2023 Hyundai IONIQ 6               | 0.8       | 1.0    | 0.889    | 4.0     |
| 2023 Hyundai Ioniq 5               | 0.857     | 0.667  | 0.75     | 9.0     |
| 2023 Hyundai Kona                  | 1.0       | 0.5    | 0.667    | 4.0     |
| 2023 Hyundai Kona Electric         | 0.818     | 0.75   | 0.783    | 12.0    |
| 2023 Hyundai Palisade              | 1.0       | 1.0    | 1.0      | 10.0    |
| 2023 Hyundai Santa Cruz            | 1.0       | 1.0    | 1.0      | 6.0     |
| 2023 Hyundai Santa Fe              | 0.786     | 0.917  | 0.846    | 12.0    |
| 2023 Hyundai Sonata                | 1.0       | 0.857  | 0.923    | 7.0     |
| 2023 Hyundai Tucson                | 0.842     | 0.941  | 0.889    | 17.0    |
| 2023 Hyundai Venue                 | 0.8       | 0.571  | 0.667    | 7.0     |
| 2023 INFINITI Q50                  | 0.667     | 0.286  | 0.4      | 7.0     |
| 2023 INFINITI QX50                 | 0.625     | 0.625  | 0.625    | 8.0     |
| 2023 INFINITI QX80                 | 1.0       | 1.0    | 1.0      | 6.0     |
| 2023 Jaguar E-Pace                 | 0.5       | 0.375  | 0.429    | 8.0     |
| 2023 Jaguar F-Pace                 | 0.667     | 0.857  | 0.75     | 14.0    |
| 2023 Jaguar F-Type                 | 0.5       | 0.333  | 0.4      | 6.0     |
| 2023 Jaguar I-Pace                 | 0.5       | 0.3    | 0.375    | 10.0    |
| 2023 Jaguar XF                     | 0.667     | 0.667  | 0.667    | 6.0     |
| 2023 Jeep Grand Cherokee           | 0.529     | 0.75   | 0.621    | 12.0    |
| 2023 MINI Cooper                   | 1.0       | 0.833  | 0.909    | 6.0     |
| 2023 Mercedes-Benz CLA Class       | 0.5       | 0.75   | 0.6      | 8.0     |
| 2023 Mercedes-Benz CLS Class       | 0.75      | 0.375  | 0.5      | 8.0     |
| 2023 Mercedes-Benz E Class         | 0.0       | 0.0    | 0.0      | 4.0     |
| 2023 Mercedes-Benz EQS             | 0.5       | 0.625  | 0.556    | 8.0     |
| 2023 Mercedes-Benz GLB Class       | 0.429     | 0.429  | 0.429    | 7.0     |
| 2023 Mercedes-Benz GLC Class       | 0.889     | 0.8    | 0.842    | 10.0    |
| 2023 Mercedes-Benz GLE Class       | 0.833     | 1.0    | 0.909    | 5.0     |
| 2023 Mercedes-Benz GLS Class       | 0.5       | 0.375  | 0.429    | 8.0     |
| 2023 Mercedes-Benz S Class         | 0.571     | 0.571  | 0.571    | 7.0     |
| 2023 Mitsubishi Mirage             | 0.714     | 0.556  | 0.625    | 9.0     |
| 2023 Mitsubishi Outlander          | 0.714     | 0.625  | 0.667    | 8.0     |
| 2023 Nissan Altima                 | 0.727     | 1.0    | 0.842    | 8.0     |
| 2023 Nissan Ariya                  | 0.5       | 0.5    | 0.5      | 8.0     |
| 2023 Nissan Kicks                  | 1.0       | 0.6    | 0.75     | 5.0     |
| 2023 Nissan Leaf                   | 1.0       | 1.0    | 1.0      | 7.0     |
| 2023 Nissan Murano                 | 0.818     | 1.0    | 0.9      | 9.0     |
| 2023 Nissan Sentra                 | 0.833     | 0.714  | 0.769    | 7.0     |
| 2023 Nissan Versa                  | 1.0       | 0.833  | 0.909    | 6.0     |
| 2023 Nissan Z                      | 0.5       | 0.818  | 0.621    | 11.0    |
| 2023 Polestar 2                    | 0.562     | 1.0    | 0.72     | 9.0     |
| 2023 Porsche 718                   | 0.75      | 0.5    | 0.6      | 6.0     |
| 2023 Porsche 911                   | 0.6       | 0.429  | 0.5      | 7.0     |
| 2023 Porsche Cayenne               | 0.429     | 0.5    | 0.462    | 6.0     |
| 2023 Porsche Taycan                | 0.75      | 0.6    | 0.667    | 5.0     |
| 2023 Ram 1500                      | 0.0       | 0.0    | 0.0      | 3.0     |
| 2023 Ram 2500                      | 0.714     | 1.0    | 0.833    | 5.0     |
| 2023 Rivian R1S                    | 1.0       | 0.75   | 0.857    | 4.0     |
| 2023 Rivian R1T                    | 0.778     | 0.778  | 0.778    | 9.0     |
| 2023 Subaru Crosstrek              | 0.833     | 0.714  | 0.769    | 7.0     |
| 2023 Subaru Outback                | 0.0       | 0.0    | 0.0      | 2.0     |
| 2023 Subaru WRX                    | 0.6       | 0.375  | 0.462    | 8.0     |
| 2023 Tesla Model 3                 | 0.75      | 0.6    | 0.667    | 5.0     |
| 2023 Tesla Model S                 | 0.857     | 0.667  | 0.75     | 9.0     |
| 2023 Tesla Model X                 | 0.6       | 0.667  | 0.632    | 9.0     |
| 2023 Tesla Model Y                 | 0.75      | 0.6    | 0.667    | 5.0     |
| 2023 Toyota 4Runner                | 0.857     | 1.0    | 0.923    | 6.0     |
| 2023 Toyota BZ4X                   | 0.833     | 0.625  | 0.714    | 8.0     |
| 2023 Toyota Camry                  | 0.846     | 1.0    | 0.917    | 11.0    |
| 2023 Toyota Corolla                | 0.833     | 0.833  | 0.833    | 6.0     |
| 2023 Toyota Crown                  | 0.778     | 1.0    | 0.875    | 7.0     |
| 2023 Toyota Highlander             | 0.333     | 0.333  | 0.333    | 3.0     |
| 2023 Toyota Highlander             | 0.333     | 0.333  | 0.333    | 3.0     |
| 2023 Toyota Prius                  | 0.538     | 0.7    | 0.609    | 10.0    |
| 2023 Toyota Sequoia                | 1.0       | 0.571  | 0.727    | 7.0     |
| 2023 Toyota Sienna                 | 0.429     | 0.6    | 0.5      | 5.0     |
| 2023 Toyota Tundra                 | 1.0       | 0.8    | 0.889    | 5.0     |
| 2023 Toyota Venza                  | 0.889     | 1.0    | 0.941    | 8.0     |
| 2023 Volkswagen Arteon             | 0.8       | 0.941  | 0.865    | 17.0    |
| 2023 Volkswagen Atlas              | 0.727     | 0.889  | 0.8      | 9.0     |
| 2023 Volkswagen Golf               | 1.0       | 0.6    | 0.75     | 5.0     |
| 2023 Volkswagen ID                 | 0.143     | 0.167  | 0.154    | 6.0     |
| 2023 Volkswagen ID.4               | 0.0       | 0.0    | 0.0      | 6.0     |
| 2023 Volkswagen Jetta              | 0.538     | 0.7    | 0.609    | 10.0    |
| 2023 Volkswagen Taos               | 0.556     | 0.5    | 0.526    | 10.0    |
| 2023 Volvo S60                     | 0.25      | 0.143  | 0.182    | 7.0     |
| 2023 Volvo S90                     | 0.2       | 0.167  | 0.182    | 6.0     |
| 2023 Volvo XC40                    | 1.0       | 0.667  | 0.8      | 6.0     |
| 2023 Volvo XC60                    | 0.75      | 0.5    | 0.6      | 6.0     |
| 2023 Volvo XC90                    | 0.833     | 0.625  | 0.714    | 8.0     |
| accuracy                           | 0.71      | 0.71   | 0.71     | 0.71    |
