# Random Forest Classifier
### Authors: Radosław Radziukiewicz, Julia Skoneczna 

---
## Project overview
Project's target was to create the implementation of the random forest classifier. <br>
We decided to use the ID3 algorithm for the creation of decision trees, which 
are the necessary part of random forest existance. <br>
Training and model evaluation was performed using the custom implementation of cross-validation technique.

---
## How to run the project
In order to run the project, one should download this repository and lunch `main.py` file. <br>
<br>
**Project dependencies**:
* numpy
* pandas

---
## Dataset
Our model was used to predict student's alcohol consumption. Dataset can be obtained from [here](https://www.kaggle.com/uciml/student-alcohol-consumption "Student Alcohol Consumption") and is present (as the part) in this repository.

---
## Results
As the metric of how-well-model-is-doing-it's-job we used accuracy. The results were rather poor. <br>
After tuning the model we have obtained around `71%` accuracy.  We belive, that such poor result is connected 
with the dataset itselt as well as with not-so-much sophisticated random forest implementation. <br>
For more datails and more elaborated conclusions, please look at the report from the project (unfortunately, available only in polish).
