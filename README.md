# Disaster Prediction from Tweets

## Project Description:

In this project, I developed an NLP pipeline to predict whether a tweet is related to a disaster or not. The pipeline involves preprocessing the tweets, extracting features, and training a machine learning model to make predictions. The project achieved an accuracy of 80% on the test set.

Skills Utilized:

Natural Language Processing (NLP)
Python
Pandas
Scikit-learn
Machine Learning

## Steps Involved:

- Data Collection: The first step was to collect a dataset of tweets related to disasters. I used the publicly available dataset provided by Kaggle for this purpose.

- Data Preprocessing: The tweets were preprocessed by removing stop words, URLs, and special characters, converting the text to lowercase, and tokenizing the text.

- Feature Extraction: The preprocessed tweets were then converted to a feature matrix using TF-IDF vectorization. This helped in representing the tweets as numerical vectors, which could be used as input to the machine learning model.

- Model Training: The feature matrix was split into training and testing sets. A machine learning model was trained on the training set using the Random Forest Classifier algorithm.

- Model Evaluation: The trained model was evaluated on the testing set to determine its accuracy.

## Usage :

Clone the repository:

```bash
git clone https://github.com/kulsumkamal/Disaster-Prediction.git
```

Install the required libraries:

```bash
pip install pandas scikit-learn nltk transformers tensorflow
```

Run the main.py script to preprocess the data, extract features, and train the machine learning model:

```bash
python main.py
```

The model will be saved as model.pkl.

To make predictions on new data, use the predict.py script:

```bash
python predict.py
```

## Results:

The project achieved an accuracy of 80% on the test set. This indicates that the NLP pipeline is effective in predicting whether a tweet is related to a disaster or not.
