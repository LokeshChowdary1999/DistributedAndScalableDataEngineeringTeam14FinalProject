# DistributedAndScalableDataEngineeringTeam14FinalProject

# Twitter Sentiment Analysis Project
#### Data Set:
Data Collection using Kaggle API:
The project begins by importing necessary libraries and setting up Kaggle API credentials.
It downloads a dataset named "twitter-tweets-sentiment-dataset" from Kaggle using the Kaggle API.
This dataset contains tweets along with their sentiments (positive, negative, neutral).

#### Data Preprocessing:
After downloading the dataset, it preprocesses the tweet text by cleaning and standardizing it.
The cleaning process involves converting text to lowercase, removing URLs, special characters, and numbers, and tokenizing the text.
Stop words (common words like "the", "is", and "and" etc.) are removed, and stemming is applied to reduce words to their root form.

#### Exploratory Data Analysis (EDA):
Once the data is preprocessed, it conducts exploratory data analysis to gain insights into the dataset.
It analyzes the distribution of sentiments (positive, negative, neutral) in the dataset using visualizations like countplots.
It explores the relationship between tweet length and sentiment using boxplots.
Word clouds are generated to visualize the most common words in each sentiment category.

#### Feature Engineering:
Text Vectorization: The tweet text is converted into a numerical format using TF-IDF (Term Frequency-Inverse Document Frequency) vectorization.
Target Encoding: The sentiment labels (positive, negative, neutral) are encoded into numerical format.
Additional Feature Creation: Features like tweet length are created and considered for modeling.

#### Model Training and Evaluation:
The dataset is split into training, validation, and testing sets.
Four machine learning models (Logistic Regression, Naive Bayes, SVM, Random Forest) are trained on the training set and evaluated on the validation set.
Model performance is assessed using accuracy, precision, recall, and F1-score metrics.
The best performing model is then evaluated on the test set to assess its performance on unseen data.

#### Conclusion:
The project concludes by summarizing the findings and insights gained from the sentiment analysis of Twitter data.
It highlights the performance of different machine learning models and provides recommendations or insights based on the analysis.
Overall, the project aims to perform sentiment analysis on Twitter data to classify tweets into different sentiment categories and provide insights into the sentiment distribution and trends in the dataset.
