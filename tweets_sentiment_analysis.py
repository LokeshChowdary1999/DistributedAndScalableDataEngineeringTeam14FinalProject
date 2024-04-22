import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import kaggle as kg
import os
import warnings
warnings.filterwarnings("ignore")
def run_tweet_etl():
    # Set Kaggle API credentials
    os.environ["KAGGLE_USERNAME"] = "lokeshdammalapati"
    os.environ["KAGGLE_KEY"] = "06a501518cca3ed1f48515162a1e929c"
    current_dir = os.getcwd()
    # Initialize Kaggle API
    kg.api.authenticate()
    kg.api.dataset_download_files(dataset="yasserh/twitter-tweets-sentiment-dataset", path=current_dir, unzip=True)
    # Tweets preprocessing 
    # Drop the 'textID' column: This column isn't necessary for our analysis.
    # Clean the 'text' and 'selected_text' columns: Remove any special characters, hyperlinks, and standardize the text format in these fields.
    # Remove stop words and apply stemming: Common stop words will be removed as they typically don't affect sentiment, and stemming will be applied to reduce words to their root form, streamlining the analysis.
    import nltk
    nltk.download('punkt')
    nltk.download('stopwords')
    # dataset 
    df = pd.read_csv(os.path.join(current_dir, 'Tweets.csv'))
    # Dropping the 'textID' column
    df.drop(columns=['textID'], inplace=True)
    # Function to clean and preprocess text
    def clean_and_preprocess_text(text):
        # Convert non-string to string
        text = str(text)
        # Lowercasing the text
        text = text.lower()
        # Removing URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        # Removing special characters and numbers
        text = re.sub(r'\W+|\d+', ' ', text)
        # Tokenization
        tokens = word_tokenize(text)
        # Removing stop words and stemming
        stop_words = set(stopwords.words('english'))
        stemmer = PorterStemmer()
        filtered_tokens = [stemmer.stem(word) for word in tokens if word not in stop_words]
        # Rejoining the tokens into a string
        return " ".join(filtered_tokens)
    # Applying the cleaning function to 'text' and 'selected_text' columns
    df['text'] = df['text'].apply(clean_and_preprocess_text)
    df['selected_text'] = df['selected_text'].apply(clean_and_preprocess_text)
    # Display 
    print(df.head(2))

    #Discover the tweets
    # 1.Sentiment Distribution: We'll examine how the sentiments (neutral, positive, negative) are distributed in the dataset.
    # 2.Tweet Length Analysis: We'll explore if there's a relationship between the length of the tweets and their sentiments.
    # 3.Common Words Visualization: Utilizing word clouds or frequency distributions, we'll identify the most common words in each sentiment category.
    # Set plot style
    sns.set(style="whitegrid")
    # Analyzing Sentiment Distribution
    plt.figure(figsize=(8, 5))
    sns.countplot(x='sentiment', data=df)
    plt.title('Distribution of Sentiments_in_Tweets')
    plt.savefig(os.path.join(current_dir, 'sentiment_distribution.png'))
    plt.close()
    # Sentiment Distribution:
    # The bar chart displays that neutral sentiments are the most prevalent in the dataset, followed by positive and then negative sentiments. This could indicate that people tend to tweet in a neutral tone more frequently.
    # The relatively balanced distribution between positive and negative sentiments shows somehow balanced. 
    # Analyzing the Relationship Between Tweet Length and Sentiment
    df['text_length'] = df['text'].apply(lambda x: len(x.split()))
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='sentiment', y='text_length', data=df)
    plt.title('Tweet Length vs Sentiment')
    plt.savefig(os.path.join(current_dir, 'tweet_length_and_sentiment.png'))
    plt.close()
    # Tweet Length vs Sentiment:
    # From the boxplot, we see that the median tweet length for neutral sentiments is slightly lower than that for positive and negative sentiments.  
    # The spread for negative tweets appears to be larger than for the other two categories.
    # Outliers are present in all three sentiment categories, indicating that there are some tweets significantly longer than the average.
    # Visualizing Common Words in Each Sentiment Category
    wordcloud_dir = os.path.join(current_dir, 'wordcloud_images')
    os.makedirs(wordcloud_dir, exist_ok=True)
    for sentiment in df['sentiment'].unique():
        subset = df[df['sentiment'] == sentiment]
        text = " ".join(subset['text'].tolist())
        wordcloud = WordCloud(width=800, height=400, background_color ='white').generate(text)
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.title(f'Most Common Words in {sentiment.capitalize()} Tweets')
        plt.axis("off")
        image_name = f'Most_Common_Words_{sentiment}.png'
        plt.savefig(os.path.join(wordcloud_dir, image_name))
        plt.close()
    # Positive Tweets:
    # Words like "love," "thank," "good," "great," and "happy" dominate, which are typically associated with positive emotions and experiences. 
    # Negative Tweets:
    # The negative word cloud is scattered with words such as "miss," "sorry," "bore," and "ugh," reflecting dissatisfaction or unpleasant states.
    # Neutral Tweets:
    # Neutral tweets contain a mix of terms that are less emotionally charged, like "work," "today," "going," and "watch." This category seems to encompass a variety of subjects.
    # Creating Feature Engineering
    # Text Vectorization:
    # We'll convert the tweet text into a numerical format using the Term Frequency-Inverse Document Frequency (TF-IDF) technique. This will quantify the importance of words in relation to the dataset as a whole.
    # Target Encoding:
    # We'll encode the 'sentiment' column, our target variable, into a numerical format that can be understood by the machine learning models.
    # Additional Feature Creation:
    # include 'tweet_length' as a feature.
    df.columns
    # begin the TF-IDF Vectorizer
    tfidf_vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
    # Apply TF-IDF to the 'text' column to have the feature matrix
    X_tfidf = tfidf_vectorizer.fit_transform(df['selected_text'])
    # Encoding the 'sentiment' column
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(df['sentiment'])
    # create additional features
    df['tweet_length'] = df['text'].apply(lambda x: len(x.split()))
    #  include 'tweet_length' as a feature:
    from scipy.sparse import hstack
    X_final = hstack((X_tfidf, df[['tweet_length']].values.astype(float)))
    # Actually the model performance fall behind around 10% when use tweet length as a feature so not use it. 
    df.head(2)
    # checking encoding 
    label_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
    print(label_mapping)
    # Constructing machine learning model
    # Partitioning the Data:
    # We will segregate the dataset into separate sets for training, testing, and validation to ensure robust evaluation and mitigate overfitting.
    # Model Training:
    # We will instruct several classifier models, each with its own strengths and theoretical foundations.
    # we are going to use  different model and then choose the best performance. 
    # Models are logistic regression, naive bayes, SVC, and random forset
    # Performance Metrics Assessment:
    # Post-training, we will employ a suite of metrics to appraise the models’ predictive prowess, focusing on accuracy, precision, recall, and the F1-score.
    # Split the data into training, validation, and testing sets
    X_train, X_temp, y_train, y_temp = train_test_split(X_tfidf, y_encoded, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    # Initialize the models
    logistic_model = LogisticRegression(max_iter=1000)
    naive_bayes_model = MultinomialNB()
    svm_model = SVC()
    random_forest_model = RandomForestClassifier(n_estimators=100, random_state=42)

    # Train the Logistic Regression model and evaluate on validation set
    logistic_model.fit(X_train, y_train)
    logistic_train_preds = logistic_model.predict(X_train)
    logistic_val_preds = logistic_model.predict(X_val)
    logistic_train_accuracy = accuracy_score(y_train, logistic_train_preds)
    logistic_val_accuracy = accuracy_score(y_val, logistic_val_preds)
    print("Logistic Regression Training Accuracy:", logistic_train_accuracy)
    print("Logistic Regression Validation Accuracy:", logistic_val_accuracy)

    # Train the Naive Bayes model and evaluate on validation set
    naive_bayes_model.fit(X_train, y_train)
    naive_bayes_train_preds = naive_bayes_model.predict(X_train)
    naive_bayes_val_preds = naive_bayes_model.predict(X_val)
    naive_bayes_train_accuracy = accuracy_score(y_train, naive_bayes_train_preds)
    naive_bayes_val_accuracy = accuracy_score(y_val, naive_bayes_val_preds)
    print("Naive Bayes Training Accuracy:", naive_bayes_train_accuracy)
    print("Naive Bayes Validation Accuracy:", naive_bayes_val_accuracy)

    # Train the SVM model and evaluate on validation set
    svm_model.fit(X_train, y_train)
    svm_train_preds = svm_model.predict(X_train)
    svm_val_preds = svm_model.predict(X_val)
    svm_train_accuracy = accuracy_score(y_train, svm_train_preds)
    svm_val_accuracy = accuracy_score(y_val, svm_val_preds)
    print("SVM Training Accuracy:", svm_train_accuracy)
    print("SVM Validation Accuracy:", svm_val_accuracy)

    # Train the Random Forest model and evaluate on validation set
    random_forest_model.fit(X_train, y_train)
    random_forest_train_preds = random_forest_model.predict(X_train)
    random_forest_val_preds = random_forest_model.predict(X_val)
    random_forest_train_accuracy = accuracy_score(y_train, random_forest_train_preds)
    random_forest_val_accuracy = accuracy_score(y_val, random_forest_val_preds)
    print("Random Forest Training Accuracy:", random_forest_train_accuracy)
    print("Random Forest Validation Accuracy:", random_forest_val_accuracy)
    # Logistic Regression:
    # This model is like a machine that's learning from data. When we tested it on the data it was trained on, it got about 86 out of 100 predictions right. When we tested it on new data (the validation set), it got about 80 out of 100 predictions right. So, it's pretty good at making predictions.
    # Naive Bayes:
    # This model, similar to Logistic Regression, is also good at making predictions. When tested on its training data, it got about 83 out of 100 predictions right. On new data (the validation set), it got about 78 out of 100 predictions right. It's doing well but not as well as Logistic Regression.
    # SVM (Support Vector Machine):
    # This model is the best performer among all. When we tested it on the data it was trained on, it got about 91 out of 100 predictions right. When we tested it on new data (the validation set), it got about 80 out of 100 predictions right. So, it's not only good at learning from data but also at making accurate predictions on new, unseen data.
    # Random Forest:
    # This model is also quite good. When tested on its training data, it got about 95 out of 100 predictions right. On new data (the validation set), it got about 79 out of 100 predictions right. It's doing well but not as consistently as the SVM model.
    # ![image.png](attachment:de08e498-8518-44e6-a3b6-075fa7eef176.png)
    # Evaluate the models on the test dataset
    # Logistic Regression
    logistic_test_preds = logistic_model.predict(X_test)
    logistic_test_report = classification_report(y_test, logistic_test_preds)
    print("Logistic Regression Test Classification Report")
    print(logistic_test_report)
    # Naive Bayes
    naive_bayes_test_preds = naive_bayes_model.predict(X_test)
    naive_bayes_test_report = classification_report(y_test, naive_bayes_test_preds)
    print("Naive Bayes Test Classification Report")
    print(naive_bayes_test_report)
    # SVM
    svm_test_preds = svm_model.predict(X_test)
    svm_test_report = classification_report(y_test, svm_test_preds)
    print("SVM Test Classification Report")
    print(svm_test_report)
    # Random Forest
    random_forest_test_preds = random_forest_model.predict(X_test)
    random_forest_test_report = classification_report(y_test, random_forest_test_preds)
    print("Random Forest Test Classification Report")
    print(random_forest_test_report)
    # Logistic Regression:
    # The Logistic Regression model performs consistently across different sentiment categories. It correctly identifies sentiments approximately 80% of the time. This means that 8 out of 10 tweets are accurately classified.
    # It exhibits a balanced performance with precision and recall ranging between 78% and 85%. This indicates that when it makes a prediction, it's correct about 78% to 85% of the time, depending on the sentiment.
    # Naive Bayes:
    # The Naive Bayes model achieves an overall accuracy of approximately 78%. This means that it accurately predicts the sentiment for about 78% of the tweets in the test dataset.
    # It excels in predicting extreme sentiments, with high precision for negative and positive classes (around 85%). However, it misses some negative tweets, with a recall of about 63%.
    # SVM (Support Vector Machine):
    # The SVM model delivers strong performance with an accuracy of approximately 80%. This means that it correctly classifies tweets as negative, neutral, or positive in 80% of the cases.
    # It particularly shines in identifying positive sentiment tweets, with a precision of around 90% and an overall balanced performance.
    # Random Forest:
    # The Random Forest model achieves an accuracy of about 79%. This implies that it accurately predicts sentiment for roughly 79% of the tweets in the test dataset.
    # It provides a balanced performance across the sentiment classes, with precision and recall scores around 76% to 85%, depending on the sentiment.
    # In summary, all models are doing a commendable job in sentiment classification, with Logistic Regression and SVM leading the way with around 80% accuracy. Naive Bayes excels in predicting extreme sentiments but misses some negative tweets. Random Forest, while robust, falls slightly short of the top performers in terms of overall accuracy.
    # Conclusion
    # In conclusion, our sentiment analysis project using a Twitter dataset has yielded promising results and valuable insights into classifying tweets into three sentiment categories: negative, neutral, and positive. We successfully preprocessed and cleaned the data, conducted exploratory data analysis, engineered relevant features, and trained and evaluated four different machine learning models.
    # The Logistic Regression and Support Vector Machine (SVM) models emerged as the frontrunners, with accuracies around 80%. These models demonstrated balanced performance across precision, recall, and accuracy metrics. Naive Bayes displayed high precision for negative and positive sentiments but struggled with recall for negative tweets. The Random Forest model, while providing balanced precision and recall, slightly lagged behind the top performers in terms of accuracy.
run_tweet_etl()