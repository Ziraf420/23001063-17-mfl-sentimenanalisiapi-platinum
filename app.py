from flask import Flask, request
from flask_restx import Api, Resource
from werkzeug.datastructures import FileStorage
from werkzeug.utils import secure_filename
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import pandas as pd
import re
import numpy as np
import nltk
import os
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# Initialize Flask app
app = Flask(__name__)
api = Api(app, version='1.0', title='Sentiment Analysis API', description='A simple API for sentiment analysis')

# Namespace
ns = api.namespace('Sentiment', description='Sentiment operations')

# Parser for text input
parser = api.parser()
parser.add_argument('text', type=str, required=True, help='Text to analyze sentiment')

# Parser for file upload
file_upload_parser = api.parser()
file_upload_parser.add_argument('file', location='files', type=FileStorage, required=True)

# Define the directory to save uploaded files
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'txt'}

# Function to check if the file extension is allowed
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Set the upload directory
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Function to ensure the upload directory exists
def ensure_upload_dir_exists():
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)

# Call the function to ensure the upload directory exists
ensure_upload_dir_exists()

# Download NLTK stopwords resource
nltk.download('stopwords')

# Load stopword list
stopwords = set(nltk.corpus.stopwords.words('indonesian'))

# Load alay dictionary
df_alay = pd.read_csv('newkamus_alay.csv', header=None, encoding='latin1')
alay_dict = dict(zip(df_alay[0], df_alay[1]))

# Function to preprocess text
def preprocess_text(text):
    # Remove non-word characters and convert to lowercase
    cleaned_text = re.sub(r'\W', ' ', text.lower())
    # Remove stopwords
    cleaned_text = ' '.join(word for word in cleaned_text.split() if word not in stopwords)
    # Replace alay words with their original form
    cleaned_text = ' '.join(alay_dict.get(word, word) for word in cleaned_text.split())
    # Stemming
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    cleaned_text = stemmer.stem(cleaned_text)
    return cleaned_text

def analyze_sentiment_file_rnn(file_path):
    results = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            result = analyze_sentiment_text_rnn(line.strip())
            results.append(result)
    return results

def analyze_sentiment_file_lstm(file_path):
    results = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            result = analyze_sentiment_text_lstm(line.strip())
            results.append(result)
    return results

def analyze_sentiment_text_rnn(text):
    df = pd.read_csv('train_preprocess.txt', sep='\t', names=['text', 'sentiment'])
    df['cleaned_text'] = df['text'].apply(preprocess_text)

    # Ekstraksi fitur menggunakan TF-IDF
    tfidf_vectorizer = TfidfVectorizer()
    X = tfidf_vectorizer.fit_transform(df['cleaned_text'])

    y = df['sentiment'].map({'positive': 2, 'neutral': 1, 'negative': 0})
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = Sequential()
    model.add(Dense(128, activation='relu', input_dim=X_train.shape[1]))
    model.add(Dense(3, activation='softmax'))

    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=10, batch_size=64, validation_data=(X_test, y_test))

    # Preprocess input text
    text = preprocess_text(text)
    text_vectorized = tfidf_vectorizer.transform([text])

    sentiment_category = np.argmax(model.predict(text_vectorized))
    sentiment_label = {2: 'positive', 1: 'neutral', 0: 'negative'}
    sentiment_result = sentiment_label.get(sentiment_category, 'unknown')

    # Classification Report
    y_pred = model.predict_classes(X_test)
    print(classification_report(y_test, y_pred))

    # Confusion Matrix Visualization
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['negative', 'neutral', 'positive'], yticklabels=['negative', 'neutral', 'positive'])
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix')
    plt.show()

    return {'text': text, 'sentiment': sentiment_result}

def analyze_sentiment_text_lstm(text):
    df = pd.read_csv('train_preprocess.txt', sep='\t', names=['text', 'sentiment'])
    df['cleaned_text'] = df['text'].apply(preprocess_text)

    # Ekstraksi fitur menggunakan TF-IDF
    tfidf_vectorizer = TfidfVectorizer()
    X = tfidf_vectorizer.fit_transform(df['cleaned_text'])

    y = df['sentiment'].map({'positive': 2, 'neutral': 1, 'negative': 0})
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = Sequential()
    model.add(Dense(128, activation='relu', input_dim=X_train.shape[1]))
    model.add(Dense(3, activation='softmax'))

    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=10, batch_size=64, validation_data=(X_test, y_test))

    # Preprocess input text
    text = preprocess_text(text)
    text_vectorized = tfidf_vectorizer.transform([text])

    sentiment_category = np.argmax(model.predict(text_vectorized))
    sentiment_label = {2: 'positive', 1: 'neutral', 0: 'negative'}
    sentiment_result = sentiment_label.get(sentiment_category, 'unknown')

    # Classification Report
    y_pred = model.predict_classes(X_test)
    print(classification_report(y_test, y_pred))

    # Confusion Matrix Visualization
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['negative', 'neutral', 'positive'], yticklabels=['negative', 'neutral', 'positive'])
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix')
    plt.show()

    return {'text': text, 'sentiment': sentiment_result}

# Endpoint for analyzing text sentiment using RNN model
@ns.route('/analyze_text_rnn')
class SentimentAnalysisTextRNN(Resource):
    @api.expect(parser)
    def post(self):
        args = parser.parse_args()
        text = args['text']
        return analyze_sentiment_text_rnn(text), 200

# Endpoint for analyzing text sentiment using LSTM model
@ns.route('/analyze_text_lstm')
class SentimentAnalysisTextLSTM(Resource):
    @api.expect(parser)
    def post(self):
        args = parser.parse_args()
        text = args['text']
        return analyze_sentiment_text_lstm(text), 200

# Endpoint for analyzing file sentiment using RNN model
@ns.route('/analyze_file_rnn')
class SentimentAnalysisFileRNN(Resource):
    @api.expect(file_upload_parser)
    def post(self):
        args = file_upload_parser.parse_args()
        uploaded_file = args['file']
        
        if uploaded_file and allowed_file(uploaded_file.filename):
            filename = secure_filename(uploaded_file.filename)
            uploaded_file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            
            # Call function to analyze sentiment from file using RNN model
            result = analyze_sentiment_file_rnn(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return result, 200
        else:
            return {'message': 'Invalid file. Please upload a .txt file.'}, 400

# Endpoint for analyzing file sentiment using LSTM model
@ns.route('/analyze_file_lstm')
class SentimentAnalysisFileLSTM(Resource):
    @api.expect(file_upload_parser)
    def post(self):
        args = file_upload_parser.parse_args()
        uploaded_file = args['file']
        
        if uploaded_file and allowed_file(uploaded_file.filename):
            filename = secure_filename(uploaded_file.filename)
            uploaded_file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            
            # Call function to analyze sentiment from file using LSTM model
            result = analyze_sentiment_file_lstm(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return result, 200
        else:
            return {'message': 'Invalid file. Please upload a .txt file.'}, 400

if __name__ == '__main__':
    app.run(debug=True)