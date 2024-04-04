from flask import Flask
from werkzeug.datastructures import FileStorage
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
import tensorflow as tf
import pandas as pd
import re
import numpy as np
from flask_restx import Api, Resource, fields

app = Flask(__name__)
api = Api(app, version='1.0', title='Sentiment Analysis API', description='A simple API for sentiment analysis')

ns = api.namespace('sentiment', description='Sentiment operations')
parser = api.parser()
parser.add_argument('text', type=str, required=True, help='Text to analyze sentiment')

@ns.route('/analyze')
class SentimentAnalysis(Resource):
    @api.expect(parser)
    def post(self):
        args = parser.parse_args()
        text = args['text']

        # Your sentiment analysis code here
        df = pd.read_csv('train_preprocess.txt', sep='\t', names=['text', 'sentiment'])
        df['cleaned_text'] = df['text'].apply(lambda x: re.sub(r'\W', ' ', x))

        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(df['cleaned_text'])

        y = df['sentiment']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = MLPClassifier(hidden_layer_sizes=(10,), max_iter=1000)
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)
        report = classification_report(y_test, predictions)

        df['sentiment'] = df['sentiment'].map({'positive': 2, 'neutral': 1,'negative': 0})

        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(df['cleaned_text'])
        sequences = tokenizer.texts_to_sequences(df['cleaned_text'])

        X = pad_sequences(sequences)
        X_train, X_test, y_train, y_test = train_test_split(X, df['sentiment'], test_size=0.2, random_state=42)

        # Remove the extra dimension
        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1])
        X_test = X_test.reshape(X_test.shape[0], X_test.shape[1])

        model = Sequential()
        model.add(Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=64))  
        model.add(LSTM(100))
        model.add(Dense(3, activation='softmax'))  

        model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.fit(X_train, y_train, epochs=10, batch_size=64, validation_data=(X_test, y_test))

        y_pred_prob = model.predict(X_test)
        y_pred = tf.argmax(y_pred_prob, axis=-1)
        lstm_report = classification_report(y_test, y_pred, target_names=['positive','neutral','negative' ])

        # Fungsi sigmoid
        def sigmoid(x):
            return 1 / (1 + np.exp(-x))

        # Data initial
        X = np.array([[1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]])  # Input data
        target = np.array([0, 1, 1, 0])  # Ground truth
        alpha = 0.1  # Learning rate
        theta = -1  # Threshold

        # Initial weights (random)
        weights = np.random.rand(3, 3)

        # Forward pass
        y4 = sigmoid(np.dot(X, weights[:, 0]))
        y5 = sigmoid(np.dot(X, weights[:, 1]))
        y6 = sigmoid(np.dot(X, weights[:, 2]))

        # Hitung error
        e = target - y6

        # Backward pass
        delta6 = e * y6 * (1 - y6)
        delta5 = np.dot(delta6, weights[1, 2]) * y5 * (1 - y5)
        delta4 = np.dot(delta6, weights[0, 2]) * y4 * (1 - y4)

        # Koreksi bobot
        delta_weights = np.zeros_like(weights)
        delta_weights[:, 2] = alpha * np.dot(X.T, delta6)
        delta_weights[:, 1] = alpha * np.dot(X.T, delta5)  # Update delta_weights for delta5
        delta_weights[:, 0] = alpha * np.dot(X.T, delta4)

        # Update weights
        weights += delta_weights

        # Nilai-nilai dari TfidfVectorizer
        X = np.array([0.7, 0.8, 0.9])

        # Forward Pass
        y4 = sigmoid(np.dot(X, weights[:, 0]))
        y5 = sigmoid(np.dot(X, weights[:, 1]))
        y6 = sigmoid(np.dot(X, weights[:, 2]))

        # Lakukan prediksi
        output = y6 > theta

        # Print hasil
        forward_pass_result = {
            "y4": y4.tolist(),
            "y5": y5.tolist(),
            "y6": y6.tolist(),
            "Error": e.tolist()
        }

        backward_pass_result = {
            "delta6": delta6.tolist(),
            "delta5": delta5.tolist(),
            "delta4": delta4.tolist()
        }

        updated_weights_result = {
            "Updated Weights": weights.tolist()
        }

        neuron_output_result = {
            "Output dari neuron 4 (y4)": y4.tolist(),
            "Output dari neuron 5 (y5)": y5.tolist(),
            "Output dari neuron 6 (y6)": y6.tolist(),
            "Prediksi akhir": output.tolist()
        }

        # Return the result
        return {
            'mlp_report': report,
            'lstm_report': lstm_report,
            'forward_pass_result': forward_pass_result,
            'backward_pass_result': backward_pass_result,
            'updated_weights_result': updated_weights_result,
            'neuron_output_result': neuron_output_result
        }, 200

if __name__ == '__main__':
    app.run(debug=True)
