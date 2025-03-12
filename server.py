from flask import Flask, request, jsonify
from pymongo import MongoClient
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from flask_cors import CORS
import logging

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "http://localhost:3000"}})
logging.basicConfig(level=logging.DEBUG)
# load keras model
model = load_model('./my_model.keras')

client = MongoClient('mongodb://localhost:27017/')
db = client['promptchecker']

tokenizer = Tokenizer(num_words=5000)

# create endpoint
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    input_string = data['input_string']
    tokenizer.fit_on_texts([input_string])
    
    # tokenize + pad input string
    sequences = tokenizer.texts_to_sequences([input_string])
    
    input_data = pad_sequences(sequences, maxlen=5000)  

    prediction = model.predict(input_data)
    prediction_number = prediction[0][0]  

    return str(prediction_number)

@app.route('/falseNegative', methods=['POST'])
def falseNegative():
    try:
        data = request.json
        input_string = data['input_string']
        # Perform some operations with data
        if not data:
            raise ValueError("Invalid data received")
        db['falseNegative'].insert_one({
            'prompt': input_string
        })
        return jsonify(message='Request successful'), 200
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        return jsonify(error=str(e)), 500

@app.route('/falsePositive', methods=['POST'])
def falsePositive():
    try:
        data = request.json
        input_string = data['input_string']
        # Perform some operations with data
        if not data:
            raise ValueError("Invalid data received")
        db['falsePositive'].insert_one({
            'prompt': input_string
        })
        return jsonify(message='Request successful'), 200
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        return jsonify(error=str(e)), 500

if __name__ == '__main__':
    app.run(debug=True)
