from flask import Flask, request, jsonify
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import math

app = Flask(__name__)

# load keras model
model = load_model('my_model.h5')

tokenizer = Tokenizer(num_words=5000)

# create endpoint
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    input_string = data['input_string']
    
    # tokenize + pad input string
    sequences = tokenizer.texts_to_sequences([input_string])
    
    input_data = pad_sequences(sequences, maxlen=1000)  

    prediction = model.predict(input_data)
    prediction_number = prediction[0][0]  

    return jsonify({'prediction': math.round(prediction_number)})

if __name__ == '__main__':
    app.run(debug=True)
