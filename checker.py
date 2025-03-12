import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os

# Set the environment variable
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

dataCSV = pd.read_csv('../cleaned_dataset.csv')
malicious = dataCSV.loc[dataCSV['Value'] == 1]
nonMalicious = dataCSV.loc[dataCSV['Value'] == 0]
minSize = min(len(malicious), len(nonMalicious))
sampleNTrain = nonMalicious.sample(minSize)
sampleMTrain = malicious.sample(minSize)
combined_df = pd.concat([sampleMTrain, sampleNTrain])
shuffled_df = combined_df.sample(frac=1).reset_index(drop=True)

data = {
    "texts": shuffled_df["Prompt"],
    "labels": shuffled_df["Value"]  
}

# preprocessing
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(data["texts"])
sequences = tokenizer.texts_to_sequences(data["texts"])
padded_sequences = pad_sequences(sequences, maxlen=5000)
labels = np.array(data["labels"])

# randomize training and test data
X_train, X_test, y_train, y_test = train_test_split(padded_sequences, labels, test_size=0.2, random_state=42)


# build model
model = Sequential()
# embedding helps turn string input to vector of fixed size
model.add(Embedding(input_dim=5000, output_dim=128))
# long short term memory nodes + dense nodes for classification
model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(64, activation='relu')) 
# one dense node for binary prediction
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# train model
model.fit(X_train, y_train, batch_size=64, verbose=1)

# check prediction accuracy
results = model.predict(X_test)
binary_predictions = (results > 0.5).astype(int)
df = pd.DataFrame(results)
reverse_sequence = ' '.join([tokenizer.index_word.get(i, '') for i in X_test[0]])
accuracy = accuracy_score(y_test, binary_predictions)

model.save('my_model.keras')