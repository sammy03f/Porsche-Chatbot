import random
import pickle
import json
import numpy as np

import nltk
from nltk.stem import WordNetLemmatizer

from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.models import Sequential

# Create a lemmatizer for words and load in the JSON file
lemmatizer = WordNetLemmatizer()
intents = json.loads(open('intents.json').read())

# Store directories and delete unnecessary punctuation
words = []
classes = []
documents = []
ignore_cap = ["!", ".", ",", "?", "/"]

# Iterate over intents file and tokenize words
for intent in intents['intents']:
    for pattern in intent['patterns']:
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)
        documents.append((word_list, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# Lemmatize words and remove duplicates
words = [lemmatizer.lemmatize(word) for word in words if word not in ignore_cap]
words = sorted(list(set(words)))
classes = sorted(list(set(classes)))

# Serialize tokenized words into a file
pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

# Add document data to training list
training = []
output_empty = [0] * len(classes)

for document in documents:
    bag = []
    word_list = document[0]
    word_list = [lemmatizer.lemmatize(word.lower()) for word in word_list]
    for word in words:
        bag.append(1) if word in word_list else bag.append(0)
    output_row = list(output_empty)
    output_row[classes.index(document[1])] = 1
    training.append([bag, output_row])

# Shuffle data and split into x and y variables
random.shuffle(training)
training = np.array(training, dtype=object)
train_x = np.array([item[0] for item in training])
train_y = np.array([item[1] for item in training])

# Begin training
model = Sequential()

# Model contains 128 neurons and is rectified into a linear unit
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))  # Prevents overfitting
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))  # Softmax scales results in output layer to add up to one

# Cast a gradient decent optimizer
sgd = SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)

# Compile and save model
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
hist = model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)
model.save('chatbot.h5', hist)
print("Model saved")

