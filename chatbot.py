import random
import pickle
import json
import numpy as np

import nltk
from nltk.stem import WordNetLemmatizer

from tensorflow.keras.models import load_model


# Open Lemmatized JSON file
lemmatizer = WordNetLemmatizer()
intents = json.loads(open('intents.json').read())

# Load in binary files
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('chatbot.h5')

# Cleaning up sentences
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words

# Bag of words (list of 0 and 1)
def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, word in enumerate(words):
            if word == s:
                bag[i] = 1
    return np.array(bag)

# Prediction
def predict(sentence):
    bow = bag_of_words(sentence)
    prediction = model.predict(np.array([bow]))[0]
    error = 0.25
    result = [[i, r] for i, r in enumerate(prediction) if r > error]
    result.sort(key=lambda x: x[1], reverse=True)  # Sorting the result

    return_list = []
    for r in result:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
    return return_list

# Get Response
def get_response(intents_list, intents_json):
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result

print("Welcome to Sammy's Porsche Manual")
print("Please ask away: ")

while True:
    message = input("")
    ints = predict(message)
    response = get_response(ints, intents)
    print(response)





