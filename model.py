import random
from collections import defaultdict

# Include your code from the previous steps
import re
import tensorflow as tf
from tensorflow.keras.datasets import imdb

# Load the full IMDB dataset
num_words = 10000  # Use the top 10,000 words
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=num_words)

# Combine training and test data
X = list(X_train) + list(X_test)
y = list(y_train) + list(y_test)

# Convert the integer sequences back to words
word_index = imdb.get_word_index()
reverse_word_index = {value: key for key, value in word_index.items()}

def decode_review(encoded_review):
    return ' '.join([reverse_word_index.get(i - 3, '?') for i in encoded_review])

# Separate positive and negative reviews
positive_reviews = []
negative_reviews = []

for i, review in enumerate(X):
    decoded_review = decode_review(review)
    if y[i] == 1:
        positive_reviews.append(decoded_review)
    else:
        negative_reviews.append(decoded_review)

# Clean the reviews
def clean_review(review):
    review = re.sub(r'\d+', '', review)  # Remove digits
    review = re.sub(r'\W+', ' ', review.lower())  # Remove special characters
    return review.strip()

positive_reviews = [clean_review(review) for review in positive_reviews]
negative_reviews = [clean_review(review) for review in negative_reviews]

# Function to train a Markov Chain model using trigrams
def train_markov_chain(text_data):
    markov_chain = defaultdict(lambda: defaultdict(int))
    
    for review in text_data:
        words = review.split()
        for i in range(len(words) - 2):
            current_state = (words[i], words[i + 1])  # Current trigram state
            next_word = words[i + 2]
            markov_chain[current_state][next_word] += 1
    
    # Convert counts to probabilities
    for current_state, next_words in markov_chain.items():
        total_count = sum(next_words.values())
        for next_word in next_words:
            next_words[next_word] /= total_count
    
    return markov_chain

# Train separate Markov Chains for positive and negative sentiments
positive_chain = train_markov_chain(positive_reviews)
negative_chain = train_markov_chain(negative_reviews)

# Function to generate text based on the chosen Markov Chain model
def generate_text(markov_chain, seed_words, length=10):
    seed_words = seed_words.lower().split()
    current_state = (seed_words[-2], seed_words[-1])
    text = seed_words.copy()
    
    for _ in range(length - 2):  # Length adjusted for the initial seed words
        next_words = markov_chain.get(current_state, None)
        if not next_words:
            break
        next_word = random.choices(list(next_words.keys()), list(next_words.values()))[0]
        text.append(next_word)
        current_state = (current_state[1], next_word)
    
    return ' '.join(text)

# Function to generate text based on user-specified sentiment
def generate_sentiment_based_text(seed_words, sentiment, length=10):
    if sentiment == 'positive':
        return generate_text(positive_chain, seed_words, length)
    elif sentiment == 'negative':
        return generate_text(negative_chain, seed_words, length)
    else:
        return "Invalid sentiment chosen. Please select 'positive' or 'negative'."
