from flask import Flask, render_template, request
from model import generate_sentiment_based_text

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate():
    seed_words = request.form['seed_words']
    sentiment = request.form['sentiment']
    generated_text = generate_sentiment_based_text(seed_words, sentiment, length=10)
    return render_template('result.html', generated_text=generated_text, seed_words=seed_words, sentiment=sentiment)

if __name__ == '__main__':
    app.run(debug=True)
