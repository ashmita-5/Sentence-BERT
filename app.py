from flask import Flask, render_template, request
from utils import BERT, SimpleTokenizer, calculate_similarity
import torch

app = Flask(__name__)

# Load your custom-trained Sentence Transformer model
model_path = "./app/models/bert_model.pkl"
bert_model = BERT(model_path)

# Function to calculate cosine similarity
def calculate_similarity(query, text):
    # Encode query and text to get embeddings
    query_embedding = bert_model.encode(query, convert_to_tensor=True)
    text_embedding = bert_model.encode(text, convert_to_tensor=True)

    # Calculate cosine similarity
    similarity_score = torch.nn.functional.cosine_similarity(query_embedding, text_embedding).item()
    return similarity_score

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    query = request.form['query']
    text = request.form['text']

    similarity_score = calculate_similarity(query, text)

    return render_template('index.html', query=query, text=text, similarity=similarity_score)

if __name__ == '__main__':
    app.run(debug=True)
