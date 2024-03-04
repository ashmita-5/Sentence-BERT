from flask import Flask, render_template, request
from utils import BERT,SimpleTokenizer,calculate_similarity
import torch
import pickle

app = Flask(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Load your custom-trained Sentence Transformer model
model_path = "./app/models/s_bertmodel.pt"
bert_model = BERT()

BERTData = pickle.load(open('./app/models/bert_model.pkl', 'rb'))
word2id = BERTData['word2id']
tokenizer = SimpleTokenizer(word2id)


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    sentence1 = request.form.get('sentence1')
    sentence2 = request.form.get('sentence2')

    similarity_score = calculate_similarity(bert_model, tokenizer, sentence1, sentence2, device)

    return render_template('index.html', similarity=similarity_score)

if __name__ == '__main__':
    app.run(debug=True)
