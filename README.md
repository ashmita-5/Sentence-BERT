# NLP-Assignment-5---BERT

## Student Info

Ashmita Phuyal(124454)

### Task 1: BERT from Scratch with Sentence Transformer <br>

The IMDb dataset is loaded using the load_dataset function from the Hugging Face datasets library. This dataset contains movie reviews labeled with sentiment polarity (positive or negative). The dataset have train, test and unsupervised subset of the data. The dataset is implemented and trained via Bidirectional Encoder Representations from Transformers (BERT) from scratch and saved the trained model on bert_model.pkl.

### Task 2: Sentence Embedding with Sentence BERT <br>

The datasets: SNLI or MNLI from Hugging Face is trained using fine-tuning a pretrained BERT model and the model is trained to generate sentence embeddings that capture semantic cosine similarity between sentences along with the accuracy of the model.

### Task 3. Evaluation and Analysis<br>

1. Sentence Comparison<br>


<img width="712" alt="Screenshot 2024-03-04 at 01 46 52" src="https://github.com/ashmita-5/NLP-Assignment-5---BERT/assets/32629216/568c6b5d-9cdd-4a53-86ab-45d439ce5d3d">



<img width="729" alt="Screenshot 2024-03-04 at 01 47 01" src="https://github.com/ashmita-5/NLP-Assignment-5---BERT/assets/32629216/f54e812e-153d-4b88-9e10-074d26438e14">

2. Performance of the model

| Model | Average Training Loss | Average Cosine Similarity | Average Accuracy |
| --- | --- | --- | --- | 
| BERT - Pretrained | 1.10 | 76.61% | 32.80%
| BERT - Scratch | 
