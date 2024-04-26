# Sentence-BERT

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

3. Hyperparameters

The performance of our BERT model might not be great because we trained it on a relatively small amount of data (1 million sentences), and we only used 6 layers of encoder. BERT typically has 12 layers. Additionally, when fine-tuning S-BERT (a variant of BERT) on our data, we had even less training data and trained for only one epoch. 

In our case, we used 6 layers of encoder, 8 attention heads, and an embedding size of 768. These selection is done based on the size of our training data. To improve performance, we might need more training data and could consider using more layers in the BERT model. Having larger datasets and deeper models could help improve the performance of our model.

The hyperparameters chosen for training our BERT model was: Training data, Embedding Size, Number of epochs, Vocab size.

4. Limitations and Improvements

During implementation, I faced challenges in tokenizing the dataset for the scratch model and pre-trained model as we need to combine them and it was quite challenging for me. Also, due to the limited amount of training data, there was impact on the performance of the model.

To improve the model, we could consider collecting more training data  We can also experiment with fine-tuning hyperparameters like learning rate could further enhance the model's performance.

### Task 4. Text similarity - Web Application Development

<img width="1251" alt="Screenshot 2024-03-04 at 23 12 37" src="https://github.com/ashmita-5/NLP-Assignment-5---BERT/assets/32629216/7db09a5a-9ab0-4885-9bf8-d3d36b827322">

