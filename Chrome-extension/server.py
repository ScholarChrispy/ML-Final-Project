# pandas
import pandas as pd

# classifers (from local python file)
from models import RNN_classifier, GRU_classifier, LSTM_classifier

# torch
import torch
from torchtext.vocab import Vocab

# nltk (tokenizer)
from nltk import word_tokenize

# tqdm (progress bar)
from tqdm.notebook import tqdm

# flask
from flask import Flask, jsonify
from flask_cors import CORS

# import vocab from disk
vocab = torch.load('vocab.pth')

#! IMPORTANT: This server WILL NOT FUNCTION PROPERLY if the models have not 
#! already been trained!
# load & configure the RNN model
RNN_model = RNN_classifier(
    vocab_size = len(vocab),
    embedding_dimension = 32, # hyperparameter
    state_dimension = 64 # hyperparameter
)
RNN_model.load_state_dict(torch.load('lightning_logs/RNN/version_0/checkpoints/epoch=19-step=26020.ckpt')['state_dict'])
RNN_model.eval()

# load & configure the GRU model
GRU_model = GRU_classifier(
    vocab_size = len(vocab),
    embedding_dimension = 512, # hyperparameter
)
GRU_model.load_state_dict(torch.load('lightning_logs/GRU/version_0/checkpoints/epoch=19-step=26020.ckpt')['state_dict'])
GRU_model.eval()

# load & configure the LSTM model
LSTM_model = LSTM_classifier(
    vocab_size = len(vocab),
    embedding_dimension = 32, # hyperparameter
    state_dimension = 64 # hyperparameter
)
LSTM_model.load_state_dict(torch.load('lightning_logs/LSTM/version_0/checkpoints/epoch=19-step=26020.ckpt')['state_dict'])
LSTM_model.eval()

# create the Flask app
app = Flask(__name__)
CORS(app)

# Receiving GET requests from the client
@app.route('/<model_type>/<tweet_content>', methods = ['GET'])
def return_sentiment(model_type, tweet_content):
    
    # tokenize the tweet_content to input it into the model
    tokens = word_tokenize(tweet_content)

    # numerically encode the tokens using the vocab object
    content =  [vocab[token] for token in tokens]

    # convert the encoded tokens into a torch Tensor
    content = torch.as_tensor(content)

    if (model_type == 'RNN'):
        value = RNN_model(torch.as_tensor(content))
        print("RNN Sentiment Probabilities: " + str(value.tolist()))  # display the sentiments
        
        return jsonify({'message': value.tolist()}), 200
    elif (model_type == 'GRU'):
        value = GRU_model(torch.as_tensor(content))
        print("GRU Sentiment Probabilities: " + str(value.tolist()))  # display the sentiments
        
        return jsonify({'message': value.tolist()}), 200
    elif (model_type == 'LSTM'):
        value = LSTM_model(torch.as_tensor(content))
        print("LSTM Sentiment Probabilities: " + str(value.tolist()))  # display the sentiments
        
        return jsonify({'message': value.tolist()}), 200
    else:
        # error message
        return jsonify({'message': 'Model does not exist!'}), 400

if __name__ == '__main__':
    app.run(debug = True)