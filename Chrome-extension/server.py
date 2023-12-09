from flask import Flask, Response

def temp(content):
    return content

# TODO: Import models from lightning_logs
RNN_model = temp
GRU_model = temp
LSTM_model = temp

# create the Flask app
app = Flask(__name__)

# Receiving GET requests from the client
@app.route('/<model_type>/<tweet_content>', methods = ['GET'])
def return_sentiment(model_type, tweet_content):
    if (model_type == 'RNN'):
        value = RNN_model(tweet_content)
        return Response(
            {
                'message': value
            },
            status = 200
        )
    elif (model_type == 'GRU'):
        value = GRU_model(tweet_content)
        return Response(
            {
                'message': value
            },
            status = 200
        )
    elif (model_type == 'LSTM'):
        value = LSTM_model(tweet_content)
        return Response(
            {
                'message': value
            },
            status = 200
        )
    else:
        return Response(
            {
                'message': 'Model type does not exist!'
            },
            status = 400
        )
    return f'type: {model_type}, content: {tweet_content}'

if __name__ == '__main__':
    app.run(debug = True)