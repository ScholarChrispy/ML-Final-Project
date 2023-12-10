from lightning import LightningModule
import torch
import torch.nn as nn
from torchmetrics import Accuracy

# a basic RNN classifier using a LightningModule
class RNN_classifier(LightningModule):
    def __init__(self, vocab_size, embedding_dimension, state_dimension):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dimension)
        self.rnn = nn.RNN(
            input_size = embedding_dimension,
            hidden_size = state_dimension,
            num_layers = 1, # hyperparameter
            batch_first = True
        )

        # possible outputs: positive, negative, neutral
        self.output = nn.Linear(state_dimension, 3)

        # activation function to provide probabilities
        self.activation = nn.Sigmoid()

        # monitors accuracy
        self.accuracy = Accuracy(task = 'multiclass', num_classes = 3)

    def forward(self, sequence_batch):
        embedded = self.embedding(sequence_batch)
        h_t, h_n = self.rnn(embedded)  # output features (h_t) and state (h_n)
        output = self.output(h_n[-1])
        output = self.activation(output)

        return output

    def loss(self, output, targets):
        loss = nn.CrossEntropyLoss()

        return loss(output, targets)

    def training_step(self, batch):
        inputs, targets = batch
        outputs = self.forward(inputs)
        loss = self.loss(outputs, targets)

        # get accuracy value
        self.accuracy(outputs, targets)

        # log the training accuracy
        self.log('training accuracy', self.accuracy, prog_bar = True)

        return loss

    def validation_step(self, batch):
        inputs, targets = batch
        outputs = self.forward(inputs)

        # get accuracy value
        self.accuracy(outputs, targets)

        # log the validation accuracy
        self.log('validation accuracy', self.accuracy, prog_bar = True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())

class GRU_classifier(LightningModule):
    def __init__(self, vocab_size, embedding_dimension):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dimension)
        self.GRU = nn.GRU(input_size = embedding_dimension, hidden_size = 100, batch_first=True)
        self.linear1 = nn.Linear(100, 62)
        self.ReLU = nn.ReLU()
        self.linear2 = nn.Linear(62, 32)

        # possible outputs: positive, negative, neutral
        self.output = nn.Linear(32, 3)

        # activation function to provide probabilities
        self.activation = nn.Sigmoid()

        # monitors accuracy
        self.accuracy = Accuracy(task = 'multiclass', num_classes = 3)

    def forward(self, sequence_batch):
        output = self.embedding(sequence_batch)
        output, h_o = self.GRU(output)
        output = self.linear1(h_o[-1])
        output = self.ReLU(output)
        output = self.linear2(output)
        output = self.ReLU(output)
        output = self.output(output)
        output = self.activation(output)

        return output

    def loss(self, output, targets):
        loss = nn.CrossEntropyLoss()

        return loss(output, targets)

    def training_step(self, batch):
        inputs, targets = batch
        outputs = self.forward(inputs)
        loss = self.loss(outputs, targets)

        # get accuracy value
        self.accuracy(outputs, targets)

        # log the training accuracy
        self.log('training accuracy', self.accuracy, prog_bar = True)

        return loss

    def validation_step(self, batch):
        inputs, targets = batch
        outputs = self.forward(inputs)

        # get accuracy value
        self.accuracy(outputs, targets)

        # log the validation accuracy
        self.log('validation accuracy', self.accuracy, prog_bar = True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())

# an LSTM classifier using a LightningModule
class LSTM_classifier(LightningModule):
    def __init__(self, vocab_size, embedding_dimension, state_dimension):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dimension)
        self.lstm = nn.LSTM(
            input_size = embedding_dimension,
            hidden_size = state_dimension,
            num_layers = 2, # hyperparameter
            batch_first = True
        )

        # possible outputs: positive, negative, neutral
        self.output = nn.Linear(state_dimension, 3)

        # activation function to provide probabilities
        self.activation = nn.Sigmoid()

        # monitors accuracy
        self.accuracy = Accuracy(task = 'multiclass', num_classes = 3)

    def forward(self, sequence_batch):
        embedded = self.embedding(sequence_batch)
        h_t, (h_n, _) = self.lstm(embedded)  # output features (h_t) and state (h_n)
        output = self.output(h_n[-1])
        output = self.activation(output)

        return output

    def loss(self, output, targets):
        loss = nn.CrossEntropyLoss()

        return loss(output, targets)

    def training_step(self, batch):
        inputs, targets = batch
        outputs = self.forward(inputs)
        loss = self.loss(outputs, targets)

        # get accuracy value
        self.accuracy(outputs, targets)

        # log the training accuracy
        self.log('training accuracy', self.accuracy, prog_bar = True)

        return loss

    def validation_step(self, batch):
        inputs, targets = batch
        outputs = self.forward(inputs)

        # get accuracy value
        self.accuracy(outputs, targets)

        # log the validation accuracy
        self.log('validation accuracy', self.accuracy, prog_bar = True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())
