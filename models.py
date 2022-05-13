####################################
# Models

# embed the input character indices using a nn.Embedding layer and then feed resulting embedding tensors into an RNN.
# After that, you can use RNN’s output feature of the last timestamp as the input of a two-layer feed-forward network,
# and obtain the probability.
####################################


import numpy as np
import torch
import torch.nn as nn

np.random.seed(42)


class RNNClsModel(nn.Module):
    def __init__(self,
                 weight_matrix=None,
                 vocab_size: int = 1000,
                 embed_dim: int = 64,
                 hidden_dim: int = 32,
                 output_dim: int = 1,
                 dropout: float = 0.1
                 ) -> None:
        super().__init__()    

        # ------------------

        self.hidden_dim = hidden_dim

        # embedding layer
        if weight_matrix is not None:  # load pretrained embeddings
            vocab_size, embed_dim = weight_matrix.shape
            self.embed = nn.Embedding(vocab_size, embed_dim)
            self.embed.load_state_dict({'weight': torch.tensor(weight_matrix)})
            self.embed.weight.requires_grad = False
        else:
            self.embed = nn.Embedding(vocab_size, embed_dim)

        # recurrent neural network
        self.rnn = nn.RNN(embed_dim, hidden_dim, dropout=dropout, batch_first=True)  # (batch, seq, feature)

        # two-layer feed-forward neural network
        self.fnn = nn.Sequential(
            # first layer
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),  # BatchNorm, normally between linear and activation
            nn.ReLU(inplace=True),  # most of time, ReLU
            # second layer
            nn.Linear(hidden_dim, output_dim),  # the output dimension of the feedforward neural network is 1
            nn.Sigmoid()  # probabilities for binary classification
        )

        # ------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        out = torch.Tensor()

        # ------------------
        # Write your code here
        # The shape of x is [batch_size, length_of_sequence], batch_first = True
        # The shape of out is [batch_size, 1]

        # embed input character indices using embedding layer
        batch_size = x.size(0)
        x = self.embed(x)  # out: [batch_size, length_of_sequence, hidden_size]
        # feed resulting embedding tensors into an RNN
        hidden = torch.zeros(1, batch_size, self.hidden_dim)
        x, hidden = self.rnn(x, hidden)
        # use RNN’s output feature of the last timestamp as the input of a two-layer feed-forward network
        out = self.fnn(x[:, -1, :])  # input: [batch_size, hidden_size]

        # ------------------

        return out


class LSTMClsModel(nn.Module):
    def __init__(self,
                 weight_matrix=None,
                 vocab_size=1000,
                 embed_size=64,
                 lstm_size=32,  # hidden dim
                 dense_size=0,  # optional: more linear layers
                 output_size=1,
                 lstm_layers=1,  # number of layers
                 dropout=0.1):
        """
        Initialize the model
        """
        super().__init__()
        self.lstm_layers = lstm_layers
        self.lstm_size = lstm_size
        self.dense_size = dense_size

        # embedding layer
        if weight_matrix is not None:  # load pretrained embeddings
            vocab_size, embed_size = weight_matrix.shape
            self.embedding = nn.Embedding(vocab_size, embed_size)
            self.embedding.load_state_dict({'weight': torch.tensor(weight_matrix)})
            self.embedding.weight.requires_grad = False
        else:
            self.embedding = nn.Embedding(vocab_size, embed_size)

        # if lstm_layers >= 2, dropout occurs except last layer
        self.lstm = nn.LSTM(embed_size, lstm_size, lstm_layers, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)  # add dropout for last layer

        # linear layers
        if dense_size == 0:
            self.fc = nn.Linear(lstm_size, output_size)
        else:  # additional linear layer: dense layer
            self.fc1 = nn.Linear(lstm_size, dense_size)  # nn.Sequential cannot pass multiple inputs/sizes
            self.fc2 = nn.Linear(dense_size, output_size)

        self.sig = nn.Sigmoid()  # can do binary classification, output is probability [0,1]

    def init_hidden(self, batch_size):
        """
        Initialize the hidden state
        """
        weight = next(self.parameters()).data
        hidden = (weight.new(self.lstm_layers, batch_size, self.lstm_size).zero_(),
                  weight.new(self.lstm_layers, batch_size, self.lstm_size).zero_())

        return hidden

    def forward(self, nn_input_text):
        """
        Perform a forward pass of the model on nn_input
        """
        batch_size = nn_input_text.size(0)
        nn_input_text = nn_input_text.long()

        # 1. embedding layer
        embeds = self.embedding(nn_input_text)

        # 2. LSTM layer
        hidden_state = self.init_hidden(batch_size)  # initialize every epoch
        lstm_out, _ = self.lstm(embeds, hidden_state)
        # Stack up LSTM outputs, apply dropout
        lstm_out = lstm_out[:, -1, :]  # because batch_first=True, (batch, seq, feature)
        lstm_out = self.dropout(lstm_out)

        # 3. Linear + Dense layer
        if self.dense_size == 0:
            out = self.fc(lstm_out)
        else:
            dense_out = self.fc1(lstm_out)
            out = self.fc2(dense_out)  # (batch, 1)

        # 4. output
        # probabilities for binary classification
        probs = self.sig(out)

        return probs