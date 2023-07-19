import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score
import numpy as np

class RNNClassifier(nn.Module):
    """
    RNNClassifier is a class that defines the RNN-based classifier model.

    Args:
        input_size (int): Size of the input features.
        hidden_size (int): Size of the hidden state in the RNN.
        num_layers (int): Number of recurrent layers in the RNN.
        output_size (int): Size of the output.
        model_arch (str): Architecture of the RNN model ('lstm', 'gru', or 'rnn').
        dropout (float): Dropout rate for the RNN layers.

    Attributes:
        hidden_size (int): Size of the hidden state in the RNN.
        num_layers (int): Number of recurrent layers in the RNN.
        model_arch (str): Architecture of the RNN model ('lstm', 'gru', or 'rnn').
        rnn (nn.Module): RNN module (LSTM, GRU, or RNN).
        fc (nn.Linear): Fully connected layer for output.
        sigmoid (nn.Sigmoid): Sigmoid activation function.
    """

    def __init__(self, input_size, hidden_size, num_layers, output_size, model_arch, dropout):
        super(RNNClassifier, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.model_arch = model_arch

        # Choose the appropriate RNN module based on the model_arch parameter
        if self.model_arch.lower() == 'lstm':
            self.rnn = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        elif self.model_arch.lower() == 'gru':
            self.rnn = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        elif self.model_arch.lower() == 'rnn':
            self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        else:
            raise ValueError("Select one of lstm, gru, rnn")

        self.fc = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Forward pass of the RNNClassifier.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, input_size).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_size).
        """
        if self.model_arch.lower() == 'lstm':
            h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
            c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
            out, _ = self.rnn(x, (h0, c0))
        else:
            h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
            out, _ = self.rnn(x, h0)

        out = self.fc(out[:, -1, :])
        out = self.sigmoid(out)

        return out


class RNNmodel():
    """
    RNNmodel is a class that represents the RNN-based classifier model.

    Args:
        input_size (int): Size of the input features.
        hidden_size (int): Size of the hidden state in the RNN.
        num_layers (int): Number of recurrent layers in the RNN.
        output_size (int): Size of the output.
        model_arch (str): Architecture of the RNN model ('lstm', 'gru', or 'rnn').
        num_epochs (int): Number of training epochs.
        batch_size (int): Batch size for training.
        dropout (float): Dropout rate for the RNN layers.
        l2 (float): L2 regularization coefficient.
        lr (float): Learning rate.

    Attributes:
        input_size (int): Size of the input features.
        hidden_size (int): Size of the hidden state in the RNN.
        num_layers (int): Number of recurrent layers in the RNN.
        output_size (int): Size of the output.
        model_arch (str): Architecture of the RNN model ('lstm', 'gru', or 'rnn').
        num_epochs (int): Number of training epochs.
        batch_size (int): Batch size for training.
        dropout (float): Dropout rate for the RNN layers.
        l2 (float): L2 regularization coefficient.
        lr (float): Learning rate.
        model (RNNClassifier): Instance of the RNNClassifier model.
    """

    def __init__(self, input_size, hidden_size, num_layers, output_size, model_arch, num_epochs, batch_size, dropout, l2, lr):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.model_arch = model_arch
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.dropout = dropout
        self.l2 = l2
        self.lr = lr
        self.model = None

    def train(self, train_dataloader, valid_dataloader):
        """
        Trains the RNN model using the provided training and validation data.

        Args:
            train_dataloader (DataLoader): DataLoader for training data.
            valid_dataloader (DataLoader): DataLoader for validation data.
        """
        self.model = RNNClassifier(self.input_size, self.hidden_size, self.num_layers, self.output_size, self.model_arch, self.dropout).to('cuda')

        # Define the loss function and optimizer
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01, weight_decay=self.l2)

        # Training loop
        prevloss = 1000
        first = True
        count = 0
        state = None
        optimstate = None

        for epoch in range(self.num_epochs):
            total_loss = 0.0
            self.model.train()

            for batch in train_dataloader:
                inputs, targets = batch
                inputs, targets = inputs.to('cuda'), targets.to('cuda')

                optimizer.zero_grad()
                outputs = self.model(inputs)
                targets = targets.view(-1, self.output_size)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                lt = loss.item()
                total_loss += loss.item()

            self.model.eval()
            test_loss = 0

            with torch.no_grad():
                for validbatch in valid_dataloader:
                    validinputs, validtargets = validbatch
                    validinputs, validtargets = validinputs.to('cuda'), validtargets.to('cuda')
                    pred = self.model(validinputs)
                    test_loss += criterion(pred, validtargets).item()

            avg_loss = total_loss / len(train_dataloader)
            test_avg_loss = test_loss / len(valid_dataloader)

            if avg_loss >= prevloss:
                if not first:
                    state = self.model.state_dict()
                    optimstate = optimizer.state_dict()
                first = True
                count += 1
                if count >= 20:
                    self.model.load_state_dict(state)
                    optimizer.load_state_dict(optimstate)
                    break
            else:
                first = False
                count = 0
                prevloss = test_avg_loss

            print(f"Epoch [{epoch+1}/{self.num_epochs}], Train Loss: {avg_loss:.4f}, Valid Loss: {test_avg_loss:.4f}")

        self.model.load_state_dict(state)

    def predict(self, test_dataloader):
        """
        Generates predictions for the test data.

        Args:
            test_dataloader (DataLoader): DataLoader for test data.

        Returns:
            tuple: A tuple containing the predicted labels and the true labels.
        """
        self.model.eval()

        with torch.no_grad():
            preds = []
            reals = []
            for batch in test_dataloader:
                predictions = self.model(batch[0].to('cuda'))
                reals.append(batch[1].to('cuda').flatten(end_dim=1))
                preds.append(predictions.flatten(end_dim=1))

        all_predictions = torch.round(torch.cat(preds, dim=0))
        all_predictions = all_predictions.detach().to('cpu').numpy()
        all_reals = torch.cat(reals, dim=0)
        all_reals = all_reals.detach().to('cpu').numpy()

        return all_predictions, all_reals

    def cross_validate(self, X, y):
        """
        Performs cross-validation on the given data.

        Args:
            X (numpy.ndarray): Input features.
            y (numpy.ndarray): Target labels.

        Returns:
            float: Mean F1 score across all cross-validation folds.
        """
        scores = []
        cv = KFold(5)
        for train, test in cv.split(X, y):
            dataloader = DataLoader(list(zip(X[train], y[train])), batch_size=self.batch_size, pin_memory=True, pin_memory_device='cuda', shuffle=True)
            dataloader_true = DataLoader(list(zip(X[test], y[test])), batch_size=self.batch_size, pin_memory=True, pin_memory_device='cuda', shuffle=True)
            self.train(dataloader, dataloader_true)
            preds, all_reals = self.predict(dataloader_true)
            score = f1_score(all_reals, preds)
            scores.append(score)

        scores = np.mean(scores)
        return scores