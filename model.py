import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score
import numpy as np
import tqdm
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

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

    def forward(self, x, lengths, mask):
        """
        Forward pass of the RNNClassifier.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, input_size).
            lengths (torch.Tensor): Tensor of sequence lengths for each sequence in the batch.
            mask (torch.Tensor): Binary mask indicating valid elements in the padded sequences.

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_size).
        """
        packed_data = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        if self.model_arch.lower() == 'lstm':
            output, (hidden, _) = self.rnn(packed_data)
        else:
            output, _ = self.rnn(packed_data)
        output, _ = pad_packed_sequence(output, batch_first=True)
        out = output * mask.unsqueeze(2)
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
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.l2)

        # Training loop
        prevloss = 1000
        count = 0
        state = None
        optimstate = None
        bar = tqdm.tqdm(total=self.num_epochs)
        for epoch in range(self.num_epochs):
            total_loss = 0.0
            self.model.train()

            for padded_data, sequence_lengths, mask, batch_target in train_dataloader:

                optimizer.zero_grad()
                outputs = self.model(padded_data, sequence_lengths, mask)
                loss = criterion(outputs.squeeze(), batch_target)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            self.model.eval()
            test_loss = 0

            with torch.no_grad():
                for padded_data, sequence_lengths, mask, batch_target in valid_dataloader:

                    pred = self.model(padded_data, sequence_lengths, mask)
                    test_loss += criterion(pred.squeeze(), batch_target).item()

            avg_loss = total_loss / len(train_dataloader)
            test_avg_loss = test_loss / len(valid_dataloader)

            if test_avg_loss >= prevloss:
                count += 1
                if count >= 40:
                    self.model.load_state_dict(state)
                    optimizer.load_state_dict(optimstate)
                    break
            else:
                bar.set_description(f"Best valid loss: {test_avg_loss}")
                count = 0
                prevloss = test_avg_loss
                state = self.model.state_dict()
                optimstate = optimizer.state_dict()
            bar.update(1)

        self.model.load_state_dict(state)
        bar.close()

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
            for padded_data, sequence_lengths, mask, batch_target in test_dataloader:
                predictions = self.model(padded_data, sequence_lengths, mask)
                reals.append(batch_target.to('cuda'))
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
            dataloader = DataLoader(list(zip([torch.tensor(X[i]) for i in train], [y[i] for i in train])), batch_size=self.batch_size,  shuffle=True, collate_fn=custom_collate_fn)
            dataloader_true = DataLoader(list(zip([torch.tensor(X[i]) for i in test], [y[i] for i in test])), batch_size=self.batch_size,  shuffle=True, collate_fn=custom_collate_fn)
            self.train(dataloader, dataloader_true)
            preds, all_reals = self.predict(dataloader_true)
            score = f1_score(all_reals, preds)
            scores.append(score)

        scores = np.mean(scores)
        return scores

def reverse_padded_sequence(tensor, lengths):
    """
    Reverses the padded sequence along the time dimension.

    Args:
        tensor (torch.Tensor): Padded tensor of shape (batch_size, max_length, *).
        lengths (torch.Tensor): Tensor of sequence lengths for each sequence in the batch.

    Returns:
        torch.Tensor: Tensor with reversed padded sequence.
    """
    lengths = lengths.to(torch.int32)
    out = np.zeros([len(tensor), max([v.shape[0] for v in tensor]), tensor[1].shape[1]])
    tensor_t = [t.cpu().numpy() for t in tensor]
    for i in range(out.shape[0]):
        out[i, -lengths[i]:, :] = tensor_t[i]
    return torch.tensor(out, dtype=torch.float32).to('cuda')

# Create a DataLoader for batching with custom collate function
def custom_collate_fn(batch):
    inputs, targets = zip(*batch)
    length = max([len(xt) for xt in inputs])
    inputs = [x.to(torch.float32).to('cuda').transpose(0, 1) for x in inputs]
    # Sort inputs by sequence length in descending order
    sorted_inputs = sorted(zip(inputs, targets), key=lambda x: len(x[0]), reverse=True)
    inputs, targets = zip(*sorted_inputs)

    # Pad the sequences to the same length within a batch
    sequence_lengths = torch.tensor([len(seq) for seq in inputs], dtype=torch.float32)
    padded_data = reverse_padded_sequence(inputs, sequence_lengths)

    # Create a binary mask indicating valid elements
    mask = torch.arange(padded_data.size(1)).expand(len(sequence_lengths), padded_data.size(1)) < sequence_lengths.unsqueeze(1)

    return padded_data, sequence_lengths, mask.to('cuda'), torch.tensor(targets, dtype=torch.float32).to('cuda')