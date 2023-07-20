import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import StratifiedKFold
import numpy as np

# Define the LSTM model
class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout_prob, output_size):
        super(LSTMClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout_prob)
        self.dropout = nn.Dropout(dropout_prob)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        out, _ = self.lstm(x, (h0, c0))
        out = self.dropout(out[:, -1, :])  # Apply dropout only to the last time step
        out = self.fc(out)
        return out

# Define a custom Dataset
class CustomDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], self.labels[index]

# Function to create batches using DataLoader
def create_batches(dataset, batch_size, shuffle=True):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

# Function for training the model
def train_model(model, train_loader, criterion, optimizer, device):
    model.train()
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

# Function for model evaluation (predict)
def predict_model(model, test_loader, device):
    model.eval()
    predictions = []
    with torch.no_grad():
        for inputs, _ in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            predictions.extend(predicted.cpu().numpy())

    return np.array(predictions)

# Function for cross-validation
def cross_validate(model, dataset, n_splits, batch_size, num_epochs, learning_rate, weight_decay, dropout_prob, device):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True)
    accuracies = []

    for train_idx, val_idx in skf.split(dataset.data, dataset.labels):
        train_data, train_labels = dataset.data[train_idx], dataset.labels[train_idx]
        val_data, val_labels = dataset.data[val_idx], dataset.labels[val_idx]

        train_dataset = CustomDataset(train_data, train_labels)
        val_dataset = CustomDataset(val_data, val_labels)

        train_loader = create_batches(train_dataset, batch_size)
        val_loader = create_batches(val_dataset, batch_size, shuffle=False)

        model.to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

        for epoch in range(num_epochs):
            train_model(model, train_loader, criterion, optimizer, device)

        predictions = predict_model(model, val_loader, device)
        accuracy = np.mean(predictions == val_labels)
        accuracies.append(accuracy)

    return accuracies

# Example usage
if __name__ == "__main__":
    # Assuming you have your data and labels in numpy arrays: data, labels
    input_size = data.shape[1]  # Adjust this according to your input data
    hidden_size = 64
    num_layers = 2
    dropout_prob = 0.2
    output_size = 2  # Binary classification (2 classes)

    model = LSTMClassifier(input_size, hidden_size, num_layers, dropout_prob, output_size)

    # Convert data and labels to PyTorch tensors
    data_tensor = torch.tensor(data, dtype=torch.float32)
    label_tensor = torch.tensor(labels, dtype=torch.long)

    dataset = CustomDataset(data_tensor, label_tensor)

    n_splits = 5
    batch_size = 64
    num_epochs = 10
    learning_rate = 0.001
    weight_decay = 0.001
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    accuracies = cross_validate(model, dataset, n_splits, batch_size, num_epochs, learning_rate, weight_decay, dropout_prob, device)
    mean_accuracy = np.mean(accuracies)
    print("Mean accuracy: {:.4f}".format(mean_accuracy))