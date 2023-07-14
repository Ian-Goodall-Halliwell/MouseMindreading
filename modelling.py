# Imports
import torch
import numpy as np
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from alsoinspectdata import preprocess
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


import pickle as pkl


testrundata = preprocess(10)
feats = testrundata['features']
labels = testrundata['labels']
mins = testrundata['cutoffs']

region='MOs'
X = feats[region]
y = labels[region]

#subsampling
for en,sample in enumerate(X):
    
    indexsample = np.random.choice(range(sample.shape[0]),size=mins[region])
    X[en] = sample[indexsample]



from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()
encoder.fit(y)
y = encoder.transform(y)
print(np.mean(y))

X_train, X_test, y_train,  y_test = train_test_split(X,y,test_size=.20,stratify=y)


X = torch.tensor(X_train,dtype=torch.float32)
y = torch.tensor(y_train,dtype=torch.float32).reshape(-1,1)
X_test = torch.tensor(X_test,dtype=torch.float32)
y_test = torch.tensor(y_test,dtype=torch.float32).reshape(-1,1)
import torch
import torch.nn as nn

# Define the RNN model
class RNNClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(RNNClassifier, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        out, _ = self.rnn(x, h0)
        out = self.fc(out[:, -1, :])
        out = self.sigmoid(out)
        
        return out

# Set the hyperparameters
input_size = 39  # Dimensionality of the input features
hidden_size = 64  # Number of units in the hidden layer
num_layers = 2  # Number of recurrent layers
output_size = 1  # Number of classes (binary)

# Create an instance of the RNN classifier
model = RNNClassifier(input_size, hidden_size, num_layers, output_size)

# Define the loss function and optimizer
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Training loop
num_epochs = 500
for epoch in range(num_epochs):
    # Forward pass
    outputs = model(X)  # inputs is your training data
    
    # Compute the loss
    loss = criterion(outputs, y)  # labels is your target
    
    # Backward and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # Print the loss for every epoch
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}")

# To make predictions, use the trained model like this:
test_outputs = model(X_test)  # test_inputs is your test data
predicted_labels = torch.round(test_outputs)  # Round the outputs to obtain binary predictions
predicted_labels = predicted_labels.detach().numpy()
true_labels = y_test.detach().numpy()
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
print(f"f1 score: {f1_score(true_labels,predicted_labels)}")
plotdata = confusion_matrix(true_labels,predicted_labels)
plot = ConfusionMatrixDisplay(plotdata)
plot.plot()
plt.show()