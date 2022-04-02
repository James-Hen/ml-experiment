# Variables
DATASET_PATH = '../data/'
RESULT_PATH = './result_full2/'
DEBUG = False
GPU = True

# Define the epochs
epochs = 500

# Import the necessary libraries
import numpy as np
import torch
import matplotlib.pyplot as plt
import json
import time

# Device settings
device = torch.device('cuda:0' if torch.cuda.is_available() and GPU else 'cpu')

# Loading the Fashion-MNIST dataset
from torchvision import datasets, transforms

# Define a transform to normalize the data
transform = transforms.Compose([transforms.ToTensor(),
                                # transforms.Normalize((0.5), (0.5))
                               ])

# Download and load the training data
trainset = datasets.FashionMNIST(DATASET_PATH, download = True, train = True, transform = transform)
testset = datasets.FashionMNIST(DATASET_PATH, download = True, train = False, transform = transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size = 8192, shuffle = True)
testloader = torch.utils.data.DataLoader(testset, batch_size = 8192, shuffle = True)

# Examine a sample if DEBUG is set
if (DEBUG):
  print("CUDA:", torch.cuda.is_available())
  dataiter = iter(trainloader)
  images, labels = dataiter.next()
  print("Checking images.")
  print("Type", type(images))
  print("Shape", images.shape)
  print("Label Shape", labels.shape)
  plt.cla()
  plt.clf()
  plt.imshow(images[1].numpy().squeeze(), cmap = 'Greys_r')
  plt.savefig(RESULT_PATH + 'sample.png')
  print("One sample image saved as `sample.png`.")

# Define the network architecture
from torch import nn, optim
import torch.nn.functional as F

model = nn.Sequential(nn.Linear(784, 256),
                      nn.ReLU(),
                      nn.Linear(256, 128),
                      nn.ReLU(),
                      nn.Linear(128, 64),
                      nn.ReLU(),
                      nn.Linear(64, 10),
                     ).to(device)

# Define the loss
criterion = nn.CrossEntropyLoss().to(device)

def to_one_hot(labels, device):
  onehot = torch.zeros(labels.shape[0], 10).to(device)
  return onehot.scatter(dim=1, index=labels.view(-1, 1).to(device), value=1.)

# Define the optimizer
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Initiate the timer to instrument the performance
timer_start = time.process_time_ns()
epoch_times = [timer_start]

train_losses, test_losses, accuracies = [], [], []

for e in range(epochs):
  running_loss = 0
  for images, labels in trainloader:
    # Flatten Fashion-MNIST images into a 784 long vector
    images = images.view(images.shape[0], -1).to(device)
    labels = to_one_hot(labels, device)
    
    # Training pass
    optimizer.zero_grad()
    
    output = model.forward(images).to(device)
    loss = criterion(output, labels).to(device)
    loss.backward()
    optimizer.step()
    
    running_loss += loss.item()
  else:
    test_loss = 0
    accuracy = 0
    
    # Turn off gradients for validation, saves memory and computation
    with torch.no_grad():
      # Set the model to evaluation mode
      model.eval()
      
      # Validation pass
      for images, labels in testloader:
        images = images.view(images.shape[0], -1).to(device)
        labels = labels.to(device)
        labels_one_hot = to_one_hot(labels, device)
        log_ps = model(images).to(device)
        test_loss += criterion(log_ps, labels_one_hot)
        
        ps = torch.exp(log_ps).to(device)
        top_p, top_class = ps.topk(1, dim = 1)
        equals = (top_class == labels.view(*top_class.shape))
        accuracy += torch.mean(equals.type(torch.FloatTensor))
    
    model.train()
    train_losses.append(running_loss/len(trainloader))
    test_losses.append(float(test_loss.cpu())/len(testloader))
    accuracies.append(float(accuracy)/len(testloader))
    
    epoch_times.append(time.process_time_ns())
    print("Epoch: {:03d}/{:03d}..".format(e+1, epochs),
          "Training loss: {:.3f}..".format(running_loss/len(trainloader)),
          "Test loss: {:.3f}..".format(test_loss/len(testloader)),
          "Test Accuracy: {:.3f}".format(accuracy/len(testloader)),
          "Cur time(ns): {}".format(epoch_times[-1]))

fig, ax = plt.subplots(figsize=(10, 8))
ax.plot(train_losses, label="Training loss")
ax.plot(test_losses, label="Validation loss")
ax.set_xlabel("Epoch")
ax.set_ylabel("MSE Loss")
ax.legend()
ax2 = ax.twinx()
ax2.plot(np.array(accuracies)*100, label="Accuracy", color='g')
ax2.set_ylabel("Percent")
plt.title("Pytorch implementation of 4-layer dense network")
plt.savefig(RESULT_PATH + 'training_proc.png', dpi = 100)

with open(RESULT_PATH + 'torch_results.json', 'w+') as f:
  json.dump({
      'train_losses': train_losses,
      'test_losses': test_losses,
      'epoch_times': epoch_times,
      'accuracies': accuracies,
    }, f)
  
# eval

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from torchvision import datasets
from sklearn.metrics import precision_recall_fscore_support

y_test, pred = [], []

for images, labels in testloader:
  images = images.view(images.shape[0], -1).to(device)
  labels = labels.to(device)
  labels_one_hot = to_one_hot(labels, device)
  log_ps = model(images).to(device)
  ps = torch.exp(log_ps).to(device)
  top_p, top_class = ps.topk(1, dim = 1)
  pred += list(top_class.cpu().numpy().flatten())
  y_test += list(labels.view(*top_class.shape).cpu().numpy().flatten())

precision, recall, fscore, _ = precision_recall_fscore_support(y_test, pred)
classes = [
        'T-shirt/top',
        'Trouser',
        'Pullover',
        'Dress',
        'Coat',
        'Sandal',
        'Shirt',
        'Sneaker',
        'Bag',
        'Ankle boot',
    ]
print("Precision \n", precision)
print("\nRecall \n", recall)
print("\nF-score \n", fscore)

print('\nMeasures', end=' ')
for c in classes:
    print('&', c, end=' ')
print('\\\\\nPrecision', end=' ')
for p in precision:
    print('& {:.3f}'.format(p), end=' ')
print('\\\\\nRecall', end=' ')
for r in recall:
    print('& {:.3f}'.format(r), end=' ')
print('\\\\\nF-score', end=' ')
for f in fscore:
    print('& {:.3f}'.format(f), end=' ')
print('\\\\')