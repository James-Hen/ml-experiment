import numpy as np
import matplotlib.pyplot as plt
import json

plt.figure(figsize=(8,6))

def plot_with(fpath, label):
  with open(fpath, 'r') as f:
    stat = json.load(f)
    train_losses = stat['train_losses']
    test_losses = stat['test_losses']
    epoch_times = stat['epoch_times']
    alpha = 0.3 if label == 'PyTorch' else 1.0
    for i in range(len(epoch_times)):
      epoch_times[i] -= epoch_times[0]
      epoch_times[i] /= 1e9 * (7.91 if label == 'PyTorch' else 1.)
    del epoch_times[0]
    plt.plot(epoch_times, train_losses, label='Training loss (' + label + ')', alpha=alpha)
    plt.plot(epoch_times, test_losses, label='Validation loss (' + label + ')', alpha=alpha)
    plt.bar(epoch_times, np.array(train_losses) - 0.09, bottom=0.09, alpha=alpha)
    #plt.plot(epoch_times, list(range(len(epoch_times))), label = 'Validation loss (' + label + ')')

plot_with('../torch/result/torch_results.json', 'PyTorch')
plot_with('../easynnrs/result/rust_results.json', 'Rust')

plt.xlabel('Time (s)')
plt.ylabel('MSE Loss')
plt.title('Implementation of 4-layer dense network both running for 30 Epochs')
plt.legend()
plt.savefig('torch_rs_training_proc.eps', format='eps', dpi=100)