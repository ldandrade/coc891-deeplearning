import matplotlib.pyplot as plt
import pandas as pd

def plot_train_loss(config, moving_average=4, figsize=(10, 6)):
    path = f"{config.logs_dir}/exp_{config.num_model}/train.csv"
    loss = pd.read_csv(path, sep=',', names=['iteration','training loss'], header=None)
    if moving_average:
        loss['moving_average'] = loss['training loss'].rolling(moving_average).mean()
    
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(loss['iteration'].values, loss['training loss'].values, color='lightsteelblue')
    if moving_average:
        ax.plot(loss['iteration'].values, loss['moving_average'].values, color='royalblue')
    ax.set_title('Training Loss')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Loss')
    ax.set_ylim([-0.5, loss['training loss'].max() + 0.5])
    plt.show()

    
def plot_val_accuracy(config, figsize=(10, 6)):
    path = f"{config.logs_dir}/exp_{config.num_model}/valid.csv"
    loss = pd.read_csv(path, sep=',', names=['epoch','accuracy'], header=None)
    
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(loss['epoch'].values, loss['accuracy'].values)
    ax.set_title('Accuracy over epochs')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Accuracy')
    ax.set_ylim(0, 1)
    plt.show()