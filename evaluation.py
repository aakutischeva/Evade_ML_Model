import torch
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate(model, data_loader):
    model.eval()
    y_pred, y_true = [], []
    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            outputs = model(X_batch)
            _, predicted = torch.max(outputs, 1)
            y_pred.extend(predicted.cpu().numpy())
            y_true.extend(y_batch.cpu().numpy())
    return accuracy_score(y_true, y_pred)

def evaluate_attack(model, X_adv, y_true_tensor):
    adv_loader = DataLoader(TensorDataset(X_adv, y_true_tensor), batch_size=32, shuffle=False)
    return evaluate(model, adv_loader)

def evaluate_f1(model, data_loader):
    model.eval()
    y_pred, y_true = [], []
    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            outputs = model(X_batch)
            _, predicted = torch.max(outputs, 1)
            y_pred.extend(predicted.cpu().numpy())
            y_true.extend(y_batch.cpu().numpy())
    return f1_score(y_true, y_pred, average='weighted')

def plot_confusion_matrix(model, X, y, title, save_path=None):
    model.eval()
    y_pred = []
    with torch.no_grad():
        outputs = model(X)
        _, predicted = torch.max(outputs, 1)
        y_pred.extend(predicted.cpu().numpy())
    cm = confusion_matrix(y, y_pred)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.show()