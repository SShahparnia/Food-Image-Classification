import matplotlib.pyplot as plt
import numpy as np

def get_class_accuracy(y_true, y_pred, num_classes):
    class_acc = np.zeros(num_classes)
    for i in range(num_classes):
        correct_predictions = sum((y_true[j] == i and y_pred[j] == i) for j in range(len(y_true)))
        total_class_samples = y_true.count(i)
        class_acc[i] = (correct_predictions / total_class_samples) * 100 if total_class_samples > 0 else 0
    return class_acc

def plot_confusion_matrix(cm, num_classes, classes):
    fig, ax = plt.subplots(figsize=(num_classes, num_classes))
    cax = ax.matshow(cm, cmap=plt.cm.Blues)
    fig.colorbar(cax)
    ax.set_title('Confusion Matrix')
    ax.set_xticks(np.arange(num_classes))
    ax.set_yticks(np.arange(num_classes))
    ax.set_xticklabels(classes, rotation=45, ha='right')
    ax.set_yticklabels(classes)
    ax.set_ylabel('True Label')
    ax.set_xlabel('Predicted Label')

    for i in range(num_classes):
        for j in range(num_classes):
            percentage = cm[i, j] / cm[i].sum() * 100
            color = 'white' if percentage > 50 else 'black'
            text = f"{percentage:.1f}%" if percentage > 0 else "-"
            ax.text(j, i, text, ha='center', va='center', color=color)
