from stat_utils import get_class_accuracy, plot_confusion_matrix
from sklearn.metrics import confusion_matrix
import torchvision.models as models
import matplotlib.pyplot as plt
import pickle as pk
import torch
import sys
import os

def main():
    print(f"CUDA version: {torch.version.cuda}")

    if not torch.cuda.is_available():
        print("CUDA is not available. Please ensure you have a compatible GPU and CUDA installed.")
        sys.exit()

    cuda_version = torch.version.cuda
    if cuda_version != '12.4':
        print(f"Warning: Your CUDA version is {cuda_version}. This script is tested with CUDA 12.4.")

    print("\nLoading validation dataset...")

    try:
        with open('pickle/validate.pkl', 'rb') as handle:
            val_dataset = pk.load(handle)
            val_loader = pk.load(handle)
            batch_size = pk.load(handle)
            num_epochs = pk.load(handle)
            classes = pk.load(handle)
    except FileNotFoundError:
        print("Validation dataset not found. Please run train.py before running evaluation.")
        sys.exit()

    print("Loading fine-tuned model...")

    try:
        model_dict = torch.load("model/resnet50_food_classification_trained.pth")
    except FileNotFoundError:
        print("Fine-tuned model not found. Please run train.py before running evaluation.")
        sys.exit()

    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    num_classes = len(classes)
    num_fc_inputs = model.fc.in_features
    model.fc = torch.nn.Linear(num_fc_inputs, num_classes)
    model.load_state_dict(model_dict)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    criterion = torch.nn.CrossEntropyLoss()

    val_loss_history = []
    val_acc_history = []
    y_pred = []
    y_true = []

    print("Starting evaluation...")
    print("\n---------------------------------------------")
    print(f"Number of Epochs: {num_epochs}")
    print(f"Batch Size: {batch_size}")
    print("---------------------------------------------\n")

    for epoch in range(num_epochs):
        model.eval()
        val_loss = 0
        val_correct = 0
        batch_count = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                batch_count += 1
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                val_correct += torch.sum(preds == labels.data)

                y_pred.extend(preds.cpu().numpy())
                y_true.extend(labels.cpu().numpy())

                val_loss = val_loss / len(val_loader.dataset)
                val_acc = val_correct.double() / len(val_loader.dataset)

                val_loss_history.append(val_loss)
                val_acc_history.append(val_acc)
                print(f'Epoch {epoch + 1} - (Batch {batch_count}): Validation Loss: {val_loss:.4f}, Validation Acc: {val_acc:.4f}')

    if not os.path.exists("val_results"):
        os.mkdir("val_results")

    cm = confusion_matrix(y_true=y_true, y_pred=y_pred)
    class_acc = get_class_accuracy(y_true=y_true, y_pred=y_pred, num_classes=num_classes)

    print("\n\nEvaluation Complete!")
    print("\nValidation Summary")
    print("----------------------------------------------")
    print(f"Number of Epochs: {num_epochs}")
    print(f"Number of Batches per Epoch: {batch_count}")
    print(f"Batch Size: {batch_size}")
    print(f"\nTotal Validation Loss: {val_loss:.4f}")
    print(f"Total Validation Accuracy: {val_acc * 100:.2f}%")
    print("----------------------------------------------")
    print("\nClass-wise Accuracy")
    print("----------------------------------------------")
    for i in range(num_classes):
        print(f"\"{classes[i]}\" Accuracy: {class_acc[i]:.2f}%")
    print("----------------------------------------------")

    print("\nPlotting Validation data...")

    plot_confusion_matrix(cm, num_classes, classes)
    plt.savefig('val_results/confusion_matrix.png')

    with open('val_results/val_summary.txt', "w") as f:
        f.write("Validation Summary\n")
        f.write("----------------------------------------------\n")
        f.write(f"Number of Epochs: {num_epochs}\n")
        f.write(f"Number of Batches per Epoch: {batch_count}\n")
        f.write(f"Batch Size: {batch_size}\n")
        f.write(f"\nTotal Validation Loss: {val_loss:.4f}\n")
        f.write(f"Total Validation Accuracy: {val_acc * 100:.2f}%\n")
        f.write("----------------------------------------------\n")
        f.write("\nClass-wise Accuracy\n")
        f.write("----------------------------------------------\n")
        for i in range(num_classes):
            f.write(f"\"{classes[i]}\" Accuracy: {class_acc[i]:.2f}%\n")
        f.write("----------------------------------------------\n")
        f.write("\nValidation History\n")
        f.write("----------------------------------------------\n")
        for epoch in range(num_epochs):
            for batch in range(batch_count):
                f.write(f'Epoch {epoch + 1} - (Batch {batch + 1}): Validation Loss: {val_loss_history[(epoch * batch_count) + batch]:.4f}, Validation Acc: {val_acc_history[(epoch * batch_count) + batch]:.4f}\n')

    print("\nValidation data saved to \"val_results\" folder")
    print("\nPlease run test.py to test model.")

if __name__ == '__main__':
    main()
