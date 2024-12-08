from stat_utils import get_class_accuracy, plot_confusion_matrix
from sklearn.metrics import confusion_matrix
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
import torchvision.models as models
import matplotlib.pyplot as plt
import pickle as pk
import torch
import torch.cuda.amp as amp
import sys
import os

# Check CUDA version
print(f"CUDA version: {torch.version.cuda}")

# Ensure CUDA is available and compatible
if not torch.cuda.is_available():
    print("CUDA is not available. Please ensure you have a compatible GPU and CUDA installed.")
    sys.exit()

cuda_version = torch.version.cuda
if cuda_version != '12.4':
    print(f"Warning: Your CUDA version is {cuda_version}. This script is tested with CUDA 12.4.")

learn_rate = 0.001
batch_size = 128
num_epochs = 5
num_workers = 4  # Increase the number of workers

print("\nLoading dataset...")

data_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224, 224), antialias=True),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # normalize images
])

try:
    dataset = ImageFolder(root='dataset', transform=data_transforms)
except FileNotFoundError:
    print("Dataset not found.")
    sys.exit()

classes = dataset.classes

# Split original dataset into train (70%), validation (10%), and test (20%)
train_size = int(0.7 * len(dataset))
val_size = int(0.1 * len(dataset))
test_size = len(dataset) - train_size - val_size
train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])

# Create data loaders for training, validation, and test sets
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

if __name__ == '__main__':
    if not os.path.exists("pickle"):
        os.mkdir("pickle")

    # Save validation set, loader, batch size, and number of epochs for later use in validate.py
    with open('pickle/validate.pkl', 'wb') as handle:
        pk.dump(val_dataset, handle)
        pk.dump(val_loader, handle)
        pk.dump(batch_size, handle)
        pk.dump(num_epochs, handle)
        pk.dump(classes, handle)

    # Save test set, loader, batch size, and number of epochs for later use in test.py
    with open('pickle/test.pkl', 'wb') as handle:
        pk.dump(test_dataset, handle)
        pk.dump(test_loader, handle)
        pk.dump(batch_size, handle)
        pk.dump(num_epochs, handle)
        pk.dump(classes, handle)

    print("Loading pre-trained model...")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    resnet50 = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2).to(device)

    # Freeze all pre-trained layers
    for param in resnet50.parameters():
        param.requires_grad = False

    # Replace the final layer
    num_fc_inputs = resnet50.fc.in_features
    num_classes = len(dataset.classes)
    resnet50.fc = torch.nn.Linear(num_fc_inputs, num_classes).to(device)

    # Create optimizer and loss function
    optimizer = torch.optim.SGD(resnet50.fc.parameters(), lr=learn_rate, momentum=0.9)
    criterion = torch.nn.CrossEntropyLoss()

    scaler = amp.GradScaler()

    train_loss_history = []
    train_acc_epoch_history = []
    train_acc_batch_history = []
    y_pred = []
    y_true = []
    print("Starting Training...")
    print("\n---------------------------------------------")
    print(f"Number of Epochs: {num_epochs}")
    print(f"Batch Size: {batch_size}")
    print(f"Learning rate: {learn_rate:.3f}")
    print("---------------------------------------------\n")

    # Training loop
    for epoch in range(num_epochs):
        resnet50.train()
        train_loss = 0
        train_correct = 0
        batch_count = 0

        for inputs, labels in train_loader:
            batch_count += 1

            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            with amp.autocast():
                outputs = resnet50(inputs)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            train_correct += torch.sum(preds == labels.data)

            y_pred += preds.cpu().numpy().tolist()
            y_true += labels.data.cpu().numpy().tolist()

            batch_acc = torch.sum(preds == labels.data).double() / inputs.size(0)
            batch_loss = loss.item()

            train_acc_batch_history.append(batch_acc)
            train_loss_history.append(batch_loss)

            print(f'Epoch {epoch + 1} - (Batch {batch_count}): Training Loss: {batch_loss:.4f}, '
                  f'Training Acc: {batch_acc:.4f}')

        train_acc = train_correct.double() / len(train_loader.dataset)
        train_loss = train_loss / len(train_loader.dataset)

        train_loss_history.append(train_loss)
        train_acc_epoch_history.append(train_acc)

        print(f'Epoch {epoch + 1} completed. Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}')

    cm = confusion_matrix(y_true=y_true, y_pred=y_pred)
    class_acc = get_class_accuracy(y_true=y_true, y_pred=y_pred, num_classes=num_classes)

    print("\n\nTraining Complete!")
    print("----------------------------------------------")
    print(f"Number of Epochs: {num_epochs}")
    print(f"Number of Batches per Epoch: {batch_count}")
    print(f"Batch Size: {batch_size}")
    print(f"Learning rate: {learn_rate:.3f}")

    print(f"\nTraining Loss: {train_loss:.4f}")
    print(f"Training Accuracy: {train_acc * 100:.2f}%")
    print("----------------------------------------------")
    print("\nClass-wise Accuracy")
    print("----------------------------------------------")
    for i in range(num_classes):
        print(f"\"{classes[i]}\" Accuracy: {class_acc[i]:.2f}%")
    print("----------------------------------------------")

    if not os.path.exists("train_results"):
        os.mkdir("train_results")

    print("\nPlotting Training data...")

    plt.plot(train_loss_history)
    plt.title('Training Loss History')
    plt.xlabel('Batches')
    plt.ylabel('Training loss')
    plt.savefig("train_results/train_loss.png")
    plt.clf()

    plt.plot([acc.cpu().numpy() for acc in train_acc_epoch_history])
    plt.title('Training Accuracy History')
    plt.xlabel('Epochs')
    plt.ylabel('Training Accuracy')
    plt.savefig("train_results/train_acc.png")

    plot_confusion_matrix(cm, num_classes, classes)
    plt.savefig('train_results/confusion_matrix.png')

    with open('train_results/training_summary.txt', "w") as f:
        f.write("Training Summary\n")
        f.write("----------------------------------------------\n")
        f.write(f"Number of Epochs: {num_epochs}\n")
        f.write(f"Number of Batches per Epoch: {batch_count}\n")
        f.write(f"Batch Size: {batch_size}\n")
        f.write(f"\nTotal Training Loss: {train_loss:.4f}\n")
        f.write(f"Total Training Accuracy: {train_acc * 100:.2f}%\n")
        f.write("----------------------------------------------\n")
        f.write("\nClass-wise Accuracy\n")
        f.write("----------------------------------------------\n")
        for i in range(num_classes):
            f.write(f"\"{classes[i]}\" Accuracy: {class_acc[i]:.2f}%\n")
        f.write("----------------------------------------------\n")
        f.write("\nTraining History\n")
        f.write("----------------------------------------------\n")
        for epoch in range(num_epochs):
            for batch in range(batch_count):
                f.write(f'Epoch {epoch + 1} - (Batch {batch + 1}): Training Loss: {train_loss_history[(epoch * batch_count) + batch]:.4f}, '
                        f'Training Acc: {train_acc_batch_history[(epoch * batch_count) + batch]:.4f}\n')

    print("Training data saved to \"train_results\" folder")
    print("Saving fine-tuned model...")

    if not os.path.exists("model"):
        os.mkdir("model")

    torch.save(resnet50.state_dict(), "model/resnet50_food_classification_trained.pth")

    print("\nPlease run validate.py to validate the newly trained model.")
