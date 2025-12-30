from pathlib import Path
from pyexpat import model
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

#Define paths
DATASET_DIR = Path("src/data/dataset")
MODEL_DIR = Path("src/models")
MODEL_DIR.mkdir(exist_ok=True)

# The training settings 
BATCH_SIZE = 16
NUM_EPOCHS = 10
LEARNING_RATE = 0.0005
IMG_SIZE = (224)

# Function to get data loaders
def get_dataloaders():

    # ImageNet normalization is standard for pretrained MobileNetV2
    norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    # Preprocessing transformations for training and validation datasets
    train_transforms = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),  # This resizes the image to 224x224 pixels
      ##  transforms.RandomHorizontalFlip(p=0.2), # This augments the data by flipping images horizontally with a probability of 20%
        transforms.ToTensor(),  # this converts the image to a PyTorch tensor
        norm
    ])

# Preprocessing transformations for validation datasets
    val_transforms = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        norm
    ])

    # Loads datasets from the respective directories
    train_dataset = datasets.ImageFolder(DATASET_DIR / "train", transform=train_transforms)
    val_dataset = datasets.ImageFolder(DATASET_DIR / "val", transform=val_transforms)

# Creates data loaders for batching and shuffling
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    return train_dataset, train_loader, val_loader

# Function to build the model
# @param num_classes: number of output classes
# @param device: device to run the model on CPU or GPU
def build_model(num_classes: int, device):
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)  # Load pre-trained MobileNetV2 model

    in_features = model.classifier[1].in_features           # Get the number of input features to the classifier layer
    model.classifier[1] = nn.Linear(in_features, num_classes)   # Replaces the classifier layer to match the number of classes, nn is a neural network module in PyTorch

# Move the model to the specified device CPU or GPU
    return model.to(device) 


# function to evaluate the model
@torch.no_grad()
def evaluate_model(model, loader, device):
    model.eval()  # Sets the model to evaluation mode
    correct = 0  # counter for correct predictions
    total = 0   # counter for total predictions
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)  # Move data to the specified device
        outputs = model(images)     # Gets the model predictions
        predicts = outputs.argmax(dim=1)  # Gets the predicted class
        correct += (predicts == labels).sum().item()  # Counts correct predictions
        total += labels.size(0)  # Total number of samples
    
    return correct / max(total, 1)  # Returns accuracy

# Functio to train the model
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Use GPU if available, else CPU
    print(f"Using device:", device) # Info for device being used

    train_dataset, train_loader, val_loader = get_dataloaders()  # Get data loaders
    class_names = train_dataset.classes  # Get class names from the training dataset
    num_classes = len(class_names)  # Number of classes

    print("Classes:", class_names)  # Print class names
    print("Number of classes:", num_classes)  # Print number of classes

    model = build_model(num_classes, device)  # Build the model

    criterion = nn.CrossEntropyLoss()  # Loss function for multi-class classification
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)  # Adam optimizer, tool from torch.optim

    best_val_accuracy = 0.0  # Variable to track the best validation accuracy
    best_path = MODEL_DIR / "mobilenetv2_best.pth"  # Path to save the best model

    for epoch in range(NUM_EPOCHS):
        model.train()  # Sets the model to training mode
        running_loss = 0.0  # Variable to track running loss

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)  # Move data to the specified device

            optimizer.zero_grad()  # Zero the gradients, this means clearing old gradients from the last step
            outputs = model(images)  # Forward pass, this computes the model output
            loss = criterion(outputs, labels)  # Compute loss
            loss.backward()  # This computes the gradient of the loss with respect to model parameters
            optimizer.step()  # Update weights

            running_loss += loss.item()  # Update running loss

        avg_loss = running_loss / max(len(train_loader), 1)  # Calculate average loss
        val_accuracy = evaluate_model(model, val_loader, device)  # Evaluate on validation set

    # Save the best model based on validation accuracy, 4f formats the float to 4 decimal places, 3f to 3 decimal places
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} | loss = {avg_loss:.4f} | val_accuracy = {val_accuracy:.3f}")

        # Save the best model if val_accuracy > best_val_accuracy:
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save({
                "model_state_dict": model.state_dict(),
                "classes": class_names
            }, best_path)   # Save the model state and class names

            # Print info about the best model saved
    print(f"Best model saved with val_accuracy to {best_path} (val_accuracy = {best_val_accuracy:.3f})")

    # Finally, validate the evaluation using the best model
    checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])  # Load the best model state
    final_acc = evaluate_model(model, val_loader, device)  # Evaluate on validation set

    print("\n Final results")
    print("Best validation accuracy:", best_val_accuracy)
    print("Val accuracy:", final_acc)
    print("Saved model:", best_path)

if __name__ == "__main__":
    train()

