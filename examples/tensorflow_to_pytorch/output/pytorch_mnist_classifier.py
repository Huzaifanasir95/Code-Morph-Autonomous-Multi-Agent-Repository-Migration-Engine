"""
PyTorch Image Classifier

This is an example of modern code using PyTorch APIs.
This code demonstrates common patterns for image classification.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms


def create_model(input_shape, num_classes):
    """
    Create a simple neural network model using PyTorch APIs

    Args:
        input_shape: Shape of input data
        num_classes: Number of output classes

    Returns:
        Model
    """
    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.fc1 = nn.Linear(input_shape, 256)
            self.fc2 = nn.Linear(256, 128)
            self.fc3 = nn.Linear(128, num_classes)

        def forward(self, x):
            x = torch.relu(self.fc1(x))
            x = torch.relu(self.fc2(x))
            x = self.fc3(x)
            return x

    return Net()


def train_model(data_dir="./data"):
    """
    Train the model using PyTorch APIs

    Args:
        data_dir: Directory containing training data
    """
    # Load data
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(data_dir, train=False, transform=transform)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=False)

    # Model parameters
    input_shape = 784
    num_classes = 10
    learning_rate = 0.001
    num_epochs = 10

    # Create model
    model = create_model(input_shape, num_classes)

    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data = data.view(-1, input_shape)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            if batch_idx % 100 == 0:
                print(f"Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item():.4f}")

        # Evaluate on test set
        model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data = data.view(-1, input_shape)
                output = model(data)
                test_loss += criterion(output, target).item()
                _, predicted = torch.max(output, 1)
                correct += (predicted == target).sum().item()

        accuracy = correct / len(test_loader.dataset)
        print(f"\nTest Loss: {test_loss / len(test_loader):.4f}, Test Accuracy: {accuracy:.4f}")

        # Save model
        torch.save(model.state_dict(), "./models/mnist_model.pth")


def evaluate_model(model_path, test_data):
    """
    Evaluate saved model

    Args:
        model_path: Path to saved model checkpoint
        test_data: Test dataset
    """
    # Load model
    model = create_model(784, 10)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Run prediction
    with torch.no_grad():
        output = model(test_data)

    return output


if __name__ == "__main__":
    # Train the model
    train_model()

    print("\nTraining complete!")