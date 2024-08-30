import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from models import VerySmallCNN, SmallCNN
from tqdm import tqdm

def main(train_dataset, 
         test_dataset, 
         train_loader, 
         test_loader, 
         optimizer, 
         model, 
         criterion, 
         num_epochs, 
         device: str = 'cuda') -> None:
    
    """
    Train a model given a dataset, loader, optimizer, model, criterion,
    number of epochs, and device.

    Args:
        train_dataset (torch.utils.data.Dataset): The dataset to be used for
            training.
        test_dataset (torch.utils.data.Dataset): The dataset to be used for
            testing.
        train_loader (torch.utils.data.DataLoader): The loader for the training
            dataset.
        test_loader (torch.utils.data.DataLoader): The loader for the testing
            dataset.
        optimizer (torch.optim.Optimizer): The optimizer to be used for training.
        model (nn.Module): The model to be trained.
        criterion (nn.Module): The loss function to be used for training.
        num_epochs (int): The number of epochs to be trained.
        device (str, optional): The device to be used for training. Defaults to
            'cuda'.

    Returns:
        None
    """
    for epoch in range(num_epochs):
        train_loss, train_acc = 0, 0
        for inputs, labels in tqdm(train_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            output = model(inputs)
            loss = criterion(output, labels)
            _, pred = torch.max(output, 1)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            train_acc += torch.sum(pred == labels.data)
        
        epoch_loss = train_loss / len(train_dataset)
        epoch_acc = (train_acc.double() / len(train_dataset)) * 100
        print(f"Epoch {epoch+1}/{num_epochs} Loss: {epoch_loss:.3f} Acc: {epoch_acc:.2f}%")

    with torch.no_grad():
        run_loss, run_acc = 0, 0
        for inputs, labels in tqdm(test_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            output = model(inputs)
            loss = criterion(output, labels)
            _, pred = torch.max(output, 1)
            run_loss += loss.item() * inputs.size(0)
            run_acc += torch.sum(pred == labels.data)
    test_loss = run_loss / len(test_dataset)
    test_acc = (run_acc.double() / len(test_dataset)) * 100
    print(f"Test Loss: {test_loss:.3f}, Test Acc: {test_acc:.2f}%")

if __name__ == "__main__":
    # This is the main function that trains a model and tests it on the test set.
    # The model and the hyperparameters are defined in the models.py file.
    # The hyperparameters are:
    #   - num_epochs: the number of epochs to train.
    #   - device: the device to use for training and testing, either 'cuda' or 'cpu'.
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Defining our transforms and creating our dataset and dataloaders.
    transform = transforms.Compose([
        transforms.Resize((32, 32,)), 
        transforms.ToTensor(), 
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=2)
    
    # Initializing the model to predict 10 classes as well as our loss and optimizer functions. 
    vs_model = VerySmallCNN(n_classes=10).to(device)
    # s_model = SmallCNN(n_classes=10).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(vs_model.parameters(), lr=1e-3, momentum=0.9)
    
    # Running the main function. Replace vs_model with s_model if SmallCNN is used. 
    num_epochs: int = 25
    main(train_dataset=train_dataset, test_dataset=test_dataset, train_loader=train_loader, test_loader=test_loader, 
         optimizer=optimizer, model=vs_model, criterion=criterion, num_epochs=num_epochs, device=device)