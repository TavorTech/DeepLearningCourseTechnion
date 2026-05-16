import torch
import torch.nn as nn
import torch.optim as optim
from losses import nt_xent_loss,info_nce_loss_from_views
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.models import resnet18
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim import Adam

def evaluate_classifier_self(classifier, encoder, test_loader, device):
    encoder.eval()
    classifier.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)

            # Pass images through the encoder to get latent representations
            latent = encoder(images)

            # Pass latent representations through the classifier
            outputs = classifier(latent)

            # Calculate predictions
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'Accuracy of the classifier on the test images: {accuracy:.2f}%')
    return accuracy

def evaluate_classifier(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy of the classifier on the test images: {100 * correct / total:.2f}%')
    
def evaluate_simclr(model, classifier, test_loader, device):
    """
    Evaluate the encoder and classifier on the test set.
    """
    model.encoder.eval()
    classifier.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)

            # Extract features using the encoder and project to 128 dimensions
            features = model.encoder(images)
            latent_features = model.latent_projection(features)  # Use the latent projection layer

            # Forward pass through the classifier
            outputs = classifier(latent_features)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Test Accuracy: {100 * correct / total:.2f}%")

def evaluate_autoencoder_on_validation(autoencoder, val_loader, device):
    autoencoder.eval()
    total_reconstruction_loss = 0
    reconstruction_criterion = nn.MSELoss()

    with torch.no_grad():
        for data in val_loader:
            img, _ = data  # Ignore labels for reconstruction
            img = img.to(device)

            # Forward pass
            reconstructed = autoencoder(img)

            # Compute reconstruction loss
            reconstruction_loss = reconstruction_criterion(reconstructed, img)
            total_reconstruction_loss += reconstruction_loss.item()

    mean_reconstruction_loss = total_reconstruction_loss / len(val_loader)
    print(f'Validation - Reconstruction Loss: {mean_reconstruction_loss:.4f}')
    return mean_reconstruction_loss

# def evaluate_autoencoder_on_validation(autoencoder, classifier, val_loader, device):
#     autoencoder.eval()
#     classifier.eval()
    
#     total_reconstruction_loss = 0
#     total_classification_loss = 0
#     correct = 0
#     total = 0
    
#     reconstruction_criterion = nn.MSELoss()
#     classification_criterion = nn.CrossEntropyLoss()
    
#     with torch.no_grad():
#         for data in val_loader:
#             img, labels = data
#             img = img.to(device)
#             labels = labels.to(device)
            
#             # Forward pass
#             latent = autoencoder.encoder(img)
#             reconstructed = autoencoder.decoder(latent)
#             classification_output = classifier(latent)
            
#             # Compute losses
#             reconstruction_loss = reconstruction_criterion(reconstructed, img)
#             classification_loss = classification_criterion(classification_output, labels)
            
#             total_reconstruction_loss += reconstruction_loss.item()
#             total_classification_loss += classification_loss.item()
            
#             # Calculate accuracy
#             _, predicted = torch.max(classification_output, 1)
#             total += labels.size(0)
#             correct += (predicted == labels).sum().item()
    
#     mean_reconstruction_loss = total_reconstruction_loss / len(val_loader)
#     mean_classification_loss = total_classification_loss / len(val_loader)
#     accuracy = 100 * correct / total
    
#     print(f'Validation - Reconstruction Loss: {mean_reconstruction_loss:.4f}, Classification Loss: {mean_classification_loss:.4f}, Accuracy: {accuracy:.2f}%')
#     return mean_reconstruction_loss, mean_classification_loss, accuracy

def evaluate_autoencoder_classifier_on_validation(autoencoder, classifier, val_loader, device):
    autoencoder.eval()
    classifier.eval()
    total_classification_loss = 0
    correct = 0
    total = 0
    classification_criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for data in val_loader:
            img, labels = data
            img = img.to(device)
            labels = labels.to(device)

            # Pass the image through the encoder to get the latent representation
            latent = autoencoder.encoder(img)

            # Forward pass through the classifier
            outputs = classifier(latent)

            # Compute classification loss
            classification_loss = classification_criterion(outputs, labels)
            total_classification_loss += classification_loss.item()

            # Calculate accuracy
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    mean_classification_loss = total_classification_loss / len(val_loader)
    accuracy = 100 * correct / total
    print(f'Validation - Classification Loss: {mean_classification_loss:.4f}, Accuracy: {accuracy:.2f}%')
    return mean_classification_loss, accuracy

def evaluate_classifier_on_validation(encoder, classifier, val_loader, device):
    """
    Evaluate the encoder and classifier on the validation set.
    Args:
        encoder: The encoder model.
        classifier: The external classifier model.
        val_loader: DataLoader for validation data.
        device: Device to run the evaluation on (e.g., 'cuda' or 'cpu').
    """
    encoder.eval()
    classifier.eval()
    correct = 0
    total = 0
    val_loss = 0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for data in val_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            latent_features = encoder.encoder(images)  # Extract latent features
            outputs = classifier(latent_features)  # Pass features to the classifier
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            # Calculate accuracy
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_loss /= len(val_loader)
    accuracy = 100 * correct / total
    print(f'Validation Loss: {val_loss:.4f}, Accuracy: {accuracy:.2f}%')

def evaluate_simclr_on_validation(model, val_loader, device, num_classes):
    # Fine-tune the encoder
    for param in model.encoder.parameters():
        param.requires_grad = True

    # Define a linear classifier
    classifier = nn.Linear(512, num_classes).to(device)  # Assuming ResNet18 outputs 512 features
    optimizer = torch.optim.Adam(list(model.encoder.parameters()) + list(classifier.parameters()), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    # Train the classifier on the validation set
    classifier.train()
    for epoch in range(5):  # Train for 5 epochs on validation data
        epoch_loss = 0
        correct = 0
        total = 0
        for data in val_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)

            # Extract features using the encoder
            features = model.encoder(images)

            # Forward pass through the classifier
            outputs = classifier(features)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print(f"Validation Epoch [{epoch+1}/5], Loss: {epoch_loss:.4f}, Accuracy: {100 * correct / total:.2f}%")

    # Evaluate the classifier on the validation set
    classifier.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in val_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)

            # Extract features using the encoder
            features = model.encoder(images)

            # Forward pass through the classifier
            outputs = classifier(features)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Validation Accuracy: {accuracy:.2f}%")
    return accuracy