import torch
import torch.nn as nn
import torch.optim as optim
from losses import nt_xent_loss,info_nce_loss_from_views
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.models import resnet18
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim import Adam
from evaluate import evaluate_classifier_on_validation,evaluate_autoencoder_classifier_on_validation,evaluate_autoencoder_on_validation,evaluate_simclr_on_validation
from torch.amp import GradScaler, autocast
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn

def train_autoencoder(autoencoder, train_loader, val_loader, device, num_epochs=15, learning_rate=5e-4):
    reconstruction_criterion = nn.L1Loss()
    # optimizer = optim.Adam(autoencoder.parameters(), lr=learning_rate)
    # optimizer = optim.SGD(autoencoder.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)
    optimizer = optim.SGD(autoencoder.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    scaler = GradScaler(device=device)  # Initialize GradScaler for mixed precision

    autoencoder.train()
    # input_tensor = torch.randn(64, 3, 32, 32).to(device)  # Batch size 64, 3 channels, 32x32 image
    # output_tensor = autoencoder(input_tensor)
    # print(f"Output Shape: {output_tensor.shape}")
    for epoch in range(num_epochs):
        epoch_loss = 0
        for data in train_loader:
            img, _ = data
            img = img.to(device, non_blocking=True)

            # Forward pass with mixed precision
            with autocast(device_type=device.type):
                reconstructed = autoencoder(img)
                loss = reconstruction_criterion(reconstructed, img)

            # Backward pass with scaled gradients
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            # torch.nn.utils.clip_grad_norm_(autoencoder.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()

        epoch_loss /= len(train_loader)
        # scheduler.step(epoch_loss)
        scheduler.step()
        print(f'Epoch [{epoch+1}/{num_epochs}], Reconstruction Loss: {epoch_loss:.4f}')
        
        #TODO: return the
        # Evaluate on validation set
        # evaluate_autoencoder_on_validation(autoencoder, val_loader, device)
    # Calculate mean reconstruction error on the training set
    mean_reconstruction_error = calculate_mean_reconstruction_error(autoencoder, train_loader, device)
    print(f"Mean Reconstruction Error (Training Set): {mean_reconstruction_error:.4f}")

#L1Loss:
    #LR + cosine + simclr + BS512 = 39.90%
    #LR + cosine + regular + BS512 = 41.46%
    #LR + SGD + simclr + BS8 = 38.8%
    #cosine + cosine + simclr + BS512 = 35.16%
    #LR + SGD + simclr + BS512 = 38.51%
    
    #regular data:
    #RMS + RMS + BS512 = 40.65%
    #SGD + SGD + BS8 = low
    #LR + COS + BS512 + no greyscale = 42.45%
    #RMSPROP + ADAM + LR + COS + BS512 + no greyscale
    #LR + COS + BS512 + greyscale + 5e-3 = 20%
    ##LR + COS + BS8 + greyscale + jitter + 1e-3 + weighted adam 1e-4 = ?
    #frozen 1e-4 =~ 35%
    #adam adam LR COS BS512 nogray = 40.15%
    #only greyscale Adam x2 LR1= 1e-3 LR2=8e-3 = 35%

#MSELoss:
    #LR + SGD + simclr + BS8 = 38.67%

def calculate_mean_reconstruction_error(autoencoder, dataloader, device):
    """
    Calculate the mean reconstruction error (MAE) for the given autoencoder and dataloader.
    Args:
        autoencoder: The trained autoencoder model.
        dataloader: DataLoader for the dataset (e.g., training or validation set).
        device: Device to run the model on (e.g., 'cuda' or 'cpu').
    Returns:
        mean_error: The mean reconstruction error (MAE) over the dataset.
    """
    autoencoder.eval()
    total_error = 0
    total_samples = 0

    with torch.no_grad():
        for data in dataloader:
            img, _ = data  # Ignore labels
            img = img.to(device)
            
            # Forward pass
            reconstructed = autoencoder(img)
            
            # Compute reconstruction error
            error = torch.abs(reconstructed - img).mean().item()
            total_error += error * img.size(0)
            total_samples += img.size(0)

    mean_error = total_error / total_samples
    return mean_error

def train_classifier_with_frozen_encoder(autoencoder, classifier, train_loader, val_loader, device, num_epochs=10, learning_rate=5e-4):
    # classification_criterion = nn.CrossEntropyLoss()
    #optimizer = optim.Adam(classifier.parameters(), lr=learning_rate)
    # optimizer = torch.optim.Adam(classifier.parameters(), lr=learning_rate)
    #optimizer = optim.RMSprop(classifier.parameters(), lr=learning_rate, weight_decay=1e-5)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)
    # scheduler = optim.SGD(classifier.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)

    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    classification_criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(classifier.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)
    scaler = GradScaler(device=device)  # Initialize GradScaler for mixed precision

    # Freeze the encoder
    for param in autoencoder.encoder.parameters():
        param.requires_grad = False

    classifier.train()
    torch.backends.cudnn.benchmark = True  # Enable cuDNN benchmark for optimized performance

    for epoch in range(num_epochs):
        epoch_loss = 0
        correct = 0
        total = 0

        for data in train_loader:
            img, labels = data
            img = img.to(device, non_blocking=True)  # Non-blocking transfer to GPU
            labels = labels.to(device, non_blocking=True)

            # Forward pass through the frozen encoder and classifier with mixed precision
            with torch.no_grad():  # Do not compute gradients for the encoder
                with autocast(device_type='cuda'):  # Enable mixed precision
                    latent = autoencoder.encoder(img)

            with autocast(device_type='cuda'):  # Enable mixed precision for the classifier
                classification_output = classifier(latent)
                loss = classification_criterion(classification_output, labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            scaler.scale(loss).backward()  # Scale the loss for mixed precision
            # torch.nn.utils.clip_grad_norm_(classifier.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            _, predicted = torch.max(classification_output, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss /= len(train_loader)
        accuracy = 100 * correct / total
        scheduler.step()  # Step the learning rate scheduler

        print(f'Epoch [{epoch+1}/{num_epochs}], Classification Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.2f}%')

        #TODO: return the evaluate function
        # Evaluate on validation set
        evaluate_autoencoder_classifier_on_validation(autoencoder,classifier, val_loader, device)

# 4e-4, w.d. 1e-3 67.3
# 3e-4 no wd 67.62
def train_encoder_classifier(encoder, classifier, train_loader, val_loader, device, num_epochs=10, learning_rate=1e-4):
    """
    Train the encoder and classifier jointly for classification.
    Args:
        encoder: The encoder model (e.g., EncoderClassifierCIFAR).
        classifier: The external classifier model.
        train_loader: DataLoader for training data.
        val_loader: DataLoader for validation data.
        device: Device to run the training on (e.g., 'cuda' or 'cpu').
        num_epochs: Number of training epochs.
        learning_rate: Learning rate for the optimizer.
    """
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(classifier.parameters()), lr=learning_rate)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    encoder.train()
    classifier.train()

    for epoch in range(num_epochs):
        epoch_loss = 0
        correct = 0
        total = 0

        for data in train_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)

            # Forward pass: Extract features using the encoder and classify using the classifier
            latent_features = encoder.encoder(images)  # Extract latent features
            outputs = classifier(latent_features)  # Pass features to the classifier
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            # Calculate accuracy
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss /= len(train_loader)
        accuracy = 100 * correct / total
        # scheduler.step(epoch_loss)
        scheduler.step()

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.2f}%')

        # Evaluate on validation set
        evaluate_classifier_on_validation(encoder, classifier, val_loader, device)


# def train_encoder_classifier(model,val_loader, train_loader, device, num_epochs=10, learning_rate=1e-3):
#     classification_criterion = nn.CrossEntropyLoss()
#     optimizer = optim.Adam(model.parameters(), lr=learning_rate)
#     scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)
    
#     model.train()
#     for epoch in range(num_epochs):
#         epoch_loss = 0
#         for data in train_loader:
#             img, labels = data
#             img = img.to(device)
#             labels = labels.to(device)
            
#             # Forward pass
#             outputs = model(img)
            
#             # Compute loss
#             loss = classification_criterion(outputs, labels)
            
#             # Backward pass and optimization
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
            
#             epoch_loss += loss.item()
        
#         epoch_loss /= len(train_loader)
#         scheduler.step(epoch_loss)
        
#         print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}')
#         evaluate_classifier_on_validation(model, val_loader, device)

#62% accuracy
# def train_simclr(model, train_loader,val_loader, device, num_epochs=10, learning_rate=1e-3, temperature=0.5):
#     optimizer = optim.Adam(model.parameters(), lr=learning_rate)
#     scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=0, last_epoch=-1)
#     model.train()
#     for epoch in range(num_epochs):
#         epoch_loss = 0
#         for data in train_loader:
#             # Extract images from the data (ignore labels)
#             images, _ = data  # Assuming train_loader returns (images, labels)
            
#             # Generate two augmented views of the same batch
#             img_1, img_2 = images, images.clone()
#             img_1, img_2 = img_1.to(device), img_2.to(device)
            
#             # Forward pass
#             z_i = model(img_1)
#             z_j = model(img_2)
            
#             # Normalize embeddings
#             z_i = torch.nn.functional.normalize(z_i, dim=1)
#             z_j = torch.nn.functional.normalize(z_j, dim=1)
            
#             # Compute contrastive loss
#             # loss = nt_xent_loss(z_i, z_j, temperature)
#             loss = info_nce_loss_from_views(z_i, z_j, temperature,device)
#             # Debugging: Print loss
#             # print(f"Loss: {loss.item()}")

#             # Backward pass and optimization
#             optimizer.zero_grad()
#             loss.backward()
#             torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)## gradients exploding?
#             optimizer.step()
            
#             epoch_loss += loss.item()
#         scheduler.step()
#         epoch_loss /= len(train_loader)
#         print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}')
#         evaluate_simclr_on_validation(model,val_loader,device,10)
# Define the augment function
# def augment(img):
#     transform = transforms.Compose([
#         transforms.RandomResizedCrop(size=32, scale=(0.2, 1.0)),  # Random crop
#         transforms.RandomHorizontalFlip(),  # Random horizontal flip
#         transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),  # Color jitter
#         transforms.RandomGrayscale(p=0.2),  # Random grayscale
#         transforms.ToTensor(),  # Convert to tensor
#         transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])  # Normalize
#     ])
#     return transform(img)
def train_classifier_with_encoder(model, classifier, train_loader, val_loader, device, num_epochs=10, learning_rate=1e-3):
    """
    Train a classifier on top of the encoder and evaluate on the validation set.
    """
    optimizer = torch.optim.Adam(list(model.encoder.parameters()) + list(classifier.parameters()), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    model.encoder.train()
    classifier.train()

    for epoch in range(num_epochs):
        # Training phase
        epoch_loss = 0
        correct = 0
        total = 0

        for data in train_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)

            # Extract features using the encoder and project to 128 dimensions
            features = model.encoder(images)
            latent_features = model.latent_projection(features)  # Use the latent projection layer

            # Forward pass through the classifier
            outputs = classifier(latent_features)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_accuracy = 100 * correct / total
        print(f"Epoch [{epoch+1}/{num_epochs}], Training Loss: {epoch_loss:.4f}, Training Accuracy: {train_accuracy:.2f}%")

        # Validation phase
        val_loss = 0
        correct = 0
        total = 0
        model.encoder.eval()
        classifier.eval()

        with torch.no_grad():
            for data in val_loader:
                images, labels = data
                images, labels = images.to(device), labels.to(device)

                # Extract features using the encoder and project to 128 dimensions
                features = model.encoder(images)
                latent_features = model.latent_projection(features)  # Use the latent projection layer

                # Forward pass through the classifier
                outputs = classifier(latent_features)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_accuracy = 100 * correct / total
        val_loss /= len(val_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%")

        # Switch back to training mode
        model.encoder.train()
        classifier.train()

    return classifier

def train_simclr(model, train_loader, val_loader, optimizer, device, num_epochs=10):
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    scaler = GradScaler(device=device) 
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0
        for data in train_loader:
            (img1, img2), _ = data
            img1, img2 = img1.to(device), img2.to(device)
            # print(img1.min(), img1.max(), img1.mean())
            # print(img2.min(), img2.max(), img2.mean())
            # Forward pass
            # Use autocast for mixed precision
            with autocast(device_type=device.type):
                z_i = model(img1)
                z_j = model(img2)
                loss = nt_xent_loss(z_i, z_j, temperature=0.05)


            # # Backward pass and optimization
            optimizer.zero_grad()
            # loss.backward()
            # # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            # optimizer.step()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()

        # Step the scheduler
        scheduler.step()

        # Print training loss
        print(f"Epoch [{epoch+1}/{num_epochs}], Training Loss: {epoch_loss/len(train_loader):.4f}")

        # # Evaluate on validation set
        # val_loss = 0
        # model.eval()
        # with torch.no_grad():
        #     for data in val_loader:
        #         (img1, img2), _ = data
        #         img1, img2 = img1.to(device), img2.to(device)

        #         # Forward pass
        #         z_i = model(img1)
        #         z_j = model(img2)

        #         # Compute contrastive loss
        #         loss = nt_xent_loss(z_i, z_j)
        #         val_loss += loss.item()

        # val_loss /= len(val_loader)
        # print(f"Epoch [{epoch+1}/{num_epochs}], Validation Loss: {val_loss:.4f}")
        # model.train()

# def train_simclr(model, train_loader, device, num_epochs=10, learning_rate=1e-3, temperature=0.05):
#     optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-6)
#     scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=0, last_epoch=-1)
#     model.train()
#     for epoch in range(num_epochs):
#         epoch_loss = 0
#         for images, _ in train_loader:
#             img_1, img_2 = images, images.clone()
#             img_1, img_2 = img_1.to(device), img_2.to(device)
#             z_i, z_j = model(img_1), model(img_2)
#             #loss = nt_xent_loss(z_i, z_j, temperature)
#             loss = info_nce_loss_from_views(z_i, z_j, temperature, device)
#             #print(f"Loss: {loss.item()}")
#             optimizer.zero_grad()
#             loss.backward()
#             torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
#             optimizer.step()
#             epoch_loss += loss.item()
#         scheduler.step()
#         print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss / len(train_loader):.4f}')
