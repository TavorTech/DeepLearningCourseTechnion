import torch
import torch.nn.functional as F
import torch.nn as nn
# def nt_xent_loss(z_i, z_j, temperature=0.5):
#     batch_size = z_i.size(0)
    
#     # Concatenate positive pairs
#     z = torch.cat([z_i, z_j], dim=0)
    
#     # Normalize embeddings to unit vectors
#     z = F.normalize(z, dim=1)
    
#     # Compute cosine similarity matrix
#     similarity_matrix = torch.matmul(z, z.T) / temperature

#     #print("Max similarity matrix value:", similarity_matrix.max().item())
#     #print("Min similarity matrix value:", similarity_matrix.min().item())

#     # Mask out self-similarity
#     mask = torch.eye(2 * batch_size, device=z.device).bool()
#     similarity_matrix.masked_fill_(mask, -5)

#     # Create labels for positive pairs
#     labels = torch.arange(batch_size, device=z.device)
#     labels = torch.cat([labels, labels], dim=0)  # Ensure they are properly paired

#     # Compute cross-entropy loss
#     loss = F.cross_entropy(similarity_matrix, labels)
#     return loss

# def nt_xent_loss(z_i, z_j, temperature=0.5):
#     batch_size = z_i.size(0)
    
#     # Concatenate positive pairs
#     z = torch.cat([z_i, z_j], dim=0)  # Shape: (2 * batch_size, embedding_dim)
    
#     # Normalize embeddings to unit vectors
#     z = F.normalize(z, dim=1)
    
#     # Compute cosine similarity matrix
#     similarity_matrix = torch.matmul(z, z.T)  # Shape: (2 * batch_size, 2 * batch_size)
#     similarity_matrix /= temperature

#     # Mask out self-similarity
#     mask = torch.eye(2 * batch_size, device=z.device).bool()
#     similarity_matrix.masked_fill_(mask, -float('inf'))

#     # Compute positive similarities (diagonal offsets)
#     positives = torch.cat([torch.arange(batch_size), torch.arange(batch_size)], dim=0).to(z.device)
#     positive_similarities = similarity_matrix[torch.arange(2 * batch_size), positives]

#     # Compute denominator (sum over all negatives)
#     exp_similarities = torch.exp(similarity_matrix)  # Exponentiate similarities
#     negative_sums = exp_similarities.sum(dim=1)  # Sum over rows (negatives)

#     # Compute NT-Xent loss
#     loss = -torch.log(torch.exp(positive_similarities) / negative_sums).mean()
#     return loss

def nt_xent_loss(z_i, z_j, temperature=0.5):
    """
    NT-Xent loss for contrastive learning.
    """
    batch_size = z_i.size(0)
    device = z_i.device  # Ensure all tensors are on the same device

    # Concatenate positive pairs and normalize embeddings
    z = torch.cat([z_i, z_j], dim=0).to(device)  # Concatenate and move to device
    z = F.normalize(z, dim=1)  # Normalize embeddings

    # Compute similarity matrix
    similarity_matrix = torch.mm(z, z.T)  # Cosine similarity

    # Create mask to exclude self-similarities
    mask = ~torch.eye(2 * batch_size, device=device).bool()  # Invert the diagonal mask

    # Apply mask to similarity matrix
    similarity_matrix = similarity_matrix[mask].view(2 * batch_size, -1)  # Exclude self-similarities

    # Scale by temperature
    logits = similarity_matrix / temperature

    # Subtract max for numerical stability
    logits -= logits.max(dim=1, keepdim=True)[0]

    # Create labels for positive pairs
    labels = torch.arange(batch_size, device=device)
    labels = torch.cat([labels, labels], dim=0)  # Positive pairs

    # Compute cross-entropy loss
    loss = F.cross_entropy(logits, labels)
    return loss


# Fine-tune the encoder and classifier
# def fine_tune_classifier(model, train_loader, val_loader, device, num_epochs=10, learning_rate=1e-3):
#     optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
#     scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
#     criterion = nn.CrossEntropyLoss()

#     model.train()
#     for epoch in range(num_epochs):
#         epoch_loss = 0
#         correct = 0
#         total = 0

#         # Training loop
#         for data in train_loader:
#             (img1, img2), labels = data  # Unpack the tuple of augmented views
#             img1, img2 = img1.to(device), img2.to(device)  # Move both views to the device
#             labels = labels.to(device)  # Move labels to the device if needed

#             # Forward pass
#             outputs = model(images)
#             loss = criterion(outputs, labels)

#             # Backward pass and optimization
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()

#             epoch_loss += loss.item()
#             _, predicted = torch.max(outputs.data, 1)
#             total += labels.size(0)
#             correct += (predicted == labels).sum().item()

#         train_accuracy = 100 * correct / total
#         print(f"Epoch [{epoch+1}/{num_epochs}], Training Loss: {epoch_loss/len(train_loader):.4f}, Training Accuracy: {train_accuracy:.2f}%")

#         # Step the scheduler
#         scheduler.step()

        # # Validation loop
        # val_loss = 0
        # correct = 0
        # total = 0
        # model.eval()
        # with torch.no_grad():
        #     for data in val_loader:
        #         images, labels = data
        #         images, labels = images.to(device), labels.to(device)

        #         # Forward pass
        #         outputs = model(images)
        #         loss = criterion(outputs, labels)

        #         val_loss += loss.item()
        #         _, predicted = torch.max(outputs.data, 1)
        #         total += labels.size(0)
        #         correct += (predicted == labels).sum().item()

        # val_accuracy = 100 * correct / total
        # val_loss /= len(val_loader)
        # print(f"Epoch [{epoch+1}/{num_epochs}], Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%")
        # model.train()

def info_nce_loss_from_views(z_i, z_j, temperature=0.1, device='cuda'):
    batch_size = z_i.size(0)
    
    # Concatenate z_i and z_j
    features = torch.cat([z_i, z_j], dim=0)  # Shape: (2 * batch_size, embedding_dim)
    
    # Create labels
    labels = torch.cat([torch.arange(batch_size) for _ in range(2)], dim=0)
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float().to(device)

    # Normalize features
    features = F.normalize(features, dim=1)

    # Compute similarity matrix
    similarity_matrix = torch.matmul(features, features.T)

    # Mask out self-similarities
    mask = torch.eye(labels.shape[0], dtype=torch.bool).to(device)
    labels = labels[~mask].view(labels.shape[0], -1)
    similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)

    # Select positives and negatives
    positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)
    negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

    # Compute logits
    logits = torch.cat([positives, negatives], dim=1)
    logits /= temperature

    # Create labels for cross-entropy loss
    labels = torch.zeros(logits.shape[0], dtype=torch.long).to(device)

    # Compute loss
    loss = F.cross_entropy(logits, labels)
    return loss