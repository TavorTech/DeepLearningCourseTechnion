import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvAutoencoder(nn.Module):
    def __init__(self, latent_dim):
        super(ConvAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
    nn.Conv2d(3, 128, kernel_size=6, stride=2, padding=2),  # 32x32x3 -> 16x16x128
    nn.ReLU(),
    nn.Conv2d(128, 256, kernel_size=6, stride=2, padding=2),  # 16x16x128 -> 8x8x256
    nn.ReLU(),
    nn.Flatten(),
    nn.Linear(8 * 8 * 256, latent_dim),  # Adjusted for 8x8x256 output
    nn.ReLU()
)
        self.decoder = nn.Sequential(
    nn.Linear(latent_dim, 8 * 8 * 256),  # Adjusted for 8x8x256 input
    nn.ReLU(),
    nn.Unflatten(1, (256, 8, 8)),
    nn.ConvTranspose2d(256, 128, kernel_size=6, stride=2, padding=2, output_padding=0),  # 8x8x256 -> 16x16x128
    nn.BatchNorm2d(128),
    nn.ReLU(),
    nn.ConvTranspose2d(128, 3, kernel_size=6, stride=2, padding=2, output_padding=0),  # 16x16x128 -> 32x32x3
    nn.Sigmoid()
)

    def forward(self, x):
        # print(f"Input shape: {x.shape}")  # Input to the encoder
        latent = self.encoder(x)
        # print(f"Latent shape: {latent.shape}")  # Output of the encoder
        reconstructed = self.decoder(latent)
        # print(f"Reconstructed shape: {reconstructed.shape}")  # Output of the decoder
        return reconstructed

#The classifier might not have enough capacity (e.g., too few layers or neurons)
# to effectively map the latent features to the output classes. - adding 4th layer
class Classifier(nn.Module):
    def __init__(self, latent_dim, num_classes):
        super(Classifier, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 1024),  # First layer: latent_dim -> 256
            nn.ReLU(),
            nn.Dropout(0.9),  # Dropout for regularization
            nn.Linear(1024, 1024),  # Second layer: 256 -> 512
            nn.ReLU(),
            nn.Dropout(0.9),  # Higher dropout for more neurons
            nn.Linear(1024, num_classes)  # Third layer: 512 -> num_classes
            # nn.Linear(latent_dim, 512),
            # nn.ReLU(),
            # nn.Dropout(0.3),
            # nn.Linear(512, 256),
            # nn.ReLU(),
            # nn.Dropout(0.3),
            # nn.Linear(256, 128),  # Additional layer
            # nn.ReLU(),
            # nn.Linear(128, num_classes)


            # nn.Linear(latent_dim, 256),
            # nn.ReLU(),
            # nn.Dropout(0.3),
            # nn.Linear(256,num_classes)
        )
    def forward(self, x):
        return self.fc(x)
    

class ConvAutoencoderMNIST(nn.Module):
    def __init__(self, latent_dim):
        super(ConvAutoencoderMNIST, self).__init__()
        # Encoder: 3 layers
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1),  # 28x28x1 -> 14x14x64
            # nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # 14x14x64 -> 7x7x128
            # nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(7*7*128, latent_dim),  # Adjusted for 7x7x128 output
            nn.ReLU()
        )
        
        # Decoder: 3 layers
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 7*7*64),  # latent_dim -> 7x7x64
            nn.ReLU(),
            nn.Unflatten(1, (64, 7, 7)),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),  # 7x7x64 -> 14x14x32
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2, padding=1, output_padding=1),  # 14x14x32 -> 28x28x1
            nn.Sigmoid()
        )

    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed
    
class EncoderClassifierMNIST(nn.Module):
    def __init__(self, latent_dim, num_classes):
        super(EncoderClassifierMNIST, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),  # 28x28x1 -> 14x14x32
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # 14x14x32 -> 7x7x64
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(7*7*64, latent_dim),  # Adjusted for 7x7x64 output
            nn.ReLU()
        )
    def forward(self, x, classifier):
        """
        Forward pass with an external classifier.
        Args:
            x: Input tensor (e.g., images).
            classifier: External classifier to process the latent features.
        Returns:
            output: Predictions from the classifier.
        """
        latent = self.encoder(x)  # Extract latent features
        output = classifier(latent)  # Pass features to the external classifier
        return output
    
    

class EncoderClassifierCIFAR(nn.Module):
    def __init__(self, latent_dim):
        super(EncoderClassifierCIFAR, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),  # 32x32x3 -> 16x16x64
            # nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # 16x16x64 -> 8x8x128
            # nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(8 * 8 * 128, latent_dim),  # Adjusted for 8x8x128 output
            nn.ReLU()
        )

    def forward(self, x, classifier):
        """
        Forward pass with an external classifier.
        Args:
            x: Input tensor (e.g., images).
            classifier: External classifier to process the latent features.
        Returns:
            output: Predictions from the classifier.
        """
        latent = self.encoder(x)  # Extract latent features
        output = classifier(latent)  # Pass features to the external classifier
        return output
    
class SimCLR(nn.Module):
    def __init__(self, base_encoder, projection_dim=128):
        super(SimCLR, self).__init__()
        self.encoder = base_encoder
        self.projection_head = nn.Sequential(
            nn.Linear(512, projection_dim),
            nn.BatchNorm1d(projection_dim),
            nn.ReLU()
        )
        self.latent_projection = nn.Linear(512, 128)  # Add this layer to project to 128 dimensions

    def forward(self, x, return_latent=False):
        features = self.encoder(x)  # Extract features from the encoder
        if return_latent:
            latent = self.latent_projection(features)  # Project features to 128 dimensions
            return latent
        projections = self.projection_head(features)
        return F.normalize(projections, dim=1)  # Ensure normalized output