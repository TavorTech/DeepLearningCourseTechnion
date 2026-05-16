import torch
from torchvision import datasets, transforms
from torchvision.models import resnet18,resnet50
import torch.nn as nn
import numpy as np
from matplotlib import pyplot as plt
from utils import plot_tsne
import random
import argparse
from models import ConvAutoencoder, ConvAutoencoderMNIST, EncoderClassifierCIFAR, EncoderClassifierMNIST, Classifier,SimCLR
from train import train_autoencoder,train_classifier_with_frozen_encoder, train_encoder_classifier,train_simclr,train_classifier_with_encoder
from dataloader import load_data,load_data_simclr,load_data_classification,load_data_self_supervised
from evaluate import evaluate_simclr, evaluate_classifier_self
# from losses import fine_tune_classifier

from sklearn.manifold import TSNE

NUM_CLASSES = 10

def freeze_seeds(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def get_args():   
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=0, type=int, help='Seed for random number generators')
    parser.add_argument('--data-path', default="/datasets/cv_datasets/data", type=str, help='Path to dataset')
    parser.add_argument('--batch-size', default=8, type=int, help='Size of each batch')
    parser.add_argument('--latent-dim', default=128, type=int, help='encoding dimension')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', type=str, help='Default device to use')
    parser.add_argument('--mnist', action='store_true', default=False,
                        help='Whether to use MNIST (True) or CIFAR10 (False) data')
    parser.add_argument('--self-supervised', action='store_true', default=False,
                        help='Whether train self-supervised with reconstruction objective, or jointly with classifier for classification objective.')
    parser.add_argument('--simclr', action='store_true', default=False,
                        help='Whether to use SimCLR augmentations')
    return parser.parse_args()

def plot_tsne_simclr(encoder, test_loader, device):
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt

    encoder.eval()
    features = []
    labels = []

    with torch.no_grad():
        for data in test_loader:
            images, lbls = data
            images = images.to(device)
            lbls = lbls.to(device)

            # Extract features using the encoder
            feats = encoder(images)
            features.append(feats.cpu())
            labels.append(lbls.cpu())

    # Concatenate all features and labels
    features = torch.cat(features, dim=0)
    labels = torch.cat(labels, dim=0)

    # Perform t-SNE
    tsne = TSNE(n_components=2, random_state=0)
    features_2d = tsne.fit_transform(features)

    # Plot t-SNE
    plt.figure(figsize=(10, 10))
    scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], c=labels, cmap='tab10', s=5)
    plt.legend(*scatter.legend_elements(), title="Classes")
    plt.title("t-SNE Visualization of SimCLR Representations")
    plt.show()





def check_encoder_decoder_output(autoencoder, train_dataset, device):
    sample = train_dataset[0][0][None].to(device)  # Get a sample from the dataset
    output = autoencoder(sample)
    print(f'Input shape: {sample.shape}')
    print(f'Output shape: {output.shape}')

def visualize_latent_space(encoder, data_loader, device):
    encoder.eval()
    latent_vectors = []
    labels = []

    with torch.no_grad():
        for data in data_loader:
            img, label = data
            img = img.to(device)
            latent = encoder(img)
            latent_vectors.append(latent.cpu())
            labels.append(label)

    latent_vectors = torch.cat(latent_vectors, dim=0)
    labels = torch.cat(labels, dim=0)

    tsne = TSNE(n_components=2, random_state=0)
    latent_2d = tsne.fit_transform(latent_vectors)

    plt.figure(figsize=(10, 10))
    scatter = plt.scatter(latent_2d[:, 0], latent_2d[:, 1], c=labels, cmap='tab10', s=5)
    plt.legend(*scatter.legend_elements(), title="Classes")
    plt.show()


if __name__ == "__main__":
    args = get_args()
    freeze_seeds(args.seed)
        
    if args.simclr:
        # Load data for SimCLR pretraining
        train_loader, val_loader, test_loader = load_data_simclr(args)
        device = torch.device(args.device)

        # Initialize SimCLR model
        base_encoder = resnet18()
        if args.mnist:
            base_encoder.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        base_encoder.fc = nn.Identity()  # Keep the encoder's output as (batch_size, 512)
        model = SimCLR(base_encoder, projection_dim=128).to(device)

        # Define optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        # Train SimCLR with validation
        train_simclr(model, train_loader, val_loader, optimizer, device, num_epochs=10)

        # Load data for classification fine-tuning
        train_loader, val_loader, test_loader = load_data_classification(args)

        # Define the classifier
        classifier = Classifier(latent_dim=args.latent_dim, num_classes=NUM_CLASSES).to(device)

        # Fine-tune the classifier with validation
        classifier = train_classifier_with_encoder(model, classifier, train_loader,val_loader, device, num_epochs=10, learning_rate=1e-3)

        # Evaluate the classifier
        evaluate_simclr(model, classifier, test_loader, device)
        plot_tsne_simclr(model.encoder, test_loader, device)
    else:
            
        # train_loader,val_loader, test_loader, input_dim = load_data(args)

        # Initialize and train the model based on the self-supervised flag
        device = torch.device(args.device)
        if args.self_supervised:
            if args.mnist:
                autoencoder = ConvAutoencoderMNIST(latent_dim=args.latent_dim).to(device)
            else:
                autoencoder = ConvAutoencoder(latent_dim=args.latent_dim).to(device)
            classifier = Classifier(latent_dim=args.latent_dim, num_classes=NUM_CLASSES).to(device)
            train_loader,val_loader, test_loader, input_dim, train_encoder_dataset = load_data_self_supervised(args)
            # Step 1: Train the autoencoder
            print("Training Autoencoder...")
            #train_autoencoder(autoencoder, train_loader, val_loader, device, num_epochs=10, learning_rate=5e-4) BEST ACCURACY SO FAR 49%+
            # train_autoencoder(autoencoder, train_loader, val_loader, device, num_epochs=10, learning_rate=0.05)
            # train_autoencoder(autoencoder, train_encoder_dataset, val_loader, device, num_epochs=20, learning_rate=0.18)
            train_autoencoder(autoencoder, train_encoder_dataset, val_loader, device, num_epochs=10, learning_rate=0.5)
            # Check the output of the encoder and decoder
            check_encoder_decoder_output(autoencoder, train_loader.dataset, device)
            # Step 2: Train the classifier with the frozen encoder
            print("Training Classifier with Frozen Encoder...")
            #no extra transformation just the normalize + 5e-4 and SGD 0.08 + (SGD+ COS) + (LR + COS) = 48.6% BS 32 dropout 0.4,0.3
            ##best training rate 5e-4 49%+
            # train_classifier_with_frozen_encoder(autoencoder, classifier, train_loader, val_loader, device, num_epochs=20, learning_rate=3e-4) # BEST BEST BEST
            train_classifier_with_frozen_encoder(autoencoder, classifier, train_loader, val_loader, device, num_epochs=10, learning_rate=3e-4)
    
            # Evaluate the classifier
            evaluate_classifier_self(classifier, autoencoder.encoder, test_loader, device)
            
            # Plot t-SNE visualizations
            #plot_tsne(autoencoder.encoder, test_loader, device)
        else:
            if args.mnist:
                encoder = ConvAutoencoderMNIST(latent_dim=args.latent_dim, num_classes=NUM_CLASSES).to(device)
            else:
                encoder = ConvAutoencoder(latent_dim=args.latent_dim).to(device)
            

            # Initialize the external classifier
            classifier = Classifier(latent_dim=args.latent_dim, num_classes=NUM_CLASSES).to(device)

            # Load the data
            train_loader, val_loader, test_loader, input_dim = load_data(args)

            # Train the encoder and classifier jointly
            train_encoder_classifier(encoder=encoder, classifier=classifier, train_loader=train_loader, val_loader=val_loader, device=device)

            # Evaluate the classifier
            evaluate_classifier_self(classifier, encoder.encoder, test_loader, device)
            # Plot t-SNE visualizations
            #plot_tsne(model.encoder, test_loader, device)