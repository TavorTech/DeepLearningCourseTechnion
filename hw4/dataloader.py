import torch
from torchvision import datasets, transforms

class DataTransforms:
    @staticmethod
    def get_simclr_pipeline_transform(size=32, s=1.0):
        """Return a set of data augmentation transformations as described in the SimCLR paper."""
        color_jitter = transforms.ColorJitter(0.2 * s, 0.2 * s, 0.2 * s, 0.1 * s)
        # data_transforms = transforms.Compose([
        #     transforms.RandomResizedCrop(size=size, scale=(0.2, 1.0)),  # Random crops
        #     transforms.RandomHorizontalFlip(),                         # Horizontal flip
        #     transforms.RandomApply([color_jitter], p=0.5),              # Color distortion
        #     transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.5),  # Gaussian blur
        #     # transforms.RandomGrayscale(p=0.2),
        #     transforms.ToTensor(),                                    # Convert to tensor
        #     transforms.Normalize(mean = [0.4914, 0.4822, 0.4465],std = [0.2470, 0.2435, 0.2616]),
        # ])
        data_transforms = transforms.Compose([
            transforms.RandomResizedCrop(size=32, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=5)], p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])
        ])
        return data_transforms
    # data_transforms = transforms.Compose([
    #         transforms.RandomResizedCrop(size=size, scale=(0.2, 1.0)),  # Random crops
    #         transforms.RandomHorizontalFlip(),                         # Horizontal flip
    #         transforms.RandomApply([color_jitter], p=0.5),              # Color distortion
    #         transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.5),  # Gaussian blur
    #         # transforms.RandomGrayscale(p=0.2),
    #         transforms.ToTensor(),                                    # Convert to tensor
    #         transforms.Normalize(mean = [0.4914, 0.4822, 0.4465],std = [0.2470, 0.2435, 0.2616]),
    #     ])
    #     return data_transforms
    @staticmethod
    def get_simclr_pipeline_transformMNIST(size=28):
        train_transform = transforms.Compose([
            transforms.RandomRotation(10),  # Reduced rotation range
            transforms.RandomCrop(28, padding=4),  # Add random cropping with padding
            transforms.ColorJitter(brightness=0.3),  # Brightness adjustment
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.5),  # Gaussian blur
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        return train_transform
def load_data_self_supervised(args):
    if args.mnist:
        train_transform = transforms.Compose([
            # transforms.RandomRotation(10),  # Reduced rotation range
            # transforms.RandomCrop(28, padding=4),  # Add random cropping with padding
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        train_transform_encoder = transforms.Compose([
            # transforms.RandomResizedCrop(size=32, scale=(0.8, 1.0)),
            # transforms.RandomRotation(10),
            # transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            transforms.ToTensor(),
            # transforms.RandomErasing(p=0.5)
            ])
        #DataTransforms.get_simclr_pipeline_transformMNIST(size=28)
        train_dataset = datasets.MNIST(root=args.data_path, train=True, download=True, transform=train_transform)
        train_encoder_dataset = datasets.MNIST(root=args.data_path, train=True, download=True, transform=train_transform_encoder)
        test_dataset = datasets.MNIST(root=args.data_path, train=False, download=True, transform=train_transform)
        input_dim = 28*28
        
    else:
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            # transforms.RandomHorizontalFlip(),
            # transforms.RandomCrop(32, padding=4),
            # transforms.RandomResizedCrop(size=32, scale=(0.8, 1.0)),
            # transforms.RandomHorizontalFlip(),
            # transforms.RandomRotation(10),
            # transforms.RandomCrop(32, padding=4),
            # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            # transforms.RandomGrayscale(p=0.1),
            
            
            # transforms.Normalize(mean = [0.4914, 0.4822, 0.4465],std = [0.2470, 0.2435, 0.2616]),
            # transforms.RandomErasing(p=0.5),
        ])
        train_transform_encoder = transforms.Compose([
            # transforms.RandomResizedCrop(size=32, scale=(0.8, 1.0)),
            # transforms.RandomRotation(10),
            # transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            transforms.ToTensor(),
            # transforms.RandomErasing(p=0.5)
            ])
        # train_transform = DataTransforms.get_simclr_pipeline_transform(size=32, s=0.5)
        train_dataset = datasets.CIFAR10(root=args.data_path, train=True, download=True, transform=train_transform)
        train_encoder_dataset = datasets.CIFAR10(root=args.data_path, train=True, download=True, transform=train_transform_encoder)
        test_dataset = datasets.CIFAR10(root=args.data_path, train=False, download=True, transform=train_transform)
        input_dim = 32*32*3
    

    # Split train_dataset into training and validation subsets
    val_size = int(0.1 * len(train_dataset))  # Use 10% of the training data for validation
    train_size = len(train_dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])

    # Create data loaders
    train_encoder_loader = torch.utils.data.DataLoader(train_encoder_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, input_dim, train_encoder_loader

def load_data(args):
    if args.mnist:
        train_transform = transforms.Compose([
            # transforms.RandomRotation(10),  # Reduced rotation range
            # transforms.RandomCrop(28, padding=4),  # Add random cropping with padding
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        #DataTransforms.get_simclr_pipeline_transformMNIST(size=28)
        train_dataset = datasets.MNIST(root=args.data_path, train=True, download=True, transform=train_transform)
        test_dataset = datasets.MNIST(root=args.data_path, train=False, download=True, transform=train_transform)
        input_dim = 28*28
    else:
        train_transform = transforms.Compose([
            # transforms.RandomHorizontalFlip(),
            # transforms.RandomCrop(32, padding=4),
            # transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            # transforms.RandomGrayscale(p=0.1),
            transforms.ToTensor(),
            # transforms.Normalize(mean = [0.4914, 0.4822, 0.4465],std = [0.2470, 0.2435, 0.2616]),
            # transforms.RandomErasing(p=0.5),
        ])
        
        # train_transform = DataTransforms.get_simclr_pipeline_transform(size=32, s=0.5)
        train_dataset = datasets.CIFAR10(root=args.data_path, train=True, download=True, transform=train_transform)
        test_dataset = datasets.CIFAR10(root=args.data_path, train=False, download=True, transform=train_transform)
        input_dim = 32*32*3
    

    # Split train_dataset into training and validation subsets
    val_size = int(0.1 * len(train_dataset))  # Use 10% of the training data for validation
    train_size = len(train_dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])

    # Create data loaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True,pin_memory=True,num_workers=4)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,pin_memory=True,num_workers=4)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,pin_memory=True,num_workers=4)

    return train_loader, val_loader, test_loader, input_dim

class SimCLRTransform:
    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        # Apply the base transform to create two augmented views
        img1 = self.base_transform(x)
        img2 = self.base_transform(x)
        return img1, img2  # Return as a tuple of tensors
    
def load_data_classification(args):
    if args.mnist:
        base_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])  # For MNIST
        ])
        validation_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])  # For MNIST
        ])
    else:
        base_transform = transforms.Compose([
            transforms.RandomResizedCrop(size=32, scale=(0.8, 1.0)),  # Mild augmentation
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])  # For CIFAR-10
        ])
        validation_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])  # For CIFAR-10
        ])


    if args.mnist:
        train_dataset = datasets.MNIST(root=args.data_path, train=True, download=True, transform=base_transform)
        test_dataset = datasets.MNIST(root=args.data_path, train=False, download=True, transform=validation_transform)
    else:
        train_dataset = datasets.CIFAR10(root=args.data_path, train=True, download=True, transform=base_transform)
        test_dataset = datasets.CIFAR10(root=args.data_path, train=False, download=True, transform=validation_transform)

    # Split the test dataset into validation and test sets
    val_size = int(0.5 * len(test_dataset))  # Use 50% of the test set for validation
    test_size = len(test_dataset) - val_size
    val_dataset, test_dataset =  torch.utils.data.random_split(test_dataset, [val_size, test_size])

    # Create data loaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=2)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    return train_loader, val_loader, test_loader

def load_data_simclr(args):
    if args.mnist:
        base_transform = DataTransforms.get_simclr_pipeline_transformMNIST(size=28)
    else:
        base_transform = DataTransforms.get_simclr_pipeline_transform(size=32, s=0.5)

    simclr_transform = SimCLRTransform(base_transform)

    if args.mnist:
        train_dataset = datasets.MNIST(root=args.data_path, train=True, download=True, transform=simclr_transform)
        test_dataset = datasets.MNIST(root=args.data_path, train=False, download=True, transform=base_transform)
    else:
        train_dataset = datasets.CIFAR10(root=args.data_path, train=True, download=True, transform=simclr_transform)
        test_dataset = datasets.CIFAR10(root=args.data_path, train=False, download=True, transform=base_transform)

    val_size = int(0.1 * len(train_dataset))
    train_size = len(train_dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=2)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    return train_loader, val_loader, test_loader