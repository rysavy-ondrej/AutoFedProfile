# 可以跑但是有點亂 (硬寫分割，不是用函式庫)

"""fedvae: A Flower app for Federated Variational Autoencoder."""

from collections import OrderedDict

import torch
import torch.nn.functional as F
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner
from torch import nn
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from typing import Dict, List, Tuple, Optional, Callable
import random


class Flatten(nn.Module):
    """Flattens input by reshaping it into a one-dimensional tensor."""

    def forward(self, input):
        return input.view(input.size(0), -1)


class UnFlatten(nn.Module):
    """Unflattens a tensor converting it to a desired shape."""

    def forward(self, input):
        return input.view(-1, 16, 6, 6)


# +
# class Net(nn.Module):
#     def __init__(self, h_dim=576, z_dim=10) -> None:
#         super().__init__()
#         self.encoder = nn.Sequential(
#             nn.Conv2d(
#                 in_channels=3, out_channels=6, kernel_size=4, stride=2
#             ),  # [batch, 6, 15, 15]
#             nn.ReLU(),
#             nn.Conv2d(
#                 in_channels=6, out_channels=16, kernel_size=5, stride=2
#             ),  # [batch, 16, 6, 6]
#             nn.ReLU(),
#             Flatten(),
#         )

#         self.fc1 = nn.Linear(h_dim, z_dim)
#         self.fc2 = nn.Linear(h_dim, z_dim)
#         self.fc3 = nn.Linear(z_dim, h_dim)

#         self.decoder = nn.Sequential(
#             UnFlatten(),
#             nn.ConvTranspose2d(in_channels=16, out_channels=6, kernel_size=5, stride=2),
#             nn.ReLU(),
#             nn.ConvTranspose2d(in_channels=6, out_channels=3, kernel_size=4, stride=2),
#             nn.Tanh(),
#         )

#     def reparametrize(self, h):
#         """Reparametrization layer of VAE."""
#         mu, logvar = self.fc1(h), self.fc2(h)
#         std = torch.exp(logvar / 2)
#         eps = torch.randn_like(std)
#         z = mu + std * eps
#         return z, mu, logvar

#     def encode(self, x):
#         """Encoder of the VAE."""
#         h = self.encoder(x)
#         z, mu, logvar = self.reparametrize(h)
#         return z, mu, logvar

#     def decode(self, z):
#         """Decoder of the VAE."""
#         z = self.fc3(z)
#         z = self.decoder(z)
#         return z

#     def forward(self, x):
#         z, mu, logvar = self.encode(x)
#         z_decode = self.decode(z)
#         return z_decode, mu, logvar
# -

# Example VAE model for tabular data (if you need to adapt your existing model)
class Net(torch.nn.Module):
    def __init__(self, input_dim=35, latent_dim=10, hidden_dim=64):
        """
        VAE for tabular data.
        
        Args:
            input_dim: Number of input features
            latent_dim: Size of the latent space
            hidden_dim: Size of the hidden layers
        """
        super(Net, self).__init__()
        
        # Encoder
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU()
        )
        
        # Latent space
        self.fc_mu = torch.nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = torch.nn.Linear(hidden_dim, latent_dim)
        
        # Decoder
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(latent_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, input_dim)
        )
    
    def encode(self, x):
        """Encode input into latent representation."""
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        """Reparameterization trick."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z
    
    def decode(self, z):
        """Decode latent representation into reconstruction."""
        return self.decoder(z)
    
    def forward(self, x):
        """Forward pass through the VAE."""
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar


fds = None  # Cache FederatedDataset


# +
# def load_data(partition_id, num_partitions):
#     """Load partition CIFAR10 data."""
#     # Only initialize `FederatedDataset` once
#     global fds
#     if fds is None:
#         partitioner = IidPartitioner(num_partitions=num_partitions)
#         fds = FederatedDataset(
#             dataset="uoft-cs/cifar10",
#             partitioners={"train": partitioner},
#         )
#     partition = fds.load_partition(partition_id)
#     # Divide data on each node: 80% train, 20% test
#     partition_train_test = partition.train_test_split(test_size=0.2, seed=42)
#     pytorch_transforms = Compose(
#         [ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
#     )

#     def apply_transforms(batch):
#         """Apply transforms to the partition from FederatedDataset."""
#         batch["img"] = [pytorch_transforms(img) for img in batch["img"]]
#         return batch

#     partition_train_test = partition_train_test.with_transform(apply_transforms)
#     trainloader = DataLoader(partition_train_test["train"], batch_size=32, shuffle=True)
#     testloader = DataLoader(partition_train_test["test"], batch_size=32)
#     return trainloader, testloader

# +
class TabularNPYDataset(Dataset):
    """Dataset wrapper for tabular NPY data with features."""
    
    def __init__(self, data: np.ndarray, target_column: int = -1, transform: Optional[Callable] = None):
        """
        Args:
            data: numpy array of shape (n_samples, n_features)
            target_column: index of the column to use as target/label (-1 for last column)
                           set to None if all columns are features with no target
            transform: optional transform to apply to the features
        """
        self.data = data
        self.transform = transform
        self.target_column = target_column
        
        # Split features and target if target_column is specified
        if target_column is not None:
            self.features = np.delete(self.data, target_column, axis=1)
            self.targets = self.data[:, target_column]
        else:
            self.features = self.data
            self.targets = None
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict:
        features = self.features[idx]
        
        # Convert to torch tensor if not already
        if not isinstance(features, torch.Tensor):
            features = torch.tensor(features, dtype=torch.float32)
            
        # Apply transforms if any
        if self.transform:
            features = self.transform(features)
            
        item = {"x": features}  # Using "x" instead of "img" for tabular data
        
        # Add label if available
        if self.targets is not None:
            target = self.targets[idx]
            if not isinstance(target, torch.Tensor):
                # Check if target is categorical or continuous
                if np.issubdtype(self.targets.dtype, np.integer):
                    target = torch.tensor(target, dtype=torch.long)  # For classification
                else:
                    target = torch.tensor(target, dtype=torch.float32)  # For regression
            item["y"] = target
            
        return item

def load_data(
    npy_file_path: str, 
    target_column: int = -1,  # Set to None if all columns are features
    partition_id: int = 0, 
    num_partitions: int = 1,
    test_size: float = 0.2,
    seed: int = 42,
    normalize: bool = True
) -> Tuple[DataLoader, DataLoader]:
    """
    Load tabular NPY data file and prepare train and test loaders.
    
    Args:
        npy_file_path: Path to the NPY file containing tabular data
        target_column: Index of column to use as target (-1 for last column, None for no target)
        partition_id: Index of the partition to use (0-based)
        num_partitions: Total number of partitions to divide the data into
        test_size: Fraction of data to use for testing
        seed: Random seed for reproducibility
        normalize: Whether to normalize the features
        
    Returns:
        Tuple of (trainloader, testloader)
    """
    # Set random seeds for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Load NPY data
    print(f"Loading data from {npy_file_path}")
    data = np.load(npy_file_path, allow_pickle=True)
    
    # Print data shape for verification
    print(f"Loaded data shape: {data.shape}")
    
    # Handle different NPY file formats
    if isinstance(data, np.ndarray) and data.ndim == 2:
        # Data is already in the right format (n_samples, n_features)
        tabular_data = data
    elif isinstance(data, dict):
        # Try to find the data in the dictionary
        for key in ['data', 'features', 'x', 'samples']:
            if key in data and isinstance(data[key], np.ndarray) and data[key].ndim == 2:
                tabular_data = data[key]
                break
        else:
            raise ValueError("Could not find tabular data in the dictionary")
    else:
        raise ValueError(f"Unsupported NPY data format: {type(data)}")
    
    # Partition the data (simulate federated scenario)
    total_samples = len(tabular_data)
    indices = list(range(total_samples))
    
    # Shuffle indices before partitioning for better distribution
    random.shuffle(indices)
    
    # Create partitions
    samples_per_partition = total_samples // num_partitions
    start_idx = partition_id * samples_per_partition
    end_idx = start_idx + samples_per_partition if partition_id < num_partitions - 1 else total_samples
    
    partition_indices = indices[start_idx:end_idx]
    partition_data = tabular_data[partition_indices]
    
    # Create normalization transform if requested
    transform = None
    if normalize:
        # Calculate mean and std from the training portion to avoid data leakage
        train_size = int((1 - test_size) * len(partition_data))
        train_data = partition_data[:train_size]
        
        if target_column is not None:
            # Exclude target column from normalization
            features = np.delete(train_data, target_column, axis=1)
        else:
            features = train_data
            
        feature_mean = features.mean(axis=0)
        feature_std = features.std(axis=0)
        # Handle zero std (constant features)
        feature_std = np.where(feature_std == 0, 1.0, feature_std)
        
        # Define normalization transform as a callable
        def normalize_features(x):
            if isinstance(x, torch.Tensor):
                mean_tensor = torch.tensor(feature_mean, dtype=torch.float32)
                std_tensor = torch.tensor(feature_std, dtype=torch.float32)
                return (x - mean_tensor) / std_tensor
            else:
                return (x - feature_mean) / feature_std
        
        transform = normalize_features
    
    # Create dataset
    full_dataset = TabularNPYDataset(partition_data, target_column=target_column, transform=transform)
    
    # Split into train and test
    dataset_size = len(full_dataset)
    train_size = int((1 - test_size) * dataset_size)
    test_size = dataset_size - train_size
    
    # Create train/test splits
    indices = list(range(dataset_size))
    random.Random(seed).shuffle(indices)
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]
    
    train_dataset = Subset(full_dataset, train_indices)
    test_dataset = Subset(full_dataset, test_indices)
    
    # Create dataloaders
    trainloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    testloader = DataLoader(test_dataset, batch_size=32)
    
    print(f"Created train dataloader with {len(train_dataset)} samples")
    print(f"Created test dataloader with {len(test_dataset)} samples")
    
    return trainloader, testloader


# +
# def train(net, trainloader, epochs, learning_rate, device):
#     """Train the network on the training set."""
#     net.to(device)  # move model to GPU if available
#     optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)
#     for _ in range(epochs):
#         # for images, _ in trainloader:
#         for batch in trainloader:
#             images = batch["img"]
#             images = images.to(device)
#             optimizer.zero_grad()
#             recon_images, mu, logvar = net(images)
#             recon_loss = F.mse_loss(recon_images, images)
#             kld_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
#             loss = recon_loss + 0.05 * kld_loss
#             loss.backward()
#             optimizer.step()
# -

def train(net, trainloader, epochs, learning_rate, device):
    """Train the VAE network on the training set using tabular data."""
    net.to(device)  # move model to GPU if available
    optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)
    
    print(f"Training for {epochs} epochs with learning rate {learning_rate}")
    
    for epoch in range(epochs):
        running_loss = 0.0
        total_batches = 0
        
        for batch in trainloader:
            # Use 'x' instead of 'img' for tabular data
            features = batch["x"]
            features = features.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass through the VAE
            recon_features, mu, logvar = net(features)
            
            # Compute reconstruction loss
            recon_loss = F.mse_loss(recon_features, features)
            
            # Compute KL divergence loss
            kld_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            
            # Total loss (beta-VAE formulation with beta=0.05)
            loss = recon_loss + 0.05 * kld_loss
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            
            # Track statistics
            running_loss += loss.item()
            total_batches += 1
        
        # Print epoch statistics
        if (epoch + 1) % 5 == 0 or epoch == 0:  # Print every 5 epochs or first epoch
            avg_loss = running_loss / total_batches
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, "
                  f"Recon: {recon_loss.item():.4f}, KLD: {kld_loss.item():.4f}")


# +
# def test(net, testloader, device):
#     """Validate the network on the entire test set."""
#     total, loss = 0, 0.0
#     with torch.no_grad():
#         # for data in testloader:
#         for batch in testloader:
#             images = batch["img"].to(device)
#             # images = data[0].to(DEVICE)
#             recon_images, mu, logvar = net(images)
#             recon_loss = F.mse_loss(recon_images, images)
#             kld_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
#             loss += recon_loss + kld_loss
#             total += len(images)
#     return loss / total
# -

def test(net, testloader, device):
    """Validate the VAE network on the entire test set using tabular data."""
    total, loss = 0, 0.0
    recon_total, kld_total = 0.0, 0.0
    
    with torch.no_grad():
        for batch in testloader:
            # Use 'x' instead of 'img' for tabular data
            features = batch["x"].to(device)
            
            # Forward pass through the VAE
            recon_features, mu, logvar = net(features)
            
            # Compute losses
            recon_loss = F.mse_loss(recon_features, features)
            kld_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            batch_loss = recon_loss + 0.05 * kld_loss
            
            # Track statistics
            loss += batch_loss.item() * len(features)
            recon_total += recon_loss.item() * len(features)
            kld_total += kld_loss.item() * len(features)
            total += len(features)
    
    avg_loss = loss / total
    avg_recon = recon_total / total
    avg_kld = kld_total / total
    
    print(f"Test Loss: {avg_loss:.4f}, Recon: {avg_recon:.4f}, KLD: {avg_kld:.4f}")
    
    return avg_loss


def generate(net, image):
    """Reproduce the input with trained VAE."""
    with torch.no_grad():
        return net.forward(image)


def get_weights(net):
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def set_weights(net, parameters):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)
