import torch
import torch.nn.functional as F
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner
from torch import nn
from torch.utils.data import DataLoader
from datasets import Dataset
from torchvision.transforms import Compose, Normalize, ToTensor
from collections import OrderedDict
import numpy as np
import glob
import json
import pandas as pd
from array import array
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler 
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# ## Constant setup

RECORD_SEQUENCE_SIZE=20
tls_columns_names = np.array([f"tls.rec.{i}" for i in range(RECORD_SEQUENCE_SIZE)])
dataset_path = "./datasets/desktop.tls/*.json" # replace with the dataset

# ## Model Definition

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

# ## Dataset Preprocessing

# +
# Extracts features from raw dataset. This will provide suitable output to the preprocessing pipeline.
# Flow related columns: 'BytesOut', 'PacketsOut', 'BytesIn', 'PacketsIn', 'Duration'
# TLS handshake columns: 'TlsClientVersion','TlsServerVersion','TlsServerCipherSuite'
# TLS record sizes: 'RecordSequence' mapped as 'TlsRecord_X'
#
# The output is a DataFrame with the above specified columns. This dataframe can beused as the input to next
# processing block (preprocessor).
#
# Loads data from the specified collection of json files. It provides raw data.
# Resize row in the array
def resize_row(row, maxlen, pad_value=0):
    current_length = len(row)
    if current_length < maxlen:
        # Calculate the amount of padding needed
        pad_width = maxlen - current_length
        # Pad at the end (you can also pad at the beginning or both sides)
        row = np.pad(row, pad_width=(0, pad_width), mode='constant', constant_values=pad_value)
    else:
        # If the row is longer than the target length, slice it
        row = row[:maxlen]
    return row
# Resize the matrix by padding or removing columns
def pad_sequences(rows, maxlen, pad_value=0):
    resized_rows = [resize_row(row, maxlen) for row in rows]
    return resized_rows


def load_json_files(json_files):
    all_data = []
    # Open the file and read each line
    for filename in json_files:
        with open(filename, "r") as file:
            # Use a list comprehension to load each line as a JSON object
            data = [json.loads(line.strip()) for line in file]
            for item in data: all_data.append(item)
    # Convert the list of dictionaries into a DataFrame
    df = pd.DataFrame(all_data)
    return df

def extract_features(df):
    # Flow data
    flow_data = df[['bs', 'ps', 'br', 'pr', 'td']].astype(float)
    # TLS handshake data
    tls_data = df[['tls.cver','tls.sver','tls.scs']].fillna(0).astype(str) 
    # TLS records 
    records_data = pd.DataFrame( pad_sequences(df['tls.rec'].values, maxlen=RECORD_SEQUENCE_SIZE), columns=tls_columns_names)
    dataset = pd.concat([flow_data, tls_data, records_data], axis=1).fillna(0)
    return dataset
#
# Fits the preprocessor that contains scalers for numerical features and OneHotEncoder for categorical data.
# The result is the Pipeline that can be used for further data processing before they are fed in the Autoencoder.
# 
def fit_preprocessor(df):
    preprocessor = ColumnTransformer(
        transformers=[
            ('num_flow', MinMaxScaler(), ['bs', 'ps', 'br', 'pr', 'td']),
            ('num_tls', MinMaxScaler(), tls_columns_names),
            ('cat', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), ['tls.cver','tls.sver','tls.scs'])
        ])
    pipeline = Pipeline(steps=[('preprocessor', preprocessor)])
    pipeline.fit(df)
    return pipeline

def get_processed_data():
    raw_df = load_json_files(glob.glob(dataset_path))
    print(f'dataset shape={raw_df.shape}')
    input_df = extract_features(raw_df)

    pipeline = fit_preprocessor(input_df)
    normal_df = pipeline.transform(input_df)

    print('Normalized row of data:')
    print(normal_df.shape)
    
    return normal_df


# -

def load_data(partition_id, num_partitions):
    """Load and partition data for federated learning with proper row/column orientation."""
    # Get the processed data - shape (14962, 35) where:
    # - 14962 rows = samples
    # - 35 columns = features
    data = get_processed_data()
    
    # Create DataFrame with default index and column handling
    # This maintains rows as samples and columns as features
    df = pd.DataFrame(data)
    
    # Convert to HuggingFace dataset
    dataset = Dataset.from_pandas(df)
    
    # Partition the data
    partitioner = IidPartitioner(num_partitions)
    partitioner.dataset = dataset
    partition = partitioner.load_partition(partition_id)
    
    # Split into train/test
    partition_train_test = partition.train_test_split(test_size=0.2, seed=42)
    
    # Define transform to ensure proper tensor format
    def format_batch(batch):
        """Transform to ensure proper tensor format for training."""
        result = {}
        # Process each sample in the batch
        for i, sample_dict in enumerate(batch):
            # Convert sample dict to a tensor with all features
            feature_values = []
            for col in range(len(sample_dict)):
                # Get column value (feature) and add to list
                col_key = str(col)  # HuggingFace datasets use string column indices
                feature_values.append(sample_dict[col_key])
            
            # Create a tensor from all features for this sample
            sample_tensor = torch.tensor(feature_values, dtype=torch.float32)
            result[str(i)] = sample_tensor
        
        return result
    
    # Custom batch collation function
    def custom_collate(batch_list):
        """Custom collate function to properly arrange samples and features."""
        # Convert list of dictionaries to single tensor of shape [batch_size, num_features]
        samples = []
        for item in batch_list:
            # Each item is a dictionary with column indices as keys
            features = [item[str(i)] for i in range(len(item))]
            sample = torch.tensor(features, dtype=torch.float32)
            samples.append(sample)
        
        # Stack into batch tensor [batch_size, num_features]
        features_tensor = torch.stack(samples)
        
        # Return tensor directly - no need for dictionary structure
        return features_tensor
    
    # Create data loaders with custom collate function
    trainloader = DataLoader(
        partition_train_test["train"], 
        batch_size=32, 
        shuffle=True,
        collate_fn=custom_collate
    )
    
    testloader = DataLoader(
        partition_train_test["test"], 
        batch_size=32,
        collate_fn=custom_collate
    )
    
    # Verify the output shape
    for batch in trainloader:
        print(f"Batch shape: {batch.shape}")
        # Should show something like torch.Size([32, 35]) 
        # where 32 is batch_size and 35 is num_features
        break
    
    return trainloader, testloader


# ## Model Training & Testing definition

def train(net, trainloader, epochs, learning_rate, device):
    """Train the VAE network on the training set using tabular data."""
    net.to(device)  # move model to GPU if available
    optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)
    
    print(f"Training for {epochs} epochs with learning rate {learning_rate}")
    
    for epoch in range(epochs):
        running_loss = 0.0
        total_batches = 0
        
        for batch in trainloader:
            # Batch is already a tensor with shape [batch_size, num_features]
            features = batch.to(device)
            
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

def test(net, testloader, device):
    """Validate the VAE network on the entire test set using tabular data."""
    total, loss = 0, 0.0
    recon_total, kld_total = 0.0, 0.0
    
    with torch.no_grad():
        for batch in testloader:
            # Batch is already a tensor with shape [batch_size, num_features]
            features = batch.to(device)
            
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

# +
# Centralized testing whether the model could train
# net = Net(input_dim=35, latent_dim=10, hidden_dim=64)
# trainloader, testloader = load_data(0, 3)
# train(net, trainloader, 5, 0.001, "cuda:0")
# test(net, testloader, "cuda:0")
