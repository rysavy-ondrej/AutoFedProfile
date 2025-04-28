# +
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import glob
import json
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler 
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from collections import OrderedDict
from pathlib import Path
from datetime import datetime
import os

from flwr.common.typing import UserConfig
from flwr.common import Context
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner
from datasets import Dataset
# -

# ## Constant setup
RECORD_SEQUENCE_SIZE = 20
tls_columns_names = np.array([f"tls.rec.{i}" for i in range(RECORD_SEQUENCE_SIZE)])
dataset_path = "/workspace/datasets/cic-aa.normal.tls/*.json"  # replace with the dataset


# ## Model Definition

class Net(keras.Model):
    def __init__(self, input_dim=54, latent_dim=10, hidden_dim=64):
        """
        VAE for tabular data.
        
        Args:
            input_dim: Number of input features
            latent_dim: Size of the latent space
            hidden_dim: Size of the hidden layers
        """
        super(Net, self).__init__()
        
        # Encoder
        self.encoder = keras.Sequential([
            keras.layers.Dense(hidden_dim, activation='relu'),
            keras.layers.Dense(hidden_dim, activation='relu')
        ])
        
        # Latent space
        self.fc_mu = keras.layers.Dense(latent_dim)
        self.fc_logvar = keras.layers.Dense(latent_dim)
        
        # Decoder
        self.decoder = keras.Sequential([
            keras.layers.Dense(hidden_dim, activation='relu'),
            keras.layers.Dense(hidden_dim, activation='relu'),
            keras.layers.Dense(input_dim)
        ])
    
    def encode(self, x):
        """Encode input into latent representation."""
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        """Reparameterization trick."""
        std = tf.exp(0.5 * logvar)
        eps = tf.random.normal(shape=std.shape)
        z = mu + eps * std
        return z
    
    def decode(self, z):
        """Decode latent representation into reconstruction."""
        return self.decoder(z)
    
    def call(self, x):
        """Forward pass through the VAE."""
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar


# ## Dataset Preprocessing

# +
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
    records_data = pd.DataFrame(pad_sequences(df['tls.rec'].values, maxlen=RECORD_SEQUENCE_SIZE), columns=tls_columns_names)
    dataset = pd.concat([flow_data, tls_data, records_data], axis=1).fillna(0)
    return dataset

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

def get_processed_data(dataset_path):
    raw_df = load_json_files(glob.glob(dataset_path))
    print(f'dataset shape={raw_df.shape}')
    input_df = extract_features(raw_df)

    pipeline = fit_preprocessor(input_df)
    normal_df = pipeline.transform(input_df)

    print('Normalized row of data:')
    print(normal_df.shape)
    
    return normal_df

def load_data(partition_id, num_partitions):
    """Load and partition data for federated learning with proper row/column orientation."""
    # Get the processed data - shape (14962, 35) where:
    # - 14962 rows = samples
    # - 35 columns = features
    data = get_processed_data(dataset_path)
    
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
    
    # Create data tensors for TensorFlow
    def create_tf_dataset(hf_dataset, batch_size=32, shuffle=False):
        # First, we need to convert the dataset to a format suitable for TensorFlow
        features_list = []
        
        # Process each sample in the dataset
        for sample in hf_dataset:
            # Each sample is a dict with keys like '0', '1', '2', ...
            # Convert to a list of feature values in correct order
            feature_values = [sample[str(i)] for i in range(len(sample))]
            features_list.append(feature_values)
        
        # Convert to NumPy array
        features_array = np.array(features_list, dtype=np.float32)
        
        # Create TensorFlow dataset
        tf_dataset = tf.data.Dataset.from_tensor_slices(features_array)
        
        # Apply shuffling if requested
        if shuffle:
            tf_dataset = tf_dataset.shuffle(buffer_size=len(features_array))
        
        # Batch the dataset
        tf_dataset = tf_dataset.batch(batch_size)
        
        return tf_dataset
    
    # Create TensorFlow datasets
    trainloader = create_tf_dataset(partition_train_test["train"], batch_size=32, shuffle=True)
    testloader = create_tf_dataset(partition_train_test["test"], batch_size=32)
    
    # Verify the output shape
    for batch in trainloader.take(1):
        print(f"Batch shape: {batch.shape}")
        # Should show something like (32, 35) 
        # where 32 is batch_size and 35 is num_features
        break
    
    return trainloader, testloader


# -

# ## Model Training & Testing definition

# +
def train(net, trainloader, epochs, learning_rate, device=None):
    """Train the VAE network on the training set using tabular data."""
    # In TF we don't need the device parameter but keep it for compatibility
    optimizer = keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9)
    
    print(f"Training for {epochs} epochs with learning rate {learning_rate}")
    
    for epoch in range(epochs):
        running_loss = 0.0
        total_batches = 0
        
        for batch in trainloader:
            with tf.GradientTape() as tape:
                # Forward pass through the VAE
                recon_features, mu, logvar = net(batch)
                
                # Compute reconstruction loss
                recon_loss = tf.reduce_mean(tf.reduce_sum(tf.square(recon_features - batch), axis=1))
                
                # Compute KL divergence loss
                kld_loss = -0.5 * tf.reduce_mean(
                    tf.reduce_sum(1 + logvar - tf.square(mu) - tf.exp(logvar), axis=1)
                )
                
                # Total loss (beta-VAE formulation with beta=0.05)
                loss = recon_loss + 0.05 * kld_loss
            
            # Compute gradients and optimize
            gradients = tape.gradient(loss, net.trainable_variables)
            optimizer.apply_gradients(zip(gradients, net.trainable_variables))
            
            # Track statistics
            running_loss += loss.numpy()
            total_batches += 1
        
        # Print epoch statistics
        if (epoch + 1) % 5 == 0 or epoch == 0:  # Print every 5 epochs or first epoch
            avg_loss = running_loss / total_batches
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

def test(net, testloader, device=None):
    """Validate the VAE network on the entire test set and visualize worst reconstructions."""
    total, loss = 0, 0.0
    recon_total, kld_total = 0.0, 0.0
    
    # Lists to store all features, reconstructions and their errors
    all_features = []
    all_reconstructions = []
    all_recon_errors = []
    
    for batch in testloader:
        # Forward pass through the VAE
        recon_features, mu, logvar = net(batch)
        
        # Compute losses
        recon_loss = tf.reduce_mean(tf.reduce_sum(tf.square(recon_features - batch), axis=1))
        kld_loss = -0.5 * tf.reduce_mean(
            tf.reduce_sum(1 + logvar - tf.square(mu) - tf.exp(logvar), axis=1)
        )
        batch_loss = recon_loss + 0.05 * kld_loss
        
        # Track statistics
        batch_size = tf.shape(batch)[0].numpy()
        loss += batch_loss.numpy() * batch_size
        recon_total += recon_loss.numpy() * batch_size
        kld_total += kld_loss.numpy() * batch_size
        total += batch_size
        
        # Calculate per-sample reconstruction error
        sample_recon_errors = tf.reduce_mean(tf.square(recon_features - batch), axis=1)
        
        # Store all samples
        for i in range(batch_size):
            all_features.append(batch[i].numpy())
            all_reconstructions.append(recon_features[i].numpy())
            all_recon_errors.append(sample_recon_errors[i].numpy())
    
    # Convert lists to numpy arrays
    all_recon_errors = np.array(all_recon_errors)
    
    # Get the indices of the 10 worst reconstructed samples
    worst10 = np.argsort(all_recon_errors)[-10:][::-1]
    
    # Calculate average reconstruction error
    avg_recon_error = np.mean(all_recon_errors)
    print(f"Average reconstruction error: {avg_recon_error:.4f}")
    
    avg_loss = loss / total
    avg_recon = recon_total / total
    avg_kld = kld_total / total
    
    print(f"Test Loss: {avg_loss:.4f}, Recon: {avg_recon:.4f}, KLD: {avg_kld:.4f}")
    
    # Visualization
    plt.figure(figsize=(20, 4))
    for i, idx in enumerate(worst10):
        # Get the feature size
        feature_size = all_features[idx].shape[0]
        
        # For features with size 35, we can use 7x5 dimensions
        if feature_size == 35:
            height, width = 7, 5
        else:
            # Try to find reasonable dimensions - prefer width > height
            factors = []
            for j in range(1, int(np.sqrt(feature_size)) + 1):
                if feature_size % j == 0:
                    factors.append((j, feature_size // j))
            
            # Choose dimensions closest to a reasonable aspect ratio
            if factors:
                factors.sort(key=lambda x: abs(x[1]/x[0] - 1.5))
                height, width = factors[0]
            else:
                height, width = 1, feature_size
        
        # Plot original image
        plt.subplot(2, 10, i + 1)
        plt.imshow(all_features[idx].reshape(height, width), cmap='gray')
        plt.title("Original")
        plt.axis('off')
        
        # Plot reconstructed image
        plt.subplot(2, 10, i + 11)
        plt.imshow(all_reconstructions[idx].reshape(height, width), cmap='gray')
        plt.title(f"RE {all_recon_errors[idx]:.3f}")
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('vae_worst_reconstructions.png')
    
    return avg_loss

def generate(net, image):
    """Reproduce the input with trained VAE."""
    return net(image)

def get_weights(net):
    """Get model weights as a list of NumPy arrays."""
    return [w.numpy() for w in net.weights]

def set_weights(net, parameters):
    """Set model weights from a list of NumPy arrays."""
    # Build the model if not already built
    if len(net.weights) == 0:
        # Create a dummy batch to build the model
        input_shape = parameters[0].shape[1]  # Get input shape from first weight matrix
        dummy_input = tf.zeros((1, input_shape))
        _ = net(dummy_input)
    
    # Set weights
    trainable_weights = parameters[:len(net.trainable_weights)]
    non_trainable_weights = parameters[len(net.trainable_weights):]
    
    for i, w in enumerate(trainable_weights):
        net.trainable_weights[i].assign(w)
    
    for i, w in enumerate(non_trainable_weights):
        if i < len(net.non_trainable_weights):
            net.non_trainable_weights[i].assign(w)

def create_run_dir(config: UserConfig) -> tuple:
    """Create a directory where to save results from this run."""
    # Create output directory given current timestamp
    current_time = datetime.now()
    run_dir = current_time.strftime("%Y-%m-%d/%H-%M-%S")
    # Save path is based on the current directory
    save_path = Path.cwd() / f"outputs/{run_dir}"
    os.makedirs(save_path, exist_ok=True)

    # Save run config as json
    with open(f"{save_path}/run_config.json", "w", encoding="utf-8") as fp:
        json.dump(config, fp)

    return save_path, run_dir
