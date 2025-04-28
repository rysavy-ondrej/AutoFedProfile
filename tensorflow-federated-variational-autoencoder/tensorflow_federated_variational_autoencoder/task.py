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
#dataset_path = "/workspace/datasets/cic-aa.normal.tls/*.json"  # replace with the dataset

# ## Model Definition

class Net(keras.Model):
    def __init__(self, input_dim=None, latent_dim=10, hidden_dim=64):
        """
        VAE for tabular data with auto-detection of input dimension.
        
        Args:
            input_dim: Number of input features (can be None and determined on first call)
            latent_dim: Size of the latent space
            hidden_dim: Size of the hidden layers
        """
        super(Net, self).__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self._is_built = False
        
        # Encoder
        self.encoder = keras.Sequential([
            keras.layers.Dense(hidden_dim, activation='relu'),
            keras.layers.Dense(hidden_dim, activation='relu')
        ])
        
        # Latent space
        self.fc_mu = keras.layers.Dense(latent_dim)
        self.fc_logvar = keras.layers.Dense(latent_dim)
        
        # Decoder will be built upon first call when input_dim is known
        if input_dim is not None:
            self.build_decoder(input_dim)
        else:
            self.decoder = None
    
    def build_decoder(self, input_dim):
        """Build the decoder once input dimension is known."""
        if self._is_built and self.input_dim == input_dim:
            # Already built with the same input dimension
            return
            
        print(f"Building decoder with input_dim={input_dim}")
        self.input_dim = input_dim
        self.decoder = keras.Sequential([
            keras.layers.Dense(self.hidden_dim, activation='relu'),
            keras.layers.Dense(self.hidden_dim, activation='relu'),
            keras.layers.Dense(input_dim)
        ])
        self._is_built = True
    
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
        if self.decoder is None:
            raise ValueError("Decoder not built yet - model needs to be called with input data first")
        return self.decoder(z)
    
    def call(self, x):
        """Forward pass through the VAE."""
        # Auto-detect input dimension if not already determined
        if not self._is_built:
            input_dim = x.shape[1]
            print(f"Auto-detected input dimension: {input_dim}")
            self.build_decoder(input_dim)
        
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar

    def build(self, input_shape):
        """
        Custom build method to ensure model is properly constructed.
        This will be called by Keras when using model.build().
        """
        # Input shape will be something like (None, input_dim)
        if len(input_shape) >= 2:
            input_dim = input_shape[1]
            if not self._is_built:
                print(f"Building model with input_shape={input_shape}, input_dim={input_dim}")
                self.build_decoder(input_dim)
        
        # Call the parent build method
        super(Net, self).build(input_shape)


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

    print('Normalized data shape:')
    print(normal_df.shape)
    
    return normal_df

def load_data(partition_id, num_partitions, dataset_path):
    """Load and partition data for federated learning with proper row/column orientation."""
    # Get the processed data
    data = get_processed_data(dataset_path)
    
    # Record the input dimension
    input_dim = data.shape[1]
    print(f"Detected input dimension: {input_dim}")
    
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
    actual_input_dim = None
    for batch in trainloader.take(1):
        print(f"Batch shape: {batch.shape}")
        # Should show something like (32, 35) 
        # where 32 is batch_size and 35 is num_features
        if len(batch.shape) >= 2:
            actual_input_dim = batch.shape[1]
            if actual_input_dim != input_dim:
                print(f"WARNING: Batch input dimension ({actual_input_dim}) doesn't match detected dimension ({input_dim})")
        break
    
    # Use the actual input dimension from the batch if available
    final_input_dim = actual_input_dim if actual_input_dim is not None else input_dim
    print(f"Final input dimension: {final_input_dim}")
    
    return trainloader, testloader, final_input_dim


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
    # First, check if we can infer the input_dim from the parameters
    input_dim = None
    if len(parameters) >= 6:  # We need enough parameters to infer the input_dim
        try:
            # Last layer weights should be (hidden_dim, input_dim)
            input_dim = parameters[-2].shape[1]
            print(f"Inferred input dimension from weights: {input_dim}")
        except (IndexError, AttributeError) as e:
            print(f"Could not infer input dimension from weights: {str(e)}")
    
    # Check if model is already built
    is_built = hasattr(net, '_is_built') and net._is_built and len(net.weights) > 0
    
    if not is_built:
        # Build the model before setting weights
        if input_dim is not None:
            # If we have inferred the input_dim, use it to build the model
            if hasattr(net, 'build_decoder'):
                print(f"Building decoder with inferred input_dim={input_dim}")
                net.build_decoder(input_dim)
            
            # Create a dummy input batch to ensure model is fully built
            dummy_input = tf.zeros((1, input_dim))
            try:
                _ = net(dummy_input)
                print("Successfully built model with dummy input")
            except Exception as e:
                print(f"Error during model building with dummy input: {str(e)}")
                # Try an alternative build method
                try:
                    net.build(input_shape=(None, input_dim))
                    print("Built model using build() method")
                except Exception as e2:
                    print(f"Error during model.build(): {str(e2)}")
        else:
            # If we don't know the input_dim, try with a default value
            # This will be corrected on the first real data batch
            default_dim = 54  # Use a reasonable default
            print(f"Using default input_dim={default_dim} for initial model building")
            dummy_input = tf.zeros((1, default_dim))
            try:
                _ = net(dummy_input)
            except Exception as e:
                print(f"Error during model building with default dimension: {str(e)}")
                # Don't fail - we'll try to set the weights anyway
    
    # Check if model is now built
    if len(net.weights) == 0:
        print("WARNING: Model has no weights after attempted building")
        return
    
    # Now set weights
    print(f"Setting weights: model has {len(net.weights)} weight tensors, received {len(parameters)} parameter arrays")
    
    # First check that we have compatible sizes
    compatible = True
    for i, (model_w, param_w) in enumerate(zip(net.weights, parameters)):
        if i < len(parameters) and model_w.shape != param_w.shape:
            print(f"Weight shape mismatch at index {i}: model={model_w.shape}, parameters={param_w.shape}")
            compatible = False
    
    if not compatible:
        print("WARNING: Weight shapes are incompatible, model might need rebuilding")
        # We could try to rebuild the model here, but for now we'll just set the compatible weights
    
    # Handle the case where we have fewer weights than the model (old model format)
    if len(parameters) < len(net.weights):
        print(f"Fewer parameters ({len(parameters)}) than model weights ({len(net.weights)})")
        for i, w in enumerate(parameters):
            try:
                net.weights[i].assign(w)
            except (ValueError, tf.errors.InvalidArgumentError) as e:
                print(f"Error setting weight at index {i}: {str(e)}")
    else:
        # Normal case - set all weights
        for i, w in enumerate(net.weights):
            if i < len(parameters):
                try:
                    w.assign(parameters[i])
                except (ValueError, tf.errors.InvalidArgumentError) as e:
                    print(f"Error setting weight at index {i}: {str(e)}")
    
    print("Completed weight setting")

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
