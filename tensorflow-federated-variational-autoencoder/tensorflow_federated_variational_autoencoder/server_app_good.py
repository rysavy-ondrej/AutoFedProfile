import tensorflow as tf
import pandas as pd
from datasets import Dataset
from tensorflow_federated_variational_autoencoder.task import (
    Net,
    get_processed_data,
    get_weights,
    set_weights,
    test,
)
from tensorflow_federated_variational_autoencoder.strategy import CustomFedAvg
from flwr.common import Context, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig


# +
def on_fit_config(server_round: int):
    """Construct `config` that clients receive when running `fit()`"""
    lr = 0.1
    # Enable a simple form of learning rate decay
    if server_round > 10:
        lr /= 2
    return {"lr": lr}

def gen_evaluate_fn(testloader, device=None):
    """Generate the function for centralized evaluation."""

    def evaluate(server_round, parameters_ndarrays, config):
        """Evaluate global model on centralized test set."""
        net = Net()
        
        # Create a dummy input to initialize the model's weights
        for batch in testloader.take(1):
            input_shape = batch.shape[1]
            break
        
        dummy_input = tf.zeros((1, input_shape))
        _ = net(dummy_input)  # Initialize the model
        
        # Set the weights from parameters
        set_weights(net, parameters_ndarrays)
        
        # Evaluate the model (device parameter is ignored in TensorFlow)
        loss = test(net, testloader)
        
        return loss, {}

    return evaluate

# Define metric aggregation function
def weighted_average(metrics):
    # Multiply loss of each client by number of examples used
    losses = [num_examples * m["loss"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"federated_evaluate_loss": sum(losses) / sum(examples)}

# Custom batch processing for TensorFlow datasets
def create_tf_dataset(data, batch_size=32):
    """Convert data to TensorFlow dataset with proper batching."""
    # Convert NumPy array to TensorFlow dataset
    dataset = tf.data.Dataset.from_tensor_slices(data)
    # Batch the dataset
    dataset = dataset.batch(batch_size)
    return dataset

def get_data(dataset_path):
    """Load and preprocess data for the central server."""
    data = get_processed_data(dataset_path)
    
    # Create DataFrame with default index and column handling
    # This maintains rows as samples and columns as features
    df = pd.DataFrame(data)
    
    # Convert to HuggingFace dataset
    dataset = Dataset.from_pandas(df)
    
    return dataset

def server_fn(context: Context):
    """Construct components for ServerApp."""
    # Read from config
    num_rounds = context.run_config["num-server-rounds"]
    fraction_fit = context.run_config["fraction-fit"]
    fraction_eval = context.run_config["fraction-evaluate"]
    server_device = context.run_config.get("server-device", None)  # Device is optional in TF
    test_dataset = context.run_config.get("test-dataset")
    
    # Initialize model parameters
    model = Net()
    # Create a dummy input to initialize the model
    dummy_input = tf.zeros((1, 54))  # Assuming input_dim=54
    _ = model(dummy_input)
    
    ndarrays = get_weights(model)
    parameters = ndarrays_to_parameters(ndarrays)

    # Prepare dataset for central evaluation
    test_dataset_raw = get_data(test_dataset)
    
    # Convert to TensorFlow dataset
    test_data_list = []
    for sample in test_dataset_raw:
        # Each sample is a dict with keys like '0', '1', '2', ...
        feature_values = [sample[str(i)] for i in range(len(sample))]
        test_data_list.append(feature_values)
    
    # Convert to NumPy array
    test_data_np = tf.convert_to_tensor(test_data_list, dtype=tf.float32)
    
    # Create TensorFlow dataset
    testloader = tf.data.Dataset.from_tensor_slices(test_data_np).batch(32)

    # Define strategy
    strategy = CustomFedAvg(
        run_config=context.run_config,
        use_wandb=context.run_config["use-wandb"],
        fraction_fit=fraction_fit,
        fraction_evaluate=fraction_eval,
        initial_parameters=parameters,
        on_fit_config_fn=on_fit_config,
        evaluate_fn=gen_evaluate_fn(testloader, device=server_device),
        evaluate_metrics_aggregation_fn=weighted_average,
    )
    
    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(strategy=strategy, config=config)

# Create ServerApp
app = ServerApp(server_fn=server_fn)
