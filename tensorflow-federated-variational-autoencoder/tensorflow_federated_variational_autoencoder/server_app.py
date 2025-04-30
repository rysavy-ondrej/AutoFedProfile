import tensorflow as tf
import pandas as pd
import numpy as np
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
    
    # Get the input dimension from context/global state if available
    # This is a safer approach than trying to import from strategy
    from flwr.server import strategy
    if hasattr(strategy, 'current_context') and strategy.current_context is not None:
        if "input_dim" in strategy.current_context.run_config:
            input_dim = strategy.current_context.run_config["input_dim"]
            print(f"Passing input_dim={input_dim} to clients")
            return {"lr": lr, "input_dim": input_dim}
    
    return {"lr": lr}

def gen_evaluate_fn(testloader, device=None):
    """Generate the function for centralized evaluation."""

    def evaluate(server_round, parameters_ndarrays, config):
        """Evaluate global model on centralized test set."""
        print(f"Starting evaluation for round {server_round}")
        
        # Try to get input dimension from various sources
        input_dim = None
        
        # 1. Try to get from config
        if isinstance(config, dict) and "input_dim" in config:
            input_dim = config["input_dim"]
            print(f"Using input_dim={input_dim} from config")
        
        # 2. Try to infer from parameters
        if input_dim is None and len(parameters_ndarrays) >= 6:
            try:
                # Last layer weights should be (hidden_dim, input_dim)
                input_dim = parameters_ndarrays[-2].shape[1]
                print(f"Inferred input_dim={input_dim} from parameters")
            except (IndexError, AttributeError) as e:
                print(f"Could not infer input dimension from parameters: {str(e)}")
        
        # 3. Try to infer from test data
        if input_dim is None:
            for batch in testloader.take(1):
                if len(batch.shape) >= 2:
                    input_dim = batch.shape[1]
                    print(f"Detected input_dim={input_dim} from test data")
                break
                
        # Create model with input_dim if available
        print(f"Creating model with input_dim={input_dim}")
        net = Net(input_dim=input_dim, latent_dim=10, hidden_dim=64)
        
        # Make sure model is built before setting weights
        if input_dim is not None:
            try:
                print("Building model with explicit dimensions")
                net.build(input_shape=(None, input_dim))
            except Exception as e:
                print(f"Error building model: {str(e)}")
                # Try with a forward pass
                try:
                    dummy_input = tf.zeros((1, input_dim))
                    _ = net(dummy_input)
                    print("Built model with forward pass")
                except Exception as e2:
                    print(f"Error during forward pass: {str(e2)}")
        
        # Set the weights
        print(f"Setting weights ({len(parameters_ndarrays)} arrays)")
        set_weights(net, parameters_ndarrays)
        
        # Verify model is built before evaluation
        if not hasattr(net, '_is_built') or not net._is_built:
            # Model is not built, try one more time with data
            print("Model not built after setting weights, trying with test data")
            for batch in testloader.take(1):
                try:
                    _ = net(batch)
                    print("Successfully built model with test data")
                except Exception as e:
                    print(f"Error building model with test data: {str(e)}")
                    return float('inf'), {"error": "Model building failed"}
                break
        
        # Evaluate the model
        try:
            print("Starting model evaluation")
            loss = test(net, testloader)
            print(f"Evaluation complete, loss: {loss}")
            return loss, {}
        except Exception as e:
            print(f"Error during evaluation: {str(e)}")
            return float('inf'), {"error": str(e)}

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
    
    # Record the input dimension
    input_dim = data.shape[1]
    print(f"Server data has input dimension: {input_dim}")
    
    # Create DataFrame with default index and column handling
    # This maintains rows as samples and columns as features
    df = pd.DataFrame(data)
    
    # Convert to HuggingFace dataset
    dataset = Dataset.from_pandas(df)
    
    return dataset, input_dim  # Return data and input dimension

def server_fn(context: Context):
    """Construct components for ServerApp."""
    # Read from config
    num_rounds = context.run_config["num-server-rounds"]
    fraction_fit = context.run_config["fraction-fit"]
    fraction_eval = context.run_config["fraction-evaluate"]
    server_device = context.run_config.get("server-device", None)  # Device is optional in TF
    dataset_path = context.run_config['dataset-path']
    
    # Prepare dataset for central evaluation and get input dimension
    try:
        # Try to get both dataset and input dimension
        test_dataset_raw, input_dim = get_data(dataset_path)
        print(f"Server detected input dimension: {input_dim}")
    except ValueError:
        # Handle case where get_data doesn't return input_dim
        test_dataset_raw = get_data(test_dataset)
        input_dim = None
        print("Server couldn't detect input dimension from get_data")
    
    # Convert raw dataset to TensorFlow format
    test_data_list = []
    for sample in test_dataset_raw:
        # Each sample is a dict with keys like '0', '1', '2', ...
        feature_values = [sample[str(i)] for i in range(len(sample))]
        test_data_list.append(feature_values)
    
    # Convert to NumPy array
    test_data_np = tf.convert_to_tensor(test_data_list, dtype=tf.float32)
    
    # Create TensorFlow dataset
    testloader = tf.data.Dataset.from_tensor_slices(test_data_np).batch(32)
    
    # Detect input dimension from batch if not already known
    if input_dim is None:
        for batch in testloader.take(1):
            if len(batch.shape) >= 2:
                input_dim = batch.shape[1]
                print(f"Server detected input dimension from batch: {input_dim}")
            break
    
    # Initialize model parameters with detected input dimension
    print(f"Initializing model with input_dim={input_dim}")
    model = Net(input_dim=input_dim, latent_dim=10, hidden_dim=64)
    
    # Initialize the model with a forward pass to build it
    try:
        for batch in testloader.take(1):
            print(f"Building model with batch shape: {batch.shape}")
            _ = model(batch)  # This will trigger the auto-detection if needed
            print("Successfully built model with forward pass")
            break
    except Exception as e:
        print(f"Error during model initialization: {str(e)}")
        # Try an alternative initialization method
        if input_dim is not None:
            try:
                print(f"Manually building model with input_dim={input_dim}")
                model.build(input_shape=(None, input_dim))
                print("Successfully built model with build() method")
            except Exception as e2:
                print(f"Error during manual model building: {str(e2)}")
    
    # Get model weights
    try:
        ndarrays = get_weights(model)
        parameters = ndarrays_to_parameters(ndarrays)
        print(f"Successfully got model weights: {len(ndarrays)} arrays")
    except Exception as e:
        print(f"Error getting model weights: {str(e)}")
        # Create dummy parameters if we couldn't get real ones
        if input_dim is not None:
            print("Creating dummy parameters")
            # Create simple random weights with correct shapes
            hidden_dim = 64
            latent_dim = 10
            
            # Create dummy model and get weight shapes
            dummy_model = Net(input_dim=input_dim, latent_dim=latent_dim, hidden_dim=hidden_dim)
            
            # Build the model explicitly
            try:
                dummy_model.build(input_shape=(None, input_dim))
                # Get weights with proper shapes even if not initialized
                dummy_weights = get_weights(dummy_model)
                # Replace weights with random normal values
                ndarrays = [np.random.normal(0, 0.01, w.shape) for w in dummy_weights]
                print(f"Created dummy weights with proper shapes: {[w.shape for w in ndarrays]}")
            except Exception as e:
                print(f"Error building dummy model: {str(e)}")
                # Fallback to manual creation if that fails
                ndarrays = [
                    # Encoder weights
                    np.random.normal(0, 0.01, (input_dim, hidden_dim)),  # w1
                    np.zeros(hidden_dim),                                # b1
                    np.random.normal(0, 0.01, (hidden_dim, hidden_dim)), # w2
                    np.zeros(hidden_dim),                                # b2
                    
                    # Latent space weights
                    np.random.normal(0, 0.01, (hidden_dim, latent_dim)), # w_mu
                    np.zeros(latent_dim),                                # b_mu
                    np.random.normal(0, 0.01, (hidden_dim, latent_dim)), # w_logvar
                    np.zeros(latent_dim),                                # b_logvar
                    
                    # Decoder weights
                    np.random.normal(0, 0.01, (latent_dim, hidden_dim)), # w3
                    np.zeros(hidden_dim),                                # b3
                    np.random.normal(0, 0.01, (hidden_dim, hidden_dim)), # w4
                    np.zeros(hidden_dim),                                # b4
                    np.random.normal(0, 0.01, (hidden_dim, input_dim)),  # w5
                    np.zeros(input_dim)                                  # b5
                ]
                print(f"Created manual dummy weights: {len(ndarrays)} arrays")
            
            parameters = ndarrays_to_parameters(ndarrays)
        else:
            raise ValueError("Cannot initialize model without input dimension")

    # Define strategy - pass input_dim as part of the config
    strategy = CustomFedAvg(
        run_config={**context.run_config, "input_dim": input_dim},
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


# -

# Create ServerApp
app = ServerApp(server_fn=server_fn)
