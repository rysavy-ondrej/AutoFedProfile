"""fedvaeexample: A Flower / PyTorch app for Federated Variational Autoencoder."""

from fedvaeexample.task import Net, get_weights

from flwr.common import Context, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg

import pandas as pd
from datasets import Dataset

import torch
from fedvaeexample.task import (
    Net,
    get_processed_data,
    #apply_eval_transforms,
    get_weights,
    set_weights,
    test,
)
from torch.utils.data import DataLoader
from fedvaeexample.strategy import CustomFedAvg
from flwr.common import Context, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig


def on_fit_config(server_round: int):
    """Construct `config` that clients receive when running `fit()`"""
    lr = 0.1
    # Enable a simple form of learning rate decay
    if server_round > 10:
        lr /= 2
    return {"lr": lr}


def gen_evaluate_fn(
    testloader: DataLoader,
    device: torch.device,
):
    """Generate the function for centralized evaluation."""

    def evaluate(server_round, parameters_ndarrays, config):
        """Evaluate global model on centralized test set."""
        net = Net()
        set_weights(net, parameters_ndarrays)
        net.to(device)
        loss = test(net, testloader, device=device)
        #return loss, {"centralized_accuracy": accuracy}
        return loss, {}

    return evaluate


# Define metric aggregation function
def weighted_average(metrics):
    # Multiply accuracy of each client by number of examples used
    #accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    #examples = [num_examples for num_examples, _ in metrics]
    losses = [num_examples * m["loss"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"federated_evaluate_loss": sum(losses) / sum(examples)}


# +
# def server_fn(context: Context) -> ServerAppComponents:
#     """Construct components for ServerApp."""

#     # Read from config
#     num_rounds = context.run_config["num-server-rounds"]
#     fraction_fit = context.run_config["fraction-fit"]
#     fraction_eval = context.run_config["fraction-evaluate"]
#     server_device = context.run_config["server-device"]

#     # Initialize model parameters
#     ndarrays = get_weights(Net())
#     parameters = ndarrays_to_parameters(ndarrays)

#     # Define the strategy
#     #strategy = FedAvg(initial_parameters=parameters)
#     strategy = CustomFedAvg(
#         run_config=context.run_config,
#         use_wandb=context.run_config["use-wandb"],
#         fraction_fit=fraction_fit,
#         fraction_evaluate=fraction_eval,
#         initial_parameters=parameters,
#         on_fit_config_fn=on_fit_config,
#         evaluate_fn=gen_evaluate_fn(testloader, device=server_device),
#         evaluate_metrics_aggregation_fn=weighted_average,
#     )
#     config = ServerConfig(num_rounds=num_rounds)

#     return ServerAppComponents(strategy=strategy, config=config)

# +
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
    
def get_data(dataset_path):
    data = get_processed_data(dataset_path)
    
    # Create DataFrame with default index and column handling
    # This maintains rows as samples and columns as features
    df = pd.DataFrame(data)
    
    # Convert to HuggingFace dataset
    dataset = Dataset.from_pandas(df)
    
    return dataset


# -

def server_fn(context: Context):
    # Read from config
    num_rounds = context.run_config["num-server-rounds"]
    fraction_fit = context.run_config["fraction-fit"]
    fraction_eval = context.run_config["fraction-evaluate"]
    server_device = context.run_config["server-device"]

    # Initialize model parameters
    ndarrays = get_weights(Net())
    parameters = ndarrays_to_parameters(ndarrays)

    # Prepare dataset for central evaluation

    # This is the exact same dataset as the one donwloaded by the clients via
    # FlowerDatasets. However, we don't use FlowerDatasets for the server since
    # partitioning is not needed.
    # We make use of the "test" split only
    # global_test_set = load_dataset("zalando-datasets/fashion_mnist")["test"]
    test_dataset = get_data("/workspace/datasets/desktop.tls/*.json")
    
    testloader = DataLoader(
        #global_test_set.with_transform(apply_eval_transforms),
        test_dataset,
        batch_size=32,
        collate_fn=custom_collate
    )

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
