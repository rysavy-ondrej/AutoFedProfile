import tensorflow as tf
from tensorflow import keras
import numpy as np
from tensorflow_federated_variational_autoencoder.task import Net, get_weights, load_data, set_weights, test, train
from flwr.client import ClientApp, NumPyClient
from flwr.common import Context


class NetworkProfilingClient(NumPyClient):
    def __init__(self, trainloader, testloader, local_epochs, learning_rate):
        # Create the model
        input_dim = 54  # Your data has 54 features
        self.net = Net(input_dim=input_dim, latent_dim=10, hidden_dim=64)
        self.trainloader = trainloader
        self.testloader = testloader
        self.local_epochs = local_epochs
        self.lr = learning_rate
        
        # Initialize the model with a forward pass
        for batch in self.trainloader.take(1):
            _ = self.net(batch)
            break
    
    def fit(self, parameters, config):
        """Train the model with data of this client."""
        set_weights(self.net, parameters)
        train(
            self.net,
            self.trainloader,
            epochs=self.local_epochs,
            learning_rate=self.lr
        )
        return get_weights(self.net), len(list(self.trainloader)), {}
    
    def evaluate(self, parameters, config):
        """Evaluate the model on the data this client has."""
        set_weights(self.net, parameters)
        loss = test(self.net, self.testloader)
        # Return the loss as part of the metrics dictionary as well
        return float(loss), len(list(self.testloader)), {"loss": float(loss)}



# +
def client_fn(context: Context):
    """Construct a Client that will be run in a ClientApp."""
    # Read the node_config to fetch data partition associated to this node
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    
    # Load data for this partition
    trainloader, testloader = load_data(partition_id, num_partitions)
    
    # Read the run_config to fetch hyperparameters relevant to this run
    local_epochs = context.run_config["local-epochs"]
    learning_rate = context.run_config["learning-rate"]
    
    return NetworkProfilingClient(trainloader, testloader, local_epochs, learning_rate).to_client()

app = ClientApp(client_fn=client_fn)
