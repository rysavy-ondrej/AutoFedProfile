# +
import json
from logging import INFO

import tensorflow as tf
import wandb
from flwr.common import logger, parameters_to_ndarrays
from flwr.common.typing import UserConfig
from flwr.server.strategy import FedAvg

from pathlib import Path
import os
# -

# Define the PROJECT_NAME (missing in original code)
PROJECT_NAME = "tensorflow_federated_variational_autoencoder"


class CustomFedAvg(FedAvg):
    """A class that behaves like FedAvg but has extra functionality.

    This strategy: (1) saves results to the filesystem, (2) saves a
    checkpoint of the global model when a new best is found, (3) logs
    results to W&B if enabled.
    """

    def __init__(self, run_config: UserConfig, use_wandb: bool, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Import Net here to avoid circular imports
        from tensorflow_federated_variational_autoencoder.task import create_run_dir, Net, set_weights
        self.Net = Net
        self.set_weights = set_weights

        # Create a directory where to save results from this run
        self.save_path, self.run_dir = create_run_dir(run_config)
        self.use_wandb = use_wandb
        
        # Initialise W&B if set
        if use_wandb:
            self._init_wandb_project()

        # Keep track of best loss
        self.best_loss_so_far = float('inf')

        # A dictionary to store results as they come
        self.results = {}

    def _init_wandb_project(self):
        # init W&B
        wandb.init(project=PROJECT_NAME, name=f"{str(self.run_dir)}-ServerApp")

    def _store_results(self, tag: str, results_dict):
        """Store results in dictionary, then save as JSON."""
        # Update results dict
        if tag in self.results:
            self.results[tag].append(results_dict)
        else:
            self.results[tag] = [results_dict]

        # Save results to disk.
        # Note we overwrite the same file with each call to this function.
        # While this works, a more sophisticated approach is preferred
        # in situations where the contents to be saved are larger.
        with open(f"{self.save_path}/results.json", "w", encoding="utf-8") as fp:
            json.dump(self.results, fp)

    def _save_model(self, round, loss, parameters):
        """Save model when a new best (lowest) loss is found."""
        logger.log(INFO, "💡 New best global model found with loss: %f", loss)

        # Convert parameters to ndarrays
        ndarrays = parameters_to_ndarrays(parameters)
        
        # Create a new model instance
        model = self.Net()  # Using the imported Net class
        
        # Make a forward pass to build the model (necessary in TensorFlow)
        dummy_input = tf.zeros((1, 54))  # Assuming input_dim=54
        _ = model(dummy_input)
        
        # Set the weights from the parameters
        self.set_weights(model, ndarrays)

        # Save the model using TensorFlow's SavedModel format
        file_name = f"model_loss_{loss:.6f}_round_{round}"
        model_path = str(self.save_path / file_name)
        tf.saved_model.save(model, model_path)
        
        # Also save weights with the correct extension
        model.save_weights(f"{model_path}.weights.h5")

    def store_results_and_log(self, server_round: int, tag: str, results_dict):
        """A helper method that stores results and logs them to W&B if enabled."""
        # Store results
        self._store_results(
            tag=tag,
            results_dict={"round": server_round, **results_dict},
        )

        if self.use_wandb:
            # Log centralized loss and metrics to W&B
            wandb.log(results_dict, step=server_round)

    def evaluate(self, server_round, parameters):
        """Run centralized evaluation if callback was passed to strategy init."""
        if self.evaluate_fn is None:
            return None

        # Wrap the original evaluation function
        parameters_ndarrays = parameters_to_ndarrays(parameters)

        # Call the evaluate_fn but handle both tuple and float returns
        eval_res = self.evaluate_fn(server_round, parameters_ndarrays, {})

        # Handle the case where eval_res is just a float (loss)
        if isinstance(eval_res, (float, int)):
            loss = eval_res
            metrics = {}  # Empty metrics
        else:
            # Handle the case where it's already a tuple
            loss, metrics = eval_res

        # Store and log results
        self.store_results_and_log(
            server_round=server_round,
            tag="centralized_evaluate",
            results_dict={"centralized_loss": loss, **metrics},
        )

        # Adapting model saving to work with loss
        if loss < self.best_loss_so_far:
            self.best_loss_so_far = loss
            self._save_model(server_round, loss, parameters)

        return loss, metrics

    def aggregate_evaluate(self, server_round, results, failures):
        """Aggregate results from federated evaluation."""
        loss, metrics = super().aggregate_evaluate(server_round, results, failures)

        # Store and log
        self.store_results_and_log(
            server_round=server_round,
            tag="federated_evaluate",
            results_dict={"federated_evaluate_loss": loss, **metrics},
        )
        return loss, metrics
