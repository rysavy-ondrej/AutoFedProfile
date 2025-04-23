---
tags: [basic, vision, fds]
dataset: [CCCS-CIC-AndMal-2020, CIC-AAGM2017]
framework: [torch, torchvision]
---

# Federated Variational Autoencoder with PyTorch and Flower

This example demonstrates how a variational autoencoder (VAE) can be trained in a federated way using the Flower framework.

## Set up the project

### Clone the project
This will create a new directory called `pytorch-federated-variational-autoencoder` with the following structure:

```shell
pytorch-federated-variational-autoencoder
├── README.md
├── fedvaeexample
│   ├── __init__.py
│   ├── client_app.py   # Defines your ClientApp
│   ├── server_app.py   # Defines your ServerApp
│   └── task.py         # Defines your model, training and data loading
└── pyproject.toml      # Project metadata like dependencies and configs
```

### Setup Environment

* pull & run the environment or build the environment by yourself
```bash
docker run -it --rm --name fl_autoencoder --gpus all --ipc=host -p 8888:8888 -v ./:/workspace allenlin316/fl_autoencoder:latest
```
* Actiate the `virtualenv` which has all the packages required by the flower

```bash
source /usr/local/venv/pytorch-FedAuto-env/bin/activate
```

## Run the Project

You can run your Flower project in both _simulation_ and _deployment_ mode without making changes to the code. If you are starting with Flower, we recommend you using the _simulation_ mode as it requires fewer components to be launched manually. By default, `flwr run` will make use of the Simulation Engine.

### Run with the Simulation Engine

> \[!NOTE\]
> Check the [Simulation Engine documentation](https://flower.ai/docs/framework/how-to-run-simulations.html) to learn more about Flower simulations and how to optimize them.

```bash
flwr run .
```

You can also override some of the settings for your `ClientApp` and `ServerApp` defined in `pyproject.toml`. For example:

```bash
flwr run . --run-config num-server-rounds=5
```

### Run with the Deployment Engine

Follow this [how-to guide](https://flower.ai/docs/framework/how-to-run-flower-with-deployment-engine.html) to run the same app in this example but with Flower's Deployment Engine. After that, you might be intersted in setting up [secure TLS-enabled communications](https://flower.ai/docs/framework/how-to-enable-tls-connections.html) and [SuperNode authentication](https://flower.ai/docs/framework/how-to-authenticate-supernodes.html) in your federation.

If you are already familiar with how the Deployment Engine works, you may want to learn how to run it using Docker. Check out the [Flower with Docker](https://flower.ai/docs/framework/docker/index.html) documentation.
