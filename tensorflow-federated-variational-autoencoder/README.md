# tensorflow-federated-variational-autoencoder: A Flower / TensorFlow app

## Setup the project 

### Clone the project
* This will create the new directory called `tensorflow-federated-variational-autoencoder` 

```shell
pytorch-federated-variational-autoencoder
├── README.md
├── tensorflow_federated_variational_autoencoder
│   ├── __init__.py
│   ├── client_app.py   # Defines your ClientApp
│   ├── server_app.py   # Defines your ServerApp
│   └── task.py         # Defines your model, training and data loading
|   └── strategy.py     # Defines your server global model performance
└── pyproject.toml      # Project metadata like dependencies and configs
```
### Setup Environment

* pull & run the environment or build the environment by yourself
```bash
docker run -it --rm --name fl_autoencoder --gpus all --ipc=host -p 8888:8888 -v ./:/workspace allenlin316/fl_autoencoder:latest
```
* Actiate the `virtualenv` which has all the packages required by the flower (once inside the docker container)

```bash
source /usr/local/venv/tensorflow-FedAuto-env/bin/activate
```

## Run with the Simulation Engine

In the `tensorflow-federated-variational-autoencoder` directory, use `flwr run` to run a local simulation:

```bash
flwr run .
```

Refer to the [How to Run Simulations](https://flower.ai/docs/framework/how-to-run-simulations.html) guide in the documentation for advice on how to optimize your simulations.

## Run with the Deployment Engine

Follow this [how-to guide](https://flower.ai/docs/framework/how-to-run-flower-with-deployment-engine.html) to run the same app in this example but with Flower's Deployment Engine. After that, you might be interested in setting up [secure TLS-enabled communications](https://flower.ai/docs/framework/how-to-enable-tls-connections.html) and [SuperNode authentication](https://flower.ai/docs/framework/how-to-authenticate-supernodes.html) in your federation.

You can run Flower on Docker too! Check out the [Flower with Docker](https://flower.ai/docs/framework/docker/index.html) documentation.

## Resources

- Flower website: [flower.ai](https://flower.ai/)
- Check the documentation: [flower.ai/docs](https://flower.ai/docs/)
- Give Flower a ⭐️ on GitHub: [GitHub](https://github.com/adap/flower)
- Join the Flower community!
  - [Flower Slack](https://flower.ai/join-slack/)
  - [Flower Discuss](https://discuss.flower.ai/)
