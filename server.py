import flwr as fl
from flwr.common import ndarrays_to_parameters
from flwr.server.strategy import FedAvg
import numpy as np

# Define a simple strategy (FedAvg is standard, requires initial parameters)
# We provide empty initial parameters since clients don't use them yet.
initial_params = ndarrays_to_parameters([])

strategy = FedAvg(
    min_fit_clients=2,        # Minimum number of clients to wait for during training
    min_evaluate_clients=2,   # Minimum number of clients to wait for during evaluation
    min_available_clients=2,  # Minimum number of clients required to start a round
    initial_parameters=initial_params,
)

# Server configuration
config = fl.server.ServerConfig(num_rounds=3) # Run for 3 rounds

# Start the Flower server
print("Starting Flower server...")
fl.server.start_server(
    server_address="0.0.0.0:8080", # Listen on all interfaces, port 8080
    config=config,
    strategy=strategy,
)

print("Server finished.")