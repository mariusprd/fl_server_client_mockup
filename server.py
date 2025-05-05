import flwr as fl
from flwr.common import ndarrays_to_parameters, Parameters
from flwr.server.strategy import FedAvg
import numpy as np
import logging

# Configure logging for the server
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(asctime)s [Server] %(message)s")
log = logging.getLogger(__name__)

# Define the shape of our simulated model parameters
MODEL_SHAPE = (1000,) # A single array of 1000 elements

# --- Create Initial Parameters ---
# Instead of empty parameters, create a NumPy array with the defined shape.
# This will be sent to clients when they first connect or when the strategy requires it.
log.info(f"Creating initial model parameters with shape {MODEL_SHAPE}")
initial_model_weights = [np.zeros(MODEL_SHAPE, dtype=np.float32)] # Start with zeros
initial_params: Parameters = ndarrays_to_parameters(initial_model_weights)
log.info("Initial parameters created.")

# --- Define Strategy ---
# FedAvg strategy using the initial parameters.
strategy = FedAvg(
    min_fit_clients=2,        # Minimum number of clients for training
    min_evaluate_clients=2,   # Minimum number of clients for evaluation
    min_available_clients=2,  # Minimum number of clients required to start
    initial_parameters=initial_params, # Pass the actual initial parameters
    # You could add server-side evaluation here if needed:
    # evaluate_fn=get_evaluate_fn(),
)
log.info("FedAvg strategy configured.")

# --- Server Configuration ---
config = fl.server.ServerConfig(num_rounds=3) # Run for 3 rounds
log.info(f"Server configured for {config.num_rounds} rounds.")

# --- Start Server ---
log.info("Starting Flower server...")
# Note: start_server is deprecated, but we use it here to match the previous setup.
# In newer Flower versions, you'd use `flwr.server.ServerApp` or the CLI.
fl.server.start_server(
    server_address="0.0.0.0:8080", # Listen on all interfaces, port 8080
    config=config,
    strategy=strategy,
)

log.info("Server finished.")

# Optional: Define a server-side evaluation function if needed
# def get_evaluate_fn():
#     """Return an evaluation function for server-side evaluation."""
#     def evaluate(server_round: int, parameters: Parameters, config):
#         # Implement server-side evaluation logic here
#         # This usually involves a central dataset not available to clients
#         log.info(f"[Server Eval] Round {server_round}: Evaluating parameters...")
#         weights = fl.common.parameters_to_ndarrays(parameters)
#         # Dummy evaluation
#         loss = float(np.mean(np.abs(weights[0]))) # Example metric
#         accuracy = float(1.0 - loss) # Example metric
#         log.info(f"[Server Eval] Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")
#         return loss, {"accuracy": accuracy}
#     return evaluate
