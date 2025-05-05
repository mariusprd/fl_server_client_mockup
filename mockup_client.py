import flwr as fl
import numpy as np
import time
import random
import logging
from typing import List, Tuple, Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(asctime)s [%(filename)s:%(lineno)d] %(message)s")
log = logging.getLogger(__name__)

# Define the shape of our simulated model parameters (must match server)
MODEL_SHAPE = (1000,)

# --- Training Simulation Helper Function ---
def simulate_training(weights: List[np.ndarray], client_id: int, config: Dict[str, Any]) -> Tuple[List[np.ndarray], float]:
    """
    Simulates the training process on a client.

    Args:
        weights: The current model weights (list of NumPy arrays).
        client_id: The ID of the client for logging.
        config: Configuration dictionary received from the server (unused here).

    Returns:
        A tuple containing:
        - The updated model weights after simulated training.
        - The simulated training duration.
    """
    log.info(f"[{client_id}] Simulating training...") # Log client ID directly
    if not weights:
        log.warning(f"[{client_id}] Received empty weights for training.")
        return [], 0.0

    # Simulate making changes to the weights
    # Example: Add small random noise to simulate gradient updates
    updated_weights = [w.copy() for w in weights] # Create a copy to modify
    noise_scale = 0.01 # Controls the magnitude of simulated changes
    for i in range(len(updated_weights)):
        # Ensure the weight array is not empty before generating noise
        if updated_weights[i].size > 0:
            noise = (np.random.rand(*updated_weights[i].shape) - 0.5) * noise_scale
            updated_weights[i] += noise.astype(updated_weights[i].dtype)
        else:
            log.warning(f"[{client_id}] Weight array at index {i} is empty, skipping noise addition.")


    # Simulate training time
    sleep_duration = random.uniform(0.5, 1.5)
    time.sleep(sleep_duration)

    log.info(f"[{client_id}] Training simulation finished after {sleep_duration:.2f}s.")
    return updated_weights, sleep_duration

# --- Define the Mockup Client ---
class MockupClient(fl.client.NumPyClient):
    def __init__(self, client_id):
        self.client_id = client_id
        # Initialize the "model" with random weights
        # Use the defined shape
        self.model_weights: List[np.ndarray] = [np.random.rand(*MODEL_SHAPE).astype(np.float32)]
        log.info(f"[{self.client_id}] Initialized with model weights shape: {[w.shape for w in self.model_weights]}")

    def get_parameters(self, config: Dict[str, Any]) -> List[np.ndarray]:
        """Return the current local model weights."""
        log.info(f"[{self.client_id}] get_parameters called. Config: {config}")
        log.info(f"[{self.client_id}] Returning {len(self.model_weights)} parameter arrays.")
        return self.model_weights

    def fit(self, parameters: List[np.ndarray], config: Dict[str, Any]) -> Tuple[List[np.ndarray], int, Dict]:
        """
        Simulate training the model.

        1. Receives global model parameters from the server.
        2. Updates local model (e.g., by simple averaging - optional).
        3. Simulates training on local data using a helper function.
        4. Returns the new local model parameters and training metrics.
        """
        log.info(f"[{self.client_id}] fit called. Config: {config}")
        log.info(f"[{self.client_id}] Received {len(parameters)} parameter arrays from server.")

        # --- Update local model with received parameters (Optional Step) ---
        # Example: Simple average between local and received weights if shapes match
        if len(parameters) == len(self.model_weights) and all(p.shape == w.shape for p, w in zip(parameters, self.model_weights)):
            log.info(f"[{self.client_id}] Averaging received parameters with local model.")
            for i in range(len(self.model_weights)):
                 # Check if arrays are non-empty before averaging
                 if self.model_weights[i].size > 0 and parameters[i].size > 0:
                    self.model_weights[i] = (self.model_weights[i] + parameters[i]) / 2.0
                 else:
                    log.warning(f"[{self.client_id}] Cannot average empty array at index {i}.")
        else:
            log.warning(f"[{self.client_id}] Received parameters shape/count mismatch or empty. Keeping local model for training simulation.")
            # Or could replace local model if shapes match:
            # if len(parameters) == len(self.model_weights) and all(p.shape == w.shape for p, w in zip(parameters, self.model_weights)):
            #     self.model_weights = parameters

        # --- Simulate Local Training ---
        # Pass the potentially updated local weights to the simulation function
        new_weights, fit_duration = simulate_training(self.model_weights, self.client_id, config)

        # Update the client's model with the result of the training simulation
        self.model_weights = new_weights

        # --- Prepare and Return Results ---
        num_examples_fit = random.randint(50, 150) # Simulate number of examples used
        metrics = {"fit_duration": fit_duration}
        log.info(f"[{self.client_id}] Returning {len(self.model_weights)} updated parameter arrays, num_examples={num_examples_fit}.")

        return self.model_weights, num_examples_fit, metrics

    def evaluate(self, parameters: List[np.ndarray], config: Dict[str, Any]) -> Tuple[float, int, Dict]:
        """
        Simulate evaluating the received model parameters.

        Args:
            parameters: The model parameters received from the server for evaluation.
            config: Configuration dictionary.

        Returns:
            A tuple containing: loss, number of examples, metrics dictionary.
        """
        log.info(f"[{self.client_id}] evaluate called. Config: {config}")
        log.info(f"[{self.client_id}] Received {len(parameters)} parameter arrays for evaluation.")

        if not parameters:
             log.warning(f"[{self.client_id}] Received empty parameters for evaluation.")
             return 1.0, 0, {"accuracy": 0.0} # Return default failure values

        # --- Simulate Evaluation Logic ---
        # Example: Calculate a dummy loss based on the received weights
        # (e.g., mean absolute value of the first layer)
        loss = 1.0 # Default loss
        try:
            # Ensure parameters is a list and has elements before accessing
            if isinstance(parameters, list) and len(parameters) > 0 and parameters[0].size > 0:
                 loss = float(np.mean(np.abs(parameters[0])))
            else:
                 log.warning(f"[{self.client_id}] Received parameters in unexpected format or empty for evaluation: {type(parameters)}")
        except Exception as e:
            log.error(f"[{self.client_id}] Error calculating dummy loss: {e}")

        accuracy = max(0.0, 1.0 - loss * random.uniform(0.8, 1.2)) # Simulate accuracy inversely related to loss
        num_examples_eval = random.randint(10, 50) # Simulate evaluation data size
        metrics = {"accuracy": accuracy, "calculated_loss": loss}
        log.info(f"[{self.client_id}] Evaluation simulation finished. Loss={loss:.4f}, Accuracy={accuracy:.4f}, NumExamples={num_examples_eval}")

        # Return non-zero num_examples to avoid potential division by zero in evaluate aggregation
        if num_examples_eval == 0:
             num_examples_eval = 1

        return float(loss), num_examples_eval, metrics

# --- Start the Client ---
if __name__ == "__main__":
    # Assign a unique ID
    cid = random.randint(1, 1000)
    log.info(f"[{cid}] Creating MockupClient instance.")
    numpy_client = MockupClient(client_id=cid)

    # Convert NumPyClient to the base Client class using .to_client()
    client = numpy_client.to_client()

    server_address = "127.0.0.1:8080"
    log.info(f"[{cid}] Attempting to connect to server at {server_address}...")

    # Use the recommended start_client function
    try:
        fl.client.start_client(
            server_address=server_address,
            client=client,
        )
    except ConnectionRefusedError:
        log.error(f"[{cid}] Connection refused. Is the server running at {server_address}?")
    except Exception as e:
        log.error(f"[{cid}] Failed to connect or client loop error: {e}", exc_info=True)

    log.info(f"[{cid}] Connection process finished or failed.")

