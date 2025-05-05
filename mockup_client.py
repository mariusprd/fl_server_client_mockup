import flwr as fl
import numpy as np
import time
import random
import logging # Use logging for better output control

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(asctime)s [%(filename)s:%(lineno)d] %(message)s")
log = logging.getLogger(__name__)

# Define the Mockup Client (inherits from NumPyClient)
class MockupClient(fl.client.NumPyClient):
    def __init__(self, client_id):
        self.client_id = client_id
        log.info(f"[Client {self.client_id}] Initialized.")

    def get_parameters(self, config):
        log.info(f"[Client {self.client_id}] get_parameters called. Config: {config}")
        # Return dummy parameters (empty list of numpy arrays)
        # In a real scenario, this would return model weights.
        return []

    def fit(self, parameters, config):
        log.info(f"[Client {self.client_id}] fit called. Config: {config}")
        log.info(f"[Client {self.client_id}] Received {len(parameters)} parameter arrays.")
        # Simulate some work (e.g., model training)
        sleep_duration = random.uniform(1, 3)
        log.info(f"[Client {self.client_id}] Simulating training for {sleep_duration:.2f} seconds...")
        time.sleep(sleep_duration)
        log.info(f"[Client {self.client_id}] fit finished simulation.")

        # *** FIX: Return a non-zero number for num_examples ***
        # We'll return 1 here as a placeholder. In a real client, this
        # should be the actual number of data samples used for training.
        num_examples_mock = 1
        log.info(f"[Client {self.client_id}] Returning num_examples = {num_examples_mock}")

        # Return dummy results: updated parameters (empty), num_examples (now 1), metrics ({})
        return [], num_examples_mock, {"fit_duration": sleep_duration}

    def evaluate(self, parameters, config):
        log.info(f"[Client {self.client_id}] evaluate called. Config: {config}")
        log.info(f"[Client {self.client_id}] Received {len(parameters)} parameter arrays for evaluation.")
        # Simulate some work (e.g., model evaluation)
        sleep_duration = random.uniform(0.5, 2)
        log.info(f"[Client {self.client_id}] Simulating evaluation for {sleep_duration:.2f} seconds...")
        time.sleep(sleep_duration)
        log.info(f"[Client {self.client_id}] evaluate finished simulation.")

        # *** FIX: Also return a non-zero number for evaluate num_examples ***
        # Although the error was in fit, evaluate also uses num_examples.
        num_examples_eval_mock = 1
        log.info(f"[Client {self.client_id}] Returning evaluate num_examples = {num_examples_eval_mock}")

        # Return dummy results: loss (0.0), num_examples (now 1), metrics ({})
        loss = random.uniform(0.5, 1.5) # Simulate some loss
        accuracy = random.uniform(0.7, 0.9) # Simulate some accuracy
        return float(loss), num_examples_eval_mock, {"accuracy": accuracy, "eval_duration": sleep_duration}

# Start the client
if __name__ == "__main__":
    # Assign a unique ID
    cid = random.randint(1, 1000) # Increased range slightly
    # Add client ID to logger format
    # Note: The adapter is useful if you want structured logging, but basic config works too.
    # log_adapter = logging.LoggerAdapter(log, {'client_id': cid}) # Optional adapter

    log.info(f"[{cid}] Creating MockupClient instance.") # Log CID directly
    numpy_client = MockupClient(client_id=cid)

    # Convert NumPyClient to the base Client class using .to_client()
    client = numpy_client.to_client()

    server_address = "127.0.0.1:8080" # Connect to the server on localhost
    log.info(f"[{cid}] Attempting to connect to server at {server_address}...")

    # Use the new recommended start_client function
    try:
        fl.client.start_client(
            server_address=server_address,
            client=client,
        )
    except ConnectionRefusedError:
        log.error(f"[{cid}] Connection refused. Is the server running at {server_address}?")
    except Exception as e:
        log.error(f"[{cid}] Failed to connect or client loop error: {e}", exc_info=True) # Log full traceback

    log.info(f"[{cid}] Connection process finished or failed.")

