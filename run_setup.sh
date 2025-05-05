#!/bin/bash

echo "Starting server in background..."
python server.py &
SERVER_PID=$!

# Give the server more time to start up fully
# Increased sleep duration from 5 to 10 seconds
echo "Waiting 10 seconds for server to initialize..."
sleep 10

echo "Starting client 1 in background..."
python mockup_client.py &
CLIENT_1_PID=$!

# Add a small delay between client starts (optional, but can help)
sleep 1

echo "Starting client 2 in background..."
python mockup_client.py &
CLIENT_2_PID=$!

echo "Server PID: $SERVER_PID"
echo "Client 1 PID: $CLIENT_1_PID"
echo "Client 2 PID: $CLIENT_2_PID"

# Wait for the server process to finish (it will run for num_rounds)
echo "Waiting for server process (PID: $SERVER_PID) to complete..."
wait $SERVER_PID
SERVER_EXIT_CODE=$?
echo "Server finished with exit code $SERVER_EXIT_CODE."

# Optionally kill clients if they are still running (they might exit cleanly)
# echo "Attempting to clean up client processes..."
# kill $CLIENT_1_PID $CLIENT_2_PID 2>/dev/null

echo "All processes should be finished."

