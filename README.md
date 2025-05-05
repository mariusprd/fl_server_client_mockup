# Flower server client mockup
In this implementation, we have a flower server and the clients are mockups that only print stuff.

## How to run
Create the venv and install the requirements:
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
Then run the script:
```bash
chmod +x run.sh
./run.sh
```
This will start the flower server and the clients. The clients will connect to the server and start training. You can see the logs in the terminal.
