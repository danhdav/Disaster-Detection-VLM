# Backend

# Package Manager

Install [uv](https://docs.astral.sh/uv/) for package management.

Uv contains a configuration file to load in the required libraries. Sync the libraries from pyproject.toml with:
`uv sync`

Finally, source your .venv:
`source .venv/bin/activate` or `.venv\bin\activate` if you are using Windows

Run the backend with Uvicorn (development with reload):
`uv run uvicorn server:app --host 0.0.0.0 --port 8000 --reload`

Run the backend with production-oriented Uvicorn settings:
`uv run python server.py`

`server.py` uses these optional environment variables for production serving:
- `HOST` (default: `0.0.0.0`)
- `PORT` (default: `8000`)
- `UVICORN_WORKERS` (default: `2`)
- `UVICORN_LOG_LEVEL` (default: `info`)
- `UVICORN_ACCESS_LOG` (default: `true`)
- `UVICORN_TIMEOUT_KEEP_ALIVE` (default: `5`)
- `UVICORN_TIMEOUT_GRACEFUL_SHUTDOWN` (default: `30`)

Example production run:
`HOST=0.0.0.0 PORT=8000 UVICORN_WORKERS=4 UVICORN_LOG_LEVEL=info uv run python server.py`

# AWS

The AWS CLI must be installed on your system. When first starting out, setup using `aws configure` is required.
Enter the Keys as prompted, and skip the region and output sections.

To know the key values, sign into the AWS console. In the navigation bar on the upper right, choose your user name, and then choose Security credentials.
In the Access keys section, you may need to generate new access and secret keys if the current given one does not work.

NOTE: these keys are not hardcoded as environmental variables nor in the program.

# Environmental Variables

An example.env is given and should be the outline for what values to include in your local development setup. Railway will handle backend environmental variables during deployment.

# Database

The existing dataset and the ground truth data are stored in the collection belonging to the cluster (see environment variables for the specific names). You can connect and see the dataset using the [Compass app](https://www.mongodb.com/try/download/shell) (if you are on Windows, MacOs, Ubuntu, or RedHat) or my preferred way via [VS Code](https://www.mongodb.com/try/download/vs-code-extension). Simply enter the connection string (and ensure the username and password within the string are correct) and it should redirect you to the cluster listing.

Redis will be used to do local caching. I am still in the process of implementing this. Install redis and ensure it is properly running with `redis-cli`.
You can test your connection with:

```
127.0.0.1:6379> ping
PONG
```

# OpenRouter

The VLM and Chatbot will both be using models from OpenRouter.
