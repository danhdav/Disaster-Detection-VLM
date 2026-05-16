# Backend

# Package Manager

1) Install [uv](https://docs.astral.sh/uv/) for package management. It contains a configuration file to load in the required libraries. Sync the libraries from pyproject.toml with:

`uv sync`

2) Then source your virtual environment: 
`source .venv/bin/activate`
or 
`.venv\bin\activate`
if you are using Windows

3) Run the backend with Uvicorn (development with reload and specified host/port):
`uv run uvicorn server:app --host 0.0.0.0 --port 8000 --reload`

You can also run the backend with production-oriented Uvicorn settings:
`uv run python run.py`

where `run.py` uses these optional environment variables for production serving:
- `HOST` (default: `0.0.0.0`)
- `PORT` (default: `8000`)
- `UVICORN_WORKERS` (default: `2`)
- `UVICORN_LOG_LEVEL` (default: `info`)
- `UVICORN_ACCESS_LOG` (default: `true`)
- `UVICORN_TIMEOUT_KEEP_ALIVE` (default: `5`)
- `UVICORN_TIMEOUT_GRACEFUL_SHUTDOWN` (default: `30`)

Example production run:
`HOST=0.0.0.0 PORT=8000 UVICORN_WORKERS=4 UVICORN_LOG_LEVEL=info uv run python run.py`

# AWS
The AWS CLI must be installed on your system in order to access the credentials page and obtain your AWS access key and secret key.

1) When first starting out, run `aws configure` to set up your AWS credentials. Enter the Keys as prompted, enter `us-east-2` as the region, and skip the output section.

2) To know the key values, sign into the AWS console. In the navigation bar on the upper right, choose your user name, and then choose Security credentials. In the Access keys section, generate new access and secret keys.  Note that these keys are NOT hardcoded as environmental variables nor in the program.

# Environmental Variables
An [example.env](https://github.com/danhdav/Disaster-Detection-VLM/blob/main/backend/example.env) file is given and should be the outline for what values to include in your local development setup.

# Database

The existing dataset and the ground truth data are stored in the collection belonging to the cluster (see environment variables for the specific names). 

1) Connect and see the dataset using the [Compass app](https://www.mongodb.com/try/download/shell) (if you are on Windows, MacOS, Ubuntu, or RedHat) or via [VS Code](https://www.mongodb.com/try/download/vs-code-extension). Simply enter the connection string (and ensure the username and password within the string are correct) and it should redirect you to the cluster listing.

2) Redis will be used to do local caching. Install [redis](https://redis.io/docs/latest/operate/oss_and_stack/install/archive/install-redis/) and ensure it is properly running with `redis-cli`. You can test your connection with:

```
127.0.0.1:6379> ping
PONG
```

# OpenRouter and Anthropic
The VLM and Chatbot will both be using models from OpenRouter and Anthropic. Token pricing is based on the model used and can be found on [OpenRouter](https://openrouter.ai/pricing) and [Anthropic](https://platform.claude.com/docs/en/about-claude/pricing)'s websites.
