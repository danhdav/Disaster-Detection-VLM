# Backend

# uv
Install [uv](https://docs.astral.sh/uv/) for package management.

Uv contains a configuration file to load in the required libraries. Sync the libraries from pyproject.toml with:
`uv sync`

Finally, source your .venv:
`source .venv/bin/activate` or `.venv\bin\activate` if you are using Windows


# AWS
The AWS CLI must be installed on your system. When first starting out, setup using `aws configure` is required.
Enter the Keys as prompted, and skip the region and output sections.

To know the key values, sign into the AWS console. In the navigation bar on the upper right, choose your user name, and then choose Security credentials.
In the Access keys section, you may need to generate new access and secret keys if the current given one does not work.

NOTE: these keys are not hardcoded in the .env file nor in the program.

# Environmental Variables
An example.env is given and should be the outline for what values to include.

Run the flask server:
`uv run python server.py`