# Backend Setup

Clone this repository

Install [uv](https://docs.astral.sh/uv/) based on your OS

Sync the libraries from pyproject.toml:
`uv sync`

Source your .venv
`source .venv/bin/activate`

or if you are using Windows:
`.venv\bin\activate`

IMPORTANT: since local development uses dotenv, request a .env file from me

Run the flask server:
`uv run flask --app server`