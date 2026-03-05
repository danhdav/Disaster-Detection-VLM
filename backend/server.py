# This file will start the flask server

from flask import Flask
from mongo import register_routes

app = Flask(__name__)

@app.route("/")
def index():
    return "Running Flask server"

register_routes(app)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=80)