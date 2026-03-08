# This file will start the flask server

from flask import Flask
from flask_cors import CORS

from disaster_routes import register_disaster_routes
from mongo import register_routes

app = Flask(__name__)
CORS(app)

@app.route("/")
def index():
    return "Running Flask server"

register_routes(app)
register_disaster_routes(app)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=80)