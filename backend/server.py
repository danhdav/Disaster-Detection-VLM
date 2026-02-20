# This file will start the flask server

from flask import Flask
from pymongo import MongoClient
import os

mongo_uri = os.getenv("MONGO_URI")
client = MongoClient(mongo_uri)
db = client["cluster"]

app = Flask(__name__)

try:
    client.admin.command("ping")
    print("Connected to MongoDB successfully!")
except Exception as e:
    print("MongoDB connection failed:", e)

@app.route("/")
def index():
    return "Running Flask server"

app.run(host="0.0.0.0", port=80)