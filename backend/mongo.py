from pymongo import MongoClient
import os
from flask import request, jsonify
from bson.objectid import ObjectId

mongo_uri = os.getenv("MONGO_URI")
client = MongoClient(mongo_uri)
db = client.disaster_data
fire_labels_collection = db.fire # Collection to store fire-related labels

try:
    client.admin.command("ping")
    print("Connected to MongoDB successfully!")
except Exception as e:
    print("MongoDB connection failed:", e)

def register_routes(app):
    """Register all routes with the Flask app"""
    
    # each file is structured like this:
    # {
    #     "_id": ObjectID,
    #     "features": {
    #         ...
    #     }
    #     "metadata": {
    #         ...
    #     }
    # }

    @app.route("/fire", methods=["GET"])
    def get_fire_labels():
        """Retrieve all fire labels from the database"""
        try:
            labels = list(fire_labels_collection.find())
            for label in labels:
                label["_id"] = str(label["_id"])
            return jsonify(labels), 200
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    # adds data from the post request body for now
    @app.route("/fire", methods=["POST"])
    def add_fire_label():
        """Add a new fire label to the database"""
        try:
            data = request.get_json()
            if not data:
                return jsonify({"error": "No data provided"}), 400
            result = fire_labels_collection.insert_one(data)
            return jsonify({"_id": str(result.inserted_id), "message": "Label added successfully"}), 201
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    # label/object ID naming convention is still to be decided
    @app.route("/fire/<label_id>", methods=["DELETE"])
    def delete_fire_label(label_id):
        """Delete a fire label by ID"""
        try:
            result = fire_labels_collection.delete_one({"_id": ObjectId(label_id)})
            if result.deleted_count == 0:
                return jsonify({"error": "Label not found"}), 404
            return jsonify({"message": "Label deleted successfully"}), 200
        except Exception as e:
            return jsonify({"error": str(e)}), 500