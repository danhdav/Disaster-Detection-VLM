from pymongo import MongoClient
import os
import json
import boto3
from fastapi import FastAPI, HTTPException
from bson.objectid import ObjectId
from pathlib import Path
from functools import cache
from typing import List, Dict, Any

'''
This file contains API endpoints for the dataset (i.e interacting with MongoDB and AWS S3)
To view the documentation UI, visit /docs
'''

mongo_uri = os.getenv("MONGO_URI")
client = MongoClient(mongo_uri)
db = client[os.getenv("MONGO_DB_NAME")]
fire_labels_collection = db[os.getenv("MONGO_COLLECTION_NAME")]

bucket_name = os.getenv("S3_BUCKET_NAME")
s3 = boto3.client('s3')

def test_mongodb_connection():
    """Test the MongoDB connection"""
    try:
        client.admin.command("ping")
        print("Connected to MongoDB successfully!")
        return True
    except Exception as e:
        print("MongoDB connection failed:", e)
        return False
    
def test_s3_connection():
    """Test the AWS S3 connection"""
    try:
        s3.head_bucket(Bucket=bucket_name)
        print("Connected to AWS S3 successfully!")
        return True
    except Exception as e:
        print("AWS S3 connection failed:", e)
        return False


app = FastAPI()

# Each file in the MongoDB cluster is structured like this:
# {
#     "_id": ObjectID,
#     "features": {
#         ...
#     }
#     "metadata": {
#         "img_name": "example.png",
#         ...
#     }
# }

@app.get("/fire", response_model=List[Dict[str, Any]])
async def get_fire_labels():
    """Retrieve all fire labels from the database"""
    try:
        labels = list(fire_labels_collection.find())
        for label in labels:
            label["_id"] = str(label["_id"])
        return labels
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Get a specific label by image name
@app.get("/fire/search/{img_name}")
async def search_fire_label(img_name: str):
    """Search for a fire label by image name"""
    try:
        label = fire_labels_collection.find_one({
            "metadata.img_name": img_name + ".png"
        })
        
        if label:
            return {"_id": str(label["_id"])}
        else:
            raise HTTPException(status_code=404, detail="Label not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Add a new label
@app.post("/fire")
async def add_fire_label(data: Dict[str, Any]):
    """Add a new fire label to the database"""
    try:
        if not data:
            raise HTTPException(status_code=400, detail="No data provided")
        result = fire_labels_collection.insert_one(data)
        return {"_id": str(result.inserted_id), "message": "Label added successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Delete a label by ID
@app.delete("/fire/{label_id}")
async def delete_fire_label(label_id: str):
    """Delete a fire label by ID"""
    try:
        result = fire_labels_collection.delete_one({"_id": ObjectId(label_id)})
        if result.deleted_count == 0:
            raise HTTPException(status_code=404, detail="Label not found")
        return {"message": "Label deleted successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

# Fetch an image URL set from S3 (pre and post)
# For context, image nomenclature is as follows:
# {disaster-name}_{id}_{pre|post}_disaster[_target].{png|json}
# an example disaster name with id is "guatemala-volcano_00000003"
# we need to get the pre and post images such as 
# "guatemala-volcano_00000003_pre_disaster.png" and "guatemala-volcano_00000003_post_disaster.png"

@app.get("/image/{disaster_name}")
async def get_image_urls(disaster_name: str):
    try:
        print("Searching for images with disaster name:", disaster_name)
        image_directory = s3.list_objects_v2(Bucket=bucket_name, Prefix='xview2-test-data/images/')

        if 'Contents' not in image_directory:
            raise HTTPException(status_code=404, detail="No files found in directory")
        
        image_files = [obj['Key'] for obj in image_directory['Contents']]
        
        urls = {}
        
        for pre_or_post in ["pre", "post"]:
            image_url = f"{disaster_name}_{pre_or_post}_disaster.png"
            
            if image_url in image_files:
                url = s3.generate_presigned_url('get_object', Params={'Bucket': bucket_name, 'Key': f'xview2-test-data/images/{image_url}'}, ExpiresIn=3600)
                urls[f"{pre_or_post}_image_url"] = url
                print(f"{pre_or_post.capitalize()}-disaster image URL:", url)
            else:
                raise HTTPException(status_code=404, detail=f"No {pre_or_post}-disaster image found for {disaster_name}")
        
        return urls
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.get("/debug/health")
async def check_disasters():
    test_mongodb_connection()
    test_s3_connection()