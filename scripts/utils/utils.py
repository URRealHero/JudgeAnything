# Evaluator/utils.py

import json
import os
import time
import enum
import base64
from pydantic import BaseModel
import typing_extensions as typing
import asyncio
from copy import deepcopy
from PIL import Image
from google.genai import types
import numpy as np
import gc
import torch
from .config import FILE_STORAGE, BASE_DIR, RESULT_DIR



def load_json(file_path):
    """
    Load JSON data from a file.
    """
    if not os.path.exists(file_path):
        return {}
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json(data, file_path):
    """
    Save JSON data to a file.
    """
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4)

def upload_media_for_task(client, media_files):
    """
    media_files should be a dict, "input_media", "model1", "model2", "model3", "model4", each will be a list
    upload_file = client.files.upload(file=audio_path) -> just append file_upload to content
    upload_file = client.files.get(name=upload_file.name)
    """
    input_media = media_files["input_media"]
    model1_media = media_files["model1"]
    model2_media = media_files["model2"]
    model3_media = media_files["model3"]
    model4_media = media_files["model4"]
    
    try:
        input_files = upload_media(client,input_media)
        model1_files = upload_media(client,model1_media)
        model2_files = upload_media(client,model2_media)
        model3_files = upload_media(client,model3_media)
        model4_files = upload_media(client,model4_media)
    except Exception as e:
        print(f"Uploading failed, error occur while uploading {e}, Please try again.")
        raise f"Uploading failed, Please try again. Error occur while uploading {e}, Please try again."
    return input_files, model1_files, model2_files, model3_files, model4_files

def upload_media(client, media_files):
    """
    Upload media files to the client, skipping images (Image.Image objects),
    and re-uploading files if needed by checking their states.
    
    Args:
        client: The client instance used for uploading files.
        media_files: A list of media files to upload.

    Returns:
        A list of successfully uploaded files.
    """
    uploaded_files = []
    file_storage = load_json(FILE_STORAGE)
    for media_file in media_files:
        # Skip uploading for images already processed
        if isinstance(media_file, Image.Image):
            uploaded_files.append(media_file)
            continue
        
        # Check if file already exists and is active
        upload_file_name = get_already_exist_files(media_file, file_storage)
        if upload_file_name:
            try:
                upload_file = client.files.get(name=upload_file_name)

                # If the file is active, append it to the list
                if upload_file.state == "ACTIVE":
                    uploaded_files.append(upload_file)
                    # print("Get uploaded")
                    continue
                else:
                    raise Exception(f"{media_file} -> {upload_file.state}")

            except Exception as e:
                # Retry uploading if the file doesn't exist or an error occurred
                upload_file = client.files.upload(file=media_file)
                if checkuploading(client, upload_file):
                    uploaded_files.append(client.files.get(name=upload_file.name))
                else:
                    raise Exception(f"{media_file} -> {upload_file.state}")

        # Upload file if it's not found or needs to be uploaded
        else:
            upload_file = client.files.upload(file=media_file)
            if checkuploading(client, upload_file):
                uploaded_files.append(client.files.get(name=upload_file.name))
            else:
                raise Exception(f"{media_file} -> {upload_file.state}")
        file_storage[media_file] = upload_file.name
        save_json(file_storage, FILE_STORAGE)
    return uploaded_files

def checkuploading(client, upload_file)-> bool:
    # Prepare the file to be uploaded
    while upload_file.state == "PROCESSING":
        print('Waiting for video to be processed.')
        time.sleep(10)
        upload_file = client.files.get(name=upload_file.name)

    if upload_file.state == "FAILED":
        print(f"Uploading failed, State {upload_file.state} Please try again.")
        return False
    print(f'Uploading processing complete: ' + upload_file.uri)
    return True

def get_already_exist_files(file_path, file_storage):
    if file_path in file_storage:
        return file_storage[file_path]
    else:
        return ""
    
    

def get_question_media(entry):
    task = entry.get("task_name")
    input_modality, output_modality = task.split("2")
    input_modality = input_modality.lower()
    image_media = []
    audio_media = []
    video_media = []
    
    if input_modality == "image":
        if isinstance(entry["image_path"],list):
            for image_path in entry["image_path"]:
                img_path = os.path.join(BASE_DIR, image_path)
                image_media.append(img_path)
        else:
            img_path = os.path.join(BASE_DIR, entry["image_path"])
            image_media.append(img_path)
    elif input_modality == "audio":
        audio_path = os.path.join(BASE_DIR, entry["audio_path"])
        audio_media.append(audio_path)
    elif input_modality == "video":
        video_path = os.path.join(BASE_DIR, entry["video_path"])
        video_media.append(video_path)
    elif input_modality == "audiovideo":
        audio_path = os.path.join(BASE_DIR, entry["audio_path"])
        video_path = os.path.join(BASE_DIR, entry["video_path"])
        audio_media.append(audio_path)
        video_media.append(video_path)
        
    return image_media, audio_media, video_media


def get_resp_media(entry):
    img_media = []
    aud_media = []
    vid_media = []
    
    model_resp = entry["response"]
    if model_resp["type"].lower() == "image":
        if isinstance(model_resp["content"],list):
            for img_path in model_resp["content"]:
                img_path = os.path.join(RESULT_DIR, img_path)
                img_media.append(img_path)
        else:
            img_path = os.path.join(RESULT_DIR, model_resp["content"])
            img_media.append(img_path)
    elif model_resp["type"].lower() == "audio":
        audio_path = os.path.join(RESULT_DIR, model_resp["content"])
        aud_media.append(audio_path)
    elif model_resp["type"].lower() == "video":
        video_path = os.path.join(RESULT_DIR, model_resp["content"])
        vid_media.append(video_path)
    return img_media, aud_media, vid_media