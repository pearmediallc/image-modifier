from flask import Flask, request, render_template, send_file, jsonify, session, after_this_request
import os
from werkzeug.utils import secure_filename
from PIL import Image, ImageDraw
import cv2
import random
import subprocess
import numpy as np
import zipfile
import io
import uuid
import tempfile
import shutil
import time
import logging
import threading
from threading import Lock
from logging.handlers import RotatingFileHandler
import redis
from pillow_heif import register_heif_opener

# Initialize Flask app first
app = Flask(__name__)
app.secret_key = 'f8c43gj7cj3i4tyusoh3'

# Set up Flask logger
if not app.debug:
    gunicorn_logger = logging.getLogger("gunicorn.error")
    app.logger.handlers = gunicorn_logger.handlers
    app.logger.setLevel(gunicorn_logger.level)

# Update Redis connection to use environment variables with fallback and error handling
def get_redis_client():
    try:
        redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379')
        client = redis.from_url(redis_url, decode_responses=True)
        # Test connection
        client.ping()
        app.logger.info("Successfully connected to Redis")
        return client
    except redis.ConnectionError:
        app.logger.warning("Could not connect to Redis, falling back to in-memory storage")
        return None
    except Exception as e:
        app.logger.error(f"Redis error: {str(e)}")
        return None

# Add fallback for when Redis is not available
class FallbackStorage:
    def __init__(self):
        self.storage = {}
    
    def hset(self, key, field, value):
        if key not in self.storage:
            self.storage[key] = {}
        self.storage[key][field] = value
        return 1
    
    def hget(self, key, field):
        return self.storage.get(key, {}).get(field)
    
    def hincrby(self, key, field, amount=1):
        if key not in self.storage:
            self.storage[key] = {}
        if field not in self.storage[key]:
            self.storage[key][field] = 0
        self.storage[key][field] += amount
        return self.storage[key][field]
    
    def hgetall(self, key):
        return self.storage.get(key, {})

# Initialize Redis client and storage
redis_client = get_redis_client()
storage = redis_client if redis_client else FallbackStorage()

# Register HEIF opener
register_heif_opener()

# Configure folders
UPLOAD_FOLDER = os.getenv('UPLOAD_FOLDER', 'uploads')
OUTPUT_FOLDER = os.getenv('OUTPUT_FOLDER', 'outputs')

# Ensure folders exist with proper permissions
try:
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    # Test write permissions
    test_file = os.path.join(UPLOAD_FOLDER, '.test')
    with open(test_file, 'w') as f:
        f.write('test')
    os.remove(test_file)
except Exception as e:
    app.logger.error(f"Failed to initialize folders: {str(e)}")
    # Use temporary directory as fallback
    UPLOAD_FOLDER = tempfile.mkdtemp()
    OUTPUT_FOLDER = tempfile.mkdtemp()
    app.logger.info(f"Using temporary folders: {UPLOAD_FOLDER}, {OUTPUT_FOLDER}")

# Update FFmpeg path for Render.com
FFMPEG_PATH = os.getenv('FFMPEG_PATH', '/usr/bin/ffmpeg')

# Initialize job tracking
job_status = {}
job_folders = {}

@app.route('/')
def index():
    return render_template('index.html')

@app.before_request
def assign_session_id():
    if 'session_id' not in session:
        session['session_id'] = str(uuid.uuid4())

@app.after_request
def add_no_cache_headers(response):
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, post-check=0, pre-check=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response

@app.route('/clear', methods=['POST'])
def clear_files():

    # Get the session ID and session directories
    session_id = session.get('session_id')
    if not session_id:
        return jsonify({"error": "No active session"}), 400

    session_upload_folder = os.path.join(UPLOAD_FOLDER, session_id)
    session_output_folder = os.path.join(OUTPUT_FOLDER, session_id)

    # Attempt to delete the session directories
    try:
        if os.path.exists(session_upload_folder):
            shutil.rmtree(session_upload_folder)
        if os.path.exists(session_output_folder):
            shutil.rmtree(session_output_folder)
        return jsonify({"success": "Files cleared successfully"}), 200
    except Exception as e:
        return jsonify({"error": f"Failed to clear files: {e}"}), 500

@app.route('/upload', methods=['POST'])
def upload_files():
    try:
        # Generate unique job_id
        job_id = str(uuid.uuid4())
        app.logger.info(f"Starting upload process with job_id: {job_id}")

        # Retrieve the number of variations
        try:
            variations = int(request.form.get('variations', 1))
            app.logger.info(f"Number of variations requested: {variations}")
        except ValueError:
            app.logger.error("Invalid variations value provided")
            return jsonify({"error": "Invalid variations value"}), 400

        # Create session-specific temporary directories
        session_upload_folder = tempfile.mkdtemp(dir=UPLOAD_FOLDER)
        session_output_folder = tempfile.mkdtemp(dir=OUTPUT_FOLDER)
        app.logger.info(f"Created temp folders - Upload: {session_upload_folder}, Output: {session_output_folder}")

        # Save folder paths in Redis
        storage.hset(f"job:{job_id}", "upload_folder", session_upload_folder)
        storage.hset(f"job:{job_id}", "output_folder", session_output_folder)

        if 'files[]' not in request.files:
            app.logger.error("No files found in request")
            return jsonify({"error": "No files uploaded"}), 400

        files = request.files.getlist('files[]')
        if not files:
            app.logger.error("Empty files list received")
            return jsonify({"error": "No selected files"}), 400

        app.logger.info(f"Received {len(files)} files for processing")

        session_id = session.get('session_id')
        if not session_id:
            app.logger.error("No active session found")
            return jsonify({"error": "No active session"}), 400

        # Store job data in Redis
        total_files = len(files) * variations
        storage.hset(f"job:{job_id}", "total_files", total_files)
        storage.hset(f"job:{job_id}", "completed_files", 0)
        storage.hset(f"job:{job_id}", "status", "pending")
        storage.hset(f"job:{job_id}", "progress", 0)

        # Process each file
        for file in files:
            filename = secure_filename(file.filename)
            file_path = os.path.join(session_upload_folder, filename)
            app.logger.info(f"Saving file: {filename} to {file_path}")
            file.save(file_path)

            for i in range(variations):
                variation_output_path = os.path.join(
                    session_output_folder, f"processed_{i+1}_{filename}"
                )
                
                if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.heic', '.webp')):
                    app.logger.info(f"Starting image processing for {filename}, variation {i+1}")
                    threading.Thread(
                        target=modify_image,
                        args=(file_path, variation_output_path, job_id, i),
                        daemon=True
                    ).start()
                elif filename.lower().endswith(('.mp4', '.avi', '.mov')):
                    app.logger.info(f"Starting video processing for {filename}, variation {i+1}")
                    threading.Thread(
                        target=modify_video,
                        args=(file_path, variation_output_path, job_id, i),
                        daemon=True
                    ).start()

        app.logger.info(f"Upload process completed for job {job_id}, processing started")
        return jsonify({"job_id": job_id})

    except Exception as e:
        app.logger.error(f"Upload process failed: {str(e)}", exc_info=True)
        return jsonify({"error": f"Upload failed: {str(e)}"}), 500


@app.route('/progress/<job_id>', methods=['GET'])
def get_progress(job_id):

    job_data = storage.hgetall(f"job:{job_id}")

    if not job_data:
        app.logger.error(f"[PROGRESS] Job ID {job_id} not found.")
        return jsonify({"error": "Invalid job ID"}), 404

    return jsonify({
        "status": job_data.get("status"),
        "progress": int(job_data.get("progress", 0))
    })

@app.route('/download/<job_id>', methods=['GET'])
def download_files(job_id):
    job_data = storage.hgetall(f"job:{job_id}")
    if not job_data:
        app.logger.error(f"[DOWNLOAD] Job ID {job_id} not found.")
        return jsonify({"error": "Invalid job ID"}), 404

    session_output_folder = job_data.get("output_folder")
    session_upload_folder = job_data.get("upload_folder")

    if not session_output_folder or not os.path.exists(session_output_folder):
        app.logger.error(f"[DOWNLOAD] Output folder for Job ID {job_id} does not exist.")
        return jsonify({"error": "Output folder not found"}), 404

    processed_files = [
        os.path.join(session_output_folder, f)
        for f in os.listdir(session_output_folder)
        if f.startswith("processed_")
    ]
    if not processed_files:
        app.logger.error(f"[DOWNLOAD] No processed files found for Job ID {job_id}.")
        return jsonify({"error": "Processed files not found"}), 404

    zip_file_path = os.path.join(session_output_folder, f"processed_files_{job_id}.zip")
    if not os.path.exists(zip_file_path):
        with zipfile.ZipFile(zip_file_path, 'w') as zipf:
            for file in processed_files:
                zipf.write(file, os.path.basename(file))

    @after_this_request
    def remove_files(response):
        """Delete uploaded and output files after the ZIP file is sent."""
        try:
            # Remove output folder
            if session_output_folder and os.path.exists(session_output_folder):
                shutil.rmtree(session_output_folder)
                app.logger.info(f"Cleaned up output folder for job {job_id}")

            # Remove upload folder
            if session_upload_folder and os.path.exists(session_upload_folder):
                shutil.rmtree(session_upload_folder)
                app.logger.info(f"Cleaned up upload folder for job {job_id}")
        except Exception as e:
            app.logger.error(f"Failed to clean up files for job {job_id}: {e}")
        return response

    return send_file(zip_file_path, as_attachment=True)




def safe_remove(file_path):
    """Safely remove a file, ignoring errors if the file doesn't exist."""
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
    except Exception as e:
        app.logger.error(f"Error removing file {file_path}: {e}")



# Function to add an invisible mesh overlay to images or frames
def add_invisible_mesh(image):
    if image.mode != "RGBA":
        image = image.convert("RGBA")
    
    width, height = image.size
    step = 10  # Mesh grid spacing

    # Create a transparent overlay
    overlay = Image.new("RGBA", (width, height), (255, 255, 255, 0))
    draw = ImageDraw.Draw(overlay)

    # Draw transparent points
    for x in range(0, width, step):
        for y in range(0, height, step):
            draw.point((x, y), fill=(0, 0, 0, 0))  # Fully transparent

    # Combine the overlay with the original image
    combined_image = Image.alpha_composite(image, overlay)
    return combined_image

def update_job_progress(job_id):
    try:
        completed_files = int(storage.hget(f"job:{job_id}", "completed_files") or 0)
        total_files = int(storage.hget(f"job:{job_id}", "total_files") or 1)
        
        progress = int((completed_files / total_files) * 100)
        app.logger.info(f"Job {job_id}: Progress {progress}% ({completed_files}/{total_files} files)")
        
        storage.hset(f"job:{job_id}", "progress", progress)

        if completed_files >= total_files:
            app.logger.info(f"Job {job_id} completed")
            storage.hset(f"job:{job_id}", "status", "completed")
    except Exception as e:
        app.logger.error(f"Error updating progress for job {job_id}: {str(e)}", exc_info=True)



# Function to modify images
def modify_image(input_path, output_path, job_id, variation):
    try:
        app.logger.info(f"Starting image modification for job {job_id}, variation {variation}")
        storage.hset(f"job:{job_id}", "status", "processing")
        
        image = Image.open(input_path)
        app.logger.info(f"Image opened successfully: {input_path}")

        if image.mode not in ["RGB", "RGBA"]:
            image = image.convert("RGB")

        # Apply modifications
        random.seed(variation)
        adjustment = random.uniform(-0.1, 0.1)
        
        image_array = np.array(image, dtype=np.float32)
        image_array = np.clip(image_array * (1 + adjustment), 0, 255).astype(np.uint8)
        modified_image = Image.fromarray(image_array)
        
        # Add invisible mesh overlay
        modified_image = add_invisible_mesh(modified_image)
        
        # Save the modified image
        if input_path.lower().endswith(('.heic', '.webp')):
            output_path = os.path.splitext(output_path)[0] + ".png"
        
        modified_image.save(output_path, format="PNG")
        app.logger.info(f"Successfully saved modified image: {output_path}")

        storage.hincrby(f"job:{job_id}", "completed_files", 1)
        update_job_progress(job_id)

    except Exception as e:
        app.logger.error(f"Error in image modification for job {job_id}: {str(e)}", exc_info=True)
        storage.hset(f"job:{job_id}", "status", f"error: {str(e)}")


# Function to modify videos
def modify_video(input_path, output_path, job_id, variation):
    try:
        # Unique temporary paths for audio and video processing
        temp_audio_path = os.path.join(OUTPUT_FOLDER, f"{job_id}_{variation}_{os.path.basename(input_path)}_temp_audio.mp3")
        temp_video_path = os.path.join(OUTPUT_FOLDER, f"{job_id}_{variation}_{os.path.basename(input_path)}_temp_video.mp4")

        storage.hset(f"job:{job_id}", "status", "processing")

        # Extract audio
        audio_extraction_command = [
            FFMPEG_PATH, "-i", input_path, "-q:a", "0", "-map", "a", temp_audio_path
        ]
        subprocess.run(audio_extraction_command, check=True, capture_output=True, text=True, timeout=3000)

        # Process video frames
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise Exception(f"Could not open video file: {input_path}")

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Adjust FPS randomly by ±5% for each variation
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        random.seed(variation)  # Ensure consistent randomness per variation
        adjusted_fps = original_fps * random.uniform(0.95, 1.05)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(temp_video_path, fourcc, adjusted_fps, (width, height))

        if not out.isOpened():
            raise Exception(f"Failed to initialize VideoWriter for: {output_path}")

        # Uniform brightness adjustment
        adjustment = random.uniform(-0.05, 0.05)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        processed_frames = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Modify the frame
            frame = frame.astype(np.float32)
            frame = np.clip(frame * (1 + adjustment), 0, 255).astype(np.uint8)

            # Add invisible mesh overlay
            frame_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            frame_image = add_invisible_mesh(frame_image)

            # Convert back and write to output
            frame = cv2.cvtColor(np.array(frame_image), cv2.COLOR_RGB2BGR)
            out.write(frame)

            processed_frames += 1

            # Update progress periodically
            if total_frames > 0:
                progress = int((processed_frames / total_frames) * 100)
                storage.hset(f"job:{job_id}", "progress", progress)

        cap.release()
        out.release()

        # Combine audio and video with randomized speed adjustment
        speed_factor = random.uniform(0.95, 1.05)  # ±5% speed adjustment
        combine_command = [
            FFMPEG_PATH, "-i", temp_video_path, "-i", temp_audio_path,
            "-filter:v", f"setpts={1/speed_factor}*PTS", "-c:a", "aac", output_path
        ]
        subprocess.run(combine_command, check=True, capture_output=True, text=True, timeout=3000)

    except Exception as e:
        app.logger.error(f"Error in video processing for job {job_id}, variation {variation}: {e}")
    finally:
        # Cleanup and progress update
        safe_remove(temp_audio_path)
        safe_remove(temp_video_path)
        storage.hincrby(f"job:{job_id}", "completed_files", 1)
        update_job_progress(job_id)
