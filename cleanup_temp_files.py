import os
import time
import shutil
import json
import logging
import redis
from datetime import datetime

# Configuration
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'outputs'
CUTOFF_TIME = 60 * 60  # 1 hour in seconds
GRACE_PERIOD = 15 * 60  # 15 minutes additional time for downloads
LOG_FILE = 'cleanup.log'

# Set up logging
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def get_redis_client():
    try:
        redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379')
        client = redis.from_url(redis_url, decode_responses=True)
        client.ping()
        return client
    except:
        logging.warning("Could not connect to Redis, using more conservative cleanup approach")
        return None

def is_job_active(directory, redis_client):
    if redis_client:
        # Check all jobs in Redis
        for key in redis_client.keys("job:*"):
            job_data = redis_client.hgetall(key)
            if (directory in job_data.get('upload_folder', '') or 
                directory in job_data.get('output_folder', '')):
                status = job_data.get('status', '')
                if status in ['pending', 'processing']:
                    return True
    return False

def should_delete_directory(dir_path, redis_client):
    now = time.time()
    dir_age = now - os.path.getmtime(dir_path)
    
    # If directory is less than CUTOFF_TIME old, keep it
    if dir_age < CUTOFF_TIME:
        return False
    
    # If directory is very old (past CUTOFF_TIME + GRACE_PERIOD), delete it
    if dir_age > (CUTOFF_TIME + GRACE_PERIOD):
        return True
    
    # For directories in the grace period, check if they're associated with active jobs
    return not is_job_active(dir_path, redis_client)

def cleanup_folder(folder_path, redis_client):
    if not os.path.exists(folder_path):
        logging.warning(f"Folder does not exist: {folder_path}")
        return

    try:
        now = time.time()
        deleted_count = 0
        skipped_count = 0

        for root, dirs, files in os.walk(folder_path, topdown=False):
            for directory in dirs:
                dir_path = os.path.join(root, directory)
                try:
                    if should_delete_directory(dir_path, redis_client):
                        shutil.rmtree(dir_path, ignore_errors=True)
                        logging.info(f"Deleted directory: {dir_path}")
                        deleted_count += 1
                    else:
                        skipped_count += 1
                except Exception as e:
                    logging.error(f"Error processing directory {dir_path}: {str(e)}")

        logging.info(f"Cleanup summary for {folder_path}: "
                    f"Deleted {deleted_count} directories, "
                    f"Skipped {skipped_count} directories")

    except Exception as e:
        logging.error(f"Error during cleanup of {folder_path}: {str(e)}")

def main():
    logging.info("Starting cleanup process...")
    
    redis_client = get_redis_client()
    
    cleanup_folder(UPLOAD_FOLDER, redis_client)
    cleanup_folder(OUTPUT_FOLDER, redis_client)
    
    logging.info("Cleanup process completed")

if __name__ == "__main__":
    main()
