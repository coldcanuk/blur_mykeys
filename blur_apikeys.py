import cv2
import pytesseract
import re
import os
from pathlib import Path
import logging
from concurrent.futures import ThreadPoolExecutor
import subprocess
from collections import defaultdict
import shutil
from fuzzywuzzy import fuzz

# Backup the existing log file if it exists.
if os.path.exists('blur_apikeys.log'):
    os.rename('blur_apikeys.log', 'blur_apikeys.log.bak')
# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', filename='blur_apikeys.log')
logger = logging.getLogger(__name__)

# Configuration - will bring in the config from blur_apikeys.yml after we confirm the code is working as intended.
config = {
    'blur_kernel_size': 51,
    'max_workers': 10,
    'video_filename': 'output.mkv',
    'fps': 30,
}

# Video file configuration
VIDEO_FILENAME = config['video_filename']
FPS = config['fps']

# Define paths
homedir = os.environ['HOME']
obsdir = os.path.join(homedir, "obs")
input_dir = os.path.join(obsdir, "frames")
output_dir = os.path.join(obsdir, "processed_frames")
pre_output_dir = os.path.join(obsdir, "pre_processed_frames")
video_path = os.path.join(obsdir, VIDEO_FILENAME)
os.makedirs(input_dir, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)
os.makedirs(pre_output_dir, exist_ok=True)

# Target prefixes for fuzzy matching
target_prefixes = ["AI_API_KEY=", "AZURE_CLIENT_SECRET="]

def prepare_image_for_ocr(image):
    """Prepare image for OCR, optimized for dark mode IDE."""
    # Convert to HSV and use V channel
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    v_channel = hsv[:, :, 2]
    # Invert for dark text on light background
    inverted = cv2.bitwise_not(v_channel)
    # Reduce noise
    filtered = cv2.bilateralFilter(inverted, 9, 75, 75)
    # Scale for better OCR (increased to 3x)
    scale_factor = 3
    scaled = cv2.resize(filtered, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)
    # Adaptive thresholding for better text clarity
    thresholded = cv2.adaptiveThreshold(scaled, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    return thresholded, scale_factor

def extract_frames():
    """Extract frames from video using FFmpeg."""
    try:
        for frame in Path(input_dir).glob("frame_*.png"):
            frame.unlink()
        ffmpeg_cmd = ['ffmpeg', '-i', video_path, '-vf', f'fps={FPS}', os.path.join(input_dir, 'frame_%06d.png')]
        logger.info(f"Extracting frames from {VIDEO_FILENAME} at {FPS} FPS")
        result = subprocess.run(ffmpeg_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode != 0:
            logger.error(f"FFmpeg failed: {result.stderr}")
            return False
        logger.info("Frame extraction completed")
        return True
    except Exception as e:
        logger.error(f"Error extracting frames: {e}")
        return False

def blur_region(image, x, y, w, h, kernel_size):
    """Apply Gaussian blur to a region."""
    kernel_size = max(1, int(kernel_size) | 1)  # Ensure odd number
    roi = image[y:y+h, x:x+w]
    blurred = cv2.GaussianBlur(roi, (kernel_size, kernel_size), 0)
    image[y:y+h, x:x+w] = blurred
    return image

def process_frame(frame_path, output_path):
    """Process a frame to blur lines containing API keys using fuzzy matching."""
    try:
        img = cv2.imread(frame_path)
        if img is None:
            raise ValueError(f"Failed to load {frame_path}")

        ocr_image, scale_factor = prepare_image_for_ocr(img)
        # Tesseract config: PSM 11 for sparse text, no whitelist to avoid limiting detection
        custom_config = r'--psm 11'
        data = pytesseract.image_to_data(ocr_image, config=custom_config, output_type=pytesseract.Output.DICT)

        # Log full extracted text for debugging
        full_text = pytesseract.image_to_string(ocr_image, config=custom_config)
        logger.info(f"Full extracted text for {frame_path}: {full_text}")
        num_words = sum(1 for text in data['text'] if text.strip())
        logger.info(f"Detected {num_words} words in {frame_path}")

        # Group words by line
        lines = defaultdict(list)
        for i in range(len(data['text'])):
            if data['text'][i].strip():
                key = (data['block_num'][i], data['par_num'][i], data['line_num'][i])
                lines[key].append({
                    'text': data['text'][i],
                    'left': data['left'][i],
                    'top': data['top'][i],
                    'width': data['width'][i],
                    'height': data['height'][i],
                    'conf': data['conf'][i]
                })

        blurred_regions = 0
        for _, words in lines.items():
            line_text = " ".join(word['text'] for word in words)
            logger.info(f"Extracted line: {line_text}")
            for prefix in target_prefixes:
                similarity = fuzz.partial_ratio(prefix.lower(), line_text.lower())
                if similarity > 85:
                    logger.info(f"Detected prefix similar to {prefix} (score: {similarity}) in line: {line_text}")
                    # Calculate bounding box for the entire line
                    x_min = int(min(word['left'] for word in words) / scale_factor)
                    y_min = int(min(word['top'] for word in words) / scale_factor)
                    x_max = int(max(word['left'] + word['width'] for word in words) / scale_factor)
                    y_max = int(max(word['top'] + word['height'] for word in words) / scale_factor)
                    w = x_max - x_min
                    h = y_max - y_min
                    # Add padding to ensure full line is blurred
                    padding = 10
                    x = max(0, x_min - padding)
                    y = max(0, y_min - padding)
                    w = min(img.shape[1] - x, w + 2 * padding)
                    h = min(img.shape[0] - y, h + 2 * padding)
                    img = blur_region(img, x, y, w, h, config['blur_kernel_size'])
                    blurred_regions += 1
                    break

        # Save preprocessed image for debugging (every 10th frame)
        frame_num = int(Path(frame_path).stem.split('_')[1])
        if frame_num % 10 == 0:
            cv2.imwrite(os.path.join(pre_output_dir, f"preprocessed_frame_{frame_num}.png"), ocr_image)

        cv2.imwrite(output_path, img)
        logger.info(f"Processed {frame_path} with {blurred_regions} regions blurred")
        return True
    except Exception as e:
        logger.error(f"Error processing {frame_path}: {e}")
        shutil.copy(frame_path, output_path)
        return False

def process_all_frames():
    """Process all frames in parallel."""
    frame_files = sorted(Path(input_dir).glob("frame_*.png"))
    logger.info(f"Processing {len(frame_files)} frames with {config['max_workers']} workers")
    with ThreadPoolExecutor(max_workers=config['max_workers']) as executor:
        futures = [executor.submit(process_frame, str(f), os.path.join(output_dir, f"processed_{f.name}")) for f in frame_files]
        processed = sum(f.result() for f in futures)
    logger.info(f"Processed {processed} frames successfully")
    return processed

def assemble_video():
    """Assemble processed frames into a video."""
    try:
        output_video_path = os.path.join(obsdir, f"processed_{VIDEO_FILENAME}")
        ffmpeg_cmd = [
            'ffmpeg', '-framerate', str(FPS), '-i', os.path.join(output_dir, 'frame_%06d.png'),
            '-c:v', 'libx264', '-pix_fmt', 'yuv420p', output_video_path
        ]
        logger.info(f"Assembling video: {output_video_path}")
        result = subprocess.run(ffmpeg_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode != 0:
            logger.error(f"FFmpeg failed: {result.stderr}")
            return False
        logger.info("Video assembled")
        return True
    except Exception as e:
        logger.error(f"Error assembling video: {e}")
        return False

if __name__ == "__main__":
    if extract_frames():
        processed = process_all_frames()
        if processed > 0:
            assemble_video()