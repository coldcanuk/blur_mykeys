import cv2
import pytesseract
import re
import os
import json
from pathlib import Path
import logging
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
from fuzzywuzzy import fuzz

# Dependency check
def check_dependencies():
    """Check if required dependencies are installed."""
    try:
        subprocess.run(['ffmpeg', '-version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        raise RuntimeError("FFmpeg is not installed. Please install it on your Ubuntu system (Hermes/Athena).")
    try:
        subprocess.run(['tesseract', '--version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        raise RuntimeError("Tesseract is not installed. Please install it on your Ubuntu system (Hermes/Athena).")

# Load configuration
def load_config():
    """Load configuration from config.json, with defaults if file doesn't exist."""
    default_config = {
        'blur_kernel_size': 51,
        'max_workers': max(2, os.cpu_count() // 2),  # Dynamic based on CPU cores
        'video_filename': 'output.mkv',
        'fps': 30,
        'target_keys': ["ai_api_key", "azure_client_secret", "cloudflare_api_key", "linode_api_token"]
    }
    config_path = 'blur_config.json'
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            user_config = json.load(f)
        default_config.update(user_config)
    return default_config

# Backup the existing log file if it exists.
if os.path.exists('blur_apikeys.log'):
    os.rename('blur_apikeys.log', 'blur_apikeys.log.bak')

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', filename='blur_apikeys.log')
logger = logging.getLogger(__name__)

# Check dependencies
check_dependencies()

# Load configuration
config = load_config()
VIDEO_FILENAME = config['video_filename']
FPS = config['fps']
target_keys = config['target_keys']

# Define paths
homedir = os.environ['HOME']
obsdir = os.path.join(homedir, "obs")
input_dir = os.path.join(obsdir, "frames")
output_dir = os.path.join(obsdir, "processed_frames")
pre_output_dir = os.path.join(obsdir, "pre_processed_frames")
video_path = os.path.join(obsdir, VIDEO_FILENAME)

# Ensure directories exist
os.makedirs(input_dir, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)
os.makedirs(pre_output_dir, exist_ok=True)

def normalize(text):
    """Normalize text by keeping only letters and underscores, convert to lowercase."""
    return re.sub(r'[^a-z_]', '', text.lower())

def prepare_image_for_ocr(image):
    """Prepare image for OCR, optimized for dark mode IDE."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    v_channel = hsv[:, :, 2]
    inverted = cv2.bitwise_not(v_channel)
    filtered = cv2.bilateralFilter(inverted, 9, 75, 75)
    scale_factor = 3
    scaled = cv2.resize(filtered, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)
    thresholded = cv2.adaptiveThreshold(scaled, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    return thresholded, scale_factor

def extract_frames():
    """Extract frames from video using FFmpeg."""
    try:
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file {video_path} does not exist.")
        for frame in Path(input_dir).glob("frame_*.png"):
            frame.unlink()
        ffmpeg_cmd = ['ffmpeg', '-i', video_path, '-vf', f'fps={FPS}', os.path.join(input_dir, 'frame_%06d.png')]
        logger.info(f"Extracting frames from {VIDEO_FILENAME} at {FPS} FPS")
        result = subprocess.run(ffmpeg_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
        logger.info("Frame extraction completed")
        return True
    except Exception as e:
        logger.error(f"Error extracting frames: {e}")
        raise

def blur_region(image, x, y, w, h, kernel_size):
    """Apply Gaussian blur to a region."""
    kernel_size = max(1, int(kernel_size) | 1)
    roi = image[y:y+h, x:x+w]
    blurred = cv2.GaussianBlur(roi, (kernel_size, kernel_size), 0)
    image[y:y+h, x:x+w] = blurred
    return image

def process_frame(frame_path, output_path):
    """Process a frame to blur lines containing API keys using fuzzy matching on key part."""
    try:
        img = cv2.imread(frame_path)
        if img is None:
            raise ValueError(f"Failed to load {frame_path}")

        ocr_image, scale_factor = prepare_image_for_ocr(img)
        custom_config = r'--psm 6'
        data = pytesseract.image_to_data(ocr_image, config=custom_config, output_type=pytesseract.Output.DICT)

        full_text = pytesseract.image_to_string(ocr_image, config=custom_config)
        logger.info(f"Full extracted text for {frame_path}: {full_text}")
        num_words = sum(1 for text in data['text'] if text.strip())
        logger.info(f"Detected {num_words} words in {frame_path}")

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
            logger.info(f"Processing line: {line_text}")
            if '=' not in line_text:
                continue
            key_part = line_text.split('=', 1)[0].strip()
            # Handle prefixes by taking the last word before '='
            key_part = key_part.split()[-1] if key_part else key_part
            normalized_key = normalize(key_part)
            for target_key in target_keys:
                similarity = fuzz.ratio(target_key, normalized_key)
                logger.info(f"Similarity score for '{normalized_key}' against '{target_key}': {similarity}")
                if similarity > 70:  # Lowered threshold to 70
                    logger.info(f"Detected key similar to {target_key} (score: {similarity}) in line: {line_text}")
                    x_min = int(min(word['left'] for word in words) / scale_factor)
                    y_min = int(min(word['top'] for word in words) / scale_factor)
                    x_max = int(max(word['left'] + word['width'] for word in words) / scale_factor)
                    y_max = int(max(word['top'] + word['height'] for word in words) / scale_factor)
                    w = x_max - x_min
                    h = y_max - y_min
                    padding = 10
                    x = max(0, x_min - padding)
                    y = max(0, y_min - padding)
                    w = min(img.shape[1] - x, w + 2 * padding + 100)  # Extend width to ensure full line is blurred
                    h = min(img.shape[0] - y, h + 2 * padding)
                    logger.info(f"Blurring region at x={x}, y={y}, w={w}, h={h}")
                    img = blur_region(img, x, y, w, h, config['blur_kernel_size'])
                    blurred_regions += 1
                    break

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
    if not frame_files:
        raise ValueError("No frames found to process.")
    logger.info(f"Processing {len(frame_files)} frames with {config['max_workers']} workers")
    with ThreadPoolExecutor(max_workers=config['max_workers']) as executor:
        futures = [executor.submit(process_frame, str(f), os.path.join(output_dir, f"processed_{f.name}")) for f in frame_files]
        processed = sum(f.result() for f in futures)
    logger.info(f"Processed {processed} frames successfully")
    return processed, len(frame_files)

def assemble_video():
    """Assemble processed frames into a video."""
    try:
        processed_files = list(Path(output_dir).glob("processed_frame_*.png"))
        if not processed_files:
            raise ValueError("No processed frames found to assemble video.")
        output_video_path = os.path.join(obsdir, f"processed_{VIDEO_FILENAME}")
        ffmpeg_cmd = [
            'ffmpeg', '-framerate', str(FPS), '-i', os.path.join(output_dir, 'processed_frame_%06d.png'),
            '-c:v', 'libx264', '-pix_fmt', 'yuv420p', output_video_path
        ]
        logger.info(f"Assembling video: {output_video_path}")
        result = subprocess.run(ffmpeg_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
        logger.info("Video assembled")
        return True
    except Exception as e:
        logger.error(f"Error assembling video: {e}")
        raise

def cleanup():
    """Clean up temporary frame directories."""
    try:
        shutil.rmtree(input_dir, ignore_errors=True)
        shutil.rmtree(output_dir, ignore_errors=True)
        shutil.rmtree(pre_output_dir, ignore_errors=True)
        logger.info("Cleaned up temporary frame directories.")
    except Exception as e:
        logger.error(f"Error during cleanup: {e}")

if __name__ == "__main__":
    try:
        if extract_frames():
            processed, total = process_all_frames()
            if processed == total and processed > 0:
                if assemble_video():
                    cleanup()
    except Exception as e:
        logger.error(f"Script execution failed: {e}")
        raise