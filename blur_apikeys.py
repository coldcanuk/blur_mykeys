import cv2
import pytesseract
import re
import os
import numpy as np
from pathlib import Path
import logging
from concurrent.futures import ThreadPoolExecutor
import subprocess
import yaml

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', filename='blur_apikeys.log')
logger = logging.getLogger(__name__)

# Load configuration
config_file = Path.cwd() / 'blur_apikeys.yml'
if config_file.exists():
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
else:
    config = {
        'min_key_length': 20,
        'max_key_length': 64,
        'blur_kernel_size': 51,
        'confidence_threshold': 50,
        'max_workers': 10,
        'video_filename': 'output.mkv',
        'fps': 30,
        'brightness_threshold': 128,
        'providers': {
            'Cloudflare': r'[A-Za-z0-9]{37,40}',
            'Azure': r'[A-Za-z0-9]{8}-[A-Za-z0-9]{4}-[A-Za-z0-9]{4}-[A-Za-z0-9]{4}-[A-Za-z0-9]{12}',
            'Runpod': r'[A-Za-z0-9_-]{20,40}',
            'Linode': r'[A-Za-z0-9]{64}'
        }
    }
    with open(config_file, 'w') as f:
        yaml.safe_dump(config, f, default_flow_style=False)

# Video file configuration
VIDEO_FILENAME = config['video_filename']
FPS = config['fps']

# Define paths
homedir = os.environ['HOME']
obsdir = os.path.join(homedir, "obs")
input_dir = os.path.join(obsdir, "frames")
output_dir = os.path.join(obsdir, "processed_frames")
video_path = os.path.join(obsdir, VIDEO_FILENAME)
os.makedirs(input_dir, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)

# Regex for potential API keys (existing pattern)
provider_patterns = "|".join(f"(?:{pattern})" for pattern in config['providers'].values())
api_key_pattern = rf"(?i)(api_key|secret|key|token)\s*[:=]\s*(?:\b(?=.*[a-zA-Z])(?=.*[0-9])[A-Za-z0-9_-]{{{config['min_key_length']},{config['max_key_length']}}}\b|{provider_patterns})\b"

# New regex pattern for API key prefixes
prefix_pattern = r"(?i)\b(APIKEY|API_KEY|CLIENT_SECRET|SECRET|TOKEN|AI_API_KEY|AI_API_KEYS|AZURE_API_KEY|AZURE_API_KEYS|ACCESS_TOKEN|ACCESS_TOKEN_SECRET|ACCESS_TOKEN_KEY|ACCESS_TOKEN_ID|ACCESS_TOKEN_SECRET|ACCESS_TOKEN_KEY|ACCESS_TOKEN_ID|AWS_ACCESS_KEY_ID|AWS_SECRET_ACCESS_KEY|AWS_ACCESS_KEY|AWS_SECRET_KEY|AWS_ACCESS_KEY_ID|AWS_SECRET_ACCESS_KEY|AWS_ACCESS_KEY|AWS_SECRET_KEY|OPENAI_APIKEY)\s*[:=]"

def detect_mode(image):
    """Detect if image is in dark mode or light mode based on average brightness."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    brightness = np.mean(hsv[:, :, 2])
    threshold = float(config.get('brightness_threshold', 128))
    is_dark_mode = brightness < threshold
    logger.info(f"Image mode detection: {'Dark' if is_dark_mode else 'Light'} mode (brightness: {brightness:.2f})")
    return is_dark_mode

def prepare_image_for_ocr(image):
    """Prepare image for OCR based on whether it's dark or light mode with enhanced preprocessing."""
    is_dark_mode = detect_mode(image)
    if is_dark_mode:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        processed = cv2.bitwise_not(gray)
    else:
        processed = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    processed = cv2.bilateralFilter(processed, 9, 75, 75)
    scale_factor = 2
    processed = cv2.resize(processed, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)
    _, processed = cv2.threshold(processed, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    debug_path = os.path.join(os.path.dirname(output_dir), 'ocr_debug')
    os.makedirs(debug_path, exist_ok=True)
    cv2.imwrite(os.path.join(debug_path, 'ocr_image.png'), processed)
    
    return processed, scale_factor

def extract_frames():
    """Extract frames from video using ffmpeg at specified FPS."""
    try:
        for frame in Path(input_dir).glob("frame_*.png"):
            frame.unlink()
        ffmpeg_cmd = [
            'ffmpeg',
            '-i', video_path,
            '-vf', f'fps={FPS}',
            os.path.join(input_dir, 'frame_%06d.png')
        ]
        logger.info(f"Extracting frames from {VIDEO_FILENAME} at {FPS} FPS")
        result = subprocess.run(ffmpeg_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode != 0:
            logger.error(f"FFmpeg failed: {result.stderr}")
            return False
        logger.info("Frame extraction completed successfully")
        return True
    except Exception as e:
        logger.error(f"Error extracting frames: {str(e)}")
        return False

def blur_region(image, x, y, w, h, kernel_size):
    """Apply Gaussian blur to a rectangular region for better obscuring."""
    kernel_size = int(kernel_size)
    if kernel_size % 2 == 0:
        kernel_size += 1
    roi = image[y:y+h, x:x+w]
    blurred = cv2.GaussianBlur(roi, (kernel_size, kernel_size), 0)
    image[y:y+h, x:x+w] = blurred
    return image

def process_frame(frame_path, output_path):
    """Process a single frame to detect and blur API keys with improved error handling."""
    try:
        img = cv2.imread(frame_path)
        if img is None:
            raise ValueError(f"Failed to load {frame_path}")

        ocr_image, scale_factor = prepare_image_for_ocr(img)
        data = pytesseract.image_to_data(ocr_image, config='--psm 6', output_type=pytesseract.Output.DICT)

        full_text = " ".join([t for t in data['text'] if t.strip()])
        logger.info(f"OCR extracted text: {full_text}")
        
        for i in range(len(data['text'])):
            if data['text'][i].strip():
                logger.info(f"Text: '{data['text'][i]}' with confidence {data['conf'][i]}")

        confidence_threshold = float(config.get('confidence_threshold', 40))
        blurred_regions = 0

        # First, check for API key prefixes and blur the entire line
        prefix_matches = re.finditer(prefix_pattern, full_text)
        for match in prefix_matches:
            prefix = match.group(1)
            logger.info(f"Detected API key prefix: {prefix}")
            for i in range(len(data['text'])):
                if prefix in data['text'][i] and float(data['conf'][i]) > 20:  # Lower threshold for prefix detection
                    x = int(data['left'][i] / scale_factor)
                    y = int(data['top'][i] / scale_factor)
                    w = int(data['width'][i] / scale_factor)
                    h = int(data['height'][i] / scale_factor)
                    # Extend width to blur the entire line (approximate)
                    x_start = max(0, x - 50)
                    x_end = min(img.shape[1], x + w + 500)
                    logger.info(f"Blurring line with prefix {prefix} at ({x_start}, {y}, {x_end - x_start}, {h})")
                    img = blur_region(img, x_start, y, x_end - x_start, h, kernel_size=int(config['blur_kernel_size']))
                    blurred_regions += 1
                    break

        # Then, apply existing API key detection logic
        matches = re.finditer(api_key_pattern, full_text)
        for match in matches:
            keyword = match.group(1) or "None"
            key = match.group(2)
            logger.info(f"Regex match: keyword='{keyword}', key='{key}'")
            for i in range(len(data['text'])):
                if key in data['text'][i] and float(data['conf'][i]) > confidence_threshold:
                    x = int(data['left'][i] / scale_factor)
                    y = int(data['top'][i] / scale_factor)
                    w = int(data['width'][i] / scale_factor)
                    h = int(data['height'][i] / scale_factor)
                    logger.info(f"Blurring key: {key} at ({x}, {y}, {w}, {h}) with confidence {data['conf'][i]}")
                    img = blur_region(img, x, y, w, h, kernel_size=int(config['blur_kernel_size']))
                    blurred_regions += 1
                    break

        cv2.imwrite(output_path, img)
        logger.info(f"Processed {frame_path} with {blurred_regions} regions blurred")
        return True
        
    except Exception as e:
        logger.error(f"Error processing {frame_path}: {str(e)}")
        import shutil
        shutil.copy(frame_path, output_path)
        return False

def process_all_frames():
    """Process all frames using parallel processing."""
    frame_files = sorted(Path(input_dir).glob("frame_*.png"))
    total_frames = len(frame_files)
    processed_count = 0
    failed_count = 0
    
    max_workers = int(config['max_workers'])
    logger.info(f"Starting processing of {total_frames} frames with {max_workers} workers")
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for frame_file in frame_files:
            output_file = os.path.join(output_dir, frame_file.name)
            futures.append(executor.submit(process_frame, str(frame_file), output_file))
        for future in futures:
            if future.result():
                processed_count += 1
            else:
                failed_count += 1
                
    logger.info(f"Completed processing: {processed_count} successful, {failed_count} failed")
    return processed_count, failed_count

def assemble_video():
    """Assemble processed frames back into a video using ffmpeg."""
    try:
        output_video_path = os.path.join(obsdir, f"processed_{VIDEO_FILENAME}")
        frame_pattern = os.path.join(output_dir, 'frame_%06d.png')
        ffmpeg_cmd = [
            'ffmpeg',
            '-framerate', str(FPS),
            '-i', frame_pattern,
            '-c:v', 'libx264',
            '-pix_fmt', 'yuv420p',
            output_video_path
        ]
        logger.info(f"Assembling processed frames into {output_video_path}")
        result = subprocess.run(ffmpeg_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode != 0:
            logger.error(f"FFmpeg failed to assemble video: {result.stderr}")
            return False
        logger.info("Video assembly completed successfully")
        return True
    except Exception as e:
        logger.error(f"Error assembling video: {str(e)}")
        return False

def cleanup():
    """Remove all frames and processed frames."""
    logger.info("Cleaning up frames and processed frames")
    for frame in Path(input_dir).glob("frame_*.png"):
        frame.unlink()
    for frame in Path(output_dir).glob("frame_*.png"):
        frame.unlink()
    logger.info("Cleanup completed")

if __name__ == "__main__":
    if extract_frames():
        processed, failed = process_all_frames()
        if processed > 0:
            assemble_video()
        else:
            logger.error("No frames were processed successfully, skipping video assembly")
    else:
        logger.error("Frame extraction failed, aborting processing")
    #cleanup()