# Video API Key Blurring Tool

This tool processes video files to detect and blur potential API keys or sensitive information visible in the frames. It uses OCR (Optical Character Recognition) to identify text that matches specific patterns and applies a Gaussian blur to obscure them.

This tool kinda works right now. It is not perfect but it is an excellent starting point on how to do OCR with video and FFMPEG.

## Overview

The script performs the following steps:
1. **Frame Extraction**: Extracts frames from a specified video file using `ffmpeg` at a defined frame rate.
2. **Text Detection**: Uses `pytesseract` for OCR to detect text in each frame, with advanced preprocessing for dark/light mode detection.
3. **Pattern Matching**: Identifies potential API keys using configurable regex patterns for various providers (e.g., Cloudflare, Azure) and specific prefix patterns (e.g., API_KEY, SECRET).
4. **Blurring**: Applies a Gaussian blur to regions identified as containing potential API keys.
5. **Parallel Processing**: Processes multiple frames concurrently to improve performance.
6. **Video Assembly**: Reassembles processed frames into a final video output.

## Configuration

The tool is configurable via a YAML file named `blur_apikeys.yml`. If this file does not exist, a default configuration is created. The configurable parameters include:

- `min_key_length`: Minimum length of text to be considered an API key (default: 20).
- `max_key_length`: Maximum length of text to be considered an API key (default: 64).
- `blur_kernel_size`: Size of the Gaussian blur kernel (default: 51).
- `confidence_threshold`: OCR confidence threshold for text detection (default: 50).
- `max_workers`: Number of parallel workers for frame processing (default: 10).
- `video_filename`: Name of the input video file (default: "output.mkv").
- `fps`: Frame rate for extraction and assembly (default: 30).
- `brightness_threshold`: Threshold for detecting dark/light mode in frames (default: 128).
- `providers`: Dictionary of regex patterns for specific API key formats from different providers like OpenAI, Anthropic, AWS, etc.

## Requirements

- Python 3.x
- OpenCV (`cv2`)
- Pytesseract
- FFmpeg installed on the system
- Additional Python libraries: `numpy`, `yaml`, `pathlib`

## Installation

1. Install FFmpeg on your system:
   - On Ubuntu: `sudo apt-get install ffmpeg`
   - On macOS: `brew install ffmpeg`
   - On Windows: Download the latest build from the FFmpeg website, extract it, and add the `bin` directory to your system's PATH.
2. Install Python dependencies:
   ```bash
   pip install opencv-python pytesseract numpy pyyaml pathlib
   ```
3. Install Tesseract OCR:
   - On Ubuntu: `sudo apt-get install tesseract-ocr`
   - On macOS: `brew install tesseract`
   - On Windows: Download and install from the GitHub repository (https://github.com/UB-Mannheim/tesseract/wiki), and ensure the installation directory is added to PATH.

## Usage

1. Ensure FFmpeg and Tesseract are installed on your system.
2. Place your video file in the `~/obs` directory (or modify the path in the script).
3. Run the script:
   ```bash
   python blur_apikeys.py
   ```
4. Frames are extracted to `~/obs/frames`, processed, and saved to `~/obs/processed_frames`. The final blurred video is saved as `processed_<original_filename>` in the `~/obs` directory.

## Input/Output

- **Input Video**: Configured via `video_filename` in `blur_apikeys.yml` (default: "output.mkv").
- **Frame Rate**: Configured via `fps` in `blur_apikeys.yml` (default: 30 FPS).
- **Output**: Processed video with blurred regions saved as `processed_<original_filename>`.

## Advanced Features

- **Dark/Light Mode Detection**: The tool automatically detects whether a frame is in dark or light mode based on brightness levels and adjusts OCR preprocessing accordingly to improve text detection accuracy.
- **Prefix Pattern Matching**: In addition to provider-specific regex patterns, the tool searches for common API key prefixes (e.g., API_KEY, SECRET, TOKEN) and blurs the entire line to ensure sensitive data is obscured.
- **Comprehensive Logging**: Detailed logs are written to `blur_apikeys.log`, including OCR results, detected API keys, blurred regions, and any errors for debugging and monitoring.

## Logging

The script logs detailed information about the process, including frame extraction status, detected API keys, and processing results, to a file named `blur_apikeys.log` for debugging and monitoring.

## Limitations

- The accuracy of API key detection depends on the quality of OCR, which can be affected by video resolution, text size, and background noise.
- The regex patterns for API keys may need adjustment based on the specific format of keys you are trying to obscure.
- Processing large videos may be time-consuming; consider adjusting `fps` or `max_workers` for performance.
- Currently, there is no mechanism to review or correct OCR results before blurring, which might lead to false positives or negatives.

## Performance Tips

- Adjust `max_workers` in the configuration based on your system's CPU cores for optimal parallel processing.
- Lower the `fps` value for very long videos to reduce the number of frames processed.
- Ensure sufficient disk space for frame extraction and processing directories.

## Troubleshooting

- **OCR Failures**: If API keys are not detected, check video clarity or adjust the `confidence_threshold` in the configuration.
- **FFmpeg Errors**: Ensure FFmpeg is installed and accessible in your system's PATH. Verify the input video file exists and is accessible.
- **Processing Failures**: Review `blur_apikeys.log` for detailed error messages.

## Future Enhancements

- **Adaptive Frame Processing**: Implement logic to process only frames with significant text content or changes.
- **Configurable Paths**: Allow customization of input and output directories in the configuration file.
- **Interactive Mode**: Add an option to preview and validate blurred regions before final video assembly.

## License

MIT