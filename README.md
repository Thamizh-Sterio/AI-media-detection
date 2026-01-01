# AI Media Detection

A full-stack web application for detecting AI-generated or modified images and videos using Azure OpenAI's vision capabilities.

## Features

- 🖼️ **Image Detection**: Analyze images for AI generation artifacts
- 🎥 **Video Detection**: Analyze videos by extracting and examining key frames
- 🌐 **Web Interface**: Simple drag-and-drop interface for easy uploads
- 🔍 **Detailed Analysis**: Get confidence scores and explanations for each detection
- ⚡ **Fast Processing**: Asynchronous processing with real-time feedback

## Technology Stack

### Backend
- **FastAPI**: Modern, fast web framework for building APIs
- **Azure OpenAI**: Vision model (GPT-4o) for AI content detection
- **OpenCV**: Video frame extraction and processing
- **Python 3.10+**: Core programming language

### Frontend
- **Vanilla HTML/CSS/JavaScript**: No dependencies, lightweight interface
- **Fetch API**: Modern HTTP requests
- **Responsive Design**: Works on desktop and mobile

## Project Structure

```
AI Media detection/
├── main.py                 # FastAPI backend application
├── requirements.txt        # Python dependencies
├── .env                    # Environment variables (create from template)
├── static/
│   └── index.html         # Frontend web interface
└── README.md              # This file
```

## Setup Instructions

### Prerequisites

- Python 3.10 or higher
- Azure OpenAI account with GPT-4o (or compatible vision model) deployment
- pip (Python package installer)

### Installation

1. **Clone or navigate to the project directory**:
   ```bash
   cd "c:\Users\azoperator\Documents\Python Scripts\AI Media detection"
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment variables**:
   
   Edit the `.env` file and add your Azure OpenAI credentials:
   ```env
   AZURE_OPENAI_ENDPOINT=https://your-resource-name.openai.azure.com/
   AZURE_OPENAI_API_KEY=your-api-key-here
   DEPLOYMENT_NAME=gpt-4o
   ```

   To get these values:
   - Go to [Azure Portal](https://portal.azure.com)
   - Navigate to your Azure OpenAI resource
   - Copy the endpoint URL and API key
   - Note your deployment name (e.g., gpt-4o, gpt-4-vision)

### Running the Application

1. **Start the FastAPI server**:
   ```bash
   uvicorn main:app --reload
   ```

   Or using Python directly:
   ```bash
   python main.py
   ```

2. **Access the application**:
   - Open your browser and visit: `http://localhost:8000/static/index.html`
   - Or navigate to: `http://127.0.0.1:8000/static/index.html`

3. **Test the API directly** (optional):
   - API documentation: `http://localhost:8000/docs`
   - Alternative docs: `http://localhost:8000/redoc`

## Usage

### Web Interface

1. Visit `http://localhost:8000/static/index.html`
2. Drag and drop an image or video file (or click to browse)
3. Click "Analyze Media"
4. View results including:
   - AI-generated or human-created classification
   - Confidence score (0-100%)
   - Detailed explanation
   - For videos: number of frames analyzed

### API Endpoints

#### Image Detection
```bash
POST /detect-image
Content-Type: multipart/form-data

file: <image file>
```

**Supported formats**: JPG, PNG, WEBP

**Response**:
```json
{
  "is_ai_generated": true,
  "confidence": 0.85,
  "explanation": "Unnatural textures and lighting inconsistencies detected"
}
```

#### Video Detection
```bash
POST /detect-video
Content-Type: multipart/form-data

file: <video file>
```

**Supported formats**: MP4, MOV, AVI

**Response**:
```json
{
  "is_ai_generated": false,
  "confidence": 0.32,
  "explanation": "Analyzed 10 frames. No significant AI artifacts detected across frames",
  "frames_analyzed": 10
}
```

## How It Works

### Image Detection
1. User uploads an image file
2. Backend encodes image to base64
3. Sends to Azure OpenAI vision model with detection prompt
4. AI analyzes for artifacts like:
   - Unnatural textures or patterns
   - Lighting inconsistencies
   - Unusual distortions
   - Unrealistic details
5. Returns JSON with detection result

### Video Detection
1. User uploads a video file
2. Backend extracts 5-10 key frames (every 30 frames)
3. Each frame is analyzed separately using the same process as images
4. Results are aggregated:
   - Average confidence calculated
   - If avg confidence > 0.5, video marked as AI-modified
   - Explanations from positive detections combined
5. Returns aggregated result with frame count

## Configuration

### Adjusting Video Frame Extraction

In `main.py`, modify the `extract_video_frames` function call:

```python
frames = extract_video_frames(
    temp_file_path, 
    frame_interval=30,  # Extract every Nth frame (lower = more frames)
    max_frames=10       # Maximum frames to analyze (higher = slower but more accurate)
)
```

### Adjusting AI Temperature

In `main.py`, modify the OpenAI API call:

```python
response = client.chat.completions.create(
    model=DEPLOYMENT_NAME,
    messages=[...],
    temperature=0.1,  # Lower = more consistent (0.0-1.0)
    max_tokens=500
)
```

## Troubleshooting

### Common Issues

1. **Module not found errors**:
   ```bash
   pip install -r requirements.txt --upgrade
   ```

2. **Azure OpenAI authentication errors**:
   - Verify your `.env` file has correct credentials
   - Check that your API key is active
   - Ensure your deployment name matches exactly

3. **Video processing fails**:
   - Install system dependencies for OpenCV:
     ```bash
     # Windows: Usually works out of the box
     # Linux: sudo apt-get install python3-opencv
     # macOS: brew install opencv
     ```

4. **CORS errors** (if using different origin):
   - Check CORS settings in `main.py`
   - For production, restrict `allow_origins` to specific domains

5. **Port 8000 already in use**:
   ```bash
   uvicorn main:app --reload --port 8001
   ```

## Limitations & Disclaimer

- **Accuracy**: Detection is approximately 80% accurate
- **Not foolproof**: Results are probabilistic and should be used as guidance, not absolute truth
- **Model dependent**: Accuracy depends on Azure OpenAI model capabilities
- **Processing time**: Videos take longer to process (proportional to frame count)
- **File size**: Large files may timeout or use significant API tokens

## Security Considerations

- The `.env` file contains sensitive credentials - never commit it to version control
- For production deployment:
  - Use environment variables or secret management services
  - Restrict CORS to specific domains
  - Implement rate limiting
  - Add authentication/authorization
  - Use HTTPS

## License

This project is provided as-is for educational and research purposes.

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Review FastAPI documentation: https://fastapi.tiangolo.com
3. Check Azure OpenAI documentation: https://learn.microsoft.com/azure/ai-services/openai/

## Future Enhancements

- [ ] Add support for batch processing
- [ ] Implement result caching
- [ ] Add more detailed frame-by-frame video analysis
- [ ] Support for additional file formats
- [ ] User authentication and history tracking
- [ ] Export results to PDF/CSV
- [ ] Confidence threshold customization
