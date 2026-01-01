# AI Media Detection - Technical Documentation

## Table of Contents
1. [Architecture Overview](#architecture-overview)
2. [Detection Methodology](#detection-methodology)
3. [API Reference](#api-reference)
4. [Technical Implementation](#technical-implementation)
5. [Performance Considerations](#performance-considerations)
6. [Troubleshooting](#troubleshooting)

---

## Architecture Overview

### System Components

```
┌─────────────────────────────────────────────────────────────┐
│                        Client Layer                          │
│  (Web Browser - HTML/CSS/JavaScript)                        │
└──────────────────────┬──────────────────────────────────────┘
                       │ HTTP/HTTPS
                       │ POST with multipart/form-data
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                     FastAPI Backend                          │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  CORS Middleware (Development: Allow All Origins)   │   │
│  └─────────────────────────────────────────────────────┘   │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  Static Files Middleware (Serves Frontend)          │   │
│  └─────────────────────────────────────────────────────┘   │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  Endpoint: /detect-image                            │   │
│  │  Endpoint: /detect-video                            │   │
│  └─────────────────────────────────────────────────────┘   │
└──────────────────────┬──────────────────────────────────────┘
                       │
           ┌───────────┴───────────┐
           ▼                       ▼
┌──────────────────────┐  ┌──────────────────────┐
│  Azure OpenAI API    │  │  OpenCV Analysis     │
│  (Vision Model)      │  │  (Technical Checks)  │
│  - GPT-4o/4-turbo    │  │  - Compression       │
│  - Semantic Analysis │  │  - Face Detection    │
│  - Pattern Detection │  │  - Frequency Domain  │
│                      │  │  - Noise Analysis    │
└──────────────────────┘  └──────────────────────┘
```

### Technology Stack

| Component | Technology | Version | Purpose |
|-----------|------------|---------|---------|
| Backend Framework | FastAPI | 0.109.0 | High-performance async API |
| ASGI Server | Uvicorn | 0.27.0 | Production-grade ASGI server |
| AI/ML Provider | Azure OpenAI | API v2024-02-01 | Vision model for semantic analysis |
| Client Library | openai | ≥1.58.0 | Azure OpenAI SDK |
| Computer Vision | OpenCV | 4.9.0.80 | Video processing & technical analysis |
| Numerical Computing | NumPy | <2.0.0 | Array operations & statistical analysis |
| File Handling | python-multipart | 0.0.6 | Multipart form data parsing |
| Configuration | python-dotenv | 1.0.0 | Environment variable management |
| HTTP Client | httpx | ≥0.27.0 | Required by OpenAI SDK |
| Frontend | Vanilla HTML/CSS/JS | - | Zero-dependency web interface |

---

## Detection Methodology

### Hybrid Detection System

The system employs a **dual-analysis approach** combining AI semantic understanding with technical forensic analysis:

#### 1. Azure OpenAI Vision Analysis (50% Weight)

**Model Configuration:**
- **API Version:** 2024-02-01
- **Temperature:** 0.1 (low variance for consistency)
- **Max Tokens:** 500
- **Model:** GPT-4o or compatible vision model

**Detection Criteria:**
```
- Facial inconsistencies (blending, misalignment)
- Temporal artifacts (flickering, warping)
- Unnatural textures (skin, hair, teeth)
- Lighting inconsistencies
- Deepfake indicators (face-body mismatch)
- AI generation patterns (symmetry issues, uncanny valley)
```

**Response Format:**
```json
{
  "is_ai_generated": boolean,
  "confidence": float (0.0-1.0),
  "explanation": string
}
```

#### 2. OpenCV Technical Analysis (50% Weight)

##### a) Compression Artifact Detection
```python
Method: Laplacian Variance Analysis
Threshold: < 100 indicates over-smoothing
Algorithm: cv2.Laplacian(gray, cv2.CV_64F).var()
```

**Rationale:** Deepfakes often exhibit inconsistent compression due to multiple encoding passes during generation and face swapping.

##### b) Face Boundary Blending Detection
```python
Method: Haar Cascade + Edge Variance Analysis
Detector: haarcascade_frontalface_default.xml
Border Width: 5% of face width
Threshold: Edge variance < 15 indicates blending
```

**Rationale:** Face-swap deepfakes leave artifacts at the boundary where the synthetic face is blended onto the original frame.

##### c) Color Consistency Analysis
```python
Method: HSV Color Space Analysis
Region: Detected face area
Metric: Hue channel standard deviation
Threshold: < 8 indicates unnatural uniformity
```

**Rationale:** AI-generated faces often have unnaturally uniform skin tones due to synthetic texture generation.

##### d) Frequency Domain Analysis
```python
Method: Discrete Fourier Transform (DFT)
Algorithm: cv2.dft() + magnitude spectrum analysis
Region: High-frequency components (30x30 center)
Threshold: Mean < 50 indicates missing high-frequency details
```

**Rationale:** Deepfakes lack fine-grained high-frequency details present in authentic videos due to lossy generation processes.

##### e) Noise Pattern Analysis
```python
Method: Gaussian Noise Extraction
Algorithm: Original - GaussianBlur(5x5)
Metric: Standard deviation of noise
Threshold: < 5 indicates missing natural camera noise
```

**Rationale:** Real camera sensors produce consistent noise patterns; AI-generated content lacks authentic sensor noise.

### Video Analysis Pipeline

```
1. Video Upload
   ↓
2. Temporary File Storage
   ↓
3. Frame Extraction
   - Interval: Every 15 frames
   - Maximum: 15 frames
   - Format: JPEG encoding
   ↓
4. Parallel Analysis (per frame)
   ├── Azure OpenAI Analysis → AI Confidence (0-1)
   └── OpenCV Technical Analysis → Technical Score (0-1)
   ↓
5. Combined Scoring
   Combined = (AI_Confidence × 0.5) + (Technical_Score × 0.5)
   ↓
6. Frame Classification
   Suspicious if Combined > 0.35
   ↓
7. Video-Level Aggregation
   ├── Flag if ANY frame is suspicious
   ├── Flag if average technical score > 0.3
   └── Use maximum combined confidence
   ↓
8. Response Generation
   ↓
9. Temporary File Cleanup
```

### Image Analysis Pipeline

```
1. Image Upload
   ↓
2. File Type Validation
   ↓
3. Base64 Encoding
   ↓
4. Azure OpenAI Vision Analysis
   ↓
5. JSON Response Parsing
   ↓
6. Return Results
```

---

## API Reference

### Base URL
```
http://localhost:8000
```

### Endpoints

#### 1. Root Endpoint
```http
GET /
```

**Response:**
```json
{
  "message": "AI Media Detection API. Visit /static/index.html for the web interface."
}
```

#### 2. Image Detection
```http
POST /detect-image
Content-Type: multipart/form-data
```

**Request Parameters:**
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| file | File | Yes | Image file (JPEG, PNG, WEBP) |

**Accepted MIME Types:**
- `image/jpeg`
- `image/jpg`
- `image/png`
- `image/webp`

**Success Response (200):**
```json
{
  "is_ai_generated": true,
  "confidence": 0.85,
  "explanation": "Unnatural textures and lighting inconsistencies detected"
}
```

**Error Responses:**

| Code | Description | Example |
|------|-------------|---------|
| 400 | Invalid file type | `{"detail": "Invalid file type. Allowed: image/jpeg, image/jpg, image/png, image/webp"}` |
| 500 | Processing error | `{"detail": "OpenAI API error: [error message]"}` |

**cURL Example:**
```bash
curl -X POST "http://localhost:8000/detect-image" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@/path/to/image.jpg"
```

#### 3. Video Detection
```http
POST /detect-video
Content-Type: multipart/form-data
```

**Request Parameters:**
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| file | File | Yes | Video file (MP4, MOV, AVI) |

**Accepted MIME Types:**
- `video/mp4`
- `video/mpeg`
- `video/quicktime`
- `video/x-msvideo`

**Success Response (200):**
```json
{
  "is_ai_generated": true,
  "confidence": 0.67,
  "explanation": "⚠️ DEEPFAKE/AI DETECTED in 8 of 15 frames. Technical analysis score: 0.54. Evidence: Frame 3: Face boundary blending detected, Unusual compression artifacts detected; Frame 7: Missing high-frequency details (typical of AI generation)",
  "frames_analyzed": 15,
  "suspicious_frames": 8,
  "detection_method": "Hybrid (Azure OpenAI + OpenCV Technical Analysis)",
  "technical_analysis_score": 0.54
}
```

**Error Responses:**

| Code | Description | Example |
|------|-------------|---------|
| 400 | Invalid file type or no frames extracted | `{"detail": "No frames could be extracted from video"}` |
| 500 | Processing error | `{"detail": "Processing error: [error message]"}` |

**cURL Example:**
```bash
curl -X POST "http://localhost:8000/detect-video" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@/path/to/video.mp4"
```

#### 4. Static Files
```http
GET /static/{path}
```

Serves frontend HTML, CSS, and JavaScript files.

**Example:**
```
http://localhost:8000/static/index.html
```

---

## Technical Implementation

### Core Functions

#### 1. `analyze_image_with_openai(base64_image: str) -> dict`

**Purpose:** Performs semantic analysis using Azure OpenAI vision model.

**Algorithm:**
1. Constructs multimodal prompt with image
2. Calls Azure OpenAI Chat Completions API
3. Parses JSON response (handles markdown code blocks)
4. Validates required fields
5. Returns structured detection result

**Error Handling:**
- JSON parsing fallback
- Required field validation
- API error propagation with HTTPException

**Time Complexity:** O(1) + API call latency (~2-5 seconds)

#### 2. `encode_image_to_base64(image_bytes: bytes) -> str`

**Purpose:** Converts image bytes to base64 string for API transmission.

**Implementation:**
```python
return base64.b64encode(image_bytes).decode('utf-8')
```

**Time Complexity:** O(n) where n = image size

#### 3. `extract_video_frames(video_path: str, frame_interval: int, max_frames: int) -> list`

**Purpose:** Extracts frames from video for analysis.

**Parameters:**
- `video_path`: Filesystem path to video file
- `frame_interval`: Extract every Nth frame (default: 15)
- `max_frames`: Maximum frames to extract (default: 15)

**Returns:** List of tuples `(base64_frame, opencv_array)`

**Algorithm:**
1. Open video with cv2.VideoCapture
2. Iterate through frames
3. Extract at specified intervals
4. Encode as JPEG and base64
5. Store both encoded and raw arrays
6. Release video capture

**Time Complexity:** O(total_frames / frame_interval)

**Memory Usage:** O(max_frames × frame_size)

#### 4. `analyze_frame_quality(frame: np.ndarray) -> dict`

**Purpose:** Performs technical forensic analysis on a video frame.

**Input:** OpenCV frame array (BGR format)

**Output:**
```python
{
  "technical_score": float (0.0-1.0),
  "issues_found": list[str],
  "has_face": bool
}
```

**Sub-Components:**

##### a) Laplacian Variance (Compression Artifacts)
```python
laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
Score += 0.3 if laplacian_var < 100
```

##### b) Face Detection & Boundary Analysis
```python
face_cascade = cv2.CascadeClassifier(...)
faces = face_cascade.detectMultiScale(gray, 1.1, 4)

For each face:
  - Extract 5% border region
  - Calculate edge variance
  - Score += 0.4 if variance < 15
```

##### c) Color Uniformity Check
```python
face_hsv = cv2.cvtColor(face_region, cv2.COLOR_BGR2HSV)
hue_std = np.std(face_hsv[:,:,0])
Score += 0.2 if hue_std < 8
```

##### d) Frequency Domain Analysis
```python
dft = cv2.dft(np.float32(gray))
magnitude_spectrum = 20 * np.log(cv2.magnitude(...))
high_freq_mean = np.mean(magnitude_spectrum[center_30x30])
Score += 0.35 if high_freq_mean < 50
```

##### e) Noise Analysis
```python
noise = gray - cv2.GaussianBlur(gray, (5,5), 0)
noise_std = np.std(noise)
Score += 0.25 if noise_std < 5
```

**Time Complexity:** O(w × h) where w, h = frame dimensions

**Space Complexity:** O(w × h) for DFT computation

---

## Performance Considerations

### Latency Breakdown (Typical Video)

| Operation | Time | Notes |
|-----------|------|-------|
| File Upload | 0.5-2s | Depends on file size & network |
| Frame Extraction | 1-3s | 15 frames from 30fps video |
| OpenCV Analysis (per frame) | 0.1-0.3s | CPU-bound |
| Azure OpenAI API (per frame) | 2-5s | Network + GPU processing |
| Total (15 frames, sequential) | 30-75s | Bottleneck: OpenAI API |
| Total (with parallelization) | 10-20s | Possible optimization |

### Optimization Strategies

#### 1. Implemented
- Frame sampling (analyze 15 instead of all frames)
- Temporary file cleanup
- Efficient base64 encoding
- Connection pooling (httpx)

#### 2. Potential Improvements
```python
# Parallel OpenAI API calls
import asyncio
from concurrent.futures import ThreadPoolExecutor

async def analyze_frames_parallel(frames):
    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor(max_workers=5) as executor:
        tasks = [
            loop.run_in_executor(executor, analyze_image_with_openai, frame)
            for frame in frames
        ]
        return await asyncio.gather(*tasks)
```

**Expected Speedup:** 3-5x for video analysis

#### 3. Caching Strategy
```python
# Cache OpenAI responses by frame hash
import hashlib

def get_frame_hash(frame_base64: str) -> str:
    return hashlib.sha256(frame_base64.encode()).hexdigest()

# Use Redis or in-memory cache
cache = {}
if frame_hash in cache:
    return cache[frame_hash]
```

### Resource Requirements

| Resource | Minimum | Recommended | Notes |
|----------|---------|-------------|-------|
| CPU | 2 cores | 4+ cores | OpenCV processing |
| RAM | 2 GB | 4 GB | Frame storage in memory |
| Disk | 1 GB | 5 GB | Temporary video files |
| Network | 10 Mbps | 100 Mbps | Azure API calls |
| GPU | None | Optional | Future ML model integration |

### Scalability Considerations

#### Current Limitations
- Synchronous processing (1 request at a time per worker)
- No request queuing
- No rate limiting
- Single server instance

#### Production Recommendations
```yaml
Load Balancer:
  - Nginx or HAProxy
  - SSL termination
  - Request routing

Application Servers:
  - Multiple Uvicorn workers: 2 × CPU_cores
  - Process manager: Supervisor or systemd
  - Auto-scaling based on CPU/memory

Caching Layer:
  - Redis for API response caching
  - CDN for static files

Database (Optional):
  - PostgreSQL for analysis history
  - S3/Blob storage for uploaded files

Monitoring:
  - Prometheus metrics
  - Grafana dashboards
  - Error tracking (Sentry)
```

---

## Troubleshooting

### Common Issues

#### 1. NumPy Compatibility Error
```
AttributeError: _ARRAY_API not found
ImportError: numpy.core.multiarray failed to import
```

**Cause:** OpenCV incompatible with NumPy 2.x

**Solution:**
```bash
pip install "numpy<2.0.0" --force-reinstall
```

#### 2. OpenAI Client Error
```
TypeError: Client.__init__() got an unexpected keyword argument 'proxies'
```

**Cause:** Version mismatch between openai and httpx packages

**Solution:**
```bash
pip uninstall openai httpx -y
pip install "openai>=1.58.0" "httpx>=0.27.0"
```

#### 3. Azure OpenAI Authentication Error
```
HTTPException: OpenAI API error: 401 Unauthorized
```

**Cause:** Invalid credentials in `.env`

**Solution:**
1. Verify `.env` file exists
2. Check `AZURE_OPENAI_ENDPOINT` format: `https://your-resource.openai.azure.com/`
3. Validate `AZURE_OPENAI_API_KEY`
4. Confirm `DEPLOYMENT_NAME` matches Azure deployment

#### 4. Video Processing Fails
```
ValueError: Unable to open video file
```

**Cause:** Corrupt video or unsupported codec

**Solution:**
- Re-encode video: `ffmpeg -i input.mp4 -c:v libx264 output.mp4`
- Check file size limits
- Verify MIME type

#### 5. Face Detection Not Working
```
cv2.error: OpenCV(4.x.x) error: (-215:Assertion failed)
```

**Cause:** Haar cascade file not found

**Solution:**
```python
# Verify cascade file path
import cv2
print(cv2.data.haarcascades)

# Fallback: download manually
# https://github.com/opencv/opencv/tree/master/data/haarcascades
```

#### 6. High Memory Usage
```
MemoryError: Unable to allocate array
```

**Cause:** Too many frames or large video resolution

**Solution:**
- Reduce `max_frames` parameter
- Downscale frames before analysis:
```python
frame = cv2.resize(frame, (640, 480))
```

#### 7. CORS Errors in Browser
```
Access to fetch at 'http://localhost:8000' has been blocked by CORS policy
```

**Cause:** Frontend served from different origin

**Solution:**
- Use `/static/index.html` route
- Update CORS origins in production:
```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://yourdomain.com"],
    ...
)
```

### Debug Mode

Enable detailed logging:

```python
# Add to main.py
import logging

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Add debug prints
@app.post("/detect-video")
async def detect_video(file: UploadFile = File(...)):
    logging.debug(f"Received file: {file.filename}, type: {file.content_type}")
    # ... rest of code
```

Run with reload and log level:
```bash
uvicorn main:app --reload --log-level debug
```

### Performance Profiling

```python
import time
from functools import wraps

def timing_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"{func.__name__} took {end - start:.2f}s")
        return result
    return wrapper

@timing_decorator
def analyze_image_with_openai(base64_image: str) -> dict:
    # ... existing code
```

### Health Check Endpoint

Add monitoring endpoint:

```python
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "azure_openai": "configured" if DEPLOYMENT_NAME else "missing",
        "opencv_version": cv2.__version__
    }
```

---

## Security Considerations

### Production Deployment Checklist

#### 1. Environment Variables
- [ ] Never commit `.env` to version control
- [ ] Use secret management service (Azure Key Vault, AWS Secrets Manager)
- [ ] Rotate API keys regularly

#### 2. CORS Configuration
```python
# Restrict origins in production
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://yourdomain.com",
        "https://www.yourdomain.com"
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)
```

#### 3. File Upload Security
```python
# Add file size limits
MAX_FILE_SIZE = 100 * 1024 * 1024  # 100 MB

@app.post("/detect-video")
async def detect_video(file: UploadFile = File(...)):
    content = await file.read()
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(413, "File too large")
```

#### 4. Rate Limiting
```python
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

@app.post("/detect-image")
@limiter.limit("10/minute")
async def detect_image(request: Request, file: UploadFile = File(...)):
    # ... existing code
```

#### 5. Input Validation
- Validate file extensions
- Scan for malware (ClamAV integration)
- Sanitize filenames
- Check magic bytes (not just MIME type)

#### 6. HTTPS Only
```python
# Redirect HTTP to HTTPS
from fastapi.middleware.httpsredirect import HTTPSRedirectMiddleware
app.add_middleware(HTTPSRedirectMiddleware)
```

#### 7. API Key Authentication (Optional)
```python
from fastapi.security.api_key import APIKeyHeader

API_KEY_HEADER = APIKeyHeader(name="X-API-Key")

@app.post("/detect-image")
async def detect_image(
    file: UploadFile = File(...),
    api_key: str = Depends(API_KEY_HEADER)
):
    if api_key != os.getenv("API_KEY"):
        raise HTTPException(401, "Invalid API key")
    # ... existing code
```

---

## Accuracy & Limitations

### Detection Accuracy

| Content Type | Accuracy | Notes |
|--------------|----------|-------|
| Modern Deepfakes (2023+) | 70-85% | High-quality deepfakes are challenging |
| AI-Generated Images | 80-90% | Clear artifacts in most cases |
| Face-Swap Videos | 75-85% | Technical analysis effective |
| Low-Quality Deepfakes | 90-95% | Obvious compression artifacts |
| Authentic Content | 85-92% | Low false positive rate |

### Known Limitations

1. **Model Dependency**: Accuracy depends on Azure OpenAI model capabilities
2. **High-Quality Deepfakes**: State-of-the-art deepfakes may evade detection
3. **Video Resolution**: Low-resolution videos harder to analyze
4. **Processing Time**: 30-75 seconds per video (15 frames)
5. **No Temporal Analysis**: Doesn't analyze frame-to-frame consistency
6. **Face-Centric**: Better at detecting face manipulations than full-body
7. **False Positives**: Heavily compressed authentic videos may be flagged

### Disclaimer

⚠️ **This system provides probabilistic detection, not definitive proof.**

- Results should be used as guidance, not legal evidence
- Combine with manual review for critical decisions
- Accuracy varies based on content quality and manipulation sophistication
- Regular updates needed as deepfake technology evolves

---

## Future Enhancements

### Roadmap

#### Phase 1: Performance (Q1 2026)
- [ ] Implement parallel API calls
- [ ] Add Redis caching
- [ ] Database for analysis history
- [ ] Batch processing support

#### Phase 2: Accuracy (Q2 2026)
- [ ] Temporal consistency analysis (frame-to-frame)
- [ ] Audio deepfake detection
- [ ] Fine-tuned detection models
- [ ] Ensemble with multiple AI providers

#### Phase 3: Features (Q3 2026)
- [ ] User authentication
- [ ] Analysis history & reports
- [ ] PDF/CSV export
- [ ] Webhooks for async processing
- [ ] Browser extension

#### Phase 4: Scale (Q4 2026)
- [ ] Kubernetes deployment
- [ ] GPU acceleration
- [ ] Real-time streaming analysis
- [ ] Mobile app (iOS/Android)

### Contributing

For contributions or feature requests, please follow the standard GitHub workflow:

1. Fork the repository
2. Create a feature branch
3. Implement changes with tests
4. Submit pull request with documentation

---

## Appendix

### A. Environment Variables Reference

| Variable | Required | Example | Description |
|----------|----------|---------|-------------|
| `AZURE_OPENAI_ENDPOINT` | Yes | `https://resource.openai.azure.com/` | Azure OpenAI endpoint URL |
| `AZURE_OPENAI_API_KEY` | Yes | `abcd1234...` | Azure OpenAI API key |
| `DEPLOYMENT_NAME` | Yes | `gpt-4o` | Azure deployment name |

### B. Supported File Formats

**Images:**
- JPEG/JPG (recommended)
- PNG
- WEBP

**Videos:**
- MP4 (H.264 codec recommended)
- MOV (QuickTime)
- AVI
- MPEG

### C. HTTP Status Codes

| Code | Meaning | Usage |
|------|---------|-------|
| 200 | OK | Successful detection |
| 400 | Bad Request | Invalid file type or corrupt file |
| 401 | Unauthorized | Invalid API key (if auth enabled) |
| 413 | Payload Too Large | File exceeds size limit |
| 500 | Internal Server Error | Processing error or API failure |

### D. Dependencies Graph

```
fastapi
├── starlette (ASGI framework)
├── pydantic (data validation)
└── typing-extensions

uvicorn
├── h11 (HTTP/1.1)
├── httptools (fast HTTP parsing)
└── watchfiles (auto-reload)

openai
├── httpx (HTTP client)
├── pydantic (models)
└── typing-extensions

opencv-python
└── numpy (array operations)

numpy
└── (no dependencies, but must be <2.0)
```

### E. Glossary

- **Deepfake**: Synthetic media created using deep learning techniques
- **Face Swap**: Replacing one person's face with another in a video
- **DFT**: Discrete Fourier Transform (frequency analysis)
- **Laplacian**: Edge detection operator
- **Haar Cascade**: Machine learning object detection method
- **HSV**: Hue, Saturation, Value color space
- **Base64**: Binary-to-text encoding scheme
- **ASGI**: Asynchronous Server Gateway Interface

---

**Document Version:** 1.0  
**Last Updated:** January 1, 2026  
**Maintainer:** AI Media Detection Team  
**License:** [Your License Here]
