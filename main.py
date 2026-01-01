from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
import os
import base64
import json
import cv2
import tempfile
import numpy as np
from pathlib import Path
from openai import AzureOpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(title="AI Media Detection API")

# Configure CORS (allow all origins for development)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Azure OpenAI client
client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version="2024-02-01",
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
)

DEPLOYMENT_NAME = os.getenv("DEPLOYMENT_NAME")

# Ensure static directory exists
static_dir = Path(__file__).parent / "static"
static_dir.mkdir(exist_ok=True)

# Mount static files
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


def analyze_image_with_openai(base64_image: str) -> dict:
    """
    Analyze an image using Azure OpenAI Vision model.
    Returns detection result with is_ai_generated, confidence, and explanation.
    """
    try:
        prompt = """Analyze this image/video frame for signs of AI generation, deepfakes, or digital manipulation. Look for:
- Facial inconsistencies (unnatural blending at face edges, misaligned features)
- Temporal artifacts (flickering, warping between frames)
- Unnatural textures or patterns (especially in skin, hair, teeth)
- Lighting inconsistencies (shadows not matching light sources)
- Unusual artifacts or distortions (glitches, blurring)
- Unrealistic details or compositions
- Deepfake indicators (face-body mismatch, inconsistent head movements)
- AI generation patterns (symmetry issues, blending errors, uncanny valley effects)

Be SENSITIVE to subtle signs. Even minor inconsistencies may indicate manipulation.

Respond ONLY with valid JSON in this exact format:
{"is_ai_generated": true/false, "confidence": 0.0-1.0, "explanation": "brief explanation"}"""

        response = client.chat.completions.create(
            model=DEPLOYMENT_NAME,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            temperature=0.1,
            max_tokens=500
        )

        # Parse the response
        content = response.choices[0].message.content.strip()
        
        # Try to extract JSON from the response
        try:
            # Remove markdown code blocks if present
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
            
            result = json.loads(content)
            
            # Validate required fields
            if not all(k in result for k in ["is_ai_generated", "confidence", "explanation"]):
                raise ValueError("Missing required fields in response")
            
            return result
        except json.JSONDecodeError:
            # Fallback if JSON parsing fails
            return {
                "is_ai_generated": False,
                "confidence": 0.0,
                "explanation": "Unable to parse AI detection response"
            }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OpenAI API error: {str(e)}")


def encode_image_to_base64(image_bytes: bytes) -> str:
    """Encode image bytes to base64 string."""
    return base64.b64encode(image_bytes).decode('utf-8')


def extract_video_frames(video_path: str, frame_interval: int = 30, max_frames: int = 10) -> list:
    """
    Extract key frames from video for analysis.
    Returns list of tuples: (base64_encoded_frame, opencv_frame_array)
    """
    frames = []
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError("Unable to open video file")
    
    frame_count = 0
    extracted_count = 0
    
    try:
        while extracted_count < max_frames:
            ret, frame = cap.read()
            
            if not ret:
                break
            
            # Extract frame at specified intervals
            if frame_count % frame_interval == 0:
                # Encode frame as JPEG for OpenAI
                _, buffer = cv2.imencode('.jpg', frame)
                frame_base64 = base64.b64encode(buffer).decode('utf-8')
                frames.append((frame_base64, frame))
                extracted_count += 1
            
            frame_count += 1
    
    finally:
        cap.release()
    
    return frames


def analyze_frame_quality(frame: np.ndarray) -> dict:
    """
    Analyze frame for technical indicators of manipulation using OpenCV.
    Returns a dict with technical analysis results.
    """
    issues = []
    score = 0.0
    
    # Convert to different color spaces for analysis
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # 1. Check for compression artifacts (deepfakes often have inconsistent compression)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    if laplacian_var < 100:  # Low variance suggests over-smoothing/heavy compression
        issues.append("Unusual compression artifacts detected")
        score += 0.3
    
    # 2. Detect face region and check for blending artifacts
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    
    if len(faces) > 0:
        for (x, y, w, h) in faces:
            # Extract face region
            face_region = gray[y:y+h, x:x+w]
            
            # Check edge consistency around face boundary
            # Deepfakes often have blending issues at face edges
            border_width = int(w * 0.05)
            if border_width > 0:
                top_edge = face_region[:border_width, :]
                bottom_edge = face_region[-border_width:, :]
                left_edge = face_region[:, :border_width]
                right_edge = face_region[:, -border_width:]
                
                # Calculate edge variance
                edge_variance = np.mean([
                    np.std(top_edge), np.std(bottom_edge),
                    np.std(left_edge), np.std(right_edge)
                ])
                
                # Low edge variance can indicate blending
                if edge_variance < 15:
                    issues.append("Face boundary blending detected")
                    score += 0.4
            
            # Check for color consistency in face region
            face_hsv = hsv[y:y+h, x:x+w]
            hue_std = np.std(face_hsv[:,:,0])
            
            # Deepfakes sometimes have unnatural color uniformity
            if hue_std < 8:
                issues.append("Unnatural skin tone uniformity")
                score += 0.2
    
    # 3. Check for frequency domain anomalies
    # Deepfakes often lack high-frequency details
    dft = cv2.dft(np.float32(gray), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shift[:,:,0], dft_shift[:,:,1]) + 1)
    
    # Check high frequency content
    rows, cols = gray.shape
    crow, ccol = rows // 2, cols // 2
    high_freq_region = magnitude_spectrum[crow-30:crow+30, ccol-30:ccol+30]
    high_freq_mean = np.mean(high_freq_region)
    
    if high_freq_mean < 50:  # Low high-frequency content
        issues.append("Missing high-frequency details (typical of AI generation)")
        score += 0.35
    
    # 4. Check for noise inconsistency
    # Real videos have consistent noise; deepfakes often don't
    noise = gray.astype(float) - cv2.GaussianBlur(gray, (5, 5), 0).astype(float)
    noise_std = np.std(noise)
    
    if noise_std < 5:  # Unnaturally low noise
        issues.append("Inconsistent or missing natural camera noise")
        score += 0.25
    
    return {
        "technical_score": min(score, 1.0),
        "issues_found": issues,
        "has_face": len(faces) > 0
    }


@app.get("/")
async def root():
    """Redirect to static frontend."""
    return {"message": "AI Media Detection API. Visit /static/index.html for the web interface."}


@app.post("/detect-image")
async def detect_image(file: UploadFile = File(...)):
    """
    Detect if an uploaded image is AI-generated.
    """
    # Validate file type
    allowed_types = ["image/jpeg", "image/jpg", "image/png", "image/webp"]
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Allowed: {', '.join(allowed_types)}"
        )
    
    try:
        # Read and encode image
        image_bytes = await file.read()
        base64_image = encode_image_to_base64(image_bytes)
        
        # Analyze with OpenAI
        result = analyze_image_with_openai(base64_image)
        
        return JSONResponse(content=result)
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")


@app.post("/detect-video")
async def detect_video(file: UploadFile = File(...)):
    """
    Detect if an uploaded video contains AI-generated or modified content.
    Analyzes multiple frames and returns aggregated results.
    """
    # Validate file type
    allowed_types = ["video/mp4", "video/mpeg", "video/quicktime", "video/x-msvideo"]
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Allowed: {', '.join(allowed_types)}"
        )
    
    temp_file = None
    
    try:
        # Save video to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name
        
        # Extract frames - more frames for better detection
        frames = extract_video_frames(temp_file_path, frame_interval=15, max_frames=15)
        
        if not frames:
            raise HTTPException(status_code=400, detail="No frames could be extracted from video")
        
        # Analyze each frame with both OpenAI and technical analysis
        detections = []
        technical_scores = []
        
        for i, (frame_base64, frame_array) in enumerate(frames, 1):
            # OpenAI vision analysis
            ai_result = analyze_image_with_openai(frame_base64)
            ai_result["frame_number"] = i
            
            # Technical analysis using OpenCV
            tech_result = analyze_frame_quality(frame_array)
            
            # Combine both analyses
            combined_confidence = (ai_result["confidence"] * 0.5) + (tech_result["technical_score"] * 0.5)
            
            detection = {
                "frame_number": i,
                "ai_confidence": ai_result["confidence"],
                "ai_detected": ai_result["is_ai_generated"],
                "ai_explanation": ai_result["explanation"],
                "technical_score": tech_result["technical_score"],
                "technical_issues": tech_result["issues_found"],
                "combined_confidence": combined_confidence,
                "is_suspicious": combined_confidence > 0.35  # Lower threshold for detection
            }
            
            detections.append(detection)
            technical_scores.append(tech_result["technical_score"])
        
        # Aggregate results - flag if ANY frame is suspicious
        suspicious_frames = [d for d in detections if d["is_suspicious"]]
        avg_technical_score = np.mean(technical_scores)
        
        if suspicious_frames or avg_technical_score > 0.3:
            # Video is likely AI-modified/deepfake
            is_ai_modified = True
            
            # Get the most suspicious frame
            max_detection = max(detections, key=lambda x: x["combined_confidence"])
            final_confidence = max_detection["combined_confidence"]
            
            # Compile evidence
            all_issues = []
            for d in suspicious_frames[:5]:  # Top 5 suspicious frames
                frame_issues = f"Frame {d['frame_number']}: "
                if d['technical_issues']:
                    frame_issues += ", ".join(d['technical_issues'][:2])
                else:
                    frame_issues += d['ai_explanation']
                all_issues.append(frame_issues)
            
            explanation_text = f"⚠️ DEEPFAKE/AI DETECTED in {len(suspicious_frames)} of {len(frames)} frames. Technical analysis score: {avg_technical_score:.2f}. Evidence: {'; '.join(all_issues)}"
        else:
            # Likely authentic
            is_ai_modified = False
            final_confidence = 1.0 - avg_technical_score
            explanation_text = f"Analyzed {len(frames)} frames with hybrid detection. No significant deepfake/AI artifacts detected. Technical authenticity score: {1.0 - avg_technical_score:.2f}"
        
        aggregated_result = {
            "is_ai_generated": is_ai_modified,
            "confidence": round(final_confidence, 2),
            "explanation": explanation_text,
            "frames_analyzed": len(frames),
            "suspicious_frames": len(suspicious_frames) if suspicious_frames else 0,
            "detection_method": "Hybrid (Azure OpenAI + OpenCV Technical Analysis)",
            "technical_analysis_score": round(avg_technical_score, 2)
        }
        
        return JSONResponse(content=aggregated_result)
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")
    
    finally:
        # Clean up temporary file
        if temp_file:
            try:
                os.unlink(temp_file_path)
            except:
                pass


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
