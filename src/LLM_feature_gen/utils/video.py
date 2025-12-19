# src/LLM_feature_gen/utils/video.py
import os
import time
import requests
import ffmpeg
import cv2
import base64
import io
import numpy as np
from PIL import Image
from typing import List, Tuple, Optional


def _get_frame_signature(image: np.ndarray) -> np.ndarray:
    """
    Creates a 'fingerprint' of the image combining color (HSV) and structure.
    Helps group similar shots (e.g., zoom-in vs zoom-out of the same building).
    """
    # 1. Color profile (HSV is robust to lighting changes)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [16, 16], [0, 180, 0, 256])
    cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)

    # 2. Structural layout (tiny thumbnail)
    small = cv2.resize(image, (16, 16))
    small_flat = small.flatten().astype(np.float32) / 255.0

    # Combine them (giving more weight to color histogram)
    return np.concatenate([hist.flatten() * 5, small_flat])


def extract_key_frames(video_path: str, frame_limit: int = 6, sharpness_threshold: float = 40.0) -> List[str]:
    """
    Selects diverse keyframes using K-Means clustering.
    Instead of looking for motion, it groups similar scenes and picks the sharpest image from each group.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return []

    fps = cap.get(cv2.CAP_PROP_FPS) or 25

    # Step 1: Gather candidates (~2 frames per second to be efficient)
    sample_rate = max(1, int(fps / 2))
    candidates = []
    frame_idx = 0

    while True:
        is_read, frame = cap.read()
        if not is_read:
            break

        frame_idx += 1
        if frame_idx % sample_rate != 0:
            continue

        # Skip blurry frames immediately
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()

        if sharpness < sharpness_threshold:
            continue

        candidates.append({
            "frame": frame,
            "timestamp": frame_idx / fps,
            "sharpness": sharpness,
            "signature": _get_frame_signature(frame)
        })

    cap.release()

    if not candidates:
        return []

    # Intelligent Selection
    if len(candidates) <= frame_limit:
        # Not enough candidates? Take them all.
        final_candidates = candidates
    else:
        # Group frames by visual similarity
        data = np.array([c["signature"] for c in candidates], dtype=np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

        # Run K-Means
        _, labels, _ = cv2.kmeans(data, frame_limit, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

        selected_indices = []
        for i in range(frame_limit):
            # Find all frames belonging to this cluster
            cluster_indices = [idx for idx, label in enumerate(labels) if label == i]

            if not cluster_indices:
                continue

            # Pick the sharpest frame from this cluster
            best_in_cluster = max(cluster_indices, key=lambda idx: candidates[idx]["sharpness"])
            selected_indices.append(best_in_cluster)

        final_candidates = [candidates[i] for i in selected_indices]

    # Sort by time and prepare output
    final_candidates.sort(key=lambda x: x["timestamp"])

    b64_list = []
    for item in final_candidates:
        frame = item["frame"]

        # Burn-in timestamp for the LLM
        seconds = int(item["timestamp"])
        time_str = f"{seconds // 60:02d}:{seconds % 60:02d}"

        # Draw text (black border + white text for readability)
        cv2.putText(frame, time_str, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(frame, time_str, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)

        # Convert to base64
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        buffered = io.BytesIO()
        img.save(buffered, format="JPEG")
        b64_list.append(base64.b64encode(buffered.getvalue()).decode('utf-8'))

    return b64_list


def transcribe_video(file_path: str) -> str:
    """
    Extracts audio using FFmpeg and calls Azure OpenAI Whisper/Speech endpoint.
    """
    endpoint = os.getenv("SPEECH_AZURE_OPENAI_ENDPOINT")
    api_key = os.getenv("SPEECH_AZURE_OPENAI_API_KEY")
    api_version = os.getenv("SPEECH_AZURE_OPENAI_API_VERSION")
    deployment = os.getenv("SPEECH_GPT4O_MINI_TRANSCRIBE_DEPLOYMENT_NAME")  # Or high_quality var

    if not (endpoint and api_key and deployment):
        return "(Transcription skipped: Missing environment variables)"

    base_name = os.path.splitext(os.path.basename(file_path))[0]
    temp_audio_path = f"temp_audio_{base_name}_{int(time.time())}.wav"

    try:
        ffmpeg.input(file_path).output(
            temp_audio_path, acodec='pcm_s16le', ac=1, ar='16k'
        ).run(quiet=True, overwrite_output=True)

        url = f"{endpoint}openai/deployments/{deployment}/audio/transcriptions?api-version={api_version}"

        with open(temp_audio_path, "rb") as f:
            response = requests.post(
                url,
                headers={"api-key": api_key},
                files={"file": (os.path.basename(temp_audio_path), f, "audio/wav")},
                data={"model": "gpt-4o-transcribe", "response_format": "json"},
                timeout=300,
            )

        if response.status_code == 200:
            return response.json().get("text", "(no text found)")
        else:
            return f"(Transcription failed: {response.status_code})"

    except Exception as e:
        return f"(Transcription error: {str(e)})"
    finally:
        if os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)