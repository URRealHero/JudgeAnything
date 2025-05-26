# scripts/utils/Phi4mm/Processor.py
import cv2
import soundfile as sf
from PIL import Image
import numpy as np
import warnings

class BaseProcessor:
    def process(self, file_paths):
        raise NotImplementedError

class AudioProcessor(BaseProcessor):
    def process(self, audio_files):
        processed = []
        for file in audio_files:
            data, sr = sf.read(str(file))
            processed.append((data, sr))
        return processed

class ImageProcessor(BaseProcessor):
    def process(self, image_files):
        processed = []
        for file in image_files:
            with Image.open(str(file)) as img:
                processed.append(img.convert("RGB").copy())
        return processed

class VideoProcessor(BaseProcessor):
    def __init__(self, target_fps=1, max_frames=256):
        self.max_frames = max_frames
        self.target_fps = target_fps

    def process(self, video_files):
        processed = []
        for file in video_files:
            frames = self._extract_frames(str(file))
            processed.extend(frames)
        return processed

    def _extract_frames(self, video_path):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return []

        original_fps = cap.get(cv2.CAP_PROP_FPS)
        if original_fps <= 0:
            warnings.warn(f"Invalid FPS for {video_path}, using default 30fps")
            original_fps = 30

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_sec = total_frames / original_fps
        num_target_frames = int(duration_sec * self.target_fps)
        num_target_frames = max(num_target_frames, 1)

        timestamps = np.linspace(0, duration_sec, num_target_frames, endpoint=False)
        frames = []

        for t in timestamps:
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
            ret, frame = cap.read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(Image.fromarray(frame_rgb))

        cap.release()

        # Uniformly sample if more than max_frames
        if len(frames) > self.max_frames:
            indices = np.linspace(0, len(frames) - 1, self.max_frames, dtype=int)
            frames = [frames[i] for i in indices]

        return frames