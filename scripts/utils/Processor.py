# scripts/utils/Processor.py
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
                processed.append(img.copy())
                img.close()
        return processed

class VideoProcessor(BaseProcessor):
    def __init__(self, target_fps=16):
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
        desired_count = min(int(total_frames * self.target_fps / original_fps), total_frames)
        desired_count = max(desired_count, 1)

        frames = []
        for _ in range(desired_count):
            pos = _ * 1000 * (1/self.target_fps)
            cap.set(cv2.CAP_PROP_POS_MSEC, pos)
            ret, frame = cap.read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(Image.fromarray(frame_rgb))

        cap.release()
        return frames