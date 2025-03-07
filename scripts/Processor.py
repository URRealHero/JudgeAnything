import os
import json
import base64
import warnings
from io import BytesIO
from PIL import Image
import cv2  # OpenCV for faster frame extraction
import audiosegment
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed  # For parallel processing
import soundfile as sf


class AudioProcessor:
    """处理音频文件的类"""
    
    def process(self, audio_files):
        """
        处理音频文件列表
        
        Args:
            audio_files (list): 音频文件路径列表
            
        Returns:
            list: 包含处理后音频数据的列表
        """
        processed = []
        for file in audio_files:
            data, samplerate = sf.read(file)
            processed.append(data)
        return processed

class ImageProcessor:
    """处理图像文件的类"""
    
    def process(self, image_files):
        """
        处理图像文件列表
        
        Args:
            image_files (list): 图像文件路径列表
            
        Returns:
            list: 包含处理后图像对象的列表
        """
        processed = []
        for file in image_files:
            with Image.open(file) as img:
                processed.append(img)
        return processed


class VideoProcessor:
    """处理视频文件的类"""

    def __init__(self, uniq_id=None, temp_dir='temp/videos'):
        self.temp_dir = temp_dir
        self.uniq_id = uniq_id
        os.makedirs(self.temp_dir, exist_ok=True)

    def process(self, video_files, target_fps=16):
        """
        处理视频文件列表，按目标FPS提取帧

        Args:
            video_files (list): 视频文件路径列表
            target_fps (int, 可选): 目标帧率，默认16fps

        Returns:
            list: 每个元素是对应视频的帧列表（PIL.Image对象）
        """
        return_list = []
        for file in video_files:
            frames = self._extract_frames(file, target_fps)
            return_list.extend(frames)
        
        return return_list

    def _extract_frames(self, video_path, target_fps=16):
        """从单个视频文件中提取帧"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return []

        original_fps = cap.get(cv2.CAP_PROP_FPS)
        if original_fps <= 0:
            warnings.warn(f"No valid FPS found for {video_path}, defaulting to 30fps")
            original_fps = 30

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / original_fps if original_fps != 0 else 0

        desired_count = min(int(duration * target_fps), total_frames)
        desired_count = max(desired_count, 1)

        interval = duration / desired_count if desired_count > 0 else 0
        frames = []

        for _ in range(desired_count):
            cap.set(cv2.CAP_PROP_POS_MSEC, round(interval * _ * 1000))
            ret, frame = cap.read()
            if not ret:
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame_rgb))

        cap.release()
        return frames