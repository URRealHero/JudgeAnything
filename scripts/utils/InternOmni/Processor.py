# processor.py
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoTokenizer, AutoFeatureExtractor, ProcessorMixin
import librosa
import warnings
import cv2 # For VideoProcessor

# Constants from the InternOmni script
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Custom WhisperProcessor class from the InternOmni script
# This class is essential for audio processing, especially for calculating
# audio_len_after_cnn and audio_token_num.
class WhisperProcessor(ProcessorMixin):
    attributes = ["feature_extractor"]
    feature_extractor_class = "WhisperFeatureExtractor" # Typically 'WhisperFeatureExtractor'
    # tokenizer_class = "AutoTokenizer" # Or a specific WhisperTokenizer

    def __init__(self, feature_extractor):
        super().__init__(feature_extractor)
        # feature_extractor and tokenizer are set by ProcessorMixin
        self.current_processor = self.feature_extractor # As per original user script
        self._in_target_context_manager = False

    def get_decoder_prompt_ids(self, task=None, language=None, no_timestamps=True):
        if self.tokenizer is None:
            raise ValueError("Tokenizer not initialized for WhisperProcessor.")
        return self.tokenizer.get_decoder_prompt_ids(task=task, language=language, no_timestamps=no_timestamps)

    def get_T_after_cnn(self, L_in, dilation=1):
        # Evaluated from "[(1,3,1)] + [(1,3,2)]" in the original script
        conv_layers_params = [(1, 3, 1), (1, 3, 2)] # (padding, kernel_size, stride)
        for (padding, kernel_size, stride) in conv_layers_params:
            L_out = L_in + 2 * padding - dilation * (kernel_size - 1) - 1
            L_out = 1 + L_out // stride
            L_in = L_out
        return L_out

    def __call__(self, *args, **kwargs):
        if self._in_target_context_manager:
            return self.current_processor(*args, **kwargs)

        audio = kwargs.pop("audio", None)
        sampling_rate = kwargs.pop("sampling_rate", None)
        text = kwargs.pop("text", None) # For if it's used as a general processor

        # Prioritize kwargs if provided
        if len(args) > 0 and audio is None and text is None:
            # Basic argument handling: assume first arg is audio if not text
            if isinstance(args[0], (np.ndarray, torch.Tensor, list)):
                 audio = args[0]
            elif isinstance(args[0], str):
                 text = args[0]
            else: # Fallback
                 audio = args[0]

        if audio is None and text is None:
            raise ValueError("You need to specify either an 'audio' or 'text' input to process.")

        processed_outputs = {}

        if audio is not None:
            if not hasattr(self.feature_extractor, 'sampling_rate'):
                raise ValueError("Feature extractor must have a 'sampling_rate' attribute.")
            if sampling_rate is None:
                sampling_rate = self.feature_extractor.sampling_rate
            
            # Ensure audio is a 1D numpy array for feature extractors like WhisperFeatureExtractor
            if isinstance(audio, list): audio = np.array(audio, dtype=np.float32)
            if isinstance(audio, torch.Tensor): audio = audio.cpu().numpy()
            if audio.ndim > 1: # Attempt to convert to mono if stereo
                if audio.shape[0] == 2: audio = audio.mean(axis=0)
                elif audio.shape[1] == 2: audio = audio.mean(axis=1)
                else: audio = audio.squeeze() # If singleton dimension
            if audio.ndim > 1:
                 raise ValueError(f"Audio input is expected to be 1D (mono), but got shape {audio.shape}")


            # Call the underlying feature extractor (e.g., WhisperFeatureExtractor)
            # It should handle padding/truncation (e.g., to 30s for Whisper)
            # Pass **kwargs so return_tensors="pt" can be included by the caller
            fe_kwargs = {k:v for k,v in kwargs.items() if k not in ['text']}
            inputs = self.feature_extractor(audio, sampling_rate=sampling_rate, **fe_kwargs)
            processed_outputs.update(inputs)

            # Calculate audio_len_after_cnn and audio_token_num
            if "input_features" in inputs and hasattr(inputs["input_features"], "shape"):
                mel_len = inputs["input_features"].shape[-1]
            elif "input_values" in inputs and hasattr(self.feature_extractor, "hop_length") and hasattr(inputs["input_values"], "shape"):
                num_samples = inputs["input_values"].shape[-1]
                mel_len = num_samples // self.feature_extractor.hop_length
            else: # Fallback, relying on standard Whisper characteristics
                n_frames = getattr(self.feature_extractor, 'n_frames', None) # e.g. Whisper n_frames = 3000
                if n_frames is None:
                    # Estimate based on typical Whisper: 30s * 16kHz / 160 hop_length = 3000
                    # This is a fallback if feature_extractor doesn't expose n_frames or input_features shape isn't available.
                    n_frames = 3000 
                    warnings.warn(f"Could not reliably determine mel_len. Defaulting to {n_frames} based on Whisper standard. Ensure your feature_extractor output matches.")
                mel_len = n_frames

            audio_len_after_cnn = self.get_T_after_cnn(mel_len)
            audio_token_num = (audio_len_after_cnn - 2) // 2 + 1 # From user's original script
            
            # Ensure these are scalar tensors if processing single audio, will be stacked later if batching outside
            processed_outputs['audio_len_after_cnn'] = torch.tensor(audio_len_after_cnn, dtype=torch.long)
            processed_outputs['audio_token_num'] = torch.tensor(audio_token_num, dtype=torch.long)

        if text is not None:
            if self.tokenizer is None:
                raise ValueError("Tokenizer not initialized for WhisperProcessor to process text.")
            # Filter out kwargs meant for audio feature_extractor
            text_kwargs = {k: v for k, v in kwargs.items() if k not in ['sampling_rate', 'audio']}
            encodings = self.tokenizer(text, **text_kwargs)
            processed_outputs.update(encodings)
        
        return processed_outputs

    def batch_decode(self, *args, **kwargs):
        if self.tokenizer is None: raise ValueError("Tokenizer not initialized.")
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        if self.tokenizer is None: raise ValueError("Tokenizer not initialized.")
        return self.tokenizer.decode(*args, **kwargs)



class InternOmniUnifiedProcessor:
    """
    Processor for the InternOmni model, handling text, multiple images, 
    multiple audios, and multiple videos.
    """
    def __init__(self, model_path_or_name, 
                 image_input_size=448, 
                 image_max_tiles_per_item=12, # Max tiles from one image or video frame
                 video_target_fps=1.0, 
                 ): # kwargs for from_pretrained methods

        self.image_input_size = image_input_size
        self.image_max_tiles_per_item = image_max_tiles_per_item
        self.video_target_fps = float(video_target_fps) # Ensure float for calculations

        self.image_transform = self._build_transform(self.image_input_size)
        
        try:
            self.audio_processor = WhisperProcessor.from_pretrained(
                model_path_or_name
            )
        except Exception as e:
            warnings.warn(f"Failed to load WhisperProcessor for audio: {e}. Audio processing will be disabled.")
            self.audio_processor = None

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path_or_name, trust_remote_code=True, use_fast=False
            )
        except Exception as e:
            warnings.warn(f"Failed to load AutoTokenizer: {e}. Text processing will be disabled.")
            self.tokenizer = None

    @staticmethod
    def _build_transform(input_size):
        return T.Compose([
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])

    @staticmethod
    def _find_closest_aspect_ratio(aspect_ratio, target_ratios_hw, width, height, image_patch_size):
        best_ratio_diff = float('inf')
        best_ratio_hw = (1, 1) # (h_blocks, w_blocks)
        for ratio_hw in target_ratios_hw: 
            target_ar = ratio_hw[1] / ratio_hw[0] if ratio_hw[0] > 0 else float('inf')
            ratio_diff = abs(aspect_ratio - target_ar)
            if ratio_diff < best_ratio_diff:
                best_ratio_diff = ratio_diff
                best_ratio_hw = ratio_hw
            elif ratio_diff == best_ratio_diff: # Tie-breaking: prefer more blocks
                if ratio_hw[0] * ratio_hw[1] > best_ratio_hw[0] * best_ratio_hw[1]:
                    best_ratio_hw = ratio_hw
        return best_ratio_hw

    @staticmethod
    def _dynamic_preprocess_single_pil_image(image_pil: Image.Image, 
                                            min_total_blocks=1, max_total_blocks=12, 
                                            image_patch_size=448, use_thumbnail=True):
        orig_width, orig_height = image_pil.size
        if orig_width == 0 or orig_height == 0: return []
        aspect_ratio = orig_width / orig_height

        target_ratio_configs = set() # Stores (h_blocks, w_blocks)
        for n_total in range(min_total_blocks, max_total_blocks + 1):
            for h_blocks in range(1, int(n_total**0.5) + 1):
                if n_total % h_blocks == 0:
                    w_blocks = n_total // h_blocks
                    target_ratio_configs.add((h_blocks, w_blocks))
                    target_ratio_configs.add((w_blocks, h_blocks)) # Add swapped
        
        if not target_ratio_configs: target_ratio_configs.add((1,1)) # Fallback if range is bad
        sorted_configs = sorted(list(target_ratio_configs), key=lambda x: (x[0] * x[1], x[0], x[1]))
        
        best_config_hw = InternOmniUnifiedProcessor._find_closest_aspect_ratio(
            aspect_ratio, sorted_configs, orig_width, orig_height, image_patch_size
        )
        h_blocks, w_blocks = best_config_hw
        grid_width = w_blocks * image_patch_size
        grid_height = h_blocks * image_patch_size
        
        try: # Pillow >= 9.1.0
            resample_method = Image.Resampling.BICUBIC
        except AttributeError: # Older Pillow
            resample_method = Image.BICUBIC
        resized_img = image_pil.resize((grid_width, grid_height), resample_method)
        
        processed_patches_pil = []
        for i_row in range(h_blocks):
            for j_col in range(w_blocks):
                left = j_col * image_patch_size
                top = i_row * image_patch_size
                patch = resized_img.crop((left, top, left + image_patch_size, top + image_patch_size))
                processed_patches_pil.append(patch)
        
        if not processed_patches_pil : # Safety fallback
            warnings.warn("Dynamic preprocess yielded no patches for an image. Using single resized version.")
            processed_patches_pil.append(image_pil.resize((image_patch_size, image_patch_size), resample_method))

        if use_thumbnail and (len(processed_patches_pil) != 1 or not processed_patches_pil):
            thumbnail_img = image_pil.resize((image_patch_size, image_patch_size), resample_method)
            processed_patches_pil.append(thumbnail_img)
        
        return processed_patches_pil

    def _extract_frames_from_video(self, video_path: str) -> list[Image.Image]:
        frames_pil = []
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                warnings.warn(f"Cannot open video file: {video_path}")
                return frames_pil

            original_fps = cap.get(cv2.CAP_PROP_FPS)
            if original_fps <= 0: original_fps = 30.0 # Default if unknown/invalid

            # Ensure target_fps is positive to avoid division by zero or negative interval
            effective_target_fps = max(0.1, self.video_target_fps) # Avoid 0 or negative fps

            frame_interval = int(round(original_fps / effective_target_fps))
            if frame_interval == 0: frame_interval = 1 # Take all frames if target_fps >= original_fps

            frame_idx = 0
            while True:
                ret, frame = cap.read()
                if not ret: break
                if frame_idx % frame_interval == 0:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames_pil.append(Image.fromarray(frame_rgb))
                frame_idx += 1
            cap.release()
        except Exception as e:
            warnings.warn(f"Error processing video {video_path}: {e}")
        return frames_pil

    def process(self, text_query: str = None, 
                image_paths: list[str] = None, 
                audio_paths: list[str] = None, 
                video_paths: list[str] = None):
        """
        Processes text, image paths, audio paths, and video paths for InternOmni.
        Returns a dictionary directly usable for the model's generation method.
        """
        final_model_inputs = {}

        # 1. Process Text Query
        final_model_inputs['question'] = text_query # Model.Audio_chat expects raw question string
        # Optionally, also provide tokenized output if user wants to manage it separately
        if text_query and self.tokenizer:
            tokenized_text = self.tokenizer(text_query, return_tensors="pt", padding="longest", truncation=True)
            final_model_inputs['tokenized_text_query'] = tokenized_text # e.g. for inspection or alternative use
        elif text_query:
            warnings.warn("Text query provided, but tokenizer is not available.")

        # 2. Process Visual Inputs (Images and Video Frames)
        source_pil_images_for_tiling = []
        if video_paths:
            for path in video_paths:
                video_frames = self._extract_frames_from_video(path)
                source_pil_images_for_tiling.extend(video_frames)
        if image_paths:
            for path in image_paths:
                try:
                    img = Image.open(path).convert('RGB')
                    source_pil_images_for_tiling.append(img)
                except Exception as e:
                    warnings.warn(f"Could not load or process image {path}: {e}")
        
        all_pixel_value_tensors_list = []
        if source_pil_images_for_tiling:
            for pil_img in source_pil_images_for_tiling:
                # Get PIL patches for this single source image/frame
                pil_patches = self._dynamic_preprocess_single_pil_image(
                    pil_img, 
                    max_total_blocks=self.image_max_tiles_per_item,
                    image_patch_size=self.image_input_size, 
                    use_thumbnail=True 
                )
                if pil_patches:
                    transformed_patches = torch.stack([self.image_transform(p) for p in pil_patches])
                    all_pixel_value_tensors_list.append(transformed_patches)
            
        if all_pixel_value_tensors_list:
            final_model_inputs['pixel_values'] = torch.cat(all_pixel_value_tensors_list, dim=0)
        else: # No valid images/frames processed, provide empty tensor
            final_model_inputs['pixel_values'] = torch.empty(0, 3, self.image_input_size, self.image_input_size, dtype=torch.float32)

        # 3. Process Audio Inputs
        if audio_paths and self.audio_processor:
            batched_audio_features = []
            batched_audio_len_cnn = []
            batched_audio_token_num = []

            for path in audio_paths:
                try:
                    audio_waveform, sr_orig = librosa.load(path, sr=None) # Load with original SR
                    target_sr = self.audio_processor.feature_extractor.sampling_rate
                    if sr_orig != target_sr:
                        audio_waveform = librosa.resample(audio_waveform, orig_sr=sr_orig, target_sr=target_sr)
                    
                    # Process single audio; WhisperProcessor's __call__ returns dict with pt tensors
                    processed_audio_dict = self.audio_processor(audio=audio_waveform, return_tensors="pt")
                    
                    batched_audio_features.append(processed_audio_dict['input_features']) # Shape (1, D, L) or (D,L)
                    batched_audio_len_cnn.append(processed_audio_dict['audio_len_after_cnn'])
                    batched_audio_token_num.append(processed_audio_dict['audio_token_num'])
                except Exception as e:
                    warnings.warn(f"Could not load or process audio {path}: {e}")
            
            if batched_audio_features:
                # Ensure all features have a batch dim before cat, WhisperFE usually adds it.
                # If any are (D,L), unsqueeze them to (1,D,L)
                squeezed_features = []
                for feat in batched_audio_features:
                    if feat.ndim == 2: squeezed_features.append(feat.unsqueeze(0))
                    elif feat.ndim == 3: squeezed_features.append(feat)
                    else: warnings.warn(f"Audio feature has unexpected ndim: {feat.ndim}. Skipping.")

                if squeezed_features:
                    final_model_inputs['audio'] = {
                        'audio_values': torch.cat(squeezed_features, dim=0), # (N_audio, D_mel, L_frames)
                        'audio_len_after_cnn': torch.stack(batched_audio_len_cnn),    # (N_audio,)
                        'audio_token_num': torch.stack(batched_audio_token_num)      # (N_audio,)
                    }
                else: # No valid audio features collected after check
                    final_model_inputs['audio'] = {
                        'audio_values': torch.empty(0, self.audio_processor.feature_extractor.feature_size, 0, dtype=torch.float32), # D_mel=feature_size, L_frames=0
                        'audio_len_after_cnn': torch.empty(0, dtype=torch.long),
                        'audio_token_num': torch.empty(0, dtype=torch.long)
                    }
            else: # No audios successfully processed
                audio_feat_size = self.audio_processor.feature_extractor.feature_size if self.audio_processor else 80 # default mel bins
                final_model_inputs['audio'] = {
                    'audio_values': torch.empty(0, audio_feat_size, 0, dtype=torch.float32),
                    'audio_len_after_cnn': torch.empty(0, dtype=torch.long),
                    'audio_token_num': torch.empty(0, dtype=torch.long)
                }
        elif audio_paths: # Paths given but no processor
            warnings.warn("Audio paths provided, but audio_processor is not available.")
            final_model_inputs['audio'] = {
                'audio_values': torch.empty(0, 80, 0, dtype=torch.float32), # Default empty structure
                'audio_len_after_cnn': torch.empty(0, dtype=torch.long),
                'audio_token_num': torch.empty(0, dtype=torch.long)
            }
        else: # No audio paths given
             audio_feat_size = self.audio_processor.feature_extractor.feature_size if self.audio_processor and hasattr(self.audio_processor.feature_extractor, 'feature_size') else 80
             final_model_inputs['audio'] = {
                'audio_values': torch.empty(0, audio_feat_size, 0, dtype=torch.float32),
                'audio_len_after_cnn': torch.empty(0, dtype=torch.long),
                'audio_token_num': torch.empty(0, dtype=torch.long)
            }

        return final_model_inputs