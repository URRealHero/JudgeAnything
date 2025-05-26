# utils/QwenOmni/local.py
import os
import json
from copy import deepcopy
import torch
import gc
import warnings

# Imports from parent directory's utils and config
from ..utils import load_json, get_question_media, get_resp_media
from ..config import BASE_DIR, RESULT_DIR, CHECKLIST_FILE, LocalPrompts # Assuming LocalPrompts might still be used for USER/ASSISTANT text structure
from ..prompt_builder import SystemPromptBuilder

# QwenOmni specific imports
from transformers import Qwen2_5OmniProcessor, Qwen2_5OmniForConditionalGeneration, GenerationConfig

# We expect process_mm_info to be available from your rebuilt qwen_omni_utils package
# This import will work if qwen_omni_utils is installed (e.g., editable install)
# and PATH_TO_YOUR_MODIFIED_QWEN_OMNI_UTILS/src is in PYTHONPATH or site-packages.
try:
    from qwen_omni_utils import process_mm_info # Or specific submodule if it's nested, e.g. qwen_omni_utils.v2_5
except ImportError:
    warnings.warn("Could not import 'process_mm_info' from 'qwen_omni_utils'. "
                  "Ensure the package is installed correctly in editable mode from your modified source.", ImportWarning)
    # Define a dummy function to allow script structure to be reviewed if import fails
    def process_mm_info(conversation, use_audio_in_video=False, **kwargs):
        warnings.warn("Using dummy 'process_mm_info'. Multimedia processing will not work.", RuntimeWarning)
        return None, None, None # audios, images, videos


class BaseLocalContentCreator:
    def __init__(self, entry: dict,
                 qwen_official_processor: Qwen2_5OmniProcessor, # Expecting the official processor
                 evaluator_model: Qwen2_5OmniForConditionalGeneration,
                 generation_config_dict: dict,
                 c_flag=False):

        self.entry = entry
        self.uniq_id = entry["uniq_id"]
        self.task_name = entry.get("task_name", "UnknownTask")

        try:
            self.input_modality, self.output_modality = self.task_name.split("2")
            self.input_modality = self.input_modality.lower()
        except ValueError:
            warnings.warn(f"Task name '{self.task_name}' for uniq_id '{self.uniq_id}' is not in 'InputModality2OutputModality' format. Using defaults.")
            self.input_modality, self.output_modality = "unknown", "unknown"

        # Determine if audio should be extracted from video for this specific task
        self.current_task_use_audio_in_video = (self.input_modality == "audiovideo")
        if self.current_task_use_audio_in_video:
            warnings.warn(f"Task {self.uniq_id} is audiovideo. Setting use_audio_in_video=True.")


        # Load all result data once and filter per uniq_id
        # Ensure RESULT_FILE is correctly defined in your config.py
        self.all_result_data = load_json(os.path.join(RESULT_DIR, "X2XBenchmarkResponse.json")) # Example, use config
        self.result_entries_for_uniq_id = [res for res in self.all_result_data if isinstance(res, dict) and res.get("uniq_id") == self.uniq_id]

        self.sys_constructor = SystemPromptBuilder()
        self.local_prompts_enum = LocalPrompts # For structuring textual parts if needed

        # Ensure CHECKLIST_FILE is correctly defined in your config.py
        self.checklist_file_path = CHECKLIST_FILE
        if c_flag:
            self.checklists_text_for_uniq_id = self._get_checklist_text_for_id(self.uniq_id)
        else:
            self.checklists_text_for_uniq_id = None

        self.evaluation_items = []  # Stores dicts, each with a 'conversation' list and 'map_key'

        self.qwen_official_processor = qwen_official_processor
        self.evaluator_model = evaluator_model
        self.generation_config_dict = generation_config_dict

    def _get_checklist_text_for_id(self, uniq_id_to_filter):
        all_checklist_data = load_json(self.checklist_file_path)
        full_checklist_str = ""
        if isinstance(all_checklist_data, list): # Ensure it's a list
            for cl_entry in all_checklist_data:
                if isinstance(cl_entry, dict) and cl_entry.get("uniq_id") == uniq_id_to_filter:
                    clt_str = f"Here is the checklist items for rubric {cl_entry.get('rubric','N/A')}:\n"
                    for i, item_str in enumerate(cl_entry.get("checklists",[])):
                        clt_str += f"{i+1}. {item_str}\n"
                    full_checklist_str += clt_str + "\n"
        return full_checklist_str if full_checklist_str else None

    def _build_media_content_list(self, image_paths, audio_paths, video_paths, is_response_media=False):
        """
        Helper to build the media part of the 'content' list for a Qwen conversation turn.
        Handles the AudioVideo2Text specific logic.
        """
        media_content_list = []
        # Determine base path for media depending on whether it's from question or response
        media_base_dir = RESULT_DIR if is_response_media else BASE_DIR

        if self.input_modality == "audiovideo" and not is_response_media: # Only for original query
            if video_paths:
                # For AudioVideo2Text, only use the video path. Audio is extracted from it.
                # The path should be absolute or resolvable by qwen_omni_utils
                full_video_path = os.path.join(media_base_dir, video_paths[0]) # Assuming one video for this task
                media_content_list.append({"type": "video", "video": full_video_path})
                # Audio path is intentionally ignored here for AudioVideo2Text query
        else:
            # For other tasks or for model responses, process each modality separately
            for img_path_rel in image_paths:
                full_img_path = os.path.join(media_base_dir, img_path_rel)
                media_content_list.append({"type": "image", "image": full_img_path}) # qwen_omni_utils uses "image" key
            for aud_path_rel in audio_paths:
                full_aud_path = os.path.join(media_base_dir, aud_path_rel)
                media_content_list.append({"type": "audio", "audio": full_aud_path})
            for vid_path_rel in video_paths:
                full_vid_path = os.path.join(media_base_dir, vid_path_rel)
                media_content_list.append({"type": "video", "video": full_vid_path})
        return media_content_list

    def construct_evaluation_items(self):
        # To be implemented by subclasses (ScoreLocalContentCreator, PairLocalContentCreator)
        # Each item in self.evaluation_items should be a dictionary:
        # {
        #     "conversation": [
        #         {"role": "system", "content": [{"type": "text", "text": "..."}]},
        #         {"role": "user", "content": [{"type": "text", "text": "..."}, {"type": "image", "image": "path"}, ...]}
        #     ],
        #     "map_key": "key_for_storing_response (e.g., model_name or pair_name)",
        #     "associated_response_id": "response_id_if_applicable (for score mode)"
        # }
        raise NotImplementedError

    def generate_feedback(self, system_prompt_text_content: str): # Renamed for clarity
        responses_map = {}
        if not self.evaluation_items:
            self.construct_evaluation_items()

        for item_spec in self.evaluation_items:
            current_conversation = item_spec["conversation"] # This is already built by subclass

            # The system prompt text is already incorporated into current_conversation[0]['content']
            # by the construct_evaluation_items method.

            logging.debug(f"Raw conversation for QwenOmni: {json.dumps(current_conversation, indent=2)}")

            # 1. Apply chat template (but don't tokenize yet)
            # The system prompt is part of the conversation list.
            # add_generation_prompt=True is crucial to cue the model for generation.
            try:
                text_for_processor = self.qwen_official_processor.apply_chat_template(
                    current_conversation,
                    add_generation_prompt=True,
                    tokenize=False
                )
                logging.debug(f"Text after apply_chat_template: {text_for_processor}")
            except Exception as e:
                logging.error(f"Error in apply_chat_template for uniq_id={self.uniq_id}, key={item_spec['map_key']}: {e}", exc_info=True)
                responses_map[item_spec["map_key"]] = {
                    "uniq_id": self.uniq_id, "task_name": self.task_name,
                    "feedback": f"[Error during chat templating: {e}]",
                    "response_id": item_spec.get("associated_response_id")
                }
                continue


            # 2. Process multimedia information using qwen_omni_utils.process_mm_info
            # This function is expected to take the conversation list and extract/process media.
            # It should return processed audios, images, videos in a format Qwen2_5OmniProcessor expects.
            # The `self.current_task_use_audio_in_video` flag is critical here.
            try:
                # process_mm_info takes the original conversation list to find media paths
                processed_audios, processed_images, processed_videos = process_mm_info(
                    current_conversation, # Pass the conversation list with media paths
                    use_audio_in_video=self.current_task_use_audio_in_video
                )
                logging.debug(f"process_mm_info results: audios_type={type(processed_audios)}, images_type={type(processed_images)}, videos_type={type(processed_videos)}")
            except Exception as e:
                logging.error(f"Error in process_mm_info for uniq_id={self.uniq_id}, key={item_spec['map_key']}: {e}", exc_info=True)
                responses_map[item_spec["map_key"]] = {
                    "uniq_id": self.uniq_id, "task_name": self.task_name,
                    "feedback": f"[Error during media processing (process_mm_info): {e}]",
                    "response_id": item_spec.get("associated_response_id")
                }
                continue

            # 3. Final processing step with Qwen2_5OmniProcessor
            try:
                inputs = self.qwen_official_processor(
                    text=text_for_processor,
                    images=processed_images, # Processed by process_mm_info
                    audios=processed_audios, # Processed by process_mm_info
                    videos=processed_videos, # Processed by process_mm_info
                    return_tensors="pt",
                    padding=True, # Or "longest"
                    use_audio_in_video=self.current_task_use_audio_in_video
                )
                inputs = inputs.to(self.evaluator_model.device).to(self.evaluator_model.dtype)
                logging.debug(f"Final processor inputs prepared for model. Keys: {list(inputs.keys())}")
            except Exception as e:
                logging.error(f"Error in final QwenOmniProcessor step for uniq_id={self.uniq_id}, key={item_spec['map_key']}: {e}", exc_info=True)
                responses_map[item_spec["map_key"]] = {
                    "uniq_id": self.uniq_id, "task_name": self.task_name,
                    "feedback": f"[Error during final Qwen processor step: {e}]",
                    "response_id": item_spec.get("associated_response_id")
                }
                continue

            # 4. Model Generation
            # The generation_config_dict should already include max_new_tokens, etc.
            # The Qwen demo uses model.generate directly.
            feedback_text_ids = None
            try:
                # Qwen demo passes use_audio_in_video also to generate
                # The generate method might return (text_ids, audio_output_waveforms)
                # We are primarily interested in text_ids for the judge's feedback.
                # Unpack carefully if model.generate returns multiple items.
                gen_outputs = self.evaluator_model.generate(
                    **inputs,
                    generation_config=GenerationConfig(**self.generation_config_dict), # Pass as GenerationConfig object
                    use_audio_in_video=self.current_task_use_audio_in_video
                )
                if isinstance(gen_outputs, tuple): # If model.generate returns (text_ids, audio_waveforms)
                    feedback_text_ids = gen_outputs[0]
                else: # Assuming it returns just text_ids (or a ModelOutput object)
                    feedback_text_ids = gen_outputs

                feedback_response_text = self.qwen_official_processor.batch_decode(
                    feedback_text_ids,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False # As per Qwen demo
                )[0]
                logging.debug(f"Generated feedback text: {feedback_response_text}")

            except Exception as e:
                logging.error(f"Error during model.generate for uniq_id={self.uniq_id}, key={item_spec['map_key']}: {e}", exc_info=True)
                feedback_response_text = f"[Error during model generation: {e}]"

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            response_details = {
                "uniq_id": self.uniq_id,
                "task_name": self.task_name,
                "feedback": feedback_response_text
            }
            if "associated_response_id" in item_spec:
                response_details["response_id"] = item_spec["associated_response_id"]

            responses_map[item_spec["map_key"]] = response_details

        return responses_map


class ScoreLocalContentCreator(BaseLocalContentCreator):
    def construct_evaluation_items(self):
        self.evaluation_items = []

        # System prompt text from SystemPromptBuilder
        # This text will be the content of the "system" role message.
        # The specific prompt (rubric/overall) is determined by the calling method.
        # Here, we just prepare the structure for the user query and model responses.

        # Get media paths from the original benchmark query
        q_img_paths, q_aud_paths, q_vid_paths = get_question_media(self.entry)

        for resp_entry in self.result_entries_for_uniq_id:
            # --- Build the 'user' turn content ---
            user_content_list = []

            # 1. Original query text
            query_text_for_user_turn = f"Task: {self.input_modality} to {self.output_modality}.\n"
            query_text_for_user_turn += f"Query: {self.entry.get('question', '')}\n"
            user_content_list.append({"type": "text", "text": query_text_for_user_turn})

            # 2. Media from original query (handles AudioVideo2Text logic via _build_media_content_list)
            user_content_list.extend(self._build_media_content_list(q_img_paths, q_aud_paths, q_vid_paths, is_response_media=False))

            # 3. Text of the model's response being evaluated
            model_response_text = f"\nModel Response (from model '{resp_entry.get('model_name', 'N/A')}'):\n"
            response_content_data = resp_entry.get("response", {})
            if response_content_data.get("type", "").lower() == "text":
                model_response_text += response_content_data.get("content", "")
            else: # Multimodal response from the model being evaluated
                if response_content_data.get("content"): # Text part of multimodal response
                    model_response_text += response_content_data.get("content", "")
                # Media from this model's response
                r_img_paths, r_aud_paths, r_vid_paths = get_resp_media(resp_entry)
                user_content_list.extend(self._build_media_content_list(r_img_paths, r_aud_paths, r_vid_paths, is_response_media=True))
                if not response_content_data.get("content") and (r_img_paths or r_aud_paths or r_vid_paths):
                     model_response_text += f"[Multimodal response content with media is provided.]"

            user_content_list.append({"type": "text", "text": model_response_text})


            # 4. Checklist text, if applicable
            if self.checklists_text_for_uniq_id:
                user_content_list.append({"type": "text", "text": f"\nEvaluation Checklist:\n{self.checklists_text_for_uniq_id}"})

            # The system prompt text (e.g., rubric instructions) will be fetched by the calling method
            # (generate_rubric_content or generate_overall_content) and prepended to the conversation.
            # The user turn is constructed here.
            # The final "ASSISTANT:" cue is handled by `add_generation_prompt=True` in `apply_chat_template`.

            # The actual system prompt text will be retrieved by the calling function
            # (generate_rubric_content / generate_overall_content)
            # and passed to self.generate_feedback, which will then prepend it.
            # For now, construct_evaluation_items will prepare the user turn and response structure.
            # The generate_feedback method will receive the system_prompt_text separately.
            # Let's adjust: construct_evaluation_items will build the full conversation list.

            system_prompt_text = "" # This will be set by the calling method (rubric/overall)
                                    # For now, this structure assumes generate_feedback gets the system prompt.
                                    # Let's change it so construct_evaluation_items gets the system prompt type.

            # The methods generate_rubric_content/generate_overall_content will now pass the system_prompt_text_content
            # directly to generate_feedback. Here we just prepare the user turn.
            # The conversation list will be assembled in generate_feedback.
            # This is slightly different from InternOmni, let's align:
            # construct_evaluation_items should build the complete conversation list.

            # The system_prompt_text_content will be passed to generate_feedback,
            # which will then form the first turn of the conversation.

            # Let's refine: `construct_evaluation_items` should prepare the *full* conversation list
            # for each item, including the system prompt.
            # The caller (generate_rubric_content) will decide *which* system prompt text to use.

            # This method is called by generate_rubric_content or generate_overall_content.
            # Those methods will determine the actual system_prompt_text.
            # We'll pass that system_prompt_text to this method.
            # This is a slight departure from the previous structure, let's stick to the old way:
            # generate_rubric_content calls self.construct_evaluation_items() then self.generate_feedback(system_prompt_text)
            # So, construct_evaluation_items prepares items that LACK the system prompt,
            # and generate_feedback prepends it.

            # For Qwen, the conversation list is the primary input.
            # `construct_evaluation_items` needs to build this list.
            # The system prompt should be the first item in this list.

            # Let's make `construct_evaluation_items` accept the system prompt text directly.
            # This is a change from the previous model's local.py.

            # No, let's stick to the pattern:
            # generate_X_content() gets sys_prompt_text, then calls construct_evaluation_items(), then generate_feedback(sys_prompt_text)
            # construct_evaluation_items() will prepare the user turn and model response part.
            # generate_feedback() will prepend the system turn.

            # For Qwen, the "conversation" is the core.
            # `construct_evaluation_items` should build the *full* list of conversation turns for each eval item.

            # This requires `generate_rubric_content` to call `construct_evaluation_items` with the system prompt type.
            # Let's simplify: `construct_evaluation_items` will be called by `generate_rubric_content` etc.
            # It will construct the user part of the conversation.
            # `generate_feedback` will then prepend the system part.

            # Final decision for Qwen:
            # `construct_evaluation_items` will create a list of dictionaries, where each dictionary
            # contains the 'user_content_list' and other metadata.
            # `generate_feedback` will then take this, prepend the system turn, and form the full conversation.

            self.evaluation_items.append({
                "user_content_list": user_content_list, # The content for the 'user' role
                "map_key": resp_entry.get("model_name", f"unknown_model_{resp_entry.get('response_id','')}"),
                "associated_response_id": resp_entry.get("response_id")
            })


    # Override generate_feedback to build the Qwen conversation structure
    def generate_feedback(self, system_prompt_text_content: str):
        responses_map = {}
        if not self.evaluation_items: # Should have been called by generate_rubric/overall_content
            # This path implies construct_evaluation_items was not called by the specific generate_ method.
            # For Qwen, construct_evaluation_items needs to be called by the specific method
            # as it doesn't know the system_prompt_text_content itself.
            # Let's ensure construct_evaluation_items is called first by the public methods.
             raise RuntimeError("construct_evaluation_items must be called by generate_rubric/overall_content before generate_feedback.")


        for item_spec in self.evaluation_items:
            # Build the full conversation list for Qwen
            current_conversation = [
                {"role": "system", "content": [{"type": "text", "text": system_prompt_text_content}]},
                {"role": "user", "content": item_spec["user_content_list"]}
            ]

            logging.debug(f"Constructed conversation for QwenOmni: {json.dumps(current_conversation, indent=2)}")

            # ... (rest of the processing: apply_chat_template, process_mm_info, processor, model.generate)
            # This is identical to the BaseLocalContentCreator.generate_feedback now.
            # To avoid duplication, we can call super().generate_feedback AFTER constructing the conversation.
            # However, the base generate_feedback expects item_spec["conversation"].
            # So, ScoreLocalContentCreator.construct_evaluation_items needs to build the full conversation.

            # Let's redefine:
            # Score/Pair.construct_evaluation_items will build the *entire* conversation list for each item
            # and store it in item_spec["conversation"].
            # Then BaseLocalContentCreator.generate_feedback can be used as is.

            # This means Score/Pair.construct_evaluation_items needs the system_prompt_text.
            # This is messy.

            # Simpler: Keep BaseLocalContentCreator.generate_feedback as the main engine.
            # Score/Pair.construct_evaluation_items will prepare 'user_content_list' and other meta.
            # The generate_feedback in BaseLocalContentCreator will take this 'user_content_list'
            # and the system_prompt_text_content and assemble the conversation.

            # This is what the current BaseLocalContentCreator.generate_feedback does,
            # assuming item_spec["conversation"] is already the full conversation.

            # Let's stick to:
            # 1. Score/Pair.generate_rubric/overall_content gets system_prompt_text.
            # 2. It calls self.construct_evaluation_items(system_prompt_text_for_item_type)
            # 3. construct_evaluation_items builds the *full* conversation list for each item and stores it.
            # 4. It then calls self.generate_feedback() (which now doesn't need system_prompt_text_content arg).

            # This is a larger refactor. For now, let's assume the current `generate_feedback` in Base
            # is the one we want to use, and `construct_evaluation_items` in subclasses
            # will populate `item_spec["conversation"]` fully.

            # The `generate_feedback` method in BaseLocalContentCreator will be called by
            # `generate_rubric_content` and `generate_overall_content` in the subclasses.
            # Those subclass methods will first call `self.construct_evaluation_items(system_prompt_text)`
            # which will populate `self.evaluation_items` with full conversation lists.

            # The following is now effectively the body of the generate_feedback in Base,
            # assuming self.evaluation_items contains {"conversation": [...], "map_key": ...}

            # --- Re-aligning with the Qwen demo flow within generate_feedback ---
            # `current_conversation` is already built by the subclass's `construct_evaluation_items`
            # and passed via `item_spec['conversation']`.

            # The following logic is now duplicated from Base.generate_feedback.
            # This indicates that the `construct_evaluation_items` in the subclasses
            # should indeed build the full conversation and store it in `item_spec['conversation']`.
            # Then, the `generate_feedback` from the Base class can be called.

            # For this file, let's assume `construct_evaluation_items` in subclasses
            # does its job of creating the full conversation list.
            # The `generate_feedback` method here will be the one from `BaseLocalContentCreator`.
            # So, this method in ScoreLocalContentCreator is not strictly needed if it just calls super.
            pass # This method will be removed; logic moved to construct_evaluation_items.


    def _construct_score_items_with_system_prompt(self, system_prompt_text: str):
        """Helper to build evaluation_items for Score mode."""
        self.evaluation_items = []
        q_img_paths, q_aud_paths, q_vid_paths = get_question_media(self.entry)

        for resp_entry in self.result_entries_for_uniq_id:
            user_content_list = []
            query_text = f"Task: {self.input_modality} to {self.output_modality}.\nQuery: {self.entry.get('question', '')}\n"
            user_content_list.append({"type": "text", "text": query_text})
            user_content_list.extend(self._build_media_content_list(q_img_paths, q_aud_paths, q_vid_paths, is_response_media=False))

            model_response_text = f"\nModel Response (from model '{resp_entry.get('model_name', 'N/A')}'):\n"
            response_content_data = resp_entry.get("response", {})
            if response_content_data.get("type", "").lower() == "text":
                model_response_text += response_content_data.get("content", "")
            else:
                if response_content_data.get("content"):
                    model_response_text += response_content_data.get("content", "")
                r_img_paths, r_aud_paths, r_vid_paths = get_resp_media(resp_entry)
                user_content_list.extend(self._build_media_content_list(r_img_paths, r_aud_paths, r_vid_paths, is_response_media=True))
                if not response_content_data.get("content") and (r_img_paths or r_aud_paths or r_vid_paths):
                     model_response_text += f"[Multimodal response content with media is provided.]"
            user_content_list.append({"type": "text", "text": model_response_text})

            if self.checklists_text_for_uniq_id:
                user_content_list.append({"type": "text", "text": f"\nEvaluation Checklist:\n{self.checklists_text_for_uniq_id}"})

            # Construct the full conversation for this item
            conversation_for_item = [
                {"role": "system", "content": [{"type": "text", "text": system_prompt_text}]},
                {"role": "user", "content": user_content_list}
            ]

            self.evaluation_items.append({
                "conversation": conversation_for_item,
                "map_key": resp_entry.get("model_name", f"unknown_model_{resp_entry.get('response_id','')}"),
                "associated_response_id": resp_entry.get("response_id")
            })

    def generate_rubric_content(self):
        system_prompt_text = self.sys_constructor.build_rubric_scoring_prompt() # Get raw text
        self._construct_score_items_with_system_prompt(system_prompt_text)
        return super().generate_feedback(system_prompt_text_content=None) # Base will use items

    def generate_overall_content(self):
        system_prompt_text = self.sys_constructor.build_overall_scoring_prompt() # Get raw text
        self._construct_score_items_with_system_prompt(system_prompt_text)
        return super().generate_feedback(system_prompt_text_content=None)


class PairLocalContentCreator(BaseLocalContentCreator):

    def _construct_pair_items_with_system_prompt(self, system_prompt_text: str):
        self.evaluation_items = []
        q_img_paths, q_aud_paths, q_vid_paths = get_question_media(self.entry)

        num_responses = len(self.result_entries_for_uniq_id)
        for i in range(0, num_responses - (num_responses % 2), 2): # Ensure pairs
            if i + 1 >= num_responses: continue

            resp_entry1 = self.result_entries_for_uniq_id[i]
            resp_entry2 = self.result_entries_for_uniq_id[i+1]
            model1_name = resp_entry1.get('model_name', 'ModelA')
            model2_name = resp_entry2.get('model_name', 'ModelB')

            user_content_list = []
            query_text = f"Task: {self.input_modality} to {self.output_modality}.\nQuery: {self.entry.get('question', '')}\n"
            user_content_list.append({"type": "text", "text": query_text})
            user_content_list.extend(self._build_media_content_list(q_img_paths, q_aud_paths, q_vid_paths, is_response_media=False))

            # Model A's response
            model_a_response_text = f"\nResponse from Model A ('{model1_name}'):\n"
            response1_data = resp_entry1.get("response", {})
            if response1_data.get("type", "").lower() == "text":
                model_a_response_text += response1_data.get("content", "")
            else:
                if response1_data.get("content"): model_a_response_text += response1_data.get("content", "")
                r1_img, r1_aud, r1_vid = get_resp_media(resp_entry1)
                user_content_list.extend(self._build_media_content_list(r1_img, r1_aud, r1_vid, is_response_media=True))
                if not response1_data.get("content") and (r1_img or r1_aud or r1_vid):
                    model_a_response_text += f"[Multimodal content for Model A provided via media.]"
            user_content_list.append({"type": "text", "text": model_a_response_text})

            # Model B's response
            model_b_response_text = f"\nResponse from Model B ('{model2_name}'):\n"
            response2_data = resp_entry2.get("response", {})
            if response2_data.get("type", "").lower() == "text":
                model_b_response_text += response2_data.get("content", "")
            else:
                if response2_data.get("content"): model_b_response_text += response2_data.get("content", "")
                r2_img, r2_aud, r2_vid = get_resp_media(resp_entry2)
                user_content_list.extend(self._build_media_content_list(r2_img, r2_aud, r2_vid, is_response_media=True))
                if not response2_data.get("content") and (r2_img or r2_aud or r2_vid):
                    model_b_response_text += f"[Multimodal content for Model B provided via media.]"
            user_content_list.append({"type": "text", "text": model_b_response_text})

            if self.checklists_text_for_uniq_id:
                user_content_list.append({"type": "text", "text": f"\nEvaluation Checklist:\n{self.checklists_text_for_uniq_id}"})

            conversation_for_item = [
                {"role": "system", "content": [{"type": "text", "text": system_prompt_text}]},
                {"role": "user", "content": user_content_list}
            ]

            self.evaluation_items.append({
                "conversation": conversation_for_item,
                "map_key": f"{model1_name}_vs_{model2_name}"
                # No associated_response_id for pairs in this structure
            })

    def generate_rubric_content(self):
        system_prompt_text = self.sys_constructor.build_rubric_pairing_prompt() # Get raw text
        self._construct_pair_items_with_system_prompt(system_prompt_text)
        return super().generate_feedback(system_prompt_text_content=None) # Base will use items

    def generate_overall_content(self):
        system_prompt_text = self.sys_constructor.build_overall_pairing_prompt() # Get raw text
        self._construct_pair_items_with_system_prompt(system_prompt_text)
        return super().generate_feedback(system_prompt_text_content=None)

