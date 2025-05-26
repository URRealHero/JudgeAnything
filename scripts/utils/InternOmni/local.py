# local.py
import os
import json
from copy import deepcopy
import torch
import gc
import warnings # Added for warnings

# Imports from parent directory's utils and config
from ..utils import load_json, get_question_media, get_resp_media # Ensure these are in your ..utils
from ..config import BASE_DIR, BENCHMARK_FILE, RESULT_DIR, RESULT_FILE, CHECKLIST_FILE, LocalPrompts # Ensure these are in ..config
from ..prompt_builder import SystemPromptBuilder # Ensure this is in ..prompt_builder

# Relative import for the InternOmni processor
from .Processor import InternOmniUnifiedProcessor
from .custom_audiochat import custom_intern_omni_audio_chat

# For type hinting if you use it
from transformers import GenerationConfig


class BaseLocalContentCreator:
    def __init__(self, entry: dict, 
                 intern_omni_processor: InternOmniUnifiedProcessor, 
                 evaluator_model, # This is the loaded InternOmni model instance
                 generation_config_dict: dict, # Expecting a dictionary
                 c_flag=False):
        
        self.entry = entry # Current benchmark entry being processed
        self.uniq_id = entry["uniq_id"]
        self.task_name = entry["task_name"]
        
        # Assuming task_name is always in "InputModality2OutputModality" format
        try:
            self.input_modality, self.output_modality = self.task_name.split("2") 
        except ValueError:
            warnings.warn(f"Task name '{self.task_name}' for uniq_id '{self.uniq_id}' is not in 'InputModality2OutputModality' format. Using defaults.")
            self.input_modality, self.output_modality = "Unknown", "Unknown"

        # Load all result data once and filter per uniq_id
        self.all_result_data = load_json(os.path.join(RESULT_DIR, RESULT_FILE))
        self.result_entries_for_uniq_id = [res for res in self.all_result_data if res.get("uniq_id") == self.uniq_id]
        
        self.sys_constructor = SystemPromptBuilder() 
        self.local_prompts_enum = LocalPrompts # Keep reference to the enum

        self.checklist_file_path = os.path.join(BASE_DIR.parent, "dataset", "Checklist", "checklist.json") # As per your config.py structure for CHECKLIST_FILE

        if c_flag:
            self.checklists_text_for_uniq_id = self._get_checklist_text_for_id(self.uniq_id)
        else:
            self.checklists_text_for_uniq_id = None
            
        self.evaluation_items = []  # Stores dicts, each with inputs for one evaluator model call

        self.intern_omni_processor = intern_omni_processor
        self.evaluator_model = evaluator_model 
        self.generation_config_dict = generation_config_dict # Should be a dict
        
    def _get_checklist_text_for_id(self, uniq_id_to_filter):
        all_checklist_data = load_json(self.checklist_file_path)
        full_checklist_str = ""
        for cl_entry in all_checklist_data: # all_checklist_data might be {} if file not found by load_json
            if isinstance(cl_entry, dict) and cl_entry.get("uniq_id") == uniq_id_to_filter:
                clt_str = f"Here is the checklist items for rubric {cl_entry.get('rubric','N/A')}:\n" # Added .get for safety
                for i, item_str in enumerate(cl_entry.get("checklists",[])):
                    clt_str += f"{i+1}. {item_str}\n"
                full_checklist_str += clt_str + "\n"
        return full_checklist_str if full_checklist_str else None # Return None if empty
    
    def construct_evaluation_items(self):
        raise NotImplementedError

    def generate_feedback(self, system_prompt_text: str):
        responses_map = {}
        if not self.evaluation_items: 
            self.construct_evaluation_items()

        for item_spec in self.evaluation_items:
            evaluator_input_text = system_prompt_text + item_spec["full_text_prompt"]
            
            processed_inputs = self.intern_omni_processor.process(
                text_query=evaluator_input_text, 
                image_paths=item_spec.get("image_paths", []),
                audio_paths=item_spec.get("audio_paths", []),
                video_paths=item_spec.get("video_paths", [])
            )

            pixel_values_for_model = None
            if processed_inputs.get('pixel_values') is not None and processed_inputs['pixel_values'].numel() > 0:
                pixel_values_for_model = processed_inputs['pixel_values'].to(self.evaluator_model.device, dtype=self.evaluator_model.dtype)

            audio_dict_for_model = None
            raw_audio_dict = processed_inputs.get('audio')
            # print(f"Raw audio dict: {raw_audio_dict}") # Debugging line to check audio dict structure
            if raw_audio_dict and raw_audio_dict.get('audio_values') is not None and raw_audio_dict['audio_values'].numel() > 0:
                audio_dict_for_model = {}
                for k, v_tensor in raw_audio_dict.items(): # Corrected variable name
                    if isinstance(v_tensor, torch.Tensor):
                        target_dtype = self.evaluator_model.dtype if v_tensor.is_floating_point() else v_tensor.dtype
                        audio_dict_for_model[k] = v_tensor.to(self.evaluator_model.device, dtype=target_dtype)
                    else: # Should not happen if processor returns tensors
                        audio_dict_for_model[k] = v_tensor 
            
            question_for_chat = processed_inputs['question'] 
            
            num_patches_list_for_chat = processed_inputs.get('num_patches_list')
            if pixel_values_for_model is None or (hasattr(pixel_values_for_model, 'numel') and pixel_values_for_model.numel() == 0):
                 num_patches_list_for_chat = [] 

            current_gen_config_dict = self.generation_config_dict
            if isinstance(self.generation_config_dict, GenerationConfig): # Handle if GenerationConfig object is passed
                current_gen_config_dict = self.generation_config_dict.to_dict()
            
            feedback_response_text = self.evaluator_model.Audio_chat(
                tokenizer=self.intern_omni_processor.tokenizer,
                pixel_values=pixel_values_for_model,
                audio=audio_dict_for_model, 
                question=question_for_chat, 
                generation_config=current_gen_config_dict,
                num_patches_list=num_patches_list_for_chat,
            )
            
            feedback_response_text = custom_intern_omni_audio_chat(
                model_instance=self.evaluator_model, # Pass the loaded InternOmni model instance
                tokenizer=self.intern_omni_processor.tokenizer,
                pixel_values=pixel_values_for_model,
                audio=audio_dict_for_model, 
                question=question_for_chat, # This is the full text prompt from your local.py
                generation_config=current_gen_config_dict, # This is already a dict
                num_patches_list=num_patches_list_for_chat,
                history=None, # Assuming no history for evaluator calls, or manage it if needed
                return_history=False,
                verbose=True # Set to True for detailed debugging printouts from custom_audio_chat
            )
            
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            response_details = {
                "uniq_id" : self.uniq_id,
                "task_name" : self.task_name,
                "feedback" : feedback_response_text
            }
            if "associated_response_id" in item_spec:
                response_details["response_id"] = item_spec["associated_response_id"]
            
            responses_map[item_spec["map_key"]] = response_details
            
        return responses_map

class ScoreLocalContentCreator(BaseLocalContentCreator):
    def construct_evaluation_items(self):
        self.evaluation_items = []
        
        original_query_text = f"{self.local_prompts_enum.USER.value}" # Use enum value
        original_query_text += f"Task: {self.input_modality} to {self.output_modality}.\nQuery: {self.entry.get('question', '')}\n"
        
        q_img_paths, q_aud_paths, q_vid_paths = get_question_media(self.entry) # From ..utils
        
        if self.checklists_text_for_uniq_id:
            original_query_text += self.checklists_text_for_uniq_id
            
        for resp_entry in self.result_entries_for_uniq_id:
            eval_prompt_for_this_response = original_query_text
            eval_prompt_for_this_response += f"\nModel Response:\n"
            
            current_eval_img_paths = list(q_img_paths) 
            current_eval_aud_paths = list(q_aud_paths) 
            current_eval_vid_paths = list(q_vid_paths) 

            response_content_data = resp_entry.get("response", {})
            if response_content_data.get("type", "").lower() == "text":
                eval_prompt_for_this_response += response_content_data.get("content", "")
            else: 
                r_img_paths, r_aud_paths, r_vid_paths = get_resp_media(resp_entry) # From ..utils
                current_eval_img_paths.extend(r_img_paths)
                current_eval_aud_paths.extend(r_aud_paths)
                current_eval_vid_paths.extend(r_vid_paths)
                
                if response_content_data.get("content"):
                    eval_prompt_for_this_response += response_content_data["content"]
                else:
                    eval_prompt_for_this_response += f"[Multimodal response content with {len(r_img_paths)} image(s), {len(r_aud_paths)} audio(s), {len(r_vid_paths)} video(s) is provided via media.]"
            
            # Cue for the evaluator model
            eval_prompt_for_this_response += f"\n{self.local_prompts_enum.END.value}{self.local_prompts_enum.ASSISTANT.value}"

            self.evaluation_items.append({
                "full_text_prompt": eval_prompt_for_this_response,
                "image_paths": current_eval_img_paths,
                "audio_paths": current_eval_aud_paths,
                "video_paths": current_eval_vid_paths,
                "map_key": resp_entry.get("model_name", f"unknown_model_{resp_entry.get('response_id','')}"), 
                "associated_response_id": resp_entry.get("response_id")
            })

    def generate_rubric_content(self):
        system_prompt = self.sys_constructor.get_local_rubric_prompts()["score"]
        return self.generate_feedback(system_prompt)
    
    def generate_overall_content(self):
        system_prompt = self.sys_constructor.get_local_overall_prompts()["score"]
        return self.generate_feedback(system_prompt)

class PairLocalContentCreator(BaseLocalContentCreator):
    def construct_evaluation_items(self):
        self.evaluation_items = []

        original_query_text = f"{self.local_prompts_enum.USER.value}"
        original_query_text += f"Task: {self.input_modality} to {self.output_modality}.\nQuery: {self.entry.get('question', '')}\n"
        q_img_paths, q_aud_paths, q_vid_paths = get_question_media(self.entry)

        if self.checklists_text_for_uniq_id:
            original_query_text += self.checklists_text_for_uniq_id

        num_responses = len(self.result_entries_for_uniq_id)
        for i in range(0, num_responses - (num_responses % 2), 2): # Ensure pairs
            if i + 1 >= num_responses: continue 

            resp_entry1 = self.result_entries_for_uniq_id[i]
            resp_entry2 = self.result_entries_for_uniq_id[i+1]
            model1_name = resp_entry1.get('model_name', 'Model1')
            model2_name = resp_entry2.get('model_name', 'Model2')

            eval_prompt_for_this_pair = original_query_text
            current_eval_img_paths = list(q_img_paths)
            current_eval_aud_paths = list(q_aud_paths)
            current_eval_vid_paths = list(q_vid_paths)

            eval_prompt_for_this_pair += f"\nResponse from Model A:\n"
            response1_content_data = resp_entry1.get("response", {})
            if response1_content_data.get("type", "").lower() == "text":
                eval_prompt_for_this_pair += response1_content_data.get("content", "")
            else:
                r1_img_paths, r1_aud_paths, r1_vid_paths = get_resp_media(resp_entry1)
                current_eval_img_paths.extend(r1_img_paths)
                current_eval_aud_paths.extend(r1_aud_paths)
                current_eval_vid_paths.extend(r1_vid_paths)
                if response1_content_data.get("content"):
                    eval_prompt_for_this_pair += response1_content_data["content"]
                else:
                    eval_prompt_for_this_pair += f"[Multimodal content for Model A provided via media.]"
            
            eval_prompt_for_this_pair += f"\n\nResponse from Model B:\n"
            response2_content_data = resp_entry2.get("response", {})
            if response2_content_data.get("type", "").lower() == "text":
                eval_prompt_for_this_pair += response2_content_data.get("content", "")
            else:
                r2_img_paths, r2_aud_paths, r2_vid_paths = get_resp_media(resp_entry2)
                current_eval_img_paths.extend(r2_img_paths)
                current_eval_aud_paths.extend(r2_aud_paths)
                current_eval_vid_paths.extend(r2_vid_paths)
                if response2_content_data.get("content"):
                    eval_prompt_for_this_pair += response2_content_data["content"]
                else:
                    eval_prompt_for_this_pair += f"[Multimodal content for Model B provided via media.]"

            eval_prompt_for_this_pair += f"\n{self.local_prompts_enum.END.value}{self.local_prompts_enum.ASSISTANT.value}"
            
            self.evaluation_items.append({
                "full_text_prompt": eval_prompt_for_this_pair,
                "image_paths": current_eval_img_paths,
                "audio_paths": current_eval_aud_paths,
                "video_paths": current_eval_vid_paths,
                "map_key": f"{model1_name}_vs_{model2_name}"
            })
            
    def generate_rubric_content(self):
        system_prompt = self.sys_constructor.get_local_rubric_prompts()["pair"]
        return self.generate_feedback(system_prompt)

    def generate_overall_content(self):
        system_prompt = self.sys_constructor.get_local_overall_prompts()["pair"]
        return self.generate_feedback(system_prompt)
