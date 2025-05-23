import os
import json
from copy import deepcopy
from .utils import *
from .config import *
from .prompt_builder import SystemPromptBuilder
from .Processor import ImageProcessor, AudioProcessor, VideoProcessor
from transformers import AutoProcessor, AutoModelForCausalLM, GenerationConfig


class BaseLocalContentCreator:
    def __init__(self, entry: dict, processor: AutoProcessor, model: AutoModelForCausalLM, generation_config: GenerationConfig, c_flag=False):
        self.benchmark_dir = BASE_DIR
        self.benchmark = load_json(BENCHMARK_FILE)
        self.result_dir = RESULT_DIR
        self.result = load_json(RESULT_FILE)
        self.checklist = CHECKLIST_FILE
        self.entry = entry
        self.uniq_id = entry["uniq_id"]
        self.task_name = entry["task_name"]
        self.input_modality, self.output_modality = self.task_name.split("2") 
        self.result_entries = self.get_result_entry(self.uniq_id)
        self.sys_constructor = SystemPromptBuilder()
        self.local_prompts = LocalPrompts
        if c_flag:
            self.checklists = self.get_checklist(self.uniq_id)
        else:
            self.checklists = None
        self.models = []
        self.contents = None  
        self.processor = processor
        self.evaluator = model
        self.generation_config = generation_config
        self.image_proc = ImageProcessor()
        self.audio_proc = AudioProcessor()
        self.video_proc = VideoProcessor()

        
    def get_result_entry(self, uniq_id):
        result_data = self.result
        results = []
        for entry in result_data:
            if entry["uniq_id"] == uniq_id:
                results.append(entry)
        return results
    
    def get_checklist(self, uniq_id):
        """
                "checklists": [strs]
        """
        checklist_data = load_json(self.checklist)
        checklist_str = ""
        for entry in checklist_data:
            if entry["uniq_id"] == uniq_id:
                clt_str = f"Here is the checklist items for rubric {entry['rubric']}:\n\n"
                for i,item_str in enumerate(entry["checklists"]):
                    clt_str += f"{i+1}. {item_str}\n"
                checklist_str += clt_str
        return checklist_str
    
    def construct_content(self):
        # client, entry, task_name, result_entry, checklists
        pass

    def generate_content(self):
        """
        Generate the content for scoring/pairing
        """
        pass
    

class ScoreLocalContentCreator(BaseLocalContentCreator):
    def __init__(self, entry, processor, model, generation_config, c_flag=False):
        super().__init__(entry, processor, model, generation_config, c_flag)
        

    def construct_content(self):
        input_str = f"{self.local_prompts.USER.value}"
        contents = []
        input_query = f"Here is the query of {self.input_modality} to {self.output_modality} generation task:\n"+self.entry["question"]
        # print(input_query)
        response_prompt = "\nHere is the model's response:\n"
        
        input_str += input_query
        
        img_media, aud_media, vid_media = get_question_media(self.entry)
        img_input = self.image_proc.process(img_media)
        audio_input = self.audio_proc.process(aud_media)
        video_input = self.video_proc.process(vid_media)
        visual_input = img_input if img_input else video_input
        

        for resp_entry in self.result_entries:
            self.models.append(resp_entry["model_name"])
                            
        img_idx = 1
        aud_idx = 1
        for input in visual_input:
            if isinstance(input, Image.Image):
                input_str += f"<|image_{img_idx}|>"
                img_idx += 1
        for _ in audio_input:
            input_str += f"<|audio_{aud_idx}|>"
            aud_idx += 1
                
        if self.checklists:
            input_str += self.checklists
        
        # print(input_str)
        
        # Now for each model's response, making four responses contents, and then generate the content, delete the cache
        for i,model in enumerate(self.models):
            model_str = deepcopy(input_str)
            vis_input_copy = deepcopy(visual_input)
            aud_input_copy = deepcopy(audio_input)
            model_resp_img_start_idx = img_idx
            model_resp_aud_start_idx = aud_idx
            model_str += response_prompt
            
            resp = self.result_entries[i]["response"]
            
            if resp["type"].lower() == "text":
                model_str += resp["content"]
            else:
                resp_img_media, resp_audio_media, resp_video_media = get_resp_media(self.result_entries[i])
                resp_img_input = self.image_proc.process(resp_img_media)
                resp_audio_input = self.audio_proc.process(resp_audio_media)
                resp_video_input = self.video_proc.process(resp_video_media)
                
                resp_visual_input = resp_img_input if resp_img_input else resp_video_input
                
                
                for input in resp_visual_input:
                    if isinstance(input, Image.Image):
                        model_str += f"<|image_{model_resp_img_start_idx}|>"
                        model_resp_img_start_idx += 1
                        vis_input_copy.append(input)
                for input in resp_audio_input:
                    model_str += f"<|audio_{model_resp_aud_start_idx}|>"
                    model_resp_aud_start_idx += 1
                    aud_input_copy.append(input)
                        
            model_str += f"{self.local_prompts.END.value}{self.local_prompts.ASSISTANT.value}"
            contents.append((model_str, vis_input_copy if vis_input_copy else None, aud_input_copy if aud_input_copy else None))
        self.contents = contents
    
    def generate_rubric_content(self):
        self.construct_content()
        self.sys_p = self.sys_constructor.get_local_rubric_prompts()["score"]
        responses = {}
        for i, content in enumerate(self.contents):
            input_str, visual_input, audio_input = content
            input_str = self.sys_p + input_str
            inputs = self.processor(text=input_str, images=visual_input, audio=audio_input, return_tensors="pt").to("cuda:0")
            generate_ids = self.evaluator.generate(**inputs, max_new_tokens = 1500, generation_config=self.generation_config)
            generate_ids = generate_ids[:, inputs["input_ids"].shape[1]:]
            
            response = self.processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
            gc.collect()
            torch.cuda.empty_cache()
            responses_dict = {
                "uniq_id" : self.uniq_id,
                "response_id" : self.result_entries[i]["response_id"],
                "task_name" : self.task_name,
                "feedback" : response
            }
            responses[self.models[i]] = responses_dict

        return responses
    
    def generate_overall_content(self):
        self.construct_content()
        self.sys_p = self.sys_constructor.get_local_overall_prompts()["score"]
        responses = {}
        for i, content in enumerate(self.contents):
            input_str, visual_input, audio_input = content
            input_str = self.sys_p + input_str
            inputs = self.processor(text=input_str, images=visual_input, audio=audio_input, return_tensors="pt").to("cuda:0")
            generate_ids = self.evaluator.generate(**inputs, max_new_tokens = 1500, generation_config=self.generation_config)
            generate_ids = generate_ids[:, inputs["input_ids"].shape[1]:]
            
            response = self.processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
            gc.collect()
            torch.cuda.empty_cache()
            responses_dict = {
                "uniq_id" : self.uniq_id,
                "response_id" : self.result_entries[i]["response_id"],
                "task_name" : self.task_name,
                "feedback" : response
            }
            responses[self.models[i]] = responses_dict

        return responses
    
            
            
            
class PairLocalContentCreator(BaseLocalContentCreator):
    def __init__(self, entry, processor, model, generation_config, c_flag=False):
        super().__init__(entry, processor, model, generation_config, c_flag)
        
    def construct_content(self):
        input_str = f"{self.local_prompts.USER.value}"
        contents = []
        input_query = f"Here is the query of {self.input_modality} to {self.output_modality} generation task:\n"+self.entry["question"]
        response1_prompt = "\nHere is the first model's response:\n"
        response2_prompt = "\nHere is the second model's response:\n"
        
        input_str += input_query
        
        img_media, aud_media, vid_media = get_question_media(self.entry)
        img_input = self.image_proc.process(img_media)
        audio_input = self.audio_proc.process(aud_media)
        video_input = self.video_proc.process(vid_media)
        visual_input = img_input if img_input else video_input
        
        for resp_entry in self.result_entries:
            self.models.append(resp_entry["model_name"])

                            
        img_idx = 1
        aud_idx = 1
        for input in visual_input:
            if isinstance(input, Image.Image):
                input_str += f"<|image_{img_idx}|>"
                img_idx += 1
        for input in audio_input:
            input_str += f"<|audio_{aud_idx}|>"
            aud_idx += 1
                
        if self.checklists:
            input_str += self.checklists
        
        
        # Now for each model's response, making two paired contents (model1 vs model2 ; model3 vs model4)
        for i in range(0,4,2):
            model_pair_str = deepcopy(input_str)
            vis_input_copy = deepcopy(visual_input)
            aud_input_copy = deepcopy(audio_input)
            model_resp_img_start_idx = img_idx
            model_resp_aud_start_idx = aud_idx
            model_pair_str += response1_prompt
            if self.result_entries[i]["response"]["type"].lower() == "text":
                model_pair_str += self.result_entries[i]["response"]["content"]
            else:
                resp_img_media, resp_audio_media, resp_video_media = get_resp_media(self.result_entries[i])
                resp_img_input = self.image_proc.process(resp_img_media)
                resp_audio_input = self.audio_proc.process(resp_audio_media)
                resp_video_input = self.video_proc.process(resp_video_media)
                resp_visual_input = resp_img_input if resp_img_input else resp_video_input

                for input in resp_visual_input:
                    if isinstance(input, Image.Image):
                        model_pair_str += f"<|image_{model_resp_img_start_idx}|>"
                        model_resp_img_start_idx += 1
                        vis_input_copy.append(input)
                for input in resp_audio_input:
                    model_pair_str += f"<|audio_{model_resp_aud_start_idx}|>"
                    model_resp_aud_start_idx += 1
                    aud_input_copy.append(input)
            model_pair_str += response2_prompt
            if self.result_entries[i+1]["response"]["type"].lower() == "text":
                model_pair_str += self.result_entries[i+1]["response"]["content"]
            else:
                resp_img_media, resp_audio_media, resp_video_media = get_resp_media(self.result_entries[i+1])
                resp_img_input = self.image_proc.process(resp_img_media)
                resp_audio_input = self.audio_proc.process(resp_audio_media)
                resp_video_input = self.video_proc.process(resp_video_media)
                resp_visual_input = resp_img_input if resp_img_input else resp_video_input

                for input in resp_visual_input:
                    if isinstance(input, Image.Image):
                        model_pair_str += f"<|image_{model_resp_img_start_idx}|>"
                        model_resp_img_start_idx += 1
                        vis_input_copy.append(input)
                for input in resp_audio_input:
                    model_pair_str += f"<|audio_{model_resp_aud_start_idx}|>"
                    model_resp_aud_start_idx += 1
                    aud_input_copy.append(input)
            model_pair_str += f"{self.local_prompts.END.value}{self.local_prompts.ASSISTANT.value}"
            contents.append((model_pair_str, vis_input_copy if vis_input_copy else None, aud_input_copy if aud_input_copy else None))

        self.contents = contents
        
    def generate_rubric_content(self):
        self.construct_content()
        self.sys_p = self.sys_constructor.get_local_rubric_prompts()["pair"]
        responses = {}
        for i, content in enumerate(self.contents):
            input_str, visual_input, audio_input = content
            input_str = self.sys_p + input_str
            inputs = self.processor(text=input_str, images=visual_input, audio=audio_input, return_tensors="pt").to("cuda:0")
            generate_ids = self.evaluator.generate(**inputs, max_new_tokens = 1500, generation_config=self.generation_config)
            generate_ids = generate_ids[:, inputs["input_ids"].shape[1]:]
            
            response = self.processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
            responses_dict = {
                "uniq_id" : self.uniq_id,
                "task_name" : self.task_name,
                "feedback" : response
            }
            gc.collect()
            torch.cuda.empty_cache()
            responses[f"{self.models[i*2]}_vs_{self.models[i*2+1]}"] = responses_dict

        return responses
    
    def generate_overall_content(self):
        self.construct_content()
        self.sys_p = self.sys_constructor.get_local_overall_prompts()["pair"]
        responses = {}
        for i, content in enumerate(self.contents):
            input_str, visual_input, audio_input = content
            input_str = self.sys_p + input_str
            inputs = self.processor(text=input_str, images=visual_input, audio=audio_input, return_tensors="pt").to("cuda:0")
            generate_ids = self.evaluator.generate(**inputs, max_new_tokens = 1500, generation_config=self.generation_config)
            generate_ids = generate_ids[:, inputs["input_ids"].shape[1]:]
            
            response = self.processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
            responses_dict = {
                "uniq_id" : self.uniq_id,
                "task_name" : self.task_name,
                "feedback" : response
            }
            gc.collect()
            torch.cuda.empty_cache()
            responses[f"{self.models[i*2]}_vs_{self.models[i*2+1]}"] = responses_dict

        return responses