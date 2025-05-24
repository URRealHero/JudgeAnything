import os
import json
from copy import deepcopy
from ..utils import *
from ..config import *
from ..schemas import *
from ..prompt_builder import SystemPromptBuilder

class BaseGoogleContentCreator:
    """
    This is the use of base, for score/pair evaluator's succeeding
    Will init benchmark data, result data, checklist data
    given an uniq_id, this will: 1. get the input query and input media files 2. get each model's response 3. construct content(Adding exp and checklist if c_flag is True) 4. get the result and save in result
    """
    def __init__(self, client, entry, c_flag=False, e_type="flash"):
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
        if c_flag:
            self.checklists = self.get_checklist(self.uniq_id)
        else:
            self.checklists = None
        self.client = client
        self.models = []
        self.contents = None
        self.evaluator = GOOGLE_MODELS[e_type]
        
    def get_result_entry(self, uniq_id):
        result_data = load_json(self.result)
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

    async def async_generate_content(self):
        pass
    
    
class ScoringGoogleContentCreator(BaseGoogleContentCreator):
    def __init__(self, client, entry, c_flag=False, e_type="flash"):
        super().__init__(client, entry, c_flag, e_type)
        
        
    def construct_content(self):
        """
        Construct the content for scoring
        """
        content = []
        contents = []
        media_files = {
            "input_media": [],
            "model1": [],
            "model2": [],
            "model3": [],
            "model4": []
        }
        text_strs={
            "model1": "",
            "model2": "",
            "model3": "",
            "model4": ""
        }
        input_query = f"Here is the query of {self.input_modality} to {self.output_modality} generation task:\n"+self.entry["question"]
        # print(input_query)
        response_prompt = "\nHere is the model's reponse:\n"
        
        content.append(input_query)
        
        if self.input_modality.lower() == "image":
            if isinstance(self.entry["image_path"],list):
                for image_path in self.entry["image_path"]:
                    img = Image.open(os.path.join(BASE_DIR, image_path))
                    media_files["input_media"].append(img)
            else:
                img = Image.open(os.path.join(BASE_DIR, self.entry["image_path"]))
                media_files["input_media"].append(img)
        elif self.input_modality.lower() == "audio":
            media_files["input_media"].append(os.path.join(BASE_DIR, self.entry["audio_path"]))
        elif self.input_modality.lower() == "video" or self.input_modality.lower() == "audiovideo":
            media_files["input_media"].append(os.path.join(BASE_DIR, self.entry["video_path"]))
        
        i = 0
        for res_entry in self.result_entries:
            self.models.append(res_entry["model_name"])
            resp = res_entry["response"]
            if resp["type"].lower() == "image":
                if isinstance(resp["content"],list):
                    for image_path in resp["content"]:
                        img = Image.open(os.path.join(RESULT_DIR, image_path))
                        media_files[f"model{i+1}"].append(img)
                else:
                    img = Image.open(os.path.join(RESULT_DIR, resp["content"]))
                    media_files[f"model{i+1}"].append(img)
            elif resp["type"].lower() == "audio":
                media_files[f"model{i+1}"].append(os.path.join(RESULT_DIR, resp["content"]))
            elif resp["type"].lower() == "video":
                media_files[f"model{i+1}"].append(os.path.join(RESULT_DIR, resp["content"]))
            elif resp["type"].lower() == "text":
                text_strs[f"model{i+1}"] = resp["content"]
            i += 1
        input_files, model1_files, model2_files, model3_files, model4_files = upload_media_for_task(self.client, media_files)
        for file in input_files:
            content.append(file)
        if self.checklists:
            content.append(self.checklists)
        
        
        # Caching Content: cached_content = client.caches.create(model=MODEL, contents=[...],config=types.CreateCachedContentConfig(system_instruction=system_instruction,ttl="3600s",))
        # response = client.models.generate_content(model=cached_content.model,contents=,config=types.GenerateContentConfig(cached_content=cached_content.name, temperature=0.7))
        # client.caches.delete(name=cached_content.name)

        # Now for each model's response, making four responses contents, and then generate the content, delete the cache
        for i,model_files in enumerate([model1_files, model2_files, model3_files, model4_files]):
            model_content = deepcopy(content)
            model_content.append(response_prompt)
            if text_strs[f"model{i+1}"] != "":
                model_content.append(text_strs[f"model{i+1}"])
            else:
                for file in model_files:
                    model_content.append(file)
            contents.append(model_content)
        
        self.contents = contents
        
    def generate_rubric_content(self):
        self.construct_content()
        results = {}
        for i,model_content in enumerate(self.contents):
            response = self.client.models.generate_content(model=self.evaluator, contents=model_content, config=types.GenerateContentConfig(system_instruction=self.sys_constructor.build_rubric_scoring_prompt(),temperature=0.7, response_mime_type="application/json", response_schema=ScoringRubrics))
            # Save the result
            res = json.loads(response.text)
            
            res["uniq_id"] = self.uniq_id
            res["task_name"] = self.task_name
            res["response_id"] = self.result_entries[i]["response_id"]
            
            results[self.models[i]] = res
        return results
    
    async def async_generate_rubric_content(self):
        self.construct_content()
        results = {}
        tasks = [
            self.client.aio.models.generate_content(
                model=self.evaluator,
                contents=model_content,
                config=types.GenerateContentConfig(
                    system_instruction=self.sys_constructor.build_rubric_scoring_prompt(),
                    temperature=0.7,
                    response_mime_type="application/json",
                    response_schema=ScoringRubrics
                )
            ) for model_content in self.contents
        ]
        responses = await asyncio.gather(*tasks)
        
        for i,response in enumerate(responses):
            res = json.loads(response.text)
            res["uniq_id"] = self.uniq_id
            res["task_name"] = self.task_name
            res["response_id"] = self.result_entries[i]["response_id"]
            results[self.models[i]] = res
        return results
    
    def generate_overall_content(self):
        self.construct_content()
        results = {}
        for i,model_content in enumerate(self.contents):
            response = self.client.models.generate_content(model=self.evaluator, contents=model_content, config=types.GenerateContentConfig(system_instruction=self.sys_constructor.build_overall_scoring_prompt(),temperature=0.7, response_mime_type="application/json", response_schema=ScoringOverall))
            # Save the result
            res = json.loads(response.text)
            res["uniq_id"] = self.uniq_id
            res["task_name"] = self.task_name
            res["response_id"] = self.result_entries[i]["response_id"]
            results[self.models[i]] = res
        return results
    
    async def async_generate_overall_content(self):
        self.construct_content()
        results = {}
        tasks = [
            self.client.aio.models.generate_content(
                model=self.evaluator,
                contents=model_content,
                config=types.GenerateContentConfig(
                    system_instruction=self.sys_constructor.build_overall_scoring_prompt(),
                    temperature=0.7,
                    response_mime_type="application/json",
                    response_schema=ScoringOverall
                )
            ) for model_content in self.contents
        ]
        responses = await asyncio.gather(*tasks)
        
        for i,response in enumerate(responses):
            res = json.loads(response.text)
            res["uniq_id"] = self.uniq_id
            res["task_name"] = self.task_name
            res["response_id"] = self.result_entries[i]["response_id"]
            results[self.models[i]] = res
        return results
    


class PairingGoogleContentCreator(BaseGoogleContentCreator):
    def __init__(self, client, entry, c_flag=False, e_type="flash"):
        super().__init__(client, entry, c_flag, e_type)
        
    def construct_content(self):
        """
        Construct the content for pairing
        """
        content = []
        contents = []
        media_files = {
            "input_media": [],
            "model1": [],
            "model2": [],
            "model3": [],
            "model4": []
        }
        text_strs={
            "model1": "",
            "model2": "",
            "model3": "",
            "model4": ""
        }
        input_query = f"Here is the query of {self.input_modality} to {self.output_modality} generation task:\n"+self.entry["question"]
        response1_prompt = "\nHere is the first model's response:\n"
        response2_prompt = "\nHere is the second model's response:\n"
        
        content.append(input_query)
        
        if self.input_modality.lower() == "image":
            if isinstance(self.entry["image_path"],list):
                for image_path in self.entry["image_path"]:
                    img = Image.open(os.path.join(BASE_DIR, image_path))
                    media_files["input_media"].append(img)
            else:
                img = Image.open(os.path.join(BASE_DIR, self.entry["image_path"]))
                media_files["input_media"].append(img)
        elif self.input_modality.lower() == "audio":
            media_files["input_media"].append(os.path.join(BASE_DIR, self.entry["audio_path"]))
        elif self.input_modality.lower() == "video" or self.input_modality.lower() == "audiovideo":
            media_files["input_media"].append(os.path.join(BASE_DIR, self.entry["video_path"]))
        
        i=0
        for res_entry in self.result_entries:
            self.models.append(res_entry["model_name"])
            resp = res_entry["response"]
            if resp["type"].lower() == "image":
                if isinstance(resp["content"],list):
                    for image_path in resp["content"]:
                        img = Image.open(os.path.join(RESULT_DIR, image_path))
                        media_files[f"model{i+1}"].append(img)
                else:
                    img = Image.open(os.path.join(RESULT_DIR, resp["content"]))
                    media_files[f"model{i+1}"].append(img)
            elif resp["type"].lower() == "audio":
                media_files[f"model{i+1}"].append(os.path.join(RESULT_DIR, resp["content"]))
            elif resp["type"].lower() == "video":
                media_files[f"model{i+1}"].append(os.path.join(RESULT_DIR, resp["content"]))
            elif resp["type"].lower() == "text":
                text_strs[f"model{i+1}"] = resp["content"]
            i += 1
        
        input_files, model1_files, model2_files, model3_files, model4_files = upload_media_for_task(self.client, media_files)
        models_files = [model1_files, model2_files, model3_files, model4_files]
        for file in input_files:
            content.append(file)
        if self.checklists:
            content.append(self.checklists)
        
        
        # Now for each model's response, making two paired contents (model1 vs model2 ; model3 vs model4)
        for i in range(0,4,2):
            model_pair_content = deepcopy(content)
            model_pair_content.append(response1_prompt)
            if text_strs[f"model{i+1}"] != "":
                model_pair_content.append(text_strs[f"model{i+1}"])
            else:
                for file in models_files[i]:
                    model_pair_content.append(file)
            model_pair_content.append(response2_prompt)
            if text_strs[f"model{i+2}"] != "":
                model_pair_content.append(text_strs[f"model{i+2}"])
            else:
                for file in models_files[i+1]:
                    model_pair_content.append(file)
            contents.append(model_pair_content)
        
        self.contents = contents
        
    def generate_rubric_content(self):
        self.construct_content()
        results = {}
        
        for i,model_pair_content in enumerate(self.contents):
            response = self.client.models.generate_content(model=self.evaluator, contents=model_pair_content, config=types.GenerateContentConfig(system_instruction=self.sys_constructor.build_rubric_pairing_prompt(), temperature=0.7, response_mime_type="application/json", response_schema=PairingRubrics))
            # Save the result
            res = json.loads(response.text)
            res["uniq_id"] = self.uniq_id
            res["task_name"] = self.task_name
            results[f"{self.models[i*2]}_vs_{self.models[i*2+1]}"] = res
        return results
    
    
    async def async_generate_rubric_content(self):
        self.construct_content()
        results = {}
        tasks = [
            self.client.aio.models.generate_content(
                model=self.evaluator,
                contents=model_pair_content,
                config=types.GenerateContentConfig(
                    system_instruction=self.sys_constructor.build_rubric_pairing_prompt(),
                    temperature=0.7,
                    response_mime_type="application/json",
                    response_schema=PairingRubrics
                )
            ) for model_pair_content in self.contents
        ]
        responses = await asyncio.gather(*tasks)
        
        for i,response in enumerate(responses):
            res = json.loads(response.text)
            res["uniq_id"] = self.uniq_id
            res["task_name"] = self.task_name
            results[f"{self.models[i*2]}_vs_{self.models[i*2+1]}"] = res
            
        return results
    
    def generate_overall_content(self):
        self.construct_content()
        results = {}
        
        for i,model_pair_content in enumerate(self.contents):
            response = self.client.models.generate_content(model=self.evaluator, contents=model_pair_content, config=types.GenerateContentConfig(system_instruction=self.sys_constructor.build_overall_pairing_prompt(), temperature=0.7, response_mime_type="application/json", response_schema=PairingOverall))
            # Save the result
            res = json.loads(response.text)
            res["uniq_id"] = self.uniq_id
            res["task_name"] = self.task_name
            
            results[f"{self.models[i*2]}_vs_{self.models[i*2+1]}"] = res
        return results
    
    async def async_generate_overall_content(self):
        self.construct_content()
        results = {}
        tasks = [
            self.client.aio.models.generate_content(
                model=self.evaluator,
                contents=model_pair_content,
                config=types.GenerateContentConfig(
                    system_instruction=self.sys_constructor.build_overall_pairing_prompt(),
                    temperature=0.7,
                    response_mime_type="application/json",
                    response_schema=PairingOverall
                )
            ) for model_pair_content in self.contents
        ]
        responses = await asyncio.gather(*tasks)
        
        for i,response in enumerate(responses):
            res = json.loads(response.text)
            res["uniq_id"] = self.uniq_id
            res["task_name"] = self.task_name
            results[f"{self.models[i*2]}_vs_{self.models[i*2+1]}"] = res
            
        return results
    
