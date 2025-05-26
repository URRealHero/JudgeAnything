# gen_intern_omni_judge.py
import os
import json
from copy import deepcopy
import argparse
import torch
import gc
import logging
from transformers.utils import logging as hf_logging
from transformers import AutoModel, GenerationConfig # AutoModel for InternOmni

from utils.InternOmni.local import ScoreLocalContentCreator, PairLocalContentCreator 
from utils.utils import load_json, save_json # Assuming these are general utils
from utils.config import BENCHMARK_FILE # Assuming this is correctly defined in your config


# Setup basic logging (similar to your Phi4mm script)
log_file = "intern_omni_judge_debug.log"
# Basic logging, customize as needed
logging.basicConfig(
    level=logging.INFO, # Changed from DEBUG for less verbosity by default
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)

def main(model_name,
         p_flag, 
         baseline):
    benchmark_data = load_json(BENCHMARK_FILE)
    if not benchmark_data:
        logging.error(f"Failed to load benchmark data from: {BENCHMARK_FILE}. Exiting.")
        return

    output_dir = os.path.join(model_name) # Example output path
    
    # --- InternOmni Model Specific Configuration ---
    # model_path = "$HF_HOME/hub/..." Official path, or your local path
    model_path = model_name
    logging.info(f"Using InternOmni model path: {model_path}")

    # Init InternOmniUnifiedProcessor
    from utils.InternOmni.Processor import InternOmniUnifiedProcessor 
    try:
        intern_omni_processor = InternOmniUnifiedProcessor(
            model_path_or_name=model_path,
        )
        logging.info(f"InternOmniUnifiedProcessor initialized from {model_path}.") 
    except Exception as e:
        logging.error(f"Failed to initialize InternOmniUnifiedProcessor: {e}")
        return

    # Loading InternOmni Model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.bfloat16 if device == "cuda" and torch.cuda.is_bf16_supported() else torch.float16
    
    try:
        intern_omni_model = AutoModel.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        ).eval().to(device)
        logging.info(f"InternOmni model loaded from {model_path} to {device} with dtype {torch_dtype}.")
    except Exception as e:
        logging.error(f"Failed to load InternOmni model: {e}")
        return

    generation_config_dict = {
        "max_new_tokens": 200,
        "do_sample": True,
        "temperature": 0.7,
    }
    logging.info(f"Using generation config: {generation_config_dict}")


    os.makedirs(output_dir, exist_ok=True)
    
    output_file_name = f"{'Pair' if p_flag else 'Score'}_{baseline}.json"
    output_file_path = os.path.join(output_dir, output_file_name)
    
    if os.path.exists(output_file_path):
        try:
            output_data = load_json(output_file_path)
            if not isinstance(output_data, list): # Ensure it's a list
                logging.warning(f"Output file {output_file_path} did not contain a list. Initializing as empty list.")
                output_data = []
        except json.JSONDecodeError:
            logging.warning(f"Could not decode JSON from {output_file_path}. Initializing as empty list.")
            output_data = []

    else:
        output_data = []
    existing_ids = {entry.get("uniq_id") for entry in output_data if isinstance(entry, dict)} # Added get and check

    c_flag = (baseline == "checklist") # Checklist flag

    # Main processing loop
    for i, entry in enumerate(benchmark_data):
        if not isinstance(entry, dict) or "uniq_id" not in entry:
            logging.warning(f"Skipping invalid benchmark entry at index {i}: {entry}")
            continue
            
        current_uniq_id = entry["uniq_id"]
        if current_uniq_id in existing_ids:
            logging.info(f"Skipping already processed uniq_id: {current_uniq_id}")
            continue

        logging.info(f"Processing entry {i+1}/{len(benchmark_data)}: uniq_id={current_uniq_id}")
        
        try:
            if p_flag: # Pairing mode
                generator = PairLocalContentCreator(
                    entry=entry, 
                    intern_omni_processor=intern_omni_processor, 
                    evaluator_model=intern_omni_model, 
                    generation_config_dict=generation_config_dict, 
                    c_flag=c_flag
                )
            else:
                generator = ScoreLocalContentCreator(
                    entry=entry, 
                    intern_omni_processor=intern_omni_processor, 
                    evaluator_model=intern_omni_model, 
                    generation_config_dict=generation_config_dict, 
                    c_flag=c_flag
                )

            if baseline == "overall":
                result_dict = generator.generate_overall_content()
            else: # "rubric" or "checklist" (checklist flag c_flag handles content part)
                result_dict = generator.generate_rubric_content()
        
        except Exception as e:
            logging.error(f"Error processing entry uniq_id={current_uniq_id}: {str(e)}", exc_info=True) # Log traceback
            continue # Skip to next entry

        # Process results from the generator
        # The generator methods (generate_rubric_content etc.) return a dictionary.
        # Keys are model_name (score) or model_pair (pair), values are dicts with feedback.
        current_batch_formatted_results = []
        if p_flag:
            for model_pair_key, response_data in result_dict.items():
                formatted_entry = {
                    "uniq_id": response_data.get("uniq_id", current_uniq_id),
                    "task_name": response_data.get("task_name", entry.get("task_name")),
                    "model_pair": model_pair_key,
                    "response": response_data.get("feedback", "Error: No feedback generated."),
                }
                current_batch_formatted_results.append(formatted_entry)
        else:
            for model_name_key, response_data in result_dict.items():
                formatted_entry = {
                    "uniq_id": response_data.get("uniq_id", current_uniq_id),
                    "task_name": response_data.get("task_name", entry.get("task_name")),
                    "model_name": model_name_key,
                    "response_id": response_data.get("response_id", "N/A"),
                    "response": response_data.get("feedback", "Error: No feedback generated."),
                }
                current_batch_formatted_results.append(formatted_entry)

        output_data.extend(current_batch_formatted_results)
        try:
            save_json(output_data, output_file_path)
            logging.info(f"Saved data to {output_file_path} after processing uniq_id={current_uniq_id}")
        except Exception as e:
            logging.error(f"Failed to save output data to {output_file_path}: {e}")

        del generator
        del result_dict
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()


    logging.info(f"Finished processing all entries. Final results saved to {output_file_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate evaluations using InternOmni model.")
    parser.add_argument("--The Huggingface Model Name. You can modify to your local dir.")
    parser.add_argument("--pair", action="store_true", 
                       help="Enable pairing mode (ModelA vs ModelB). Default is scoring mode.")
    parser.add_argument("--baseline", type=str, choices=["overall", "rubric", "checklist"], 
                       default="rubric", 
                       help="Evaluation baseline type: 'overall', 'rubric', or 'checklist' (which uses rubric scoring with checklist text).")
    
    args = parser.parse_args()
    
    # Ensure BENCHMARK_FILE path is correct based on your config.py
    if not os.path.exists(BENCHMARK_FILE):
        print(f"ERROR: BENCHMARK_FILE not found at configured path: {BENCHMARK_FILE}")
        print("Please ensure your config.py points to the correct benchmark JSON file.")
        exit(1)
        
    main(model_name=args.model_name,
         p_flag=args.pair, 
         baseline=args.baseline)