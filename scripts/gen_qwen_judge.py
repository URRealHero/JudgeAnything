# gen_qwen_omni_judge.py
import os
import json
import argparse
import torch
import gc
import logging
from transformers.utils import logging as hf_logging

# QwenOmni specific imports from Hugging Face Transformers
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor, GenerationConfig

from utils.utils import load_json, save_json
from utils.config import BENCHMARK_FILE # BENCHMARK_FILE path from your config.py

log_file_path = "qwen_omni_judge_debug.log" # Changed filename for clarity
logging.basicConfig(
    level=logging.INFO, # Default to INFO, can be changed via args if needed
    format="%(asctime)s [%(levelname)s] %(name)s:%(lineno)d - %(message)s",
    handlers=[
        logging.FileHandler(log_file_path, mode='a'), # Append mode
        logging.StreamHandler()
    ]
)
# hf_logging.set_verbosity_info() # Set verbosity for Hugging Face logs if desired

def main(model_name: str,
         p_flag: bool,
         baseline: str):

    logging.info(f"Starting QwenOmni evaluation run: {model_name}")
    logging.info(f"Evaluation mode: {'Pairing' if p_flag else 'Scoring'}, Baseline: {baseline}")

    benchmark_data = load_json(BENCHMARK_FILE)
    if not benchmark_data:
        logging.error(f"Failed to load benchmark data from: {BENCHMARK_FILE}. Exiting.")
        return

    output_dir = os.path.join(model_name)
    os.makedirs(output_dir, exist_ok=True)
    logging.info(f"Output will be saved in: {output_dir}")

    # model_path = "Qwen/Qwen2.5-Omni-7B" # Default QwenOmni model path, can be overridden by CLI args
    # --- QwenOmni Model and Official Processor Initialization ---
    logging.info(f"Loading QwenOmni model and processor from: {model_name}")

    try:
        # 1. Load Qwen2_5OmniProcessor (Official)
        # This processor handles text tokenization and works with process_mm_info for multimedia.
        qwen_official_processor = Qwen2_5OmniProcessor.from_pretrained(
            model_name
        )
        logging.info("Qwen2_5OmniProcessor loaded successfully.")

        # 2. Load Qwen2_5OmniForConditionalGeneration Model (Official)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        # Qwen models often perform well with bf16 on compatible GPUs, fallback to fp16 or fp32.
        torch_dtype = torch.bfloat16 if device == "cuda" and torch.cuda.is_bf16_supported() else torch.float16
        if device == "cpu":
            torch_dtype = torch.float32
        
        logging.info(f"Attempting to load model on device: {device} with dtype: {torch_dtype}")
        qwen_omni_model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            device_map="auto",  # Handles multi-GPU or sends to 'device' if single GPU/CPU
            attn_implementation="flash_attention_2", # As per Qwen demo, if available
        ).eval() # Set to evaluation mode
        logging.info(f"Qwen2_5OmniForConditionalGeneration model loaded successfully to device(s): {qwen_omni_model.device_map if hasattr(qwen_omni_model, 'device_map') else qwen_omni_model.device}.")
        logging.info(f"Model is using dtype: {next(qwen_omni_model.parameters()).dtype}")


    except Exception as e:
        logging.error(f"Fatal error during QwenOmni model or processor initialization: {e}", exc_info=True)
        return

    # 3. GenerationConfig
    # Start with some defaults, then try to load from model path and merge.
    generation_config_dict = {
        "max_new_tokens": 256,       # Max tokens for the evaluator's feedback
        "do_sample": True,           # Enable sampling
        "temperature": 0.7,          # Common temperature for creative/evaluative tasks
    }

    
    from utils.QwenOmni.local import ScoreLocalContentCreator, PairLocalContentCreator

    # --- Output File Handling & Loop Setup ---
    output_file_name = f"{'Pair' if p_flag else 'Score'}_{baseline}.json"
    output_file_path = os.path.join(output_dir, output_file_name)

    output_data = []
    if os.path.exists(output_file_path):
        try:
            output_data = load_json(output_file_path)
            if not isinstance(output_data, list):
                logging.warning(f"Output file {output_file_path} was not a list. Re-initializing.")
                output_data = []
        except json.JSONDecodeError:
            logging.warning(f"Could not decode JSON from {output_file_path}. Re-initializing.")
            output_data = []
    existing_uniq_ids = {entry.get("uniq_id") for entry in output_data if isinstance(entry, dict)}

    c_flag = (baseline == "checklist") # True if baseline is "checklist"

    # --- Main Processing Loop ---
    processed_count = 0
    for i, entry in enumerate(benchmark_data):
        if not isinstance(entry, dict) or "uniq_id" not in entry:
            logging.warning(f"Skipping invalid benchmark entry at index {i} (missing 'uniq_id' or not a dict): {str(entry)[:200]}")
            continue

        current_uniq_id = entry["uniq_id"]
        if current_uniq_id in existing_uniq_ids:
            continue

        logging.info(f"Processing entry {i+1}/{len(benchmark_data)}: uniq_id='{current_uniq_id}', task='{entry.get('task_name', 'N/A')}'")

        generator_instance = None
        try:
            if p_flag:
                generator_instance = PairLocalContentCreator(
                    entry=entry,
                    qwen_official_processor=qwen_official_processor, # Pass the official Qwen processor
                    evaluator_model=qwen_omni_model,
                    generation_config_dict=generation_config_dict,
                    c_flag=c_flag
                    # Add use_audio_in_video_flag if it needs to be passed down
                )
            else:
                generator_instance = ScoreLocalContentCreator(
                    entry=entry,
                    qwen_official_processor=qwen_official_processor, # Pass the official Qwen processor
                    evaluator_model=qwen_omni_model,
                    generation_config_dict=generation_config_dict,
                    c_flag=c_flag
                    # Add use_audio_in_video_flag if it needs to be passed down
                )

            if baseline == "overall":
                result_dict_from_generator = generator_instance.generate_overall_content()
            else: # "rubric" or "checklist"
                result_dict_from_generator = generator_instance.generate_rubric_content()

        except Exception as e:
            logging.error(f"Error processing entry uniq_id='{current_uniq_id}': {str(e)}", exc_info=True)
            continue # Skip to the next entry

        # Format results (same as your previous scripts)
        current_run_formatted_results = []
        if p_flag:
            for model_pair_key, response_data in result_dict_from_generator.items():
                formatted_item = {
                    "uniq_id": response_data.get("uniq_id", current_uniq_id),
                    "task_name": response_data.get("task_name", entry.get("task_name")),
                    "model_pair": model_pair_key,
                    "response": response_data.get("feedback", "[Error: No feedback generated]"),
                }
                current_run_formatted_results.append(formatted_item)
        else: # Scoring mode
            for model_name_key, response_data in result_dict_from_generator.items():
                formatted_item = {
                    "uniq_id": response_data.get("uniq_id", current_uniq_id),
                    "task_name": response_data.get("task_name", entry.get("task_name")),
                    "model_name": model_name_key,
                    "response_id": response_data.get("response_id", "N/A"),
                    "response": response_data.get("feedback", "[Error: No feedback generated]"),
                }
                current_run_formatted_results.append(formatted_item)

        output_data.extend(current_run_formatted_results)
        try:
            save_json(output_data, output_file_path)
            logging.info(f"Saved data to {output_file_path} after processing uniq_id='{current_uniq_id}' ({len(current_run_formatted_results)} new result(s)).")
            processed_count +=1
        except Exception as e:
            logging.error(f"Failed to save output data to {output_file_path}: {e}")

        # Clean up
        del generator_instance
        del result_dict_from_generator
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    logging.info(f"Finished processing all entries. Total new entries processed in this run: {processed_count}. Results saved to {output_file_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate evaluations using a QwenOmni model.")
    parser.add_argument("--model_name_tag", type=str, default="Qwen2.5-Omni-7B_Evaluation",
                       help="A tag for this evaluation run, used for naming the output directory.")
    parser.add_argument("--qwen_model_path", type=str, default="Qwen/Qwen2.5-Omni-7B", # Default to a common Qwen Omni model
                       help="Hugging Face Hub path or local path to the QwenOmni model.")
    parser.add_argument("--pair", action="store_true",
                       help="Enable pairing mode (ModelA vs ModelB). Default is scoring mode.")
    parser.add_argument("--baseline", type=str, choices=["overall", "rubric", "checklist"],
                       default="rubric",
                       help="Evaluation baseline type: 'overall', 'rubric', or 'checklist'.")
    # You might want to add an argument for --use_audio_in_video if you want to control it from CLI

    args = parser.parse_args()

    # Basic check for benchmark file existence
    if not os.path.exists(BENCHMARK_FILE):
        print(f"ERROR: BENCHMARK_FILE not found at configured path: {BENCHMARK_FILE}")
        print(f"Please ensure your ..utils/config.py points to the correct benchmark JSON file (current: {os.path.abspath(BENCHMARK_FILE)}).")
        exit(1)

    main(cli_model_name_tag=args.model_name_tag,
         qwen_model_hub_path=args.qwen_model_path,
         p_flag=args.pair,
         baseline=args.baseline)