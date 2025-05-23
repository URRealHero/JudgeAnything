import os
from utils.local import *
from utils.utils import *
from utils.config import *
from copy import deepcopy
import argparse
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig,pipeline,AutoTokenizer
import torch
import gc

def main(model_name, p_flag, baseline, batch_size=1):
    benchmark = load_json(BENCHMARK_FILE)
    output_dir = f"{model_name}"
    

    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="cuda",
        torch_dtype="auto",
        trust_remote_code=True,
        attn_implementation='flash_attention_2',
    ).cuda()
    generation_config = GenerationConfig.from_pretrained(model_name)
    generation_config.temperature = 0.7

    os.makedirs(output_dir, exist_ok=True)
    

    output_file = os.path.join(output_dir, f"{'Pair' if p_flag else 'Score'}_{baseline}.json")
    

    if os.path.exists(output_file):
        output_data = load_json(output_file)
    else:
        output_data = []
    existing_ids = {entry["uniq_id"] for entry in output_data}


    c_flag = (baseline == "checklist")


    for i, entry in enumerate(benchmark):
        if entry["uniq_id"] in existing_ids:
            continue


        if p_flag:
            generator = PairLocalContentCreator(
                entry, processor, model, generation_config, c_flag)
        else:
            generator = ScoreLocalContentCreator(
                entry, processor, model, generation_config, c_flag)

        try:
            if baseline == "overall":
                result = generator.generate_overall_content()
            else:
                result = generator.generate_rubric_content()
        except Exception as e:
            print(f"Error processing entry {i+1}: {str(e)}")
            continue

        batch_results = []
        if p_flag:
            for model_pair, response in result.items():
                ent = {
                    "uniq_id": response["uniq_id"],
                    "task_name": response["task_name"],
                    "model_pair": model_pair,
                    "response": response["feedback"],
                }
                batch_results.append(ent)
        else:
            for model_name, response in result.items():
                ent = {
                    "uniq_id": response["uniq_id"],
                    "task_name": response["task_name"],
                    "model_name": model_name,
                    "response": response["feedback"],
                }
                batch_results.append(ent)

        output_data.extend(batch_results)
        save_json(output_data, output_file)
        print(f"Saved data after processing entry {i+1}/{len(benchmark)}")


    print("Finished processing all entries.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="microsoft/Phi-4-multimodal-instruct", 
                       help="Model name")
    parser.add_argument("--pair", action="store_true", 
                       help="Enable pairing mode")
    parser.add_argument("--baseline", type=str, choices=["overall", "rubric", "checklist"], 
                       default="rubric", help="Evaluation baseline type")
    
    args = parser.parse_args()
    main(args.model, args.pair, args.baseline)