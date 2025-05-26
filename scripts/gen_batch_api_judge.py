import os
from scripts.utils.Gemini2.api import *
from utils.utils import *
from utils.config import *
from google import genai
from copy import deepcopy
import asyncio
import argparse

client = genai.Client(api_key="YOUR-API-KEY")


async def main(p_flag, baseline, e_type, batch_size):
    benchmark = load_json(BENCHMARK_FILE)
    output_dir = e_type
    os.makedirs(output_dir, exist_ok=True)
    if p_flag:
        output_file = os.path.join(output_dir, f"Pair_{baseline}.json")
    else:
        output_file = os.path.join(output_dir, f"Score_{baseline}.json")
    if os.path.exists(output_file):
        output_data = load_json(output_file)
    else:
        output_data = []
    existing_idx = {entry["uniq_id"] for entry in output_data}
    # 100 batches one time?
    batch_size = batch_size
    tasks = []
    batch_start_index = 0
    if baseline != "checklist":
        c_flag = False
    else:
        c_flag = True
    # Loop over the benchmark entries in batches
    for i in range(0, len(benchmark), batch_size):
        batch = benchmark[i:i+batch_size]
        batch_results = []
        
        # Create async tasks for each entry in the batch
        for entry in batch:
            if entry["uniq_id"] in existing_idx:
                continue
            # Instantiate the appropriate content creator (Pairing or Scoring)
            if p_flag:
                generator = PairingGoogleContentCreator(client, entry, c_flag=c_flag, e_type=e_type)
            else:
                generator = ScoringGoogleContentCreator(client, entry, c_flag=c_flag, e_type=e_type)
            # Create the async task for content generation and append to tasks list
            if baseline == "overall":
                task = asyncio.create_task(generator.async_generate_overall_content())
            else:
                task = asyncio.create_task(generator.async_generate_rubric_content())
            tasks.append(task)
        if len(tasks) == 0:
            print(f"Saved data after processing batch {batch_start_index + 1} to {batch_start_index + batch_size}")
            batch_start_index += batch_size
            continue
        # Wait for all tasks to complete and gather the results for the batch
        results = await asyncio.gather(*tasks)

        # Process the results for the batch and append to output_data
        for result in results:
            if p_flag:
                for model_pair, response in result.items():
                    ent_template = {
                        "uniq_id": response["uniq_id"],
                        "task_name": response["task_name"],
                        "model_pair": model_pair,
                        "rubric_name": None,
                        "choice": None,
                        "comment": None,
                    }
                    # 根据 baseline 确定要处理的 rubric 类型
                    current_rubrics = ["overall_score"] if baseline == "overall" else RUBRICS
                    for rubric_name in current_rubrics:
                        ent = deepcopy(ent_template)
                        ent["rubric_name"] = rubric_name
                        # 处理字段前缀
                        rubric_prefix = "overall" if rubric_name == "overall_score" else rubric_name
                        ent["choice"] = response[f"{rubric_prefix}_choice"]
                        ent["comment"] = response.get(f"{rubric_prefix}_comment", "No comment provided.")
                        batch_results.append(ent)
            else:
                for model_name, response in result.items():
                    ent_template = {
                        "uniq_id": response["uniq_id"],
                        "task_name": response["task_name"],
                        "model_name": model_name,
                        "rubric_name": None,
                        "score": None,
                        "comment": None,
                    }
                    # 根据 baseline 确定要处理的 rubric 类型
                    current_rubrics = ["overall_score"] if baseline == "overall" else RUBRICS
                    for rubric_name in current_rubrics:
                        ent = deepcopy(ent_template)
                        ent["rubric_name"] = rubric_name
                        # 处理字段前缀
                        rubric_prefix = "overall" if rubric_name == "overall_score" else rubric_name
                        ent["score"] = response[f"{rubric_prefix}_score"]
                        ent["comment"] = response.get(f"{rubric_prefix}_comment", "No comment provided.")
                        batch_results.append(ent)

        # Save the output data for this batch
        output_data.extend(batch_results)
        save_json(output_data, output_file)  # Save after each batch
        print(f"Saved data after processing batch {batch_start_index + 1} to {batch_start_index + batch_size}")
        batch_start_index += batch_size  # Update the batch counter

    print("Finished processing all batches.")
        
        
if __name__ == "__main__":
    # get pair/score and using-checklist flag
    parser = argparse.ArgumentParser()
    # --pair means set a True flag
    parser.add_argument("--pair", action="store_true", help="Enable pairing mode")
    parser.add_argument("--evaluator", type=str, default="flash", help="Evaluator Type: [pro, flash,lite]")
    parser.add_argument("--bs", type=int, default=25, help="Batch size")
    parser.add_argument("--baseline", type=str, default="overall", help="Baselines: [overall, rubric, checklist]")
    
    args = parser.parse_args()
    pair = args.pair
    baseline = args.baseline
    e_type = args.evaluator
    batch_size = args.bs
    asyncio.run(main(pair,baseline,e_type,batch_size))