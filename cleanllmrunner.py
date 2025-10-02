import torch
import json
import os
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import gc
import re
from CodeExtract import extract_function_def
import traceback
import matplotlib.pyplot as plt
from gridDraw import plot_grid

# Pick Model and tokenizer
# "Qwen/Qwen2.5-Coder-7B-Instruct"
# "Qwen/Qwen3-Coder-30B-A3B-Instruct"
#"Qwen/Qwen3-14B"

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

model_id = "Qwen/Qwen3-Coder-30B-A3B-Instruct"# Model directory
max_new_tokens = 2048  # Limit for generated tokens

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=quantization_config, 
    device_map="auto",
    
)
model.eval()

def generate_code(taskData: dict) -> str:
    train_str = json.dumps(
        {"train": taskData.get("train", [])}, 
        separators=(",", ":")
    )

    # Stage 1 
    stage1_prompt = (
            "You are an expert algorithm designer specializing in grid-based puzzles. Your task is to analyze pairs of input/output grids and deduce the transformation rule. "
            "Based on the provided training examples, describe the transformation as a precise, step-by-step algorithm that can be implemented in Python. "
            "Focus on grid properties, object shapes, colors, and movements. Your description should be unambiguous and detailed enough for a programmer to follow. "
            "For example, instead of 'move the shape', specify 'find the largest contiguous object of color blue (2) and shift all its pixels down by 3 rows'.\n\n"
            "CRITICAL: Enclose your final algorithmic description ONLY within <reasoning>...</reasoning> tags. Do not add any other text, greetings, or explanations outside these tags.\n\n"
            f"TASK EXAMPLES:\n{train_str}\n\n"
        )
    inputs1 = tokenizer(stage1_prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output1 = model.generate(
            **inputs1,
            max_new_tokens=max_new_tokens,
            num_return_sequences=1,
            temperature=0.1,  
            
        )

    # Decode only new tokens to avoid prompt contamination
    response1_raw = tokenizer.decode(output1[0][len(inputs1.input_ids[0]):], skip_special_tokens=True)
    reasoning_match = re.search(r"<reasoning>([\s\S]*?)</reasoning>", response1_raw, re.IGNORECASE | re.DOTALL)
    reasoning_content = reasoning_match.group(1).strip() if reasoning_match else response1_raw.strip()
    print("Extracted REASONING:\n", reasoning_content)

    # Stage 2
    stage2_prompt = (
            "You are a robotic, highly-disciplined Python code writer. You will be given a set of REASONING steps. "
            "Your one and only job is to translate these steps into a single, complete Python function named `p`. "
            "The function must adhere strictly to the signature `def p(grid: list[list[int]]) -> list[list[int]]:`.\n"
            "Your output must be **only the code**. Do not write any explanations, comments, markdown, or any text whatsoever. "
            "Your response must begin with `def p` and nothing else. The function body must be fully implemented, not a placeholder.\n\n"
            "--- PERFECT EXAMPLE ---\n"
            "REASONING:\n"
            "1. Get the dimensions of the input grid (height and width).\n"
            "2. Create a new grid of the same dimensions, filled with zeros.\n"
            "3. Iterate through each cell (r, c) of the input grid.\n"
            "4. If the cell's value is 2 (blue), place a 4 (red) at the corresponding location in the new grid.\n"
            "5. Otherwise, copy the original cell's value to the new grid.\n"
            "6. Return the new grid.\n\n"
            "def p(grid: list[list[int]]) -> list[list[int]]:\n"
            "    height = len(grid)\n"
            "    width = len(grid[0])\n"
            "    new_grid = [[0 for _ in range(width)] for _ in range(height)]\n"
            "    for r in range(height):\n"
            "        for c in range(width):\n"
            "            if grid[r][c] == 2:\n"
            "                new_grid[r][c] = 4\n"
            "            else:\n"
            "                new_grid[r][c] = grid[r][c]\n"
            "    return new_grid\n"
            "--- END OF EXAMPLE ---\n\n"
            "--- YOUR TASK ---\n"
            "REASONING:\n"
            f"{reasoning_content}\n\n"
            "CODE:"
        )

    inputs2 = tokenizer(stage2_prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output2 = model.generate(
            **inputs2,
            max_new_tokens=max_new_tokens,
            num_return_sequences=1,
            temperature=0.1,  

        )
    # Decode only new tokens
    response2_raw = tokenizer.decode(output2[0][len(inputs2.input_ids[0]):], skip_special_tokens=True)

    code = extract_function_def(response2_raw)
    return code

def evaluate_code(code: str, taskData: dict, is_train: bool = True) -> dict:

    # Prepare the environment for code execution
    local_env = {}
    try:
        exec(code, {}, local_env)
        func = local_env.get("p")
        if not callable(func):
            print("Function 'p' not found in the provided code.")
            raise ValueError("No function 'p' defined.")
    except Exception as e:
        return {"error": f"Code execution error: {e}", "predicted": None, "expected": None}

    examples = taskData.get('train' if is_train else 'test', [])

    predicted = None
    expected_output = None
    for idx, example in enumerate(examples):
        input_grid = example.get('input')
        expected_output = example.get('output')
        try:
            predicted = func(input_grid)
        except Exception as e:
            # report the runtime error but keep trying next examples
            print(f"could not run function on input (example #{idx}): {e}")
            predicted = None

    return {"error": None, "predicted": predicted, "expected": expected_output}



def print_grid(label: str, grid: list) -> None:
    print(label)
    if grid is None:
        print("None")
    else:
        for row in grid:
            print(' '.join(map(str, row)))
    print()


def batch_evaluate(directory: str):
    log_file = 'generated_functions.txt'
    failed_tasks = []

    if os.path.exists(log_file):
        os.remove(log_file)

    keys= (['8e1813be.json', '694f12f3.json', '8be77c9e.json', '0ca9ddb6.json', '264363fd.json', '1cf80156.json', '2dee498d.json', '2bee17df.json', 'f35d900a.json', '93b581b8.json', '28bf18c6.json', 'ddf7fa4f.json', 'b548a754.json', 'bd4472b8.json', 'b60334d2.json', 'ea32f347.json', 'e50d258f.json', 'a79310a0.json', 'dae9d2b5.json', '25d487eb.json', 'c9e6f938.json', 'd687bc17.json', '09629e4f.json', '23581191.json', '67385a82.json', '56dc2b01.json', '890034e9.json', '3c9b0459.json', '46442a0e.json', '780d0b14.json', '3af2c5a8.json', '8efcae92.json', 'e9614598.json', '7b6016b9.json', 'ce4f8723.json', '941d9a10.json', 'd22278a0.json', '72322fa7.json', '3428a4f5.json', '9d9215db.json', '3ac3eb23.json', '9ecd008a.json', '4be741c5.json', 'dc0a314f.json', '88a62173.json', '7447852a.json', '5c0a986e.json', '5c2c9af4.json', '9565186b.json', '2281f1f4.json'])
    for filename in sorted(os.listdir(directory)):
        if filename.endswith(".json"): # for peter dataset add "and filename in keys"
            filepath = os.path.join(directory, filename)
            with open(filepath, 'r') as f:
                try:
                    taskData = json.load(f)
                    print(f"\nProcessing task: {filename}")
                    generated_code = generate_code(taskData)
                    with open(log_file, 'a') as log:
                        log.write(f"Task: {filename}\n{generated_code}\n\n{'-'*40}\n")

                    train_eval = evaluate_code(generated_code, taskData, is_train=True)

                    # If there was a catastrophic code execution error
                    if train_eval.get("error"):
                        print(f"  -> Reason: {train_eval['error']}")
                        failed_tasks.append(filename)
                        continue
                    else:
                        # print the predicted grid (may be None)
                        plot_grid(filename, train_eval.get("predicted"), filename=f"{filename}_predicted.png")
                        plot_grid(f"{filename} - Expected", train_eval.get("expected"), filename=f"{filename}_expected.png")

                except Exception as e:
                    print(f"Error processing {filename}: {e}")
                    traceback.print_exc()
                    failed_tasks.append(filename)

                # --- Clear memory between tasks to avoid OOM ---
                gc.collect()
                if torch is not None and torch.cuda.is_available():
                    torch.cuda.empty_cache()

# --- MAIN ENTRY POINT ---
if __name__ == "__main__":
    default_dir = os.path.expanduser("data/train")
    parser = argparse.ArgumentParser(description="ARC-AGI Task Solver")
    parser.add_argument("--dir", type=str, default=default_dir, help="Directory with JSON task files")
    args = parser.parse_args()

    if not os.path.isdir(args.dir):
        print(f"Error: Directory '{args.dir}' not found.")
    else:
        print(f"Running evaluation on '{args.dir}'...\n")
        batch_evaluate(args.dir)