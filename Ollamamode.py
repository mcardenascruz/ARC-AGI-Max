import json
import os
import argparse
import gc
import re
from CodeExtract import extract_function_def
import traceback
import matplotlib.pyplot as plt
from gridDraw import plot_grid
from langchain_ollama import OllamaLLM

# global counter
total_matches = 0  

llm = OllamaLLM(
    model="dengcao/Qwen3-30B-A3B-Instruct-2507:latest",
    num_predict= 1024,
    num_ctx=4096,
    temperature=0.4,
    reasoning=True
)

llm_code = OllamaLLM(
    model="qwen3-coder:30b",
    num_predict= 1024,
    num_ctx=4096,
    temperature=0.1
)

def generate_code(taskData: dict) -> str:

    train_str = json.dumps(
        {"train": taskData.get("train", [])}, 
        separators=(",", ":")
    )
    stage1_prompt = (
    "You are an expert algorithm designer solving ARC-style intelligence puzzles (like IQ tests for grids). "
    "Each puzzle consists of input/output grid pairs. Your task is to determine the transformation and describe it as a precise, step-by-step algorithm that can be implemented in Python. "
    "Provide only actionable instructions. Do not include explanations, reasoning steps, guesses, or examples. "
    "Focus on all relevant grid operations: moving objects, merging or splitting shapes, changing colors, rotations, reflections, or other transformations needed to produce the output from the input. "
    "Be unambiguous and specific enough for a programmer to implement directly.\n\n"
    "CRITICAL: Enclose ONLY the algorithmic instructions within <reasoning>...</reasoning> tags. Output nothing outside these tags.\n\n"
    f"TASK EXAMPLES:\n{train_str}"
)

    output1 = llm.invoke(stage1_prompt)

    # Clean response
    response1_raw = output1
    cleaned_text = re.sub(r'<think>.*?</think>', '', response1_raw, flags=re.DOTALL | re.IGNORECASE)
    cleaned_text = re.sub(r'\n\s*\n', '\n\n', cleaned_text).strip()
    match = re.search(r'<reasoning>([\s\S]*?)</reasoning>', cleaned_text, re.IGNORECASE | re.DOTALL)

    if match:
        reasoning_content = match.group(1).strip()
        reasoning_content = re.sub(r'\n\s*\n', '\n\n', reasoning_content)
    else:
        reasoning_content = f"MODEL OUTPUT MISSING TAGS:\n{cleaned_text}"
        print("WARNING: No <reasoning> tags found. Using raw output instead.")

    print("Extracted REASONING:\n", reasoning_content)



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

    output2 = llm_code.invoke(stage2_prompt)
    response2_raw = output2

    code = extract_function_def(response2_raw)
    return code

def evaluate_code(code: str, taskData: dict) -> dict:
    global total_matches

    local_env = {}
    try:
        exec(code, {}, local_env)
        func = local_env.get("p")
        if not callable(func):
            raise ValueError("No function 'p' defined.")
    except Exception as e:
        return {"error": f"Code execution error: {e}", "predicted": None, "expected": None}
    match = False
    for example in taskData.get("train", []):
        try:
            if func(example["input"]) == example["output"]:
                test_example = taskData.get("test", [])[0]
                predicted_test = func(test_example["input"])
                expected_test = test_example["output"]
                match = True
                if predicted_test == expected_test:
                    total_matches += 1

                return {"error": None, "predicted": predicted_test, "expected": expected_test,  "match": match}
        except Exception:
            continue

    return {"error": None, "predicted": None, "expected": None}




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
    t=0
    for filename in sorted(os.listdir(directory)):
        if filename.endswith(".json"):
            filepath = os.path.join(directory, filename)
            with open(filepath, 'r') as f:
                try:
                    taskData = json.load(f)
                    t+=1
                    print(f"\nProcessing task: {filename}")
                    print(f"Current score = {total_matches}/{t} ({total_matches / t * 100:.1f}%)")
                    match = False
                    count = 0
                    while count < 10:
                        generated_code = generate_code(taskData)
                        with open(log_file, 'a') as log:
                            log.write(f"Task: {filename}\n{generated_code}\n\n{'-'*40}\n")

                        train_eval = evaluate_code(generated_code, taskData)

                        if train_eval.get("error"):
                            print(f"  -> Reason: {train_eval['error']}")
                            failed_tasks.append(filename)
                            continue
                        else:
                            if train_eval.get("match"):
                                plot_grid(filename, train_eval.get("predicted"), filename=f"{filename}_predicted.png")
                                plot_grid(f"{filename} - Expected", train_eval.get("expected"), filename=f"{filename}_expected.png")
                                count = 100
                                match = True
                            else:
                                count += 1
                        
                    if match == False:
                        print(f"  -> Attempt {count}: Train Mismatch")
                        plot_grid(filename, train_eval.get("predicted"), filename=f"{filename}_predicted.png")
                        plot_grid(f"{filename} - Expected", train_eval.get("expected"), filename=f"{filename}_expected.png")

                except Exception as e:
                    print(f"Error processing {filename}: {e}")
                    traceback.print_exc()
                    failed_tasks.append(filename)

                gc.collect()

if __name__ == "__main__":
    default_dir = os.path.expanduser("/home/epochvipc4/Desktop/projects/arc_agi1/ARC-AGI/data/training")
    parser = argparse.ArgumentParser(description="ARC-AGI Task Solver")
    parser.add_argument("--dir", type=str, default=default_dir, help="Directory with JSON task files")
    args = parser.parse_args()

    if not os.path.isdir(args.dir):
        print(f"Error: Directory '{args.dir}' not found.")
    else:
        print(f"Running evaluation on '{args.dir}'...\n")
        batch_evaluate(args.dir)