import re

def extract_function_def(generated_text: str) -> str:
    # Extract code from fenced block if present
    code_block_match = re.search(r"```(?:python)?([\s\S]*?)```", generated_text, re.DOTALL | re.IGNORECASE)
    if code_block_match:
        code = code_block_match.group(1).strip()
    else:
        code = generated_text.strip()

    # Remove stray fences
    code = re.sub(r'^```(?:python)?\s*', '', code, flags=re.IGNORECASE | re.MULTILINE)
    code = re.sub(r'\s*```$', '', code, flags=re.MULTILINE)

    # Match a function definition (p, transform, or any function)
    def_pattern = re.compile(
        r"(def\s+(?:p|transform|[a-zA-Z_]\w*)\s*\([^)]*\)\s*(?:->\s*[^:]+)?\s*:[\s\S]*?)(?=\n\s*def\s+[a-zA-Z_]\w*\s*\(|\Z)",
        re.DOTALL
    )

    match = def_pattern.search(code)
    if match:
        final_code = match.group(1).strip()
    else:
        final_code = code

    # 🚨 Remove trailing junk like "--- END OF TASK ---" or "Task: xyz.json"
    final_code = re.sub(r"---.*?---", "", final_code, flags=re.DOTALL)
    final_code = re.sub(r"Task:.*", "", final_code, flags=re.DOTALL)
    
    # Cleanup again (in case junk left extra blank lines)
    final_code = final_code.strip()

    return final_code
