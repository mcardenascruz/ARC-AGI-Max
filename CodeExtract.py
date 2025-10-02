
import re

def extract_function_def(generated_text: str) -> str:

    code_block_match = re.search(r"```(?:python)?([\s\S]*?)```", generated_text, re.DOTALL | re.IGNORECASE)
    if code_block_match:
        code = code_block_match.group(1).strip()
    else:
        code = generated_text.strip()


    # Remove any stray leading or trailing fences
    code = re.sub(r'^```(?:python)?\s*', '', code, re.IGNORECASE)
    code = re.sub(r'\s*```$', '', code)

    # Common header pattern with optional return type
    header_pattern = r"def\s+{name}\s*\([^)]*\)\s*(->\s*[^:]+)?\s*:"
    body_pattern = r"[\s\S]+?"
    lookahead = r"(?=\n\s*def\s+[a-zA-Z_]\w*\s*\(|$)"

    # Prefer def p
    p_pattern = header_pattern.format(name="p") + body_pattern + lookahead
    func_match = re.search(p_pattern, code, re.DOTALL)

    if not func_match:
        transform_pattern = header_pattern.format(name="transform") + body_pattern + lookahead
        func_match = re.search(transform_pattern, code, re.DOTALL)

    if not func_match:
        any_pattern = r"def\s+[a-zA-Z_]\w*\s*\([^)]*\)\s*(->\s*[^:]+)?\s*:" + body_pattern + lookahead
        func_match = re.search(any_pattern, code, re.DOTALL)

    if func_match:
        final_code = func_match.group(0).strip()
    else:
        final_code = code
    final_code = re.sub(r"```.*?```", "", final_code, re.DOTALL).strip()

    return final_code