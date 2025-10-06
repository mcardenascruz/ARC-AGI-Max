def get_stage1_prompt(representation_type: str, train_str: str) -> str:
    stage1_prompt_json = (
        "You are an expert algorithm designer solving ARC-style intelligence puzzles (like IQ tests for grids). "
        "Each puzzle consists of multiple input/output grid pairs as training examples. Your task is to analyze all examples to identify the consistent transformation rule that applies across all of them, then describe it as a precise, step-by-step algorithm that can be implemented in Python. "
        "Provide only actionable instructions. Do not include explanations, reasoning steps, guesses, or examples beyond what's necessary for clarity. "
        "Focus on all relevant grid operations: identifying objects, moving objects, merging or splitting shapes, changing colors, rotations, reflections, symmetry detection, pattern filling, counting, or other transformations needed to produce the output from the input. "
        "Be unambiguous and specific enough for a programmer to implement directly, including handling grid sizes, color values (0-9), and edge cases observed in the examples.\n\n"
        "CRITICAL: Enclose ONLY the algorithmic instructions within <reasoning>...</reasoning> tags. Output nothing outside these tags.\n\n"
        f"TASK EXAMPLES:\n{train_str}"
    )
    stage1_prompt_images = (
        "You are an expert in visual reasoning and algorithmic pattern recognition for ARC puzzles. "
        "Each puzzle example consists of INPUT and OUTPUT grid images encoded as base64 PNGs. "
        "Each pixel represents one cell in a discrete grid; its color corresponds to a specific value (typically 0-9, where 0 is background).\n\n"
        "Your goal is to analyze all examples to infer the consistent transformation that converts every INPUT image into its corresponding OUTPUT image. "
        "Describe the transformation as clear, step-by-step operations suitable for a Python implementation — such as object detection, movement, reflection, resizing, recoloring, pattern replication, or shape manipulation. "
        "Do not include any explanations, color descriptions, or visual commentary. Focus only on precise algorithmic steps that work for all examples.\n\n"
        "CRITICAL: Enclose ONLY the algorithmic instructions within <reasoning>...</reasoning> tags. Output nothing outside these tags.\n\n"
        f"TASK EXAMPLES (image-based):\n{train_str}"
    )

    stage1_prompt_rle = (
        "You are an expert in grid transformation reasoning for ARC puzzles. "
        "Each example represents the INPUT and OUTPUT grids using Run-Length Encoding (RLE). "
        "Each RLE string encodes rows of the grid as counts of consecutive cell values (colors 0-9). "
        "For example, '3x0,2x1' means three background cells (0) followed by two colored cells (1).\n\n"
        "Your goal is to analyze all examples to determine the consistent transformation that maps every input grid to its output grid. "
        "Describe the steps as explicit grid operations that can be implemented in Python — such as modifying sequences, shifting regions, recoloring, pattern-based transformations, or decompressing/recompressing RLE. "
        "Do not describe reasoning or guesses — only executable steps that apply to all examples.\n\n"
        "CRITICAL: Enclose ONLY the algorithmic instructions within <reasoning>...</reasoning> tags. Output nothing outside these tags.\n\n"
        f"TASK EXAMPLES (RLE-based):\n{train_str}"
    )

    stage1_prompt_coords = (
        "You are an expert in geometric and spatial reasoning for grid-based transformations in ARC puzzles. "
        "Each example provides INPUT and OUTPUT data as coordinate mappings — where each color ID (0-9) is associated with a list of (x, y) coordinates representing its cells on the grid.\n\n"
        "Your goal is to analyze all examples to infer the deterministic transformation that maps the INPUT coordinates to the OUTPUT coordinates consistently. "
        "Describe the process as a precise, step-by-step algorithm involving operations like translation, rotation, mirroring, scaling, recoloring, object extraction, or connectivity analysis. "
        "Avoid speculative explanations or reasoning — focus only on unambiguous algorithmic steps that work for all examples.\n\n"
        "CRITICAL: Enclose ONLY the algorithmic instructions within <reasoning>...</reasoning> tags. Output nothing outside these tags.\n\n"
        f"TASK EXAMPLES (coordinate-based):\n{train_str}"
    )

    prompts = {
        "json": stage1_prompt_json,
        "images": stage1_prompt_images,
        "rle": stage1_prompt_rle,
        "coords": stage1_prompt_coords
    }

    # Return the requested prompt, default to JSON if unknown type
    return prompts.get(representation_type, stage1_prompt_json)