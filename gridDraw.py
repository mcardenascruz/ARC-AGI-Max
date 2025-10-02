import os
import matplotlib.pyplot as plt

def plot_grid(label: str, grid: list, filename: str = "grid.png") -> None:
    os.makedirs("mappings", exist_ok=True)

    plt.figure(figsize=(6, 6))
    plt.title(label)
    plt.imshow(grid, cmap="viridis", interpolation="nearest")
    plt.colorbar(label="Value")

    save_path = os.path.join("mappings", filename)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()

    print(f"Grid saved to {save_path}")
