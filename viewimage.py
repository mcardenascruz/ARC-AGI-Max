import base64

b64_data = """
iVBORw0KGgoAAAANSUhEUgAAACQAAAAkCAYAAADhAJiYAAAAUUlEQVR4nO3XQQoAIAgAwYw+7svtCYuB0GHnbixepKiq1ZGZvYGmPfn4C4OIQcQgYhAxiJzp29T13YYMIgYRg4hBxCAS/suAQcQgYhAxiBhELoP1Dcn9PA0WAAAAAElFTkSuQmCC
""".strip()

image_data = base64.b64decode(b64_data)

with open("decoded_grid.png", "wb") as f:
    f.write(image_data)

print("✅ Image saved as decoded_grid.png")