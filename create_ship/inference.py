import torch
from diffusers import DiffusionPipeline
import os

# Define the list of prompt variations
prompts = [
    "saar6 sailing on a vast ocean, night vision, epic cinematic wide shot, captured from a distant boat",
    "saar6 on the distant horizon, night vision, full view, photo taken from a high altitude, detailed, 8k"
]

# Define the models and output paths
models_and_paths = {
    "D:\ofir\output1": "D:\ofir\create_ship\output1"
}

# Loop through each model and its corresponding output path
for model_path, output_dir in models_and_paths.items():
    print(f"Loading model from: {model_path}")

    # Load the pipeline
    pipe = DiffusionPipeline.from_pretrained(model_path, torch_dtype=torch.float16)
    pipe = pipe.to("cuda")

    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    print(f"Generating images for model in: {model_path}")

    # Loop through each prompt and generate 200 images
    for i, prompt in enumerate(prompts):
        sanitized_prompt = prompt.replace(" ", "_").replace(",", "")[:30]
        print(f"Generating images for prompt: '{prompt}'")
        for j in range(200):
            image = pipe(prompt).images[0]

            # Create a unique filename based on the prompt and a number
            filename = f"{sanitized_prompt}_{j + 1}.png"
            image.save(os.path.join(output_dir, filename))

            if (j + 1) % 10 == 0:
                print(f"  Saved {j + 1} images for this prompt.")

    print(f"Finished generating all images for model: {model_path}\n")

print("All image generation tasks are complete!")