from datasets import load_dataset

print("Downloading")
dataset = load_dataset("tsbpp/fall2025_deeplearning")

print(f"Train images: {len(dataset['train'])}")

# Saving
dataset.save_to_disk("/scratch/ap9283/deep_learning/data")
print("Finished")


