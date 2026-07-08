from datasets import load_dataset

dataset = load_dataset("mohanty/PlantVillage", "default")

dataset.save_to_disk("data")
