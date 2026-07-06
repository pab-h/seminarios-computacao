import kagglehub

path = kagglehub.dataset_download(
    "sadmansakibmahi/plant-disease-expert",
    output_dir = "data/kaggle"
)

print("Path to dataset files:", path)