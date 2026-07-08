import os
import random
import keras
import tensorflow as tf

from keras import layers
from tqdm import tqdm 

DATASET_DIR = 'data/PlantVillage/raw/color'
OUTPUT_DIR  = 'data/PlantVillageSubset'

IMG_HEIGHT_RESIZE = 128
IMG_WIDTH_RESIZE  = 128

TARGET_SIZE  = 3000  
SUBSET_SIZE  = 10

def get_top_classes(top): 

    all_classes = {}

    for class_name in os.listdir(DATASET_DIR):

        class_path = os.path.join(DATASET_DIR, class_name)

        if os.path.isdir(class_path):
            files                   = [os.path.join(class_path, f) for f in os.listdir(class_path) if os.path.isfile(os.path.join(class_path, f))]
            all_classes[class_name] = files

    top_classes = sorted(all_classes.items(), key=lambda x: len(x[1]), reverse=True)[: top]

    return top_classes

def load_image_to_numpy(file_path):

    img = tf.io.read_file(file_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [IMG_HEIGHT_RESIZE, IMG_WIDTH_RESIZE])

    return img.numpy()

def apply_augmentation_and_noise(image_tensor):

    augment_pipeline = keras.Sequential([
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomRotation(factor=0.2, fill_mode="reflect"),
    ])

    img_normalized = image_tensor / 255.0
    img_augmented  = augment_pipeline(tf.expand_dims(img_normalized, 0), training=True)[0]
    
    noise          = tf.random.normal(shape=tf.shape(img_augmented), mean=0.0, stddev=0.05, dtype=tf.float32)
    img_with_noise = tf.clip_by_value(img_augmented + noise, 0.0, 1.0)
    
    return (img_with_noise.numpy() * 255).astype('uint8')

def create_subset(top_classes):

    for class_name, files in top_classes:

        num_samples = len(files)
        
        class_output_dir = os.path.join(OUTPUT_DIR, class_name)
        os.makedirs(class_output_dir, exist_ok=True)
        
        print(f"\nProcessando classe: {class_name}")
        
        if num_samples >= TARGET_SIZE:

            sampled_files = random.sample(files, TARGET_SIZE)
            
            for idx, file_path in enumerate(tqdm(sampled_files, desc = "Subamostragem")):
                img = load_image_to_numpy(file_path)
                output_file_path = os.path.join(class_output_dir, f"img_{idx}.jpg")
                keras.utils.save_img(output_file_path, img)
                
        else:

            loaded_tensors = []

            for idx, file_path in enumerate(tqdm(files, desc="Copiando Originais")):

                img = load_image_to_numpy(file_path)
                loaded_tensors.append(img)
                
                output_file_path = os.path.join(class_output_dir, f"img_{idx}.jpg")
                keras.utils.save_img(output_file_path, img)
                
            shortfall = TARGET_SIZE - num_samples

            for idx in tqdm(range(shortfall), desc="Gerando Augmentations"):

                base_img_tensor = random.choice(loaded_tensors)
                augmented_img   = apply_augmentation_and_noise(base_img_tensor)
                
                output_file_path = os.path.join(class_output_dir, f"aug_{idx}.jpg")

                keras.utils.save_img(output_file_path, augmented_img)


if __name__ == "__main__":

    os.makedirs(OUTPUT_DIR, exist_ok = True)

    top_classes = get_top_classes(top = SUBSET_SIZE)

    print("--- Classes Selecionadas e Contagem Original ---")
    for class_name, files in top_classes:
        print(f"- {class_name}: {len(files)} amostras")
    print("-" * 50)

    print("\nIniciando o processamento e salvamento em disco...")

    create_subset(top_classes)

    print("\n--- PROCESSO CONCLUÍDO COM SUCESSO ---")
    print(f"O dataset balanceado foi salvo em: {OUTPUT_DIR}")
