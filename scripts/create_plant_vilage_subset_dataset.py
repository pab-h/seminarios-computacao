import os
import random
import keras
import tensorflow as tf

from keras                   import layers
from tqdm                    import tqdm 
from sklearn.model_selection import train_test_split 

DATASET_DIR = 'data/PlantVillage/raw/color'
OUTPUT_DIR  = 'data/PlantVillageSubset'

IMG_HEIGHT_RESIZE = 128
IMG_WIDTH_RESIZE  = 128

TARGET_SIZE  = 3000  
SUBSET_SIZE  = 10

TRAIN_SPLIT = 0.70

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

def save_images_to_splits(class_name, all_images_numpy):

    train_imgs, temp_imgs = train_test_split(
        all_images_numpy, 
        train_size   = TRAIN_SPLIT, 
        random_state = 42
    )
    
    val_imgs, test_imgs = train_test_split(
        temp_imgs, 
        test_size    = 0.5, 
        random_state = 42
    )

    splits = {
        'train': train_imgs,
        'val':   val_imgs,
        'test':  test_imgs
    }

    for split_name, images in splits.items():

        split_dir = os.path.join(OUTPUT_DIR, split_name, class_name)
        os.makedirs(split_dir, exist_ok=True)
        
        for idx, img in enumerate(images):
            
            output_file_path = os.path.join(split_dir, f"img_{idx}.jpg")
            keras.utils.save_img(output_file_path, img)

def create_subset(top_classes):

    for class_name, files in top_classes:

        num_samples = len(files)
        print(f"\nProcessando classe: {class_name}")
        
        final_class_images = []
        
        if num_samples >= TARGET_SIZE:
            sampled_files = random.sample(files, TARGET_SIZE)
            
            for file_path in tqdm(sampled_files, desc="Carregando Subamostragem"):
                img = load_image_to_numpy(file_path)
                final_class_images.append(img)
                
        else:

            for file_path in tqdm(files, desc="Carregando Originais"):
                img = load_image_to_numpy(file_path)
                final_class_images.append(img)
                
            shortfall = TARGET_SIZE - num_samples
            
            for _ in tqdm(range(shortfall), desc="Gerando Augmentations"):
                base_img_tensor = random.choice(final_class_images[:num_samples])
                augmented_img   = apply_augmentation_and_noise(base_img_tensor)
                final_class_images.append(augmented_img)

        print(f"Dividindo e salvando {len(final_class_images)} imagens...")
        save_images_to_splits(class_name, final_class_images)


if __name__ == "__main__":

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    top_classes = get_top_classes(top=SUBSET_SIZE)

    print("--- Classes Selecionadas e Contagem Original ---")
    for class_name, files in top_classes:
        print(f"- {class_name}: {len(files)} amostras")
    print("-" * 50)

    print("\nIniciando o processamento, divisão (70/15/15) e salvamento...")
    create_subset(top_classes)

    print("\n--- PROCESSO CONCLUÍDO COM SUCESSO ---")
    print(f"O dataset foi dividido e salvo em: {OUTPUT_DIR}")