import os
import random
import glob
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array, array_to_img

def set_seed(seed=42):
    """Garante reprodutibilidade nas divisões aleatórias e aumentos."""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

def apply_augmentation(img_arr):
    """
    Aplica as técnicas de Data Augmentation descritas no artigo de forma aleatória:
    Flips horizontais/verticais, rotações de 90° e Ruído Gaussiano.
    """
    # Flips aleatórios
    if random.random() > 0.5:
        img_arr = tf.image.flip_left_right(img_arr).numpy()
    if random.random() > 0.5:
        img_arr = tf.image.flip_up_down(img_arr).numpy()
    
    # Rotação aleatória em múltiplos de 90 graus
    img_arr = tf.image.rot90(img_arr, k=random.randint(0, 3)).numpy()
    
    # Ruído Gaussiano (Artigo menciona média 0 e desvio padrão 0.02)
    if random.random() > 0.5:
        noise = np.random.normal(0, 0.02, img_arr.shape)
        img_arr = np.clip(img_arr + noise, 0.0, 1.0)
        
    return img_arr

def process_and_save_dataset(base_dir, output_dir, target_train=1000, target_val=100):
    """
    Lê o dataset original, filtra as classes excluídas, divide em treino, validação e teste,
    aplica o balanceamento/augmentation e salva na nova estrutura de diretórios.
    """
    input_dataset_path = os.path.join(base_dir, "kaggle", "Image Data base", "Image Data base")
    
    if not os.path.exists(input_dataset_path):
        raise FileNotFoundError(f"Diretório de entrada não encontrado: {input_dataset_path}")
        
    all_classes = sorted(os.listdir(input_dataset_path))
    exclude_classes = ["Nitrogen deficiency in plant", "Waterlogging in plant"]
    target_classes = [c for c in all_classes if c not in exclude_classes and os.path.isdir(os.path.join(input_dataset_path, c))]
    
    print(f"Total de classes a processar: {len(target_classes)}")
    
    for cls_name in target_classes:
        print(f"Processando classe: {cls_name}...")
        cls_dir = os.path.join(input_dataset_path, cls_name)
        
        # Coleta e filtra arquivos válidos de imagem
        images = glob.glob(os.path.join(cls_dir, "*.*"))
        images = [img for img in images if img.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
        
        random.shuffle(images)
        n_total = len(images)
        
        if n_total == 0:
            print(f" [Aviso] Classe {cls_name} não contém imagens válidas.")
            continue
            
        # 1. Separação do Teste: min(65, 10%)
        n_test = min(65, int(0.10 * n_total))
        test_images = images[:n_test]
        rest_images = images[n_test:]
        
        # 2. Definição do Alvo para Treino + Validação (1100 imagens no total)
        total_required = target_train + target_val
        
        # Determina a lista base de imagens que usaremos para gerar o treino/validação
        if len(rest_images) >= total_required:
            # Caso tenha imagens suficientes, fazemos uma subamostragem aleatória
            selected_base = random.sample(rest_images, total_required)
        else:
            # Caso falte, usamos todas as disponíveis e precisaremos complementar com Augmentation
            selected_base = rest_images
            
        # Divisão proporcional de 90% para treino e 10% para validação baseada na seleção
        split_idx = int(len(selected_base) * 0.9)
        train_base = selected_base[:split_idx]
        val_base = selected_base[split_idx:]
        
        # 3. Criar os caminhos de destino compatíveis com o image_dataset_from_directory
        for split in ['train', 'val', 'test']:
            os.makedirs(os.path.join(output_dir, split, cls_name), exist_ok=True)
            
        # --- SALVAR TESTE (Sem alterações ou redimensionamentos físicos obrigatórios no disco além do 128x128) ---
        for idx, img_path in enumerate(test_images):
            img = load_img(img_path, target_size=(128, 128))
            img.save(os.path.join(output_dir, 'test', cls_name, f"test_{idx}.jpg"))
            
        # --- SALVAR VALIDAÇÃO (Até atingir exatamente target_val) ---
        val_idx = 0
        while val_idx < target_val:
            # Se a lista base acabou, reamostramos dela para complementar aplicando augmentation
            if val_idx < len(val_base):
                img_path = val_base[val_idx]
                img = load_img(img_path, target_size=(128, 128))
                img.save(os.path.join(output_dir, 'val', cls_name, f"val_{val_idx}.jpg"))
            else:
                img_path = random.choice(val_base)
                img = load_img(img_path, target_size=(128, 128))
                img_arr = img_to_array(img) / 255.0
                img_arr = apply_augmentation(img_arr)
                img_aug = array_to_img(img_arr * 255.0)
                img_aug.save(os.path.join(output_dir, 'val', cls_name, f"val_aug_{val_idx}.jpg"))
            val_idx += 1
            
        # --- SALVAR TREINO (Até atingir exatamente target_train) ---
        train_idx = 0
        while train_idx < target_train:
            if train_idx < len(train_base):
                # Mantém as originais redimensionadas primeiro
                img_path = train_base[train_idx]
                img = load_img(img_path, target_size=(128, 128))
                img.save(os.path.join(output_dir, 'train', cls_name, f"train_{train_idx}.jpg"))
            else:
                # Complementa aplicando as técnicas de distorção/ruído do artigo
                img_path = random.choice(train_base)
                img = load_img(img_path, target_size=(128, 128))
                img_arr = img_to_array(img) / 255.0
                img_arr = apply_augmentation(img_arr)
                img_aug = array_to_img(img_arr * 255.0)
                img_aug.save(os.path.join(output_dir, 'train', cls_name, f"train_aug_{train_idx}.jpg"))
            train_idx += 1

if __name__ == "__main__":
    set_seed(42)
    
    BASE_INPUT_DIR = "data"
    BASE_OUTPUT_DIR = "data/plant_disease_expert_processed" 
    
    print("Iniciando o pré-processamento offline do dataset...")
    process_and_save_dataset(
        base_dir=BASE_INPUT_DIR,
        output_dir=BASE_OUTPUT_DIR,
        target_train=1000, # Fixado pelo artigo por classe
        target_val=100     # Fixado pelo artigo por classe
    )
    print(f"\nPré-processamento concluído com sucesso! Os dados salvos em '{BASE_OUTPUT_DIR}' estão prontos para uso.")