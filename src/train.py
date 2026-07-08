import os
import random
import numpy as np
import tensorflow as tf
from keras.callbacks import EarlyStopping

# Importando a classe do seu modelo MobRes (definida em model.py)
from model import MobRes

import os

import tensorflow as tf

def setup_gpu():
    """Configura o TensorFlow para alocar memória dinamicamente."""
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("Crescimento de memória da GPU ativado com sucesso.")
        except RuntimeError as e:
            print(f"Erro ao configurar GPU: {e}")

def set_seed(seed=42):
    """Garante a reprodutibilidade dos experimentos."""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def load_datasets(processed_dir, batch_size=16):
    """
    Carrega os datasets de Treino, Validação e Teste que foram salvos 
    pelo script de pré-processamento offline.
    
    Aplica também o redimensionamento padrão (128x128), a codificação 
    one-hot das classes (label_mode='categorical') e a normalização dos pixels em [0, 1].
    """
    train_path = os.path.join(processed_dir, "train")
    val_path = os.path.join(processed_dir, "val")
    test_path = os.path.join(processed_dir, "test")

    # Verifica se as pastas existem
    for path in [train_path, val_path, test_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Diretório não encontrado: {path}. Certifique-se de rodar o pré-processamento primeiro.")

    # Carrega o conjunto de treino
    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_path,
        image_size=(128, 128),
        batch_size=batch_size,
        label_mode="categorical",
        shuffle=True
    )
    
    # Carrega o conjunto de validação
    val_ds = tf.keras.utils.image_dataset_from_directory(
        val_path,
        image_size=(128, 128),
        batch_size=batch_size,
        label_mode="categorical",
        shuffle=False
    )

    # Carrega o conjunto de teste
    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_path,
        image_size=(128, 128),
        batch_size=batch_size,
        label_mode="categorical",
        shuffle=False
    )
    
    # Captura a lista das classes mapeadas e sua quantidade
    class_names = train_ds.class_names
    num_classes = len(class_names)
    
    # Mapeamento para normalizar os valores dos pixels de [0, 255] para [0, 1]
    # Conforme as restrições e normalizações descritas no artigo
    normalization_layer = lambda x, y: (x / 255.0, y)
    
    train_ds = train_ds.map(normalization_layer, num_parallel_calls=tf.data.AUTOTUNE)
    val_ds = val_ds.map(normalization_layer, num_parallel_calls=tf.data.AUTOTUNE)
    test_ds = test_ds.map(normalization_layer, num_parallel_calls=tf.data.AUTOTUNE)
    
    # Prefetch para otimizar o uso da GPU/CPU durante o treinamento
    train_ds = train_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    test_ds = test_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    
    return train_ds, val_ds, test_ds, num_classes, class_names


def run_training(train_dataset, val_dataset, num_classes):
    """Instancia, compila e executa o ciclo de treinamento do MobRes."""
    
    # Inicializa o modelo no estilo subclassificado
    model = MobRes(input_shape=(128, 128, 3), num_classes=num_classes)
    
    # Otimizador Adam com taxa de aprendizado inicial padrão de 0.001
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    
    # Compilação usando a perda de entropia cruzada categórica
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Callback de Early Stopping monitorando 'val_loss' por até 5 épocas
    # Restaura os melhores pesos obtidos ao fim do treinamento
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        verbose=1
    )
    
    print(f"\nIniciando o treinamento do modelo MobRes para {num_classes} classes...")
    
    # Treinamento configurado para até 40 épocas conforme o artigo
    history = model.fit(
        train_dataset,
        epochs=40,
        validation_data=val_dataset,
        callbacks=[early_stopping]
    )
    
    return model, history


def evaluate_model(model, test_dataset):
    """Avalia a performance final do modelo treinado utilizando o conjunto de teste isolado."""
    print("\n--- Avaliando o modelo no conjunto de teste final ---")
    loss, accuracy = model.evaluate(test_dataset)
    print(f"Perda no Teste (Test Loss): {loss:.4f}")
    print(f"Acurácia no Teste (Test Accuracy): {accuracy * 100:.2f}%")


# ==========================================
# BLOCO PRINCIPAL DE EXECUÇÃO
# ==========================================
if __name__ == "__main__":
    # Define a semente global para garantir a reprodutibilidade
    setup_gpu()
    set_seed(42)
    
    # Caminho onde o script anterior salvou a estrutura balanceada
    PROCESSED_DATA_DIR = "data/plant_disease_expert_processed" 
    
    # 1. Carrega os conjuntos pré-processados e normalizados do disco
    train_ds, val_ds, test_ds, num_classes, class_names = load_datasets(
        processed_dir=PROCESSED_DATA_DIR, 
        batch_size=16  # Batch size fixado em 16 conforme o artigo
    )
    
    print(f"Dataset carregado com sucesso!")
    print(f"Quantidade de classes encontradas: {num_classes}")
    
    # 2. Executa a rotina de treinamento por 40 épocas com EarlyStopping
    model, history = run_training(
        train_dataset=train_ds,
        val_dataset=val_ds,
        num_classes=num_classes
    )
    
    # 3. Salva os pesos finais do modelo treinado
    weights_filename = "dist/mobres.h5"
    model.save_weights(weights_filename)
    print(f"Pesos de treinamento salvos localmente em: '{weights_filename}'")
    
    # 4. Avaliação opcional do modelo no conjunto de teste limpo
    evaluate_model(model, test_ds)