import os
import keras

import tensorflow        as tf
import matplotlib.pyplot as plt

from mobres import create_model

DATASET_DIR = 'data/PlantVillageSubset'
IMG_SIZE    = 128  
NUM_CLASSES = 10  

MODEL_DIST_PATH = "dist/mobresv1"
MODEL_SAVE_PATH = f"{MODEL_DIST_PATH}/mobresv1.keras"
EPOCHS          = 100
BATCH_SIZE      = 16
LEARNING_RATE   = 1e-4

def load_datasets(dataset_dir, img_size, batch_size):
    
    print("-> Carregando datasets do disco...")
    
    train_path = os.path.join(dataset_dir, 'train')
    val_path   = os.path.join(dataset_dir, 'val')
    
    train_ds = keras.utils.image_dataset_from_directory(
        train_path,
        seed             = 42,
        image_size       = (img_size, img_size),
        batch_size       = batch_size,
        label_mode       = "categorical"
    )

    val_ds = keras.utils.image_dataset_from_directory(
        val_path,
        seed             = 42,
        image_size       = (img_size, img_size),
        batch_size       = batch_size,
        label_mode       = "categorical"
    )
    
    train_ds = train_ds.prefetch(buffer_size = tf.data.AUTOTUNE)
    val_ds   = val_ds.prefetch(buffer_size = tf.data.AUTOTUNE)
    
    return train_ds, val_ds

def get_callbacks(model_save_path):

    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    
    callbacks = [

        keras.callbacks.EarlyStopping(
            monitor              = "val_loss", 
            patience             = 3, 
            restore_best_weights = True
        ),

        keras.callbacks.ModelCheckpoint(
            filepath       = model_save_path, 
            monitor        = "val_accuracy", 
            save_best_only = True
        )

    ]

    return callbacks

def plot_training_results(history):
    
    os.makedirs(MODEL_DIST_PATH, exist_ok=True)

    acc      = history.history["accuracy"]
    val_acc  = history.history["val_accuracy"]
    loss     = history.history["loss"]
    val_loss = history.history["val_loss"]
    epochs   = range(1, len(acc) + 1)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(epochs, acc, label="Treino")
    ax.plot(epochs, val_acc, label="Validação")
    ax.set_title("Acurácia de Treino vs Validação")
    ax.set_xlabel("Épocas")
    ax.set_ylabel("Acurácia")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(f"{MODEL_DIST_PATH}/accuracy.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(epochs, loss, label="Treino")
    ax.plot(epochs, val_loss, label="Validação")
    ax.set_title("Perda (Loss) de Treino vs Validação")
    ax.set_xlabel("Épocas")
    ax.set_ylabel("Perda")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(f"{MODEL_DIST_PATH}/loss.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

if __name__ == "__main__":

    train_dataset, val_dataset = load_datasets(DATASET_DIR, IMG_SIZE, BATCH_SIZE)
    
    mobres_model = create_model(
        num_classes = NUM_CLASSES
    )
    
    callbacks_list = get_callbacks(MODEL_SAVE_PATH)
    
    print(f"Iniciando o treinamento por {EPOCHS} épocas...")

    history = mobres_model.fit(
        train_dataset,
        validation_data = val_dataset,
        epochs          = EPOCHS,
        callbacks       = callbacks_list
    )
    
    print(f"Treinamento Concluído! O melhor modelo foi salvo em: {MODEL_SAVE_PATH}")
    
    plot_training_results(history)