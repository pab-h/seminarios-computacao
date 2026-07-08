import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

import keras
import tensorflow as tf
from tf_keras_vis.utils.scores import CategoricalScore
from tf_keras_vis.gradcam import Gradcam
from tf_keras_vis.gradcam_plus_plus import GradcamPlusPlus
from lime import lime_image
from skimage.segmentation import mark_boundaries

DATASET_DIR     = 'data/PlantVillageSubset'
IMG_SIZE        = 128
BATCH_SIZE      = 16
MODEL_DIST_PATH = "dist/mobresv1"
MODEL_SAVE_PATH = f"{MODEL_DIST_PATH}/mobresv1.keras"
OUTPUT_EVAL_DIR = f"{MODEL_DIST_PATH}/evaluation"

def load_test_dataset(dataset_dir, img_size, batch_size):
    print("-> Carregando dataset de teste...")
    test_path = os.path.join(dataset_dir, 'test')
    
    test_ds = keras.utils.image_dataset_from_directory(
        test_path,
        shuffle          = False,  
        image_size       = (img_size, img_size),
        batch_size       = batch_size,
        label_mode       = "categorical"
    )
    class_names = test_ds.class_names
    test_ds = test_ds.prefetch(buffer_size=tf.data.AUTOTUNE)

    return test_ds, class_names

def evaluate_metrics(model, test_ds, class_names):

    print("-> Calculando métricas de avaliação...")
    
    y_true = np.concatenate([y for x, y in test_ds], axis=0)
    y_true_indices = np.argmax(y_true, axis=1)
    
    y_pred = model.predict(test_ds)
    y_pred_indices = np.argmax(y_pred, axis=1)
    
    report = classification_report(y_true_indices, y_pred_indices, target_names=class_names)
    print("\n--- Relatório de Classificação ---")
    print(report)
    
    with open(os.path.join(OUTPUT_EVAL_DIR, "classification_report.txt"), "w") as f:
        f.write(report)
        
    cm = confusion_matrix(y_true_indices, y_pred_indices)
    plt.figure(figsize=(10, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap=plt.cm.Blues, xticks_rotation='vertical')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_EVAL_DIR, "confusion_matrix.png"))
    plt.close()
    print(f"Métricas salvas em: {OUTPUT_EVAL_DIR}")

def get_sample_image(test_ds):

    for images, labels in test_ds.take(1):
        img = images[0].numpy()
        label_idx = np.argmax(labels[0].numpy())
        return img, label_idx

def run_gradcam_variants(model, img, label_idx, class_names):

    print("-> Executando Grad-CAM e Grad-CAM++...")
    
    img_input = np.expand_dims(img, axis=0)
    score = CategoricalScore([label_idx])
    
    target_layer = None
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            target_layer = layer
            break
            
    if target_layer is None:
        for layer in reversed(model.layers):
            if len(layer.output_shape) == 4:
                target_layer = layer
                break

    print(f" [XAI] Utilizando a camada '{target_layer.name}' para gerar os mapas de calor.")

    gradcam    = Gradcam(model)
    gradcam_pp = GradcamPlusPlus(model)
    
    cam    = gradcam(score, img_input, penultimate_layer=target_layer.name)
    cam_pp = gradcam_pp(score, img_input, penultimate_layer=target_layer.name)
    
    img_display = img.astype('uint8')
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(img_display)
    axes[0].set_title(f"Original: {class_names[label_idx]}")
    axes[0].axis('off')
    
    axes[1].imshow(img_display)
    axes[1].imshow(cam[0], cmap='jet', alpha=0.5)
    axes[1].set_title("Grad-CAM")
    axes[1].axis('off')
    
    axes[2].imshow(img_display)
    axes[2].imshow(cam_pp[0], cmap='jet', alpha=0.5)
    axes[2].set_title("Grad-CAM++")
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_EVAL_DIR, "gradcam_comparison.png"))
    plt.close()

def run_lime(model, img, class_names):

    print("-> Executando LIME...")
    
    explainer = lime_image.LimeImageExplainer()
    
    def predict_fn(images):
        return model.predict(images, verbose=0)
    
    explanation = explainer.explain_instance(
        img.astype('double'), 
        predict_fn, 
        top_labels=1, 
        hide_color=0, 
        num_samples=1000
    )
    
    top_label = explanation.top_labels[0]
    
    temp, mask = explanation.get_image_and_mask(
        top_label, 
        positive_only=True, 
        num_features=5, 
        hide_rest=False
    )
    
    img_bound = mark_boundaries(temp.astype('uint8'), mask)
    
    plt.figure(figsize=(6, 6))
    plt.imshow(img_bound)
    plt.title(f"LIME Explicação para: {class_names[top_label]}")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_EVAL_DIR, "lime_explanation.png"))
    plt.close()

if __name__ == "__main__":

    os.makedirs(OUTPUT_EVAL_DIR, exist_ok=True)
    
    test_dataset, classes = load_test_dataset(DATASET_DIR, IMG_SIZE, BATCH_SIZE)
    
    if not os.path.exists(MODEL_SAVE_PATH):
        raise FileNotFoundError(f"Modelo não encontrado em: {MODEL_SAVE_PATH}. Treine o modelo primeiro.")
        
    print(f"-> Carregando o modelo de: {MODEL_SAVE_PATH}")
    mobres_model = keras.models.load_model(MODEL_SAVE_PATH)
    
    evaluate_metrics(mobres_model, test_dataset, classes)
    
    sample_img, true_label_idx = get_sample_image(test_dataset)
    
    run_gradcam_variants(mobres_model, sample_img, true_label_idx, classes)
    run_lime(mobres_model, sample_img, classes)
    
    print(f"\n[SUCESSO] Avaliação concluída. Todos os gráficos e relatórios foram salvos em '{OUTPUT_EVAL_DIR}'")