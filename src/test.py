import os
import keras

import numpy             as np
import matplotlib.pyplot as plt
import tensorflow        as tf

from tf_keras_vis.utils.scores      import CategoricalScore
from tf_keras_vis.gradcam           import Gradcam
from tf_keras_vis.gradcam_plus_plus import GradcamPlusPlus

from lime import lime_image

from skimage.segmentation import mark_boundaries
from sklearn.metrics      import classification_report
from sklearn.metrics      import confusion_matrix
from sklearn.metrics      import ConfusionMatrixDisplay

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
    
    y_true         = np.concatenate([y for x, y in test_ds], axis=0)
    y_true_indices = np.argmax(y_true, axis=1)
    
    y_pred         = model.predict(test_ds)
    y_pred_indices = np.argmax(y_pred, axis=1)
    
    report = classification_report(y_true_indices, y_pred_indices, target_names=class_names)
    
    print("\n--- Relatório de Classificação ---")
    print(report)
    
    with open(os.path.join(OUTPUT_EVAL_DIR, "classification_report.txt"), "w") as f:
        f.write(report)
        
    cm = confusion_matrix(y_true_indices, y_pred_indices)

    fig, ax = plt.subplots(figsize=(12, 10))
    
    disp = ConfusionMatrixDisplay(
        confusion_matrix = cm, 
        display_labels   = class_names
    )
    
    disp.plot(
        cmap            = plt.cm.Blues, 
        xticks_rotation = 'vertical',
        ax              = ax,
        values_format   = 'd' 
    )
    
    ax.tick_params(axis='both', which='major', labelsize=9)
    ax.set_title("Matriz de Confusão - PlantVillage", fontsize=14, pad=20)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_EVAL_DIR, "confusion_matrix.png"), bbox_inches='tight', dpi=150)
    plt.close()
    
    print(f"Métricas salvas em: {OUTPUT_EVAL_DIR}")

def run_gradcam_variants_for_all_classes(model, test_ds, class_names):
    print("\n-> Iniciando Grad-CAM e Grad-CAM++ para todas as classes...")
    
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

    print(f"[XAI] Utilizando a camada '{target_layer.name}' para os mapas de calor.")

    gradcam = Gradcam(model)
    gradcam_pp = GradcamPlusPlus(model)
    
    processed_classes = set()
    
    for images, labels in test_ds:

        if len(processed_classes) == len(class_names):
            break
            
        images_np = images.numpy()
        labels_np = labels.numpy()
        
        preds = model.predict(images, verbose=0)
        
        for i in range(len(images_np)):

            true_label_idx = np.argmax(labels_np[i])
            pred_label_idx = np.argmax(preds[i])
            
            if true_label_idx == pred_label_idx and true_label_idx not in processed_classes:
                
                class_name = class_names[true_label_idx]
                print(f"Generating heatmaps for class: {class_name}")
                
                img = images_np[i]
                img_input = np.expand_dims(img, axis=0)
                score = CategoricalScore([true_label_idx])
                
                cam = gradcam(score, img_input, penultimate_layer=target_layer.name)
                cam_pp = gradcam_pp(score, img_input, penultimate_layer=target_layer.name)
                
                img_display = img.astype('uint8')
                
                fig, axes = plt.subplots(1, 3, figsize=(15, 6))
                axes[0].imshow(img_display)
                axes[0].set_title(f"Original: {class_name}")
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
                
                xai_output_dir = os.path.join(OUTPUT_EVAL_DIR, "gradcam_classes")
                os.makedirs(xai_output_dir, exist_ok=True)
                
                plt.savefig(os.path.join(xai_output_dir, f"gradcam_{class_name}.png"))
                plt.close()
                
                processed_classes.add(true_label_idx)

def run_lime_for_all_classes(model, test_ds, class_names):

    print("\n-> Iniciando LIME para todas as classes...")
    
    explainer = lime_image.LimeImageExplainer()
    
    def predict_fn(images):
        return model.predict(images, verbose=0)
    
    processed_classes = set()
    
    for images, labels in test_ds:

        if len(processed_classes) == len(class_names):
            break
            
        images_np = images.numpy()
        labels_np = labels.numpy()
        
        preds = model.predict(images, verbose = 0)
        
        for i in range(len(images_np)):

            true_label_idx = np.argmax(labels_np[i])
            pred_label_idx = np.argmax(preds[i])
            
            if true_label_idx == pred_label_idx and true_label_idx not in processed_classes:
                
                class_name = class_names[true_label_idx]
                print(f"Generating LIME explanation for class: {class_name}")
                
                img = images_np[i]
                
                explanation = explainer.explain_instance(
                    img.astype('double'), 
                    predict_fn, 
                    top_labels  = 1, 
                    hide_color  = 0, 
                    num_samples = 1000 
                )
                
                top_label = explanation.top_labels[0]
                
                temp, mask = explanation.get_image_and_mask(
                    top_label, 
                    positive_only = True, 
                    num_features  =  5, 
                    hide_rest     =  False
                )
                
                img_bound = mark_boundaries(temp.astype('uint8'), mask)
                
                plt.figure(figsize = (6, 6))
                plt.imshow(img_bound)
                plt.title(f"LIME Explicação para: {class_name}")
                plt.axis('off')
                plt.tight_layout()
                
                lime_output_dir = os.path.join(OUTPUT_EVAL_DIR, "lime_classes")
                os.makedirs(lime_output_dir, exist_ok = True)
                
                plt.savefig(os.path.join(lime_output_dir, f"lime_{class_name}.png"))
                plt.close()
                
                processed_classes.add(true_label_idx)

if __name__ == "__main__":

    os.makedirs(OUTPUT_EVAL_DIR, exist_ok=True)
    
    test_dataset, classes = load_test_dataset(DATASET_DIR, IMG_SIZE, BATCH_SIZE)
    
    if not os.path.exists(MODEL_SAVE_PATH):
        raise FileNotFoundError(f"Modelo não encontrado em: {MODEL_SAVE_PATH}. Treine o modelo primeiro.")
        
    print(f"-> Carregando o modelo de: {MODEL_SAVE_PATH}")
    mobres_model = keras.models.load_model(MODEL_SAVE_PATH)
    
    evaluate_metrics(mobres_model, test_dataset, classes)
    
    run_gradcam_variants_for_all_classes(mobres_model, test_dataset, classes)
    run_lime_for_all_classes(mobres_model, test_dataset, classes)
    
    print(f"\n[SUCESSO] Avaliação concluída. Todos os gráficos e relatórios foram salvos em '{OUTPUT_EVAL_DIR}'")