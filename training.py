import os
import shutil
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import kagglehub

# ============================================================================
# 1. DESCARGAR Y PREPARAR EL DATASET
# ============================================================================

print("Descargando dataset de PlantVillage...")
path = kagglehub.dataset_download("mohitsingh1804/plantvillage")
print("Path to dataset files:", path)

# Crear directorio para datos filtrados
base_dir = 'potato_dataset'
train_dir = os.path.join(base_dir, 'train')
test_dir = os.path.join(base_dir, 'test')

os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# ============================================================================
# Apuntar directamente a la carpeta de entrenamiento
# ============================================================================
dataset_path = os.path.join(path, 'PlantVillage', 'train')
print(f"Dataset forzado a: {dataset_path}")

# Verificar que existe el directorio
if not os.path.exists(dataset_path):
    # Intentar buscar alternativas
    print("Buscando rutas alternativas...")
    for root, dirs, files in os.walk(path):
        if 'train' in dirs:
            dataset_path = os.path.join(root, 'train')
            print(f"Dataset encontrado en: {dataset_path}")
            break

# Filtrar solo las clases de papa
potato_classes = []
for item in os.listdir(dataset_path):
    if 'Potato' in item and os.path.isdir(os.path.join(dataset_path, item)):
        potato_classes.append(item)

print(f"\nClases de papa encontradas: {potato_classes}")

# Copiar y dividir datos (80% entrenamiento, 20% prueba)
from random import shuffle, seed
seed(42)

print("\n" + "="*60)
print("DISTRIBUCIÓN DE DATOS")
print("="*60)

total_images_per_class = {}

for class_name in potato_classes:
    source_dir = os.path.join(dataset_path, class_name)

    # Crear directorios para train y test
    train_class_dir = os.path.join(train_dir, class_name)
    test_class_dir = os.path.join(test_dir, class_name)
    os.makedirs(train_class_dir, exist_ok=True)
    os.makedirs(test_class_dir, exist_ok=True)

    # Obtener todas las imágenes
    images = [f for f in os.listdir(source_dir) if f.endswith(('.jpg', '.JPG', '.png', '.PNG'))]
    shuffle(images)

    total_images_per_class[class_name] = len(images)

    # Dividir 80-20
    split_idx = int(len(images) * 0.8)
    train_images = images[:split_idx]
    test_images = images[split_idx:]

    # Copiar imágenes
    for img in train_images:
        shutil.copy2(os.path.join(source_dir, img), os.path.join(train_class_dir, img))

    for img in test_images:
        shutil.copy2(os.path.join(source_dir, img), os.path.join(test_class_dir, img))

    print(f"{class_name}:")
    print(f"  Total: {len(images)} imágenes")
    print(f"  Train: {len(train_images)} imágenes")
    print(f"  Test:  {len(test_images)} imágenes")

print("\n" + "="*60)
print(f"TOTAL DE IMÁGENES: {sum(total_images_per_class.values())}")
print("="*60)

# ============================================================================
# 2. PREPARACIÓN DE DATOS CON DATA AUGMENTATION
# ============================================================================

IMG_SIZE = 224
BATCH_SIZE = 32

# Data augmentation para entrenamiento (MÁS AGRESIVO para compensar desbalanceo)
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.3,
    height_shift_range=0.3,
    shear_range=0.3,
    zoom_range=0.3,
    horizontal_flip=True,
    vertical_flip=True,
    brightness_range=[0.7, 1.3],
    fill_mode='nearest'
)

# Solo rescaling para validación
test_datagen = ImageDataGenerator(rescale=1./255)

# Generadores
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

print(f"\nClases detectadas: {train_generator.class_indices}")
print(f"Total imágenes de entrenamiento: {train_generator.samples}")
print(f"Total imágenes de prueba: {test_generator.samples}")

num_classes = len(train_generator.class_indices)

# Verificar si hay desbalanceo crítico
class_counts = {}
for class_name in train_generator.class_indices:
    class_dir = os.path.join(train_dir, class_name)
    class_counts[class_name] = len(os.listdir(class_dir))

print("\n" + "="*60)
print("VERIFICACIÓN DE BALANCE DE CLASES")
print("="*60)
for class_name, count in class_counts.items():
    percentage = (count / train_generator.samples) * 100
    print(f"{class_name}: {count} imágenes ({percentage:.1f}%)")

min_class = min(class_counts.values())
max_class = max(class_counts.values())
imbalance_ratio = max_class / min_class

if imbalance_ratio > 3:
    print(f"\n⚠️  ADVERTENCIA: Desbalanceo detectado (ratio {imbalance_ratio:.1f}:1)")
    print("Se aplicará class_weight para compensar")

    # Calcular pesos de clase
    total_samples = sum(class_counts.values())
    class_weights = {}
    for idx, (class_name, count) in enumerate(class_counts.items()):
        class_weights[idx] = total_samples / (num_classes * count)

    print("\nPesos de clase calculados:")
    for idx, weight in class_weights.items():
        class_name = list(train_generator.class_indices.keys())[idx]
        print(f"  {class_name}: {weight:.2f}")
else:
    class_weights = None
    print("\n✓ Balance de clases aceptable")

# ============================================================================
# 3. CONSTRUCCIÓN DEL MODELO CNN CON TRANSFER LEARNING
# ============================================================================

# Usar MobileNetV2 como base (eficiente y preciso)
base_model = keras.applications.MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights='imagenet'
)

# Congelar el modelo base
base_model.trainable = False

# Construir el modelo completo con más regularización
model = keras.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.BatchNormalization(),
    layers.Dropout(0.4),
    layers.Dense(256, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01)),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    layers.Dense(128, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01)),
    layers.Dropout(0.2),
    layers.Dense(num_classes, activation='softmax')
])

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()]
)

print("\n" + "="*60)
print("ARQUITECTURA DEL MODELO")
print("="*60)
model.summary()

# ============================================================================
# 4. ENTRENAMIENTO DEL MODELO
# ============================================================================

# Callbacks mejorados
early_stopping = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=8,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=4,
    min_lr=1e-7,
    verbose=1
)

checkpoint = keras.callbacks.ModelCheckpoint(
    'best_potato_model.keras',
    monitor='val_accuracy',
    save_best_only=True,
    mode='max',
    verbose=1
)

print("\n" + "="*60)
print("INICIANDO ENTRENAMIENTO - FASE 1")
print("="*60)

history = model.fit(
    train_generator,
    epochs=30,
    validation_data=test_generator,
    callbacks=[early_stopping, reduce_lr, checkpoint],
    class_weight=class_weights,
    verbose=1
)

# ============================================================================
# 5. FINE-TUNING
# ============================================================================

print("\n" + "="*60)
print("FINE-TUNING DEL MODELO - FASE 2")
print("="*60)

# Descongelar las últimas capas del modelo base
base_model.trainable = True
for layer in base_model.layers[:-30]:
    layer.trainable = False

# Recompilar con learning rate más bajo
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=5e-6),
    loss='categorical_crossentropy',
    metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()]
)

history_fine = model.fit(
    train_generator,
    epochs=15,
    validation_data=test_generator,
    callbacks=[early_stopping, reduce_lr, checkpoint],
    class_weight=class_weights,
    verbose=1
)

# ============================================================================
# 6. EVALUACIÓN COMPLETA DEL MODELO
# ============================================================================

print("\n" + "="*60)
print("EVALUACIÓN FINAL DEL MODELO")
print("="*60)

# Cargar el mejor modelo
model = keras.models.load_model('best_potato_model.keras')

# Evaluar
test_loss, test_accuracy, test_precision, test_recall = model.evaluate(test_generator, verbose=1)
f1_score = 2 * (test_precision * test_recall) / (test_precision + test_recall)

print("\n" + "="*60)
print("MÉTRICAS FINALES")
print("="*60)
print(f"Accuracy:  {test_accuracy*100:.2f}%")
print(f"Precision: {test_precision*100:.2f}%")
print(f"Recall:    {test_recall*100:.2f}%")
print(f"F1-Score:  {f1_score*100:.2f}%")
print(f"Loss:      {test_loss:.4f}")

# Predicciones
print("\nGenerando predicciones...")
predictions = model.predict(test_generator, verbose=1)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = test_generator.classes
class_labels = list(train_generator.class_indices.keys())

# Reporte de clasificación detallado
print("\n" + "="*60)
print("REPORTE DE CLASIFICACIÓN POR CLASE")
print("="*60)
print(classification_report(true_classes, predicted_classes, target_names=class_labels, digits=4))

# Matriz de confusión
cm = confusion_matrix(true_classes, predicted_classes)
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_labels, yticklabels=class_labels,
            cbar_kws={'label': 'Número de predicciones'})
plt.title('Matriz de Confusión - Clasificación de Enfermedades en Papa', fontsize=14, fontweight='bold')
plt.ylabel('Clase Real', fontsize=12)
plt.xlabel('Clase Predicha', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
print("\n✓ Matriz de confusión guardada como 'confusion_matrix.png'")

# Calcular accuracy por clase
print("\n" + "="*60)
print("ACCURACY POR CLASE")
print("="*60)
for i, class_name in enumerate(class_labels):
    class_mask = true_classes == i
    class_acc = np.mean(predicted_classes[class_mask] == true_classes[class_mask])
    print(f"{class_name}: {class_acc*100:.2f}%")

# ============================================================================
# 7. VISUALIZACIÓN DE RESULTADOS
# ============================================================================

# Combinar historiales
all_accuracy = history.history['accuracy'] + history_fine.history['accuracy']
all_val_accuracy = history.history['val_accuracy'] + history_fine.history['val_accuracy']
all_loss = history.history['loss'] + history_fine.history['loss']
all_val_loss = history.history['val_loss'] + history_fine.history['val_loss']

# Gráficas de entrenamiento
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

# Accuracy
ax1.plot(all_accuracy, label='Train', linewidth=2)
ax1.plot(all_val_accuracy, label='Validation', linewidth=2)
ax1.axvline(x=len(history.history['accuracy']), color='r', linestyle='--', label='Fine-tuning start')
ax1.set_title('Accuracy del Modelo', fontsize=14, fontweight='bold')
ax1.set_xlabel('Época', fontsize=12)
ax1.set_ylabel('Accuracy', fontsize=12)
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)

# Loss
ax2.plot(all_loss, label='Train', linewidth=2)
ax2.plot(all_val_loss, label='Validation', linewidth=2)
ax2.axvline(x=len(history.history['loss']), color='r', linestyle='--', label='Fine-tuning start')
ax2.set_title('Loss del Modelo', fontsize=14, fontweight='bold')
ax2.set_xlabel('Época', fontsize=12)
ax2.set_ylabel('Loss', fontsize=12)
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)

# Precision
if 'precision' in history.history:
    all_precision = history.history['precision'] + history_fine.history['precision_1'] # Corrected: used 'precision_1' for history_fine
    all_val_precision = history.history['val_precision'] + history_fine.history['val_precision_1'] # Corrected: used 'val_precision_1' for history_fine
    ax3.plot(all_precision, label='Train', linewidth=2)
    ax3.plot(all_val_precision, label='Validation', linewidth=2)
    ax3.axvline(x=len(history.history['precision']), color='r', linestyle='--', label='Fine-tuning start')
    ax3.set_title('Precision del Modelo', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Época', fontsize=12)
    ax3.set_ylabel('Precision', fontsize=12)
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)

# Recall
if 'recall' in history.history:
    all_recall = history.history['recall'] + history_fine.history['recall_1'] # Corrected: used 'recall_1' for history_fine
    all_val_recall = history.history['val_recall'] + history_fine.history['val_recall_1'] # Corrected: used 'val_recall_1' for history_fine
    ax4.plot(all_recall, label='Train', linewidth=2)
    ax4.plot(all_val_recall, label='Validation', linewidth=2)
    ax4.axvline(x=len(history.history['recall']), color='r', linestyle='--', label='Fine-tuning start')
    ax4.set_title('Recall del Modelo', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Época', fontsize=12)
    ax4.set_ylabel('Recall', fontsize=12)
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
print("✓ Gráficas de entrenamiento guardadas como 'training_history.png'")

# ============================================================================
# 8. GUARDAR MODELO FINAL Y METADATOS
# ============================================================================

model.save('potato_disease_model_final.keras')
print("\n✓ Modelo final guardado como 'potato_disease_model_final.keras'")

# Guardar las clases y metadatos
import json
metadata = {
    'class_indices': train_generator.class_indices,
    'total_train_samples': train_generator.samples,
    'total_test_samples': test_generator.samples,
    'img_size': IMG_SIZE,
    'num_classes': num_classes,
    'test_accuracy': float(test_accuracy),
    'test_precision': float(test_precision),
    'test_recall': float(test_recall),
    'f1_score': float(f1_score),
    'class_distribution': class_counts
}

with open('model_metadata.json', 'w') as f:
    json.dump(metadata, f, indent=4)
print("✓ Metadatos guardados como 'model_metadata.json'")

# ============================================================================
# 9. FUNCIÓN DE PREDICCIÓN MEJORADA
# ============================================================================

def predict_potato_disease(image_path, model_path='potato_disease_model_final.keras'):
    """
    Predice la enfermedad de una imagen de hoja de papa
    """
    # Cargar modelo
    model = keras.models.load_model(model_path)

    # Cargar y preprocesar imagen
    img = keras.preprocessing.image.load_img(image_path, target_size=(IMG_SIZE, IMG_SIZE))
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    # Predecir
    predictions = model.predict(img_array)
    predicted_class_idx = np.argmax(predictions[0])
    confidence = predictions[0][predicted_class_idx] * 100

    # Cargar metadatos
    with open('model_metadata.json', 'r') as f:
        metadata = json.load(f)

    class_names = {v: k for k, v in metadata['class_indices'].items()}
    predicted_class = class_names[predicted_class_idx]

    # Todas las probabilidades
    all_probs = {class_names[i]: float(predictions[0][i] * 100) for i in range(len(predictions[0]))}

    return {
        'predicted_class': predicted_class,
        'confidence': confidence,
        'all_probabilities': all_probs
    }

print("\n" + "="*60)
print("¡ENTRENAMIENTO COMPLETADO EXITOSAMENTE!")
print("="*60)
print("\nArchivos generados:")
print("  ✓ best_potato_model.keras")
print("  ✓ potato_disease_model_final.keras")
print("  ✓ model_metadata.json")
print("  ✓ confusion_matrix.png")
print("  ✓ training_history.png")
print("\n" + "="*60)