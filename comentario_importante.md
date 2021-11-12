Hola, quizás esto les puede servir como referencia para la tarea:

Acabo de subir a Material Docente las anotaciones del dataset QuickDraw-Animals, por si alguien estaba complicado con generar los archivos "train.txt" y "test.txt".

En la etapa de entrenamiento de las redes para clasificación, obtuve un accuracy cercano al 75% en validación con ambos modelos (sin usar pesos pre-entrenados). El archivo de configuración que utilicé es el siguiente:

[SKETCH]
NUM_EPOCHS = 20
NUM_CLASSES = 12
BATCH_SIZE = 128
VALIDATION_STEPS = 19
LEARNING_RATE = 0.01
DECAY_STEPS = 10000
SNAPSHOT_DIR = /content/convnet2/snapshots
DATA_DIR = /content/convnet2/data
CHANNELS = 3
IMAGE_TYPE = SKETCH
IMAGE_WIDTH = 224
IMAGE_HEIGHT = 224
USE_MULTITHREADS = False
#CKPFILE = /content/convnet2/snapshots/alexnet_020.h5

La última línea se debe descomentar al momento de la búsqueda por similitud, indicando el checkpoint de cada modelo.

Saludos,
Cristóbal.
