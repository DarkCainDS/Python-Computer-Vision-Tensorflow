import cv2
import tensorflow as tf
from object_detection.utils import visualization_utils as viz_utils


# Carga del modelo
model_path = 'C:/Users/Lenovo/Documents/Proyectos_Jupyter_Notebooks/Functions/Dataset/Perros_Gatos_personas/fine_tuned_model/saved_model'
model = tf.saved_model.load(model_path)
detect_fn = model.signatures['serving_default']

# Category_index
category_index = {1:{'name': 'Perro',
           'id': 1},
          2:{'name': 'Gato',
           'id': 2},
         3: {'name': 'Persona',
           'id': 3}}


# Configuración de la cámara web
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
        
    # Convertir la imagen a un tensor y ejecutar la detección de objetos
    input_tensor = tf.convert_to_tensor(frame)
    input_tensor = input_tensor[tf.newaxis, ...]
    detections = detect_fn(input_tensor)
    
    # Extraer los resultados de la detección
    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0,:num_detections].numpy()
                  for key, value in detections.items()}
    detections['num_detections'] = num_detections
    detections['detection_classes'] = detections['detection_classes'].astype(int)
    
    # Visualización de los resultados
    viz_utils.visualize_boxes_and_labels_on_image_array(
        frame,
        detections['detection_boxes'],
        detections['detection_classes'],
        detections['detection_scores'],
        category_index,
        use_normalized_coordinates=True,
        max_boxes_to_draw=10,
        min_score_thresh=.5,
        agnostic_mode=False)
    
    try:
        cv2.imshow('object detection', cv2.resize(frame, (640, 480)))
    except cv2.error:
        break
    
    if cv2.waitKey(1) == ord('q'):
        break
        
cap.release()
cv2.destroyAllWindows()
