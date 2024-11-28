from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os

# Inicializa Flask
app = Flask(__name__)

# Ruta al modelo TFLite
model_path = os.path.join(os.path.dirname(__file__), "modelo_billetes_final.tflite")

# Carga el modelo TFLite
try:
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()  # Inicializa el modelo
    print("Modelo cargado correctamente.")
except Exception as e:
    print(f"Error al cargar el modelo: {e}")
    raise e

# Obtén detalles del modelo
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Verifica que se envió una imagen
        if 'image' not in request.files:
            return jsonify({"error": "No se envió ninguna imagen"}), 400

        # Procesa la imagen recibida
        image_file = request.files['image']
        image = Image.open(io.BytesIO(image_file.read())).convert('RGB')
        image = image.resize((224, 224))
        input_data = np.expand_dims(np.array(image, dtype=np.float32) / 255.0, axis=0)

        # Realiza la predicción
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])

        # Obtiene la clase con mayor probabilidad
        labels = ["Billete_10", "Billete_20", "Billete_50", "Billete_100"]
        predicted_index = np.argmax(output_data[0])
        confidence = output_data[0][predicted_index]

        # Retorna la predicción como JSON
        return jsonify({
            "prediction": labels[predicted_index],
            "confidence": float(confidence)
        })
    except Exception as e:
        print(f"Error durante la predicción: {e}")
        return jsonify({"error": str(e)}), 500

# Corre la aplicación Flask al final del script
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
