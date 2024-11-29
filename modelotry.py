from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os

# Inicializa Flask
app = Flask(__name__)

# Ruta al modelo H5
model_path = os.path.join(os.path.dirname(__file__), "modelo_billetesh5.h5")

# Carga el modelo H5
try:
    model = tf.keras.models.load_model(model_path)
    print("Modelo cargado correctamente.")
except Exception as e:
    print(f"Error al cargar el modelo: {e}")
    raise e

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Verifica que se envió una imagen
        if 'image' not in request.files:
            return jsonify({"error": "No se envió ninguna imagen"}), 400

        # Procesa la imagen recibida
        image_file = request.files['image']
        image = Image.open(io.BytesIO(image_file.read())).convert('RGB')
        image = image.resize((224, 224))  # Ajusta según el tamaño de entrada del modelo
        input_data = np.expand_dims(np.array(image, dtype=np.float32) / 255.0, axis=0)

        # Realiza la predicción
        predictions = model.predict(input_data)
        
        # Obtiene la clase con mayor probabilidad
        labels = ["Billete_10", "Billete_20", "Billete_50", "Billete_100"]
        predicted_index = np.argmax(predictions[0])
        confidence = predictions[0][predicted_index]

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
    # Establece el puerto predeterminado en 5000 o selecciona uno aleatorio.
    port = int(os.environ.get('PORT', 5000))  # Usa una variable de entorno o el 5000 por defecto.
    app.run(debug=True, host='0.0.0.0', port=port)
