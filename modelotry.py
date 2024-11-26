from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import io

# Inicializa Flask
app = Flask(__name__)

# Carga el modelo TFLite
interpreter = tf.lite.Interpreter(model_path="BillDetector\modelo_billetes.tflite")


interpreter.allocate_tensors()

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
        labels = ["Billete_10", "Billete_100","Billete_20" ,"Billete_50" ]
        predicted_index = np.argmax(output_data[0])
        confidence = output_data[0][predicted_index]
        
        return jsonify({
            "prediction": labels[predicted_index],
            "confidence": float(confidence)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
