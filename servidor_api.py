from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import io
import base64

app = Flask(__name__)
CORS(app)  # Permitir peticiones desde el celular

# Cargar modelo al iniciar el servidor
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Servidor usando: {device}")

# Recrear arquitectura del modelo
model = models.efficientnet_b0(weights='IMAGENET1K_V1')
num_features = model.classifier[1].in_features
model.classifier = nn.Sequential(
    nn.Dropout(p=0.3, inplace=True),
    nn.Linear(num_features, 256),
    nn.ReLU(),
    nn.Dropout(p=0.3),
    nn.Linear(256, 4)
)

# Cargar pesos entrenados
model.load_state_dict(torch.load('mejor_modelo_heridas.pth', map_location=device))
model = model.to(device)
model.eval()

# Nombres de clases
class_names = ['ABRASIONES', 'HEMATOMA', 'LACERACIONES', 'QUEMADURAS']

# Transformación de imagen
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

@app.route('/health', methods=['GET'])
def health():
    """Endpoint para verificar que el servidor está funcionando"""
    return jsonify({
        'status': 'ok',
        'message': 'Servidor de clasificación de heridas funcionando',
        'device': str(device)
    })

@app.route('/clasificar', methods=['POST'])
def clasificar():
    """Endpoint para clasificar heridas - Compatible con app Flutter"""
    try:
        # Obtener datos JSON
        data = request.get_json()
        
        if not data or 'image' not in data:
            return jsonify({'error': 'No se recibió ninguna imagen'}), 400
        
        # Decodificar imagen base64
        image_base64 = data['image']
        image_data = base64.b64decode(image_base64)
        image = Image.open(io.BytesIO(image_data)).convert('RGB')
        
        # Transformar imagen
        image_tensor = transform(image).unsqueeze(0).to(device)
        
        # Predecir
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilidades = torch.nn.functional.softmax(outputs, dim=1)
            confianza, prediccion = torch.max(probabilidades, 1)
        
        # Preparar respuesta
        clase_predicha = class_names[prediccion.item()]
        confianza_pct = confianza.item() * 100
        
        # Todas las probabilidades
        todas_probs = {
            class_names[i]: float(probabilidades[0][i].item() * 100)
            for i in range(len(class_names))
        }
        
        return jsonify({
            'prediccion': clase_predicha,
            'confianza': round(confianza_pct, 2),
            'probabilidades': todas_probs
        })
    
    except Exception as e:
        print(f"Error en clasificación: {str(e)}")
        return jsonify({
            'error': str(e)
        }), 500

@app.route('/predict', methods=['POST'])
def predict():
    """Endpoint alternativo (mantener compatibilidad)"""
    try:
        # Recibir imagen (puede ser base64 o archivo)
        if 'image' in request.files:
            # Imagen como archivo
            file = request.files['image']
            image = Image.open(file.stream).convert('RGB')
        elif 'image_base64' in request.json:
            # Imagen como base64
            image_base64 = request.json['image_base64']
            image_data = base64.b64decode(image_base64)
            image = Image.open(io.BytesIO(image_data)).convert('RGB')
        elif 'image' in request.json:
            # Formato de la app Flutter
            image_base64 = request.json['image']
            image_data = base64.b64decode(image_base64)
            image = Image.open(io.BytesIO(image_data)).convert('RGB')
        else:
            return jsonify({'error': 'No se recibió ninguna imagen'}), 400
        
        # Transformar imagen
        image_tensor = transform(image).unsqueeze(0).to(device)
        
        # Predecir
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilidades = torch.nn.functional.softmax(outputs, dim=1)
            confianza, prediccion = torch.max(probabilidades, 1)
        
        # Preparar respuesta
        clase_predicha = class_names[prediccion.item()]
        confianza_pct = confianza.item() * 100
        
        # Todas las probabilidades
        todas_probs = {
            class_names[i]: float(probabilidades[0][i].item() * 100)
            for i in range(len(class_names))
        }
        
        return jsonify({
            'success': True,
            'prediccion': clase_predicha,
            'confianza': round(confianza_pct, 2),
            'probabilidades': todas_probs
        })
    
    except Exception as e:
        print(f"Error en clasificación: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🚀 SERVIDOR API DE CLASIFICACIÓN DE HERIDAS")
    print("="*60)
    print(f"✓ Modelo cargado en {device}")
    print(f"✓ Clases: {class_names}")
    print("\nEndpoints disponibles:")
    print("  GET  /health    - Verificar estado del servidor")
    print("  POST /clasificar - Clasificar herida (app Flutter)")
    print("  POST /predict   - Clasificar herida (alternativo)")
    print("\nIniciando servidor en http://0.0.0.0:5000")
    print("="*60 + "\n")
    
    import os
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
