from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import io
import base64
import os
import mysql.connector
from datetime import datetime

app = Flask(__name__)
CORS(app)

# ====================================
# CONFIGURACIÓN DE BASE DE DATOS MySQL
# ====================================
def get_db_connection():
    """Crear conexión a la base de datos MySQL"""
    try:
        connection = mysql.connector.connect(
            host=os.environ.get('MYSQLHOST'),
            port=int(os.environ.get('MYSQLPORT', 3306)),
            database=os.environ.get('MYSQLDATABASE'),
            user=os.environ.get('MYSQLUSER'),
            password=os.environ.get('MYSQLPASSWORD')
        )
        return connection
    except Exception as e:
        print(f"Error conectando a MySQL: {e}")
        return None

def init_database():
    """Inicializar tablas de la base de datos"""
    conn = get_db_connection()
    if not conn:
        print("⚠️ No se pudo conectar a MySQL - funcionando sin base de datos")
        return
    
    try:
        cursor = conn.cursor()
        
        # Tabla para puntos de encuentro predefinidos
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS puntos_encuentro (
                id INT AUTO_INCREMENT PRIMARY KEY,
                nombre VARCHAR(100) NOT NULL,
                descripcion VARCHAR(255),
                latitud DECIMAL(10, 8) NOT NULL,
                longitud DECIMAL(11, 8) NOT NULL,
                activo BOOLEAN DEFAULT TRUE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Tabla para llamadas de emergencia
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS llamadas_emergencia (
                id INT AUTO_INCREMENT PRIMARY KEY,
                latitud_usuario DECIMAL(10, 8) NOT NULL,
                longitud_usuario DECIMAL(11, 8) NOT NULL,
                punto_encuentro_id INT,
                tipo_herida VARCHAR(50),
                confianza DECIMAL(5, 2),
                timestamp DATETIME NOT NULL,
                estado VARCHAR(50) DEFAULT 'pendiente',
                tiempo_estimado VARCHAR(50),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (punto_encuentro_id) REFERENCES puntos_encuentro(id)
            )
        """)
        
        # Tabla para clasificaciones de heridas
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS clasificaciones (
                id INT AUTO_INCREMENT PRIMARY KEY,
                tipo_herida VARCHAR(50) NOT NULL,
                confianza DECIMAL(5, 2) NOT NULL,
                probabilidades TEXT,
                timestamp DATETIME NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        conn.commit()
        
        # Insertar puntos de encuentro por defecto si no existen
        cursor.execute("SELECT COUNT(*) FROM puntos_encuentro")
        count = cursor.fetchone()[0]
        
        if count == 0:
            print("⚠️ No hay puntos de encuentro. Insertando ejemplos...")
            cursor.execute("""
                INSERT INTO puntos_encuentro (nombre, descripcion, latitud, longitud) VALUES
                ('Entrada Ingeniería', 'Puerta principal del edificio de Ingeniería', 4.682894, -74.041940),
                ('Cafetería Central', 'Frente a la cafetería principal del campus', 4.683500, -74.042200),
                ('Biblioteca', 'Entrada principal de la biblioteca', 4.684200, -74.041800),
                ('Canchas Deportivas', 'Zona de canchas deportivas', 4.739828, -74.035760),
                ('Auditorio Principal', 'Entrada del auditorio principal', 4.683800, -74.042500)
            """)
            conn.commit()
            print("✓ Puntos de encuentro de ejemplo insertados")
        
        print("✓ Base de datos MySQL inicializada correctamente")
    except Exception as e:
        print(f"Error inicializando base de datos: {e}")
    finally:
        cursor.close()
        conn.close()

# ====================================
# CARGAR MODELO
# ====================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Servidor usando: {device}")

model = models.efficientnet_b0(weights='IMAGENET1K_V1')
num_features = model.classifier[1].in_features
model.classifier = nn.Sequential(
    nn.Dropout(p=0.3, inplace=True),
    nn.Linear(num_features, 256),
    nn.ReLU(),
    nn.Dropout(p=0.3),
    nn.Linear(256, 4)
)

model.load_state_dict(torch.load('mejor_modelo_heridas.pth', map_location=device))
model = model.to(device)
model.eval()

class_names = ['ABRASIONES', 'HEMATOMA', 'LACERACIONES', 'QUEMADURAS']

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Inicializar base de datos al arrancar
init_database()

# ====================================
# FUNCIONES AUXILIARES
# ====================================

def calcular_distancia(lat1, lon1, lat2, lon2):
    """Calcular distancia entre dos coordenadas GPS usando fórmula de Haversine"""
    from math import radians, sin, cos, sqrt, atan2
    
    R = 6371000  # Radio de la Tierra en metros
    
    lat1_rad = radians(lat1)
    lat2_rad = radians(lat2)
    delta_lat = radians(lat2 - lat1)
    delta_lon = radians(lon2 - lon1)
    
    a = sin(delta_lat/2)**2 + cos(lat1_rad) * cos(lat2_rad) * sin(delta_lon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    
    distancia = R * c
    return distancia

def encontrar_punto_mas_cercano(latitud, longitud):
    """Encontrar el punto de encuentro más cercano a las coordenadas dadas"""
    try:
        conn = get_db_connection()
        if not conn:
            return None
        
        cursor = conn.cursor(dictionary=True)
        cursor.execute("""
            SELECT id, nombre, descripcion, latitud, longitud 
            FROM puntos_encuentro 
            WHERE activo = TRUE
        """)
        puntos = cursor.fetchall()
        cursor.close()
        conn.close()
        
        if not puntos:
            return None
        
        # Calcular distancia a cada punto
        punto_mas_cercano = None
        distancia_minima = float('inf')
        
        for punto in puntos:
            distancia = calcular_distancia(
                latitud, longitud,
                float(punto['latitud']), float(punto['longitud'])
            )
            
            if distancia < distancia_minima:
                distancia_minima = distancia
                punto_mas_cercano = punto
                punto_mas_cercano['distancia_metros'] = round(distancia, 2)
        
        return punto_mas_cercano
        
    except Exception as e:
        print(f"Error buscando punto más cercano: {e}")
        return None

# ====================================
# ENDPOINTS
# ====================================

@app.route('/health', methods=['GET'])
def health():
    """Endpoint para verificar que el servidor está funcionando"""
    # Verificar conexión a base de datos
    conn = get_db_connection()
    db_status = "conectada" if conn else "desconectada"
    if conn:
        conn.close()
    
    return jsonify({
        'status': 'ok',
        'message': 'Servidor de clasificación de heridas funcionando',
        'device': str(device),
        'database': db_status
    })

@app.route('/clasificar', methods=['POST'])
def clasificar():
    """Endpoint para clasificar heridas - Compatible con app Flutter"""
    try:
        import gc
        
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
        
        # Guardar en base de datos
        try:
            conn = get_db_connection()
            if conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO clasificaciones 
                    (tipo_herida, confianza, probabilidades, timestamp)
                    VALUES (%s, %s, %s, %s)
                """, (
                    clase_predicha,
                    confianza_pct,
                    str(todas_probs),
                    datetime.now()
                ))
                conn.commit()
                cursor.close()
                conn.close()
                print(f"✓ Clasificación guardada en BD: {clase_predicha}")
        except Exception as db_error:
            print(f"⚠️ Error guardando en BD: {db_error}")
        
        # Liberar memoria
        del image_tensor, outputs, probabilidades
        gc.collect()
        
        return jsonify({
            'prediccion': clase_predicha,
            'confianza': round(confianza_pct, 2),
            'probabilidades': todas_probs
        })
    
    except Exception as e:
        print(f"Error en clasificación: {str(e)}")
        import gc
        gc.collect()
        return jsonify({
            'error': str(e)
        }), 500

@app.route('/llamar_carro', methods=['POST'])
def llamar_carro():
    """Endpoint para llamar al carro autónomo con ubicación GPS"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No se recibieron datos'}), 400
        
        # Obtener datos de la llamada
        latitud = data.get('latitud')
        longitud = data.get('longitud')
        tipo_herida = data.get('tipo_herida', 'Desconocido')
        confianza = data.get('confianza', 0)
        timestamp_str = data.get('timestamp')
        
        if latitud is None or longitud is None:
            return jsonify({'error': 'Falta latitud o longitud'}), 400
        
        # Convertir timestamp
        try:
            timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
        except:
            timestamp = datetime.now()
        
        # ⭐ BUSCAR PUNTO DE ENCUENTRO MÁS CERCANO
        punto_cercano = encontrar_punto_mas_cercano(latitud, longitud)
        
        if not punto_cercano:
            return jsonify({
                'success': False,
                'error': 'No se encontraron puntos de encuentro disponibles'
            }), 500
        
        # Log de la solicitud
        print(f"\n{'='*60}")
        print(f"🚨 LLAMADA DE EMERGENCIA")
        print(f"{'='*60}")
        print(f"📍 Ubicación usuario: {latitud}, {longitud}")
        print(f"🎯 Punto asignado: {punto_cercano['nombre']} (ID: {punto_cercano['id']})")
        print(f"📏 Distancia: {punto_cercano['distancia_metros']} metros")
        print(f"🩹 Tipo de herida: {tipo_herida} ({confianza}% confianza)")
        print(f"🕐 Timestamp: {timestamp}")
        print(f"{'='*60}\n")
        
        # Guardar en base de datos
        llamada_id = None
        try:
            conn = get_db_connection()
            if conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO llamadas_emergencia 
                    (latitud_usuario, longitud_usuario, punto_encuentro_id, tipo_herida, 
                     confianza, timestamp, estado, tiempo_estimado)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    latitud,
                    longitud,
                    punto_cercano['id'],
                    tipo_herida,
                    confianza,
                    timestamp,
                    'en_ruta',
                    '5-7 minutos'
                ))
                conn.commit()
                llamada_id = cursor.lastrowid
                cursor.close()
                conn.close()
                print(f"✓ Llamada guardada en BD con ID: {llamada_id}")
        except Exception as db_error:
            print(f"⚠️ Error guardando llamada en BD: {db_error}")
        
        # TODO: Aquí enviarías el comando al carro físico
        # enviar_comando_carro(punto_cercano['id'], tipo_herida, urgencia)
        
        # Respuesta para la app
        response = {
            'success': True,
            'mensaje': f'Carro en camino a {punto_cercano["nombre"]}',
            'llamada_id': llamada_id,
            'punto_encuentro': {
                'id': punto_cercano['id'],
                'nombre': punto_cercano['nombre'],
                'descripcion': punto_cercano['descripcion'],
                'latitud': float(punto_cercano['latitud']),
                'longitud': float(punto_cercano['longitud']),
                'distancia_metros': punto_cercano['distancia_metros']
            },
            'ubicacion_usuario': {
                'latitud': latitud,
                'longitud': longitud
            },
            'tiempo_estimado': '5-7 minutos',
            'estado_carro': 'En ruta'
        }
        
        return jsonify(response)
    
    except Exception as e:
        print(f"Error en llamada de carro: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/estadisticas', methods=['GET'])
def estadisticas():
    """Endpoint para obtener estadísticas de uso"""
    try:
        conn = get_db_connection()
        if not conn:
            return jsonify({'error': 'Base de datos no disponible'}), 500
        
        cursor = conn.cursor(dictionary=True)
        
        # Total de llamadas
        cursor.execute("SELECT COUNT(*) as total FROM llamadas_emergencia")
        total_llamadas = cursor.fetchone()['total']
        
        # Llamadas por tipo de herida
        cursor.execute("""
            SELECT tipo_herida, COUNT(*) as cantidad 
            FROM llamadas_emergencia 
            GROUP BY tipo_herida
        """)
        por_tipo = cursor.fetchall()
        
        # Llamadas recientes (últimas 10)
        cursor.execute("""
            SELECT id, latitud, longitud, tipo_herida, confianza, timestamp, estado
            FROM llamadas_emergencia 
            ORDER BY created_at DESC 
            LIMIT 10
        """)
        recientes = cursor.fetchall()
        
        cursor.close()
        conn.close()
        
        return jsonify({
            'total_llamadas': total_llamadas,
            'por_tipo_herida': por_tipo,
            'llamadas_recientes': recientes
        })
    
    except Exception as e:
        print(f"Error obteniendo estadísticas: {str(e)}")
        return jsonify({
            'error': str(e)
        }), 500

@app.route('/predict', methods=['POST'])
def predict():
    """Endpoint alternativo (mantener compatibilidad)"""
    try:
        # Recibir imagen (puede ser base64 o archivo)
        if 'image' in request.files:
            file = request.files['image']
            image = Image.open(file.stream).convert('RGB')
        elif 'image_base64' in request.json:
            image_base64 = request.json['image_base64']
            image_data = base64.b64decode(image_base64)
            image = Image.open(io.BytesIO(image_data)).convert('RGB')
        elif 'image' in request.json:
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

@app.route('/rpi/siguiente_llamada', methods=['GET'])
def siguiente_llamada_rpi():
    """
    Endpoint para que la Raspberry Pi consulte la siguiente llamada pendiente.
    Retorna el ID del punto de encuentro al que debe ir.
    """
    try:
        conn = get_db_connection()
        if not conn:
            return jsonify({'error': 'Base de datos no disponible'}), 500
        
        cursor = conn.cursor(dictionary=True)
        
        # Buscar la llamada más antigua que esté pendiente o en ruta
        cursor.execute("""
            SELECT 
                l.id as llamada_id,
                l.punto_encuentro_id,
                l.tipo_herida,
                l.confianza,
                l.estado,
                p.id as punto_id,
                p.nombre as punto_nombre,
                p.descripcion as punto_descripcion,
                p.latitud as punto_latitud,
                p.longitud as punto_longitud
            FROM llamadas_emergencia l
            INNER JOIN puntos_encuentro p ON l.punto_encuentro_id = p.id
            WHERE l.estado IN ('pendiente', 'en_ruta')
            ORDER BY l.created_at ASC
            LIMIT 1
        """)
        
        llamada = cursor.fetchone()
        
        if not llamada:
            cursor.close()
            conn.close()
            return jsonify({
                'hay_llamada': False,
                'mensaje': 'No hay llamadas pendientes'
            })
        
        # Actualizar estado a "en_proceso"
        cursor.execute("""
            UPDATE llamadas_emergencia 
            SET estado = 'en_proceso'
            WHERE id = %s
        """, (llamada['llamada_id'],))
        conn.commit()
        
        cursor.close()
        conn.close()
        
        # Respuesta para la RPI
        return jsonify({
            'hay_llamada': True,
            'llamada_id': llamada['llamada_id'],
            'punto_destino': {
                'id': llamada['punto_id'],
                'nombre': llamada['punto_nombre'],
                'descripcion': llamada['punto_descripcion'],
                'latitud': float(llamada['punto_latitud']),
                'longitud': float(llamada['punto_longitud'])
            },
            'tipo_herida': llamada['tipo_herida'],
            'confianza': float(llamada['confianza']) if llamada['confianza'] else 0
        })
    
    except Exception as e:
        print(f"Error obteniendo siguiente llamada: {e}")
        return jsonify({
            'error': str(e)
        }), 500

@app.route('/rpi/completar_llamada/<int:llamada_id>', methods=['POST'])
def completar_llamada_rpi(llamada_id):
    """
    Endpoint para que la RPI marque una llamada como completada
    """
    try:
        conn = get_db_connection()
        if not conn:
            return jsonify({'error': 'Base de datos no disponible'}), 500
        
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE llamadas_emergencia 
            SET estado = 'completado'
            WHERE id = %s
        """, (llamada_id,))
        conn.commit()
        cursor.close()
        conn.close()
        
        print(f"✓ Llamada #{llamada_id} marcada como completada")
        
        return jsonify({
            'success': True,
            'mensaje': f'Llamada #{llamada_id} completada'
        })
    
    except Exception as e:
        print(f"Error completando llamada: {e}")
        return jsonify({
            'error': str(e)
        }), 500

@app.route('/puntos_encuentro', methods=['GET'])
def listar_puntos():
    """Endpoint para listar todos los puntos de encuentro"""
    try:
        conn = get_db_connection()
        if not conn:
            return jsonify({'error': 'Base de datos no disponible'}), 500
        
        cursor = conn.cursor(dictionary=True)
        cursor.execute("""
            SELECT id, nombre, descripcion, latitud, longitud, activo
            FROM puntos_encuentro
            ORDER BY nombre
        """)
        puntos = cursor.fetchall()
        cursor.close()
        conn.close()
        
        return jsonify({
            'puntos': puntos,
            'total': len(puntos)
        })
    
    except Exception as e:
        print(f"Error listando puntos: {e}")
        return jsonify({
            'error': str(e)
        }), 500

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🚀 SERVIDOR API DE CLASIFICACIÓN DE HERIDAS")
    print("="*60)
    print(f"✓ Modelo cargado en {device}")
    print(f"✓ Clases: {class_names}")
    print("\nEndpoints disponibles:")
    print("  GET  /health                     - Verificar estado del servidor")
    print("  POST /clasificar                 - Clasificar herida (app Flutter)")
    print("  POST /llamar_carro               - Llamar carro de emergencia")
    print("  GET  /estadisticas               - Ver estadísticas de uso")
    print("  GET  /puntos_encuentro           - Listar puntos de encuentro")
    print("  GET  /rpi/siguiente_llamada      - Consultar siguiente llamada (RPI)")
    print("  POST /rpi/completar_llamada/:id  - Marcar llamada como completada (RPI)")
    print("  POST /predict                    - Clasificar herida (alternativo)")
    print("\nIniciando servidor en http://0.0.0.0:5000")
    print("="*60 + "\n")
    
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
