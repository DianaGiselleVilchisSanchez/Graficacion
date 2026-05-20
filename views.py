import os
import json
from io import BytesIO

# Importaciones de Django
from django.shortcuts import render
from django.http import HttpResponse, JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.conf import settings
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
from PIL import Image, ImageDraw, ImageFilter
from io import BytesIO

# Librerías de IA y manejo de imágenes
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image, ImageDraw, ImageFont # Importamos ImageDraw y ImageFont
import numpy as np

# -----------------------------------------------------
# 🛠️ 1. CONFIGURACIÓN Y CARGA DE MODELOS
# -----------------------------------------------------

MODELOS_DIR = os.path.join(settings.BASE_DIR, 'api_analisis', 'modelos')
RESNET_PATH = os.path.join(MODELOS_DIR, 'resnet_melanoma_classifier.keras')
ALEXNET_PATH = os.path.join(MODELOS_DIR, 'alexnet_melanoma_classifier.keras')

ALEXNET_F1_SCORE = 0.9250
RESNET_F1_SCORE = 0.9310

IMAGE_SIZE_ALEXNET = 227
IMAGE_SIZE_RESNET = 224
THRESHOLD = 0.45            
CLASS_NAMES = ["Benigno", "Maligno"]

resnet_model = None
alexnet_model = None
MODELO_CARGADO = False

# Función para cargar modelos (sin cambios)
def cargar_modelos():
    global resnet_model, alexnet_model, MODELO_CARGADO
    if not MODELO_CARGADO:
        try:
            print("Cargando modelos de Keras... (Solo una vez)")
            resnet_model = load_model(RESNET_PATH, compile=False) 
            alexnet_model = load_model(ALEXNET_PATH, compile=False)
            resnet_model.predict(np.zeros((1, IMAGE_SIZE_RESNET, IMAGE_SIZE_RESNET, 3)))
            alexnet_model.predict(np.zeros((1, IMAGE_SIZE_ALEXNET, IMAGE_SIZE_ALEXNET, 3)))
            MODELO_CARGADO = True
            print("✅ Modelos de ResNet y AlexNet cargados correctamente.")
        except Exception as e:
            print(f"❌ Error crítico al cargar modelos: {e}")
            MODELO_CARGADO = False
            return False
    return True

# Función de preprocesamiento (sin cambios)
def preprocess_image(image_bytes, target_size):
    try:
        img = Image.open(image_bytes).convert('RGB')
        img = img.resize((target_size, target_size), Image.Resampling.LANCZOS)
        img_array = np.array(img, dtype=np.float32)
        img_array /= 255.0
        return np.expand_dims(img_array, axis=0)
    except Exception as e:
        print(f"Error en el preprocesamiento de la imagen: {e}")
        return None

# -----------------------------------------------------
# NUEVA FUNCIÓN: Generar imagen con recuadro de detección (simulado)


# api_analisis/views.py (Reemplaza la función generar_imagen_deteccion_simulada)

# api_analisis/views.py (función modificada)

def generar_imagen_deteccion_simulada(original_image_bytes, output_filename, confianza_maligno):
    """
    Genera una imagen en blanco y negro simulando una máscara de segmentación.
    Esto se basa en la intensidad de color y la ubicación (simulación).
    """
    try:
        img = Image.open(original_image_bytes).convert('RGB')
        width, height = img.size
        
        # 1. Crear una versión en escala de grises
        gray_img = img.convert('L')
        
        # 2. Aplicar un umbral para crear una máscara binaria (Simulación de segmentación)
        # Los valores bajos (oscuros) se vuelven blancos (lesión), los valores altos (claros) se vuelven negros (piel)
        # Usamos un umbral alto para asegurarnos de capturar la mancha oscura
        THRESHOLD_VALUE = 80 # Píxeles con valor 0-80 se vuelven blancos (lesión)

        # 3. Crear la máscara binaria
        mask_data = []
        for pixel_value in gray_img.getdata():
            if pixel_value < THRESHOLD_VALUE:
                # Píxel oscuro (posible lesión) -> Blanco (255)
                mask_data.append(255) 
            else:
                # Píxel claro (piel) -> Negro (0)
                mask_data.append(0)
        
        mask_simulada = Image.new('L', (width, height))
        mask_simulada.putdata(mask_data)

        # 4. Superposición y Limpieza (para que se vea más profesional)
        
        # Opcional: Aplicar un filtro de apertura/cierre o erosión para suavizar bordes y eliminar ruido
        # mask_simulada = mask_simulada.filter(ImageFilter.MaxFilter(3)) 

        # 5. Combinar la máscara con una capa de color rojo semitransparente (para la detección)
        # Creamos una imagen roja (detección)
        detection_overlay = Image.new('RGB', (width, height), (255, 0, 0))
        # Usamos la máscara binaria como alfa
        detection_overlay.putalpha(mask_simulada.point(lambda p: p * 0.5)) # 50% de opacidad para el color

        # 6. Combinar la capa roja con la imagen original
        final_img = Image.composite(detection_overlay, img, detection_overlay)
        
        # 7. ELIMINAMOS TODA LA SECCIÓN QUE DIBUJA EL TEXTO Y SU FONDO ROJO
        # draw = ImageDraw.Draw(final_img)
        # try:
        #     font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
        #     font_size = max(16, int(height * 0.04))
        #     font = ImageFont.truetype(font_path, font_size)
        # except IOError:
        #     font = ImageFont.load_default()
        #     font_size = 16
        #
        # text_label = f"Maligno: {confianza_maligno:.2f}%"
        # text_x = int(0.2 * width) 
        # text_y = int(0.15 * height)
        # 
        # text_bbox = draw.textbbox((0,0), text_label, font=font)
        # text_width = text_bbox[2] - text_bbox[0]
        # text_height = text_bbox[3] - text_bbox[1]
        #
        # draw.rectangle([text_x, text_y, text_x + text_width + 10, text_y + text_height + 5], fill=(255, 0, 0))
        # draw.text((text_x + 5, text_y + 2), text_label, font=font, fill=(255, 255, 255))

        # Guardar la imagen modificada
        buffer = BytesIO()
        final_img.save(buffer, format="JPEG")
        image_detection_name = default_storage.save('analisis_images/detection_' + output_filename, ContentFile(buffer.getvalue()))
        return default_storage.url(image_detection_name)

    except Exception as e:
        print(f"Error al generar imagen de detección simulada: {e}")
        return None


# -----------------------------------------------------
# 3. VISTA ÚNICA (MANEJA GET Y POST)
# -----------------------------------------------------

@csrf_exempt
def analisis_completo(request):
    """
    Maneja la solicitud GET (muestra formulario) y POST (procesa y muestra resultado).
    """
    context = {
        'api_url': '/',
        'mostrar_resultado': False,
        'error': None,
        'imagen_deteccion_url': None, # Nueva variable para la imagen detectada
    }
    
    if request.method == 'GET':
        return render(request, 'api_analisis/index.html', context)

    if request.method == 'POST':
        if not cargar_modelos():
            context['error'] = 'Error interno: Modelos de IA no inicializados.'
            return render(request, 'api_analisis/index.html', context)

        if 'imagen_piel' not in request.FILES:
            context['error'] = 'Falta el archivo "imagen_piel" en la solicitud.'
            return render(request, 'api_analisis/index.html', context)
        
        uploaded_file = request.FILES['imagen_piel']
        file_content = uploaded_file.read()

        try:
            # Guardar imagen original para mostrarla
            image_original_name = default_storage.save('analisis_images/original_' + uploaded_file.name, ContentFile(file_content))
            context['imagen_url'] = default_storage.url(image_original_name)

            # --- PREDICCIÓN ---
            file_obj_resnet = BytesIO(file_content)
            processed_resnet = preprocess_image(file_obj_resnet, IMAGE_SIZE_RESNET)
            resnet_preds = resnet_model.predict(processed_resnet, verbose=0)
            confianza_resnet_maligno = resnet_preds[0][1] 
            
            file_obj_alexnet = BytesIO(file_content)
            processed_alexnet = preprocess_image(file_obj_alexnet, IMAGE_SIZE_ALEXNET)
            alexnet_preds = alexnet_model.predict(processed_alexnet, verbose=0)
            confianza_alexnet_maligno = alexnet_preds[0][1] 
            
            promedio_confianza_maligno_pct = (confianza_resnet_maligno + confianza_alexnet_maligno) / 2 * 100
            
            if promedio_confianza_maligno_pct / 100 >= THRESHOLD:
                resultado_clase = CLASS_NAMES[1] 
                mensaje = "¡ALERTA! Posible lesión de Cáncer de Piel (Maligno)."
                color = "danger"
                
                # 🚨 Generar la imagen con detección si es maligno
                context['imagen_deteccion_url'] = generar_imagen_deteccion_simulada(
                    BytesIO(file_content), 
                    uploaded_file.name, 
                    promedio_confianza_maligno_pct
                )

            else:
                resultado_clase = CLASS_NAMES[0] 
                mensaje = "Lesión de piel probablemente Benigna (No Cáncer)."
                color = "success"
                # Si es benigno, no hay imagen de detección
                context['imagen_deteccion_url'] = None
                
            # --- AGREGAR CONTEXTO DE RESULTADOS AL DICCIONARIO BASE ---
            context.update({
                "mostrar_resultado": True,
                "resultado_final": resultado_clase,
                "mensaje": mensaje,
                "color": color, 
                "promedio_confianza_maligno": round(float(promedio_confianza_maligno_pct), 2), 
                "promedio_confianza_benigno": round(100 - float(promedio_confianza_maligno_pct), 2),

                # Métricas de ResNet
                "resnet_confianza_maligno": round(float(confianza_resnet_maligno * 100), 2),
                "resnet_confianza_benigno": round(100 - float(confianza_resnet_maligno * 100), 2),
                "resnet_f1": RESNET_F1_SCORE,
                
                # Métricas de AlexNet
                "alexnet_confianza_maligno": round(float(confianza_alexnet_maligno * 100), 2),
                "alexnet_confianza_benigno": round(100 - float(confianza_alexnet_maligno * 100), 2),
                "alexnet_f1": ALEXNET_F1_SCORE,
            })
            
            return render(request, 'api_analisis/index.html', context)

        except Exception as e:
            print(f"Error en POST de analisis_completo: {e}")
            context['error'] = f'Error al procesar la solicitud: {e}'
            return render(request, 'api_analisis/index.html', context)
            
@csrf_exempt
def api_predict(request):

    if request.method != 'POST':
        return JsonResponse({
            'error': 'Solo se permiten solicitudes POST'
        }, status=405)

    if not cargar_modelos():
        return JsonResponse({
            'error': 'Error cargando modelos'
        }, status=500)

    if 'imagen_piel' not in request.FILES:
        return JsonResponse({
            'error': 'No se recibió imagen_piel'
        }, status=400)

    try:
        uploaded_file = request.FILES['imagen_piel']
        file_content = uploaded_file.read()

        # RESNET
        file_obj_resnet = BytesIO(file_content)
        processed_resnet = preprocess_image(file_obj_resnet, IMAGE_SIZE_RESNET)

        resnet_preds = resnet_model.predict(processed_resnet, verbose=0)
        confianza_resnet_maligno = resnet_preds[0][1]

        # ALEXNET
        file_obj_alexnet = BytesIO(file_content)
        processed_alexnet = preprocess_image(file_obj_alexnet, IMAGE_SIZE_ALEXNET)

        alexnet_preds = alexnet_model.predict(processed_alexnet, verbose=0)
        confianza_alexnet_maligno = alexnet_preds[0][1]

        # PROMEDIO
        promedio_confianza_maligno_pct = (
            (confianza_resnet_maligno + confianza_alexnet_maligno) / 2
        ) * 100

        if promedio_confianza_maligno_pct / 100 >= THRESHOLD:
            resultado = "Maligno"
            mensaje = "⚠️ Posible melanoma detectado"
        else:
            resultado = "Benigno"
            mensaje = "✅ Lesión probablemente benigna"

        return JsonResponse({
            'resultado': resultado,
            'mensaje': mensaje,
            'confianza_maligno': round(float(promedio_confianza_maligno_pct), 2),
            'confianza_benigno': round(100 - float(promedio_confianza_maligno_pct), 2)
        })

    except Exception as e:
        return JsonResponse({
            'error': str(e)
        }, status=500)
