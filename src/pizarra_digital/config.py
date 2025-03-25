"""
Módulo de configuración.

Este módulo contiene todas las configuraciones globales para la aplicación.
"""
import logging
from typing import Dict, Tuple, List, Any

# Configuración del logger
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuración de la cámara
CAMERA_INDEX: int = 0  # Índice de la cámara (0 = cámara predeterminada)
CAMERA_WIDTH: int = 640  # Ancho de la captura de la cámara
CAMERA_HEIGHT: int = 480  # Alto de la captura de la cámara
CAMERA_FPS: int = 30  # FPS objetivo para la cámara
CAMERA_BUFFER_SIZE: int = 1  # Tamaño del buffer de la cámara (menor = más reciente)

# Configuración MediaPipe
MEDIAPIPE_MAX_HANDS: int = 1  # Número máximo de manos a detectar
MEDIAPIPE_DETECTION_CONFIDENCE: float = 0.5  # Umbral de confianza para detección
MEDIAPIPE_TRACKING_CONFIDENCE: float = 0.5  # Umbral de confianza para seguimiento

# Configuración del lienzo
CANVAS_WIDTH: int = 640  # Ancho del lienzo
CANVAS_HEIGHT: int = 480  # Alto del lienzo
CANVAS_BACKGROUND: Tuple[int, int, int] = (0, 0, 0)  # Color de fondo (BGR)

# Colores para dibujo (en formato BGR)
DRAWING_COLORS: Dict[str, Tuple[int, int, int]] = {
    "negro": (0, 0, 0),
    "blanco": (255, 255, 255),
    "rojo": (0, 0, 255),
    "verde": (0, 255, 0),
    "azul": (255, 0, 0),
    "amarillo": (0, 255, 255)
}
DEFAULT_DRAWING_COLOR: Tuple[int, int, int] = DRAWING_COLORS["blanco"]
DRAWING_THICKNESS: int = 5  # Grosor del trazo

# Configuración de la aplicación
APP_NAME: str = "Pizarra Digital"  # Nombre de la ventana
EXIT_KEY: str = "q"  # Tecla para salir de la aplicación

# Configuración de gestos
# Distancia mínima (en píxeles) entre puntos para considerarlos diferentes
MIN_DRAWING_DISTANCE: int = 5
# Distancia máxima entre puntos para interpolar en movimientos rápidos
MAX_INTERPOLATION_DISTANCE: int = 50

# Optimización de rendimiento
OPTIMIZATION_RESIZE_FACTOR: float = 0.75  # Factor de escala para procesamiento
OPTIMIZATION_SKIP_FRAMES: int = 0  # Número de fotogramas a saltar (0 = procesar todos)

# Configuración de métricas
FPS_HISTORY_SIZE: int = 30  # Número de fotogramas para calcular FPS promedio
PERFORMANCE_METRICS_LOGGING: bool = True  # Registrar métricas de rendimiento

# Configuración de optimización para detección de manos
HAND_DETECTION_SKIP_FRAMES: int = 0  # Saltar fotogramas para detección (0 = detectar todos)
PREDICTION_FRAMES_THRESHOLD: int = 3  # Umbral de fotogramas para activar predicción
PREDICTION_TIME_THRESHOLD: float = 0.05  # Umbral de tiempo (segundos) para predicción
PREDICTION_DISTANCE_THRESHOLD: int = 100  # Distancia máxima para usar predicción

# Configuración de optimización para dibujo
DRAWING_BATCH_SIZE: int = 5  # Número de operaciones a acumular antes de dibujar
ADAPTIVE_THICKNESS: bool = True  # Ajustar grosor según velocidad de movimiento
THICKNESS_VELOCITY_THRESHOLD: int = 500  # Umbral de velocidad para ajuste de grosor

# Configuración para depuración
DEBUG_MODE: bool = False  # Modo de depuración
SHOW_LANDMARKS: bool = True  # Mostrar landmarks de la mano
SHOW_FINGERTIPS: bool = True  # Resaltar las puntas de los dedos
SHOW_PERFORMANCE_METRICS: bool = True  # Mostrar métricas en pantalla

# Configuración para la interfaz de usuario
UI_BUTTON_SIZE: int = 40  # Tamaño de los botones
UI_BUTTON_MARGIN: int = 10  # Margen entre botones
UI_BUTTON_RADIUS: int = 20  # Radio de los botones (para esquinas redondeadas)
UI_PANEL_HEIGHT: int = 60  # Altura del panel de la interfaz
UI_PANEL_COLOR: Tuple[int, int, int] = (50, 50, 50)  # Color del panel (BGR)
UI_PANEL_ALPHA: float = 0.7  # Transparencia del panel
UI_TEXT_COLOR: Tuple[int, int, int] = (255, 255, 255)  # Color del texto
UI_TEXT_THICKNESS: int = 1  # Grosor del texto

# Configuración para optimización de multihilo
THREAD_QUEUE_SIZE: int = 3  # Tamaño de la cola para frames capturados
THREAD_SLEEP_TIME: float = 0.001  # Tiempo de espera entre iteraciones (segundos)

logger.debug("Configuración cargada correctamente")
