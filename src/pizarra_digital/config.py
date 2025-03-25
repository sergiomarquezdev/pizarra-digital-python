"""
Módulo de configuración centralizada para la aplicación Pizarra Digital.

Este módulo contiene todas las constantes y parámetros configurables
que controlan el comportamiento de la aplicación.
"""
import logging
from typing import Dict, Tuple, Any

# Configuración del logger
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuración de la cámara
CAMERA_WIDTH: int = 640  # Aumentado de 320 a 640
CAMERA_HEIGHT: int = 480  # Aumentado de 240 a 480
CAMERA_INDEX: int = 0  # Índice de la cámara a utilizar (0 = cámara por defecto)
CAMERA_FPS: int = 60  # FPS objetivo para la captura de video
CAMERA_BUFFER_SIZE: int = 3  # Tamaño del buffer de la cámara
CAMERA_MIRROR_MODE: bool = True  # Mostrar la cámara en modo espejo (efecto de espejo)

# Configuración MediaPipe
MEDIAPIPE_MAX_HANDS: int = 1  # Detectar solo una mano para mejor rendimiento
MEDIAPIPE_DETECTION_CONFIDENCE: float = 0.7  # Umbral de confianza para detección
MEDIAPIPE_TRACKING_CONFIDENCE: float = 0.5  # Umbral de confianza para seguimiento
MEDIAPIPE_MANO_IZQUIERDA: bool = True  # Detectar mano izquierda en lugar de la derecha

# Configuración del lienzo
CANVAS_WIDTH: int = 640  # Aumentado para coincidir con la cámara
CANVAS_HEIGHT: int = 480  # Aumentado para coincidir con la cámara
CANVAS_BACKGROUND_COLOR: Tuple[int, int, int] = (255, 255, 255)  # Blanco
CANVAS_LINE_THICKNESS: int = 2  # Grosor de línea adecuado para mayor resolución
CANVAS_CIRCLE_RADIUS: int = 5  # Radio de círculo

# Colores para dibujo
DEFAULT_COLOR: Tuple[int, int, int] = (0, 0, 255)  # Color de dibujo predeterminado (BGR: Rojo)
COLORS: Dict[str, Tuple[int, int, int]] = {
    "rojo": (0, 0, 255),     # BGR (OpenCV usa BGR)
    "verde": (0, 255, 0),
    "azul": (255, 0, 0),
    "amarillo": (0, 255, 255),
    "magenta": (255, 0, 255),
    "cian": (255, 255, 0),
    "negro": (0, 0, 0),
    "blanco": (255, 255, 255)
}

# Parámetros optimización de rendimiento
OPTIMIZATION_RESIZE_FACTOR: float = 1.0  # Factor de redimensionamiento (1.0 = sin cambios)
OPTIMIZATION_SKIP_FRAMES: int = 1  # Saltar frames para mejor rendimiento (0 = no saltar)
OPTIMIZATION_PREDICTION_THRESHOLD: float = 0.05  # Tiempo límite para activar predicción (segundos)
OPTIMIZATION_MAX_PREDICTION_FRAMES: int = 3  # Máximo de frames para predicción
OPTIMIZATION_USE_ASYNC_CAPTURE: bool = False  # Usar captura asíncrona
OPTIMIZATION_SHOW_METRICS: bool = True  # Mostrar métricas de rendimiento
OPTIMIZATION_QUALITY: float = 0.7  # Factor de calidad (0-1)
OPTIMIZATION_BUFFER_SIZE: int = 10  # Tamaño del buffer para operaciones de dibujo
OPTIMIZATION_MEMORY_LIMIT_MB: int = 100  # Límite de memoria en MB para el buffer de operaciones
OPTIMIZATION_SOLO_MANO_DERECHA: bool = True  # Solo detectar la mano derecha

# Configuración de la interfaz de usuario
UI_BUTTON_SIZE: int = 40  # Aumentado para mayor resolución
UI_BUTTON_SPACING: int = 10  # Espaciado entre botones
UI_BUTTON_RADIUS: int = 3  # Radio de esquinas redondeadas
UI_PANEL_HEIGHT: int = 50  # Altura del panel de herramientas
UI_FOOTER_HEIGHT: int = 30  # Altura del footer para métricas en la parte inferior
UI_PANEL_ALPHA: float = 0.7  # Transparencia del panel (0-1)
UI_SHOW_PALETTE: bool = True  # Mostrar paleta de colores
UI_BUTTON_CORNER_RADIUS: int = 10
UI_COLOR_BUTTON_SIZE: int = 30

# Configuración de multihilo
MULTITHREAD_ENABLED: bool = True  # Habilitar procesamiento multihilo
MULTITHREAD_MAX_WORKERS: int = 4  # Número máximo de workers para procesamiento paralelo

# Configuración de depuración
DEBUG_ENABLED: bool = False  # Habilitar modo de depuración
DEBUG_DRAW_LANDMARKS: bool = True  # Dibujar landmarks de manos
DEBUG_DRAW_HAND_CONNECTIONS: bool = True  # Dibujar conexiones entre landmarks
DEBUG_LOG_PERFORMANCE: bool = True  # Registrar métricas de rendimiento

# Configuración para optimización adaptativa de dibujo
ADAPTIVE_DRAWING_ENABLED: bool = True  # Habilitar dibujo adaptativo
ADAPTIVE_THICKNESS_MIN: int = 1  # Reducido: Grosor mínimo
ADAPTIVE_THICKNESS_MAX: int = 5  # Reducido: Grosor máximo
ADAPTIVE_VELOCITY_THRESHOLD: float = 50.0  # Umbral de velocidad para grosor adaptativo

# Configuración para el grosor adaptativo basado en velocidad
DRAWING_SPEED_SMOOTH_FACTOR: float = 0.3  # Factor de suavizado para cálculo de velocidad
DRAWING_MIN_THICKNESS: int = 1  # Reducido: Grosor mínimo para dibujo (movimientos rápidos)
DRAWING_MAX_THICKNESS: int = 5  # Reducido: Grosor máximo para dibujo (movimientos lentos)
DRAWING_SPEED_THRESHOLD_LOW: float = 100.0  # Umbral de velocidad baja (píxeles/segundo)
DRAWING_SPEED_THRESHOLD_HIGH: float = 800.0  # Umbral de velocidad alta (píxeles/segundo)

logger.debug("Configuración cargada correctamente")
