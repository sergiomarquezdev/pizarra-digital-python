"""
Configuración global para la aplicación de pizarra digital.

Este módulo contiene constantes y parámetros de configuración
utilizados en toda la aplicación.
"""
from typing import Dict, Tuple, List

# Configuración de la cámara
CAMERA_INDEX: int = 0  # Cambiado de 0 a 1 para probar con cámara externa o secundaria
CAMERA_WIDTH: int = 640  # Ancho de la captura de la cámara
CAMERA_HEIGHT: int = 480  # Alto de la captura de la cámara

# Configuración de MediaPipe Hands
MAX_NUM_HANDS: int = 1  # Número máximo de manos a detectar
MIN_DETECTION_CONFIDENCE: float = 0.5  # Reducido de 0.7 a 0.5 para mayor tolerancia
MIN_TRACKING_CONFIDENCE: float = 0.5  # Umbral mínimo de confianza para el seguimiento

# Configuración del lienzo
CANVAS_WIDTH: int = CAMERA_WIDTH  # Ancho del lienzo (igual al de la cámara)
CANVAS_HEIGHT: int = CAMERA_HEIGHT  # Alto del lienzo (igual al de la cámara)
CANVAS_BACKGROUND: Tuple[int, int, int] = (0, 0, 0)  # Color de fondo del lienzo (negro)

# Configuración del dibujo
DEFAULT_DRAWING_COLOR: Tuple[int, int, int] = (255, 255, 255)  # Color de dibujo predeterminado (blanco)
DRAWING_THICKNESS: int = 4  # Grosor de la línea de dibujo

# Paleta de colores disponibles
COLOR_PALETTE: Dict[str, Tuple[int, int, int]] = {
    "blanco": (255, 255, 255),
    "negro": (0, 0, 0),
    "rojo": (0, 0, 255),  # BGR en OpenCV
    "verde": (0, 255, 0),
    "azul": (255, 0, 0),  # BGR en OpenCV
}

# Configuración de la interfaz
BUTTON_WIDTH: int = 50  # Ancho de los botones
BUTTON_HEIGHT: int = 30  # Alto de los botones
BUTTON_MARGIN: int = 10  # Margen entre botones
BUTTON_COLOR: Tuple[int, int, int] = (200, 200, 200)  # Color de fondo de los botones
BUTTON_TEXT_COLOR: Tuple[int, int, int] = (0, 0, 0)  # Color del texto de los botones
BUTTON_FONT_SCALE: float = 0.5  # Escala de la fuente para el texto de los botones

# Índices de los puntos de referencia de la mano en MediaPipe
# Basado en la documentación: https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker/
INDEX_FINGER_TIP: int = 8  # Índice de la punta del dedo índice

# Configuración de la aplicación
APP_NAME: str = "Pizarra Digital"  # Nombre de la aplicación
EXIT_KEY: str = "q"  # Tecla para salir de la aplicación
