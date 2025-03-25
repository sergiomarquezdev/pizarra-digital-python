"""
Módulo para la captura de video mediante OpenCV.

Este módulo proporciona funciones para inicializar la cámara web,
capturar fotogramas y gestionar los recursos de la cámara.
"""
import logging
import cv2
import numpy as np
from typing import Tuple, Optional, Dict, Any

from ..config import CAMERA_INDEX, CAMERA_WIDTH, CAMERA_HEIGHT

# Configuración del logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CameraError(Exception):
    """Excepción personalizada para errores relacionados con la cámara."""
    pass

def inicializar_camara() -> cv2.VideoCapture:
    """
    Inicializa y configura la cámara web.

    Intenta diferentes índices de cámara si el configurado inicialmente falla.

    Returns:
        VideoCapture: Objeto de captura de video de OpenCV.

    Raises:
        CameraError: Si no se puede abrir ninguna cámara.
    """
    # Lista de índices de cámara a intentar, empezando por el configurado
    indices_camara = [CAMERA_INDEX, 0, 1, 2]
    # Eliminar duplicados manteniendo el orden
    indices_camara = list(dict.fromkeys(indices_camara))

    for indice in indices_camara:
        try:
            logger.info(f"Intentando abrir la cámara con índice {indice}...")
            # Intentar abrir la cámara con el índice actual
            cap = cv2.VideoCapture(indice)

            # Verificar si la cámara se abrió correctamente
            if cap.isOpened():
                # Configurar la resolución de la cámara
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

                # Optimizaciones para mejorar el rendimiento
                cap.set(cv2.CAP_PROP_FPS, 30)  # Limitar a 30 FPS para un rendimiento más estable
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reducir el buffer para obtener frames más recientes
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))  # MJPG suele ser más rápido que YUY2

                # Leer un fotograma de prueba para asegurarse de que funciona
                ret, _ = cap.read()
                if not ret:
                    logger.warning(f"Se pudo abrir la cámara {indice} pero no se pudo leer un fotograma")
                    cap.release()
                    continue

                # Registrar que la cámara se ha inicializado correctamente
                actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                logger.info(f"Cámara inicializada con resolución: {actual_width}x{actual_height}")

                return cap
            else:
                logger.warning(f"No se pudo abrir la cámara con índice {indice}")

        except cv2.error as e:
            logger.warning(f"Error de OpenCV al abrir la cámara {indice}: {e}")
        except Exception as e:
            logger.warning(f"Error inesperado al abrir la cámara {indice}: {e}")

    # Si llegamos aquí, no se pudo abrir ninguna cámara
    logger.error("No se pudo abrir ninguna cámara disponible")
    raise CameraError("No se pudo abrir ninguna cámara disponible. Verifica que tu cámara esté conectada y no esté siendo utilizada por otra aplicación.")

def leer_fotograma(cap: cv2.VideoCapture, escala: float = 1.0) -> Tuple[bool, Optional[np.ndarray]]:
    """
    Lee un fotograma de la cámara con opciones de optimización.

    Args:
        cap: Objeto de captura de video de OpenCV.
        escala: Factor de escala para redimensionar el fotograma (1.0 = tamaño original,
                0.5 = mitad del tamaño). Valores menores mejoran el rendimiento.

    Returns:
        Tupla que contiene un booleano indicando si la lectura fue exitosa
        y el fotograma capturado (None si la lectura falló).
    """
    try:
        # Si hay fotogramas en buffer, descartar todos menos el último
        if cap.get(cv2.CAP_PROP_BUFFERSIZE) > 1:
            for _ in range(int(cap.get(cv2.CAP_PROP_BUFFERSIZE)) - 1):
                cap.grab()

        ret, frame = cap.read()
        if not ret:
            logger.warning("No se pudo leer el fotograma de la cámara")
            return False, None

        # Voltear el fotograma horizontalmente para una experiencia tipo espejo
        frame = cv2.flip(frame, 1)

        # Redimensionar si se solicita una escala diferente a 1.0
        if escala != 1.0:
            ancho = int(frame.shape[1] * escala)
            alto = int(frame.shape[0] * escala)
            # Usar INTER_AREA para reducción (más rápido y mejor calidad para downscaling)
            frame = cv2.resize(frame, (ancho, alto), interpolation=cv2.INTER_AREA)

        return True, frame
    except cv2.error as e:
        logger.error(f"Error de OpenCV al leer fotograma: {e}")
        return False, None
    except Exception as e:
        logger.error(f"Error inesperado al leer fotograma: {e}")
        return False, None

def ajustar_parametros_camara(cap: cv2.VideoCapture, parametros: Dict[str, Any]) -> bool:
    """
    Ajusta los parámetros de la cámara para optimizar el rendimiento.

    Args:
        cap: Objeto de captura de video de OpenCV.
        parametros: Diccionario con los parámetros a configurar.
                   Claves válidas: 'fps', 'exposicion', 'brillo', 'contraste', 'saturation'.

    Returns:
        Booleano indicando si al menos un parámetro se configuró con éxito.
    """
    if not cap.isOpened():
        logger.error("No se puede ajustar parámetros: la cámara no está abierta")
        return False

    exito = False

    # Mapa de nombres de parámetros a propiedades de OpenCV
    param_map = {
        'fps': cv2.CAP_PROP_FPS,
        'exposicion': cv2.CAP_PROP_EXPOSURE,
        'brillo': cv2.CAP_PROP_BRIGHTNESS,
        'contraste': cv2.CAP_PROP_CONTRAST,
        'saturation': cv2.CAP_PROP_SATURATION,
        'buffer': cv2.CAP_PROP_BUFFERSIZE
    }

    for nombre, valor in parametros.items():
        if nombre in param_map:
            prop = param_map[nombre]
            result = cap.set(prop, valor)
            if result:
                logger.info(f"Parámetro '{nombre}' configurado a {valor}")
                exito = True
            else:
                logger.warning(f"No se pudo configurar el parámetro '{nombre}'")

    return exito

def liberar_camara(cap: cv2.VideoCapture) -> None:
    """
    Libera los recursos de la cámara.

    Args:
        cap: Objeto de captura de video de OpenCV a liberar.
    """
    try:
        if cap is not None:
            cap.release()
            logger.info("Recursos de la cámara liberados correctamente")
    except Exception as e:
        logger.error(f"Error al liberar los recursos de la cámara: {e}")
