"""
Módulo para la captura de video mediante OpenCV.

Este módulo proporciona funciones para inicializar la cámara web,
capturar fotogramas y gestionar los recursos de la cámara.
"""
import logging
import cv2
import numpy as np
from typing import Tuple, Optional

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

def leer_fotograma(cap: cv2.VideoCapture) -> Tuple[bool, Optional[np.ndarray]]:
    """
    Lee un fotograma de la cámara.

    Args:
        cap: Objeto de captura de video de OpenCV.

    Returns:
        Tupla que contiene un booleano indicando si la lectura fue exitosa
        y el fotograma capturado (None si la lectura falló).
    """
    try:
        ret, frame = cap.read()
        if not ret:
            logger.warning("No se pudo leer el fotograma de la cámara")
            return False, None

        # Voltear el fotograma horizontalmente para una experiencia tipo espejo
        frame = cv2.flip(frame, 1)

        return True, frame
    except cv2.error as e:
        logger.error(f"Error de OpenCV al leer fotograma: {e}")
        return False, None
    except Exception as e:
        logger.error(f"Error inesperado al leer fotograma: {e}")
        return False, None

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
