"""
Módulo para la detección y seguimiento de manos usando MediaPipe.

Este módulo proporciona funciones para inicializar MediaPipe Hands,
procesar fotogramas para detectar manos y extraer las coordenadas
de los puntos de referencia (landmarks).
"""
import logging
import cv2
import mediapipe as mp
import numpy as np
from typing import Tuple, List, Optional, Dict, Any, Union

from ..config import (
    MAX_NUM_HANDS,
    MIN_DETECTION_CONFIDENCE,
    MIN_TRACKING_CONFIDENCE,
    INDEX_FINGER_TIP
)

# Configuración del logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Tipo personalizado para los puntos de referencia de la mano
# Cambiado para evitar la referencia a la estructura interna de MediaPipe
HandLandmarks = Any  # Lista de landmarks de mano

class DetectorManos:
    """Clase para la detección y seguimiento de manos usando MediaPipe Hands."""

    def __init__(self,
                max_num_hands: int = MAX_NUM_HANDS,
                min_detection_confidence: float = MIN_DETECTION_CONFIDENCE,
                min_tracking_confidence: float = MIN_TRACKING_CONFIDENCE) -> None:
        """
        Inicializa el detector de manos con MediaPipe.

        Args:
            max_num_hands: Número máximo de manos a detectar.
            min_detection_confidence: Umbral mínimo de confianza para la detección.
            min_tracking_confidence: Umbral mínimo de confianza para el seguimiento.
        """
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        # Inicializar el modelo de MediaPipe Hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )

        logger.info(f"Detector de manos inicializado con parámetros: "
                   f"max_num_hands={max_num_hands}, "
                   f"min_detection_confidence={min_detection_confidence}, "
                   f"min_tracking_confidence={min_tracking_confidence}")

    def procesar_fotograma(self, frame: np.ndarray, dibujar_landmarks: bool = False) -> Tuple[np.ndarray, Optional[List[Dict[str, Any]]]]:
        """
        Procesa un fotograma para detectar manos y extraer los puntos de referencia.

        Args:
            frame: Fotograma BGR de OpenCV a procesar.
            dibujar_landmarks: Si es True, dibuja los landmarks en el fotograma.

        Returns:
            Tupla que contiene el fotograma procesado (con landmarks dibujados si
            dibujar_landmarks es True) y una lista de diccionarios con los datos
            de las manos detectadas, o None si no se detectan manos.
        """
        if frame is None:
            logger.warning("Se recibió un fotograma nulo para procesar")
            return frame, None

        # Convertir el fotograma a RGB (MediaPipe requiere RGB)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Procesar el fotograma con MediaPipe Hands
        # To improve performance, mark the image as not writeable
        frame_rgb.flags.writeable = False
        results = self.hands.process(frame_rgb)
        frame_rgb.flags.writeable = True

        # Crear una copia del fotograma para dibujar
        processed_frame = frame.copy()

        # Lista para almacenar los datos de las manos detectadas
        manos_detectadas: List[Dict[str, Any]] = []

        # Verificar si se detectaron manos
        if results.multi_hand_landmarks:
            # Iterar sobre todas las manos detectadas
            for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                # Dibujar los landmarks si se solicita
                if dibujar_landmarks:
                    self.mp_drawing.draw_landmarks(
                        processed_frame,
                        hand_landmarks,
                        self.mp_hands.HAND_CONNECTIONS,
                        self.mp_drawing_styles.get_default_hand_landmarks_style(),
                        self.mp_drawing_styles.get_default_hand_connections_style()
                    )

                # Extraer la información de la mano
                hand_info: Dict[str, Any] = {}

                # Obtener la lateralidad de la mano (izquierda/derecha)
                if results.multi_handedness and idx < len(results.multi_handedness):
                    handedness = results.multi_handedness[idx]
                    hand_info["lado"] = handedness.classification[0].label
                    hand_info["confianza_lado"] = handedness.classification[0].score

                # Extraer los landmarks normalizados (valores entre 0 y 1)
                landmarks_normalizados = []
                for landmark in hand_landmarks.landmark:
                    landmarks_normalizados.append({
                        "x": landmark.x,
                        "y": landmark.y,
                        "z": landmark.z
                    })
                hand_info["landmarks_normalizados"] = landmarks_normalizados

                # Convertir los landmarks a coordenadas de píxeles
                altura, anchura, _ = processed_frame.shape
                landmarks_pixeles = []
                for landmark in hand_landmarks.landmark:
                    x_px = int(landmark.x * anchura)
                    y_px = int(landmark.y * altura)
                    landmarks_pixeles.append({
                        "x": x_px,
                        "y": y_px,
                        "z": landmark.z
                    })
                hand_info["landmarks_pixeles"] = landmarks_pixeles

                # Añadir la información de la mano a la lista
                manos_detectadas.append(hand_info)

            logger.debug(f"Se detectaron {len(manos_detectadas)} manos en el fotograma")
            return processed_frame, manos_detectadas
        else:
            # No se detectaron manos
            return processed_frame, None

    def obtener_punta_indice(self, mano: Dict[str, Any]) -> Optional[Tuple[int, int]]:
        """
        Obtiene las coordenadas de píxeles de la punta del dedo índice.

        Args:
            mano: Diccionario con la información de una mano detectada.

        Returns:
            Tupla con las coordenadas (x, y) de la punta del dedo índice,
            o None si no se puede obtener.
        """
        if mano and "landmarks_pixeles" in mano:
            if INDEX_FINGER_TIP < len(mano["landmarks_pixeles"]):
                punto = mano["landmarks_pixeles"][INDEX_FINGER_TIP]
                return (punto["x"], punto["y"])
        return None

    def es_indice_extendido(self, mano: Dict[str, Any]) -> bool:
        """
        Determina si el dedo índice está extendido.

        Esta función analiza la posición relativa de las articulaciones
        del dedo índice para determinar si está extendido.

        Args:
            mano: Diccionario con la información de una mano detectada.

        Returns:
            True si el dedo índice está extendido, False en caso contrario.
        """
        if not mano or "landmarks_normalizados" not in mano:
            return False

        # Índices de los landmarks del dedo índice
        INDICE_MCP = 5  # Articulación metacarpofalángica (base del dedo)
        INDICE_PIP = 6  # Articulación interfalángica proximal (primera articulación)
        INDICE_DIP = 7  # Articulación interfalángica distal (segunda articulación)
        INDICE_TIP = 8  # Punta del dedo

        # Obtener las coordenadas y de los landmarks del dedo índice
        landmarks = mano["landmarks_normalizados"]

        # Verificar que están todos los landmarks necesarios
        if len(landmarks) <= max(INDICE_MCP, INDICE_PIP, INDICE_DIP, INDICE_TIP):
            return False

        # Calcular las posiciones relativas en el eje Y
        y_mcp = landmarks[INDICE_MCP]["y"]
        y_pip = landmarks[INDICE_PIP]["y"]
        y_dip = landmarks[INDICE_DIP]["y"]
        y_tip = landmarks[INDICE_TIP]["y"]

        # El dedo índice está extendido si la punta está por encima (valor Y menor)
        # que las articulaciones intermedias
        return y_tip < y_dip < y_pip

    def liberar_recursos(self) -> None:
        """
        Libera los recursos utilizados por el detector de manos.
        """
        if hasattr(self, 'hands') and self.hands:
            self.hands.close()
            logger.info("Recursos del detector de manos liberados correctamente")
