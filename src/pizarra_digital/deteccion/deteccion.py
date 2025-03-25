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
import time
from typing import Tuple, List, Optional, Dict, Any, Union, Deque
from collections import deque

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

# Configuración para optimización del rendimiento
PREDICTION_FRAMES = 3  # Número de frames para predecir movimiento
SKIP_FRAME_THRESHOLD = 0.05  # Umbral de tiempo (segundos) para activar salto de frames
MAX_SKIPPABLE_FRAMES = 2  # Máximo número de frames que se pueden saltar
PREDICTION_SMOOTHING = 0.7  # Factor de suavizado para predicción (0-1)

class DetectorManos:
    """Clase para la detección y seguimiento de manos usando MediaPipe Hands."""

    def __init__(self,
                max_num_hands: int = MAX_NUM_HANDS,
                min_detection_confidence: float = MIN_DETECTION_CONFIDENCE,
                min_tracking_confidence: float = MIN_TRACKING_CONFIDENCE,
                enable_optimizations: bool = True) -> None:
        """
        Inicializa el detector de manos con MediaPipe.

        Args:
            max_num_hands: Número máximo de manos a detectar.
            min_detection_confidence: Umbral mínimo de confianza para la detección.
            min_tracking_confidence: Umbral mínimo de confianza para el seguimiento.
            enable_optimizations: Activa optimizaciones de rendimiento (predicción, salto de frames).
        """
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.enable_optimizations = enable_optimizations

        # Inicializar el modelo de MediaPipe Hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )

        # Variables para optimización
        self.ultima_deteccion_completa = 0.0
        self.frames_desde_ultima_deteccion = 0
        self.ultima_posicion_manos: Optional[List[Dict[str, Any]]] = None
        self.historial_posiciones: Deque[Optional[List[Dict[str, Any]]]] = deque(maxlen=PREDICTION_FRAMES)
        self.velocidades_landmarks: Optional[List[Dict[str, List[Dict[str, float]]]]] = None
        self.ultima_estimacion: Optional[List[Dict[str, Any]]] = None

        # Métricas de rendimiento
        self.tiempos_procesamiento: Deque[float] = deque(maxlen=30)  # Últimos 30 tiempos
        self.frames_saltados = 0
        self.frames_procesados = 0
        self.ultima_actualizacion_metricas = time.time()

        logger.info(f"Detector de manos inicializado con parámetros: "
                   f"max_num_hands={max_num_hands}, "
                   f"min_detection_confidence={min_detection_confidence}, "
                   f"min_tracking_confidence={min_tracking_confidence}, "
                   f"enable_optimizations={enable_optimizations}")

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

        # Variables para el tiempo y optimización
        tiempo_inicio = time.time()
        tiempo_transcurrido = tiempo_inicio - self.ultima_deteccion_completa
        realizar_deteccion_completa = True
        manos_detectadas = None

        # Decidir si usar predicción o hacer detección completa
        if (self.enable_optimizations and
            tiempo_transcurrido < SKIP_FRAME_THRESHOLD and
            self.frames_desde_ultima_deteccion < MAX_SKIPPABLE_FRAMES and
            self.ultima_posicion_manos is not None):

            # Usar predicción de movimiento en lugar de detección completa
            realizar_deteccion_completa = False
            self.frames_saltados += 1
            self.frames_desde_ultima_deteccion += 1

            # Predecir nuevas posiciones basadas en velocidad de movimiento
            manos_detectadas = self._predecir_posiciones()

        if realizar_deteccion_completa:
            # Convertir el fotograma a RGB (MediaPipe requiere RGB)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Procesar el fotograma con MediaPipe Hands
            # To improve performance, mark the image as not writeable
            frame_rgb.flags.writeable = False
            results = self.hands.process(frame_rgb)
            frame_rgb.flags.writeable = True

            # Actualizar contadores y tiempo de la última detección completa
            self.ultima_deteccion_completa = tiempo_inicio
            self.frames_desde_ultima_deteccion = 0
            self.frames_procesados += 1

            # Verificar si se detectaron manos
            if results.multi_hand_landmarks:
                # Lista para almacenar los datos de las manos detectadas
                manos_detectadas = []

                # Iterar sobre todas las manos detectadas
                for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
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
                    altura, anchura, _ = frame.shape
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

                # Actualizar historial de posiciones para predicción
                self._actualizar_historial(manos_detectadas)
            else:
                # No se detectaron manos, actualizar historial con None
                self._actualizar_historial(None)
                manos_detectadas = None

        # Crear una copia del fotograma para dibujar
        processed_frame = frame.copy()

        # Dibujar landmarks si se solicita y hay manos detectadas
        if dibujar_landmarks and manos_detectadas:
            self._dibujar_landmarks(processed_frame, manos_detectadas)

        # Actualizar métricas de rendimiento
        tiempo_procesamiento = time.time() - tiempo_inicio
        self.tiempos_procesamiento.append(tiempo_procesamiento)

        # Actualizar última estimación para referencia
        self.ultima_posicion_manos = manos_detectadas
        self.ultima_estimacion = manos_detectadas

        return processed_frame, manos_detectadas

    def _predecir_posiciones(self) -> Optional[List[Dict[str, Any]]]:
        """
        Predice la posición de las manos basándose en movimientos anteriores.

        Este método implementa una predicción lineal simple basada en la
        velocidad de movimiento de los landmarks entre frames.

        Returns:
            Lista de diccionarios con información predicha de las manos,
            o None si no hay suficientes datos para predecir.
        """
        if (self.ultima_posicion_manos is None or
            len(self.historial_posiciones) < 2 or
            self.historial_posiciones[-1] is None):
            return None

        # Crear lista para las manos predichas
        manos_predichas = []

        # Para cada mano en la última detección
        for mano_idx, ultima_mano in enumerate(self.ultima_posicion_manos):
            # Si no hay suficientes datos en el historial para esta mano, usar la última posición conocida
            if any(h is None for h in self.historial_posiciones):
                manos_predichas.append(ultima_mano)
                continue

            # Creamos una nueva estructura para la mano predicha
            mano_predicha = ultima_mano.copy()

            # Predecir landmarks normalizados
            if "landmarks_normalizados" in ultima_mano:
                landmarks_originales = ultima_mano["landmarks_normalizados"]
                landmarks_predichos = []

                # Obtener posiciones anteriores para calcular velocidad
                posiciones_anteriores = []
                for h in reversed(self.historial_posiciones):
                    if h is not None and len(h) > mano_idx:
                        if "landmarks_normalizados" in h[mano_idx]:
                            posiciones_anteriores.append(h[mano_idx]["landmarks_normalizados"])
                    if len(posiciones_anteriores) >= 2:
                        break

                # Si no hay suficientes posiciones anteriores, usar la última posición conocida
                if len(posiciones_anteriores) < 2:
                    landmarks_predichos = landmarks_originales
                else:
                    # Calcular velocidad para cada landmark
                    for i, landmark in enumerate(landmarks_originales):
                        if i >= len(posiciones_anteriores[0]):
                            # Si el índice está fuera de rango, solo copiar
                            landmarks_predichos.append(landmark.copy())
                            continue

                        # Calcular velocidad (diferencia entre últimas posiciones)
                        vx = landmark["x"] - posiciones_anteriores[0][i]["x"]
                        vy = landmark["y"] - posiciones_anteriores[0][i]["y"]
                        vz = landmark["z"] - posiciones_anteriores[0][i]["z"]

                        # Aplicar suavizado a la predicción
                        nuevo_x = landmark["x"] + vx * PREDICTION_SMOOTHING
                        nuevo_y = landmark["y"] + vy * PREDICTION_SMOOTHING
                        nuevo_z = landmark["z"] + vz * PREDICTION_SMOOTHING

                        landmarks_predichos.append({
                            "x": nuevo_x,
                            "y": nuevo_y,
                            "z": nuevo_z
                        })

                mano_predicha["landmarks_normalizados"] = landmarks_predichos

            # Predecir landmarks en píxeles si existen
            if "landmarks_pixeles" in ultima_mano:
                # Mismo procedimiento pero con coordenadas de píxeles
                landmarks_originales = ultima_mano["landmarks_pixeles"]
                landmarks_predichos = []

                # Obtener posiciones anteriores para calcular velocidad
                posiciones_anteriores = []
                for h in reversed(self.historial_posiciones):
                    if h is not None and len(h) > mano_idx:
                        if "landmarks_pixeles" in h[mano_idx]:
                            posiciones_anteriores.append(h[mano_idx]["landmarks_pixeles"])
                    if len(posiciones_anteriores) >= 2:
                        break

                # Si no hay suficientes posiciones anteriores, usar la última posición conocida
                if len(posiciones_anteriores) < 2:
                    landmarks_predichos = landmarks_originales
                else:
                    # Calcular velocidad para cada landmark
                    for i, landmark in enumerate(landmarks_originales):
                        if i >= len(posiciones_anteriores[0]):
                            # Si el índice está fuera de rango, solo copiar
                            landmarks_predichos.append(landmark.copy())
                            continue

                        # Calcular velocidad (diferencia entre últimas posiciones)
                        vx = landmark["x"] - posiciones_anteriores[0][i]["x"]
                        vy = landmark["y"] - posiciones_anteriores[0][i]["y"]
                        vz = landmark["z"] - posiciones_anteriores[0][i]["z"]

                        # Aplicar suavizado a la predicción
                        nuevo_x = int(landmark["x"] + vx * PREDICTION_SMOOTHING)
                        nuevo_y = int(landmark["y"] + vy * PREDICTION_SMOOTHING)
                        nuevo_z = landmark["z"] + vz * PREDICTION_SMOOTHING

                        landmarks_predichos.append({
                            "x": nuevo_x,
                            "y": nuevo_y,
                            "z": nuevo_z
                        })

                mano_predicha["landmarks_pixeles"] = landmarks_predichos

            # Añadir la mano predicha a la lista
            manos_predichas.append(mano_predicha)

        return manos_predichas if manos_predichas else None

    def _actualizar_historial(self, manos: Optional[List[Dict[str, Any]]]) -> None:
        """
        Actualiza el historial de posiciones de las manos.

        Args:
            manos: Lista de información de manos detectadas o None si no se detectaron.
        """
        # Hacer una copia profunda para evitar problemas de referencia
        if manos is not None:
            # Copia simplificada (suficiente para nuestro uso)
            manos_copia = []
            for mano in manos:
                mano_copia = {}
                for key, value in mano.items():
                    if isinstance(value, list):
                        # Copiar listas de landmarks
                        mano_copia[key] = [item.copy() for item in value]
                    else:
                        # Copiar valores simples
                        mano_copia[key] = value
                manos_copia.append(mano_copia)
            self.historial_posiciones.append(manos_copia)
        else:
            self.historial_posiciones.append(None)

        # Calcular velocidades si hay suficientes datos
        if len(self.historial_posiciones) >= 2:
            self._actualizar_velocidades()

    def _actualizar_velocidades(self) -> None:
        """
        Actualiza las velocidades de movimiento de los landmarks
        basándose en las últimas posiciones.
        """
        # Necesitamos al menos dos frames con manos detectadas
        ultimas_pos = self.historial_posiciones[-1]
        penultimas_pos = self.historial_posiciones[-2]

        if ultimas_pos is None or penultimas_pos is None:
            self.velocidades_landmarks = None
            return

        # Inicializar lista de velocidades
        velocidades = []

        # Para cada mano, calcular velocidades de los landmarks
        for i in range(min(len(ultimas_pos), len(penultimas_pos))):
            ultima_mano = ultimas_pos[i]
            penultima_mano = penultimas_pos[i]

            # Inicializar diccionario para esta mano
            vel_mano = {"landmarks_normalizados": [], "landmarks_pixeles": []}

            # Calcular velocidades para landmarks normalizados
            if ("landmarks_normalizados" in ultima_mano and
                "landmarks_normalizados" in penultima_mano):
                ult_landmarks = ultima_mano["landmarks_normalizados"]
                pen_landmarks = penultima_mano["landmarks_normalizados"]

                for j in range(min(len(ult_landmarks), len(pen_landmarks))):
                    vx = ult_landmarks[j]["x"] - pen_landmarks[j]["x"]
                    vy = ult_landmarks[j]["y"] - pen_landmarks[j]["y"]
                    vz = ult_landmarks[j]["z"] - pen_landmarks[j]["z"]

                    vel_mano["landmarks_normalizados"].append({
                        "vx": vx, "vy": vy, "vz": vz
                    })

            # Calcular velocidades para landmarks en píxeles
            if ("landmarks_pixeles" in ultima_mano and
                "landmarks_pixeles" in penultima_mano):
                ult_landmarks = ultima_mano["landmarks_pixeles"]
                pen_landmarks = penultima_mano["landmarks_pixeles"]

                for j in range(min(len(ult_landmarks), len(pen_landmarks))):
                    vx = ult_landmarks[j]["x"] - pen_landmarks[j]["x"]
                    vy = ult_landmarks[j]["y"] - pen_landmarks[j]["y"]
                    vz = ult_landmarks[j]["z"] - pen_landmarks[j]["z"]

                    vel_mano["landmarks_pixeles"].append({
                        "vx": vx, "vy": vy, "vz": vz
                    })

            velocidades.append(vel_mano)

        self.velocidades_landmarks = velocidades

    def _dibujar_landmarks(self, frame: np.ndarray, manos: List[Dict[str, Any]]) -> None:
        """
        Dibuja los landmarks de las manos en el fotograma.

        Args:
            frame: Fotograma donde dibujar los landmarks.
            manos: Lista de información de manos detectadas.
        """
        altura, anchura, _ = frame.shape

        for mano in manos:
            if "landmarks_pixeles" not in mano:
                continue

            landmarks = mano["landmarks_pixeles"]

            # Dibujar conexiones entre landmarks
            connections = self.mp_hands.HAND_CONNECTIONS
            for connection in connections:
                start_idx, end_idx = connection
                if (start_idx < len(landmarks) and end_idx < len(landmarks)):
                    start_point = (landmarks[start_idx]["x"], landmarks[start_idx]["y"])
                    end_point = (landmarks[end_idx]["x"], landmarks[end_idx]["y"])
                    cv2.line(frame, start_point, end_point, (0, 255, 0), 2)

            # Dibujar landmarks
            for landmark in landmarks:
                point = (landmark["x"], landmark["y"])
                cv2.circle(frame, point, 5, (0, 0, 255), -1)

            # Destacar la punta del dedo índice
            if INDEX_FINGER_TIP < len(landmarks):
                point = (landmarks[INDEX_FINGER_TIP]["x"], landmarks[INDEX_FINGER_TIP]["y"])
                cv2.circle(frame, point, 8, (255, 0, 0), -1)

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

    def obtener_metricas_rendimiento(self) -> Dict[str, Any]:
        """
        Obtiene métricas de rendimiento del detector de manos.

        Returns:
            Diccionario con varias métricas de rendimiento.
        """
        # Calcular tiempo de procesamiento promedio
        tiempo_promedio = sum(self.tiempos_procesamiento) / max(1, len(self.tiempos_procesamiento))

        # Calcular porcentaje de frames saltados
        total_frames = self.frames_procesados + self.frames_saltados
        porcentaje_saltados = (self.frames_saltados / max(1, total_frames)) * 100

        # Calcular FPS de procesamiento
        fps_procesamiento = 1.0 / max(0.001, tiempo_promedio)

        # Crear y devolver diccionario de métricas
        return {
            "tiempo_procesamiento_ms": tiempo_promedio * 1000,
            "fps_procesamiento": fps_procesamiento,
            "frames_procesados": self.frames_procesados,
            "frames_saltados": self.frames_saltados,
            "porcentaje_frames_saltados": porcentaje_saltados,
            "tamano_historial": len(self.historial_posiciones)
        }

    def liberar_recursos(self) -> None:
        """
        Libera los recursos utilizados por el detector de manos.
        """
        if hasattr(self, 'hands') and self.hands:
            self.hands.close()
            logger.info("Recursos del detector de manos liberados correctamente")
