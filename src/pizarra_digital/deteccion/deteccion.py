"""
Módulo para la detección de manos con MediaPipe.

Este módulo proporciona funciones para detectar y rastrear manos
usando la biblioteca MediaPipe.
"""
import logging
import time
import numpy as np
import cv2
import mediapipe as mp
from typing import Tuple, List, Dict, Any, Optional, Deque
from collections import deque

from ..config import (
    MEDIAPIPE_MAX_HANDS,
    MEDIAPIPE_DETECTION_CONFIDENCE,
    MEDIAPIPE_TRACKING_CONFIDENCE,
    MEDIAPIPE_MANO_IZQUIERDA,
    OPTIMIZATION_PREDICTION_THRESHOLD,
    OPTIMIZATION_MAX_PREDICTION_FRAMES,
    DEBUG_DRAW_LANDMARKS,
    DEBUG_DRAW_HAND_CONNECTIONS,
    CAMERA_HEIGHT,
    CAMERA_WIDTH
)

# Configuración del logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuración para predicción de landmarks durante movimientos rápidos
PREDICTION_FRAMES: int = OPTIMIZATION_MAX_PREDICTION_FRAMES  # Número de frames para activar predicción
SKIP_FRAME_THRESHOLD: float = OPTIMIZATION_PREDICTION_THRESHOLD  # Tiempo límite para saltar detección (segundos)
MAX_SKIPPABLE_FRAMES: int = 5  # Máximo de frames consecutivos que se pueden saltar
PREDICTION_SMOOTHING: float = 0.7  # Factor de suavizado para predicción (0-1)

# Índice del punto de la punta del dedo índice
INDEX_FINGER_TIP: int = 8
THUMB_TIP: int = 4  # Índice de la punta del pulgar
PINCH_DISTANCE_THRESHOLD: float = 0.05  # Umbral relativo al tamaño de la mano
MIN_PINCH_DISTANCE_PX: int = 10  # Distancia mínima en píxeles para el gesto de pinza
PINKY_FINGER_TIP: int = 20  # Índice de la punta del dedo meñique

class DetectorManos:
    """
    Clase para detectar y seguir manos utilizando MediaPipe Hands.
    """

    def __init__(self,
                max_num_hands: int = MEDIAPIPE_MAX_HANDS,
                min_detection_confidence: float = MEDIAPIPE_DETECTION_CONFIDENCE,
                min_tracking_confidence: float = MEDIAPIPE_TRACKING_CONFIDENCE,
                solo_mano_derecha: bool = False,
                mano_izquierda: bool = MEDIAPIPE_MANO_IZQUIERDA,
                enable_optimizations: bool = True,
                mirror_mode: bool = True):
        """
        Inicializa el detector de manos con los parámetros especificados.

        Args:
            max_num_hands: Número máximo de manos a detectar
            min_detection_confidence: Confianza mínima para detección
            min_tracking_confidence: Confianza mínima para seguimiento
            solo_mano_derecha: Si es True, solo detecta la mano derecha
            mano_izquierda: Si es True, detecta la mano izquierda en lugar de la derecha
            enable_optimizations: Si es True, aplica optimizaciones de rendimiento
            mirror_mode: Si es True, la imagen está en modo espejo, lo que invierte la detección de manos
        """
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.max_num_hands = max_num_hands
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.solo_mano_derecha = solo_mano_derecha
        self.mano_izquierda = mano_izquierda
        self.enable_optimizations = enable_optimizations
        self.mirror_mode = mirror_mode

        # Configurar model_complexity según optimizaciones
        model_complexity = 0 if enable_optimizations else 1

        # Inicializar detector de manos de MediaPipe
        # Nota: MediaPipe Hands no permite filtrar por tipo de mano (derecha/izquierda) directamente
        # en la inicialización. El filtrado se hace en el método _filtrar_mano_correcta
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            model_complexity=model_complexity
        )

        # Métricas de rendimiento
        self.metricas = {
            "tiempo_deteccion_ms": 0.0,
            "predicciones_totales": 0,
            "ultimo_tiempo_deteccion": time.time()
        }

        # Resultados guardados para predicción y velocidad
        self.ultimos_resultados = None
        self.ultima_velocidad = (0.0, 0.0)  # Inicializar velocidad

        # Inicializar variables para historial y predicción
        self.historial_manos = deque(maxlen=PREDICTION_FRAMES+1)
        self.historia_velocidades = deque(maxlen=PREDICTION_FRAMES)
        self.frames_sin_deteccion = 0

        # Variables para métricas
        self.tiempo_inicio_metricas = time.time()
        self.num_detecciones_completas = 0
        self.num_detecciones_predichas = 0
        self.tiempo_total_deteccion = 0.0

        logger.info(f"Detector de manos inicializado con optimizaciones: {enable_optimizations}, "
                   f"solo mano derecha: {solo_mano_derecha}, "
                   f"mano izquierda: {mano_izquierda}, "
                   f"modo espejo: {mirror_mode}")

    def _filtrar_mano_correcta(self, results) -> List[Any]:
        """
        Filtra los resultados para obtener solo la mano correcta según la configuración.
        Cuando el modo espejo está activado, invierte la detección de manos (izquierda/derecha).

        Args:
            results: Resultados de la detección de manos

        Returns:
            Lista con la mano correcta, o lista vacía si no se encontró
        """
        if not results.multi_handedness or not results.multi_hand_landmarks:
            return []

        manos_filtradas = []

        for idx, handedness in enumerate(results.multi_handedness):
            # Obtener si es mano derecha o izquierda según MediaPipe
            es_derecha_segun_mediapipe = handedness.classification[0].label == "Right"

            # IMPORTANTE: En modo espejo, las manos se invierten
            # La "Right" que detecta MediaPipe en realidad es la mano izquierda del usuario
            # La "Left" que detecta MediaPipe en realidad es la mano derecha del usuario
            if self.mirror_mode:
                es_derecha_real = not es_derecha_segun_mediapipe
            else:
                es_derecha_real = es_derecha_segun_mediapipe

            logger.debug(f"Mano detectada: MediaPipe dice {'derecha' if es_derecha_segun_mediapipe else 'izquierda'}, "
                        f"pero en realidad es {'derecha' if es_derecha_real else 'izquierda'} del usuario (mirror={self.mirror_mode})")

            # Determinar si debemos incluir esta mano
            if self.solo_mano_derecha:
                # Solo incluir si es la mano derecha real del usuario
                if es_derecha_real:
                    manos_filtradas.append(results.multi_hand_landmarks[idx])
            elif self.mano_izquierda:
                # Solo incluir si es la mano izquierda real del usuario
                if not es_derecha_real:
                    manos_filtradas.append(results.multi_hand_landmarks[idx])
            else:
                # Incluir todas las manos
                manos_filtradas.append(results.multi_hand_landmarks[idx])

        return manos_filtradas

    def procesar_fotograma(self,
                          frame: np.ndarray,
                          dibujar_landmarks: bool = DEBUG_DRAW_LANDMARKS,
                          usar_prediccion: bool = False) -> Tuple[np.ndarray, List[Any]]:
        """
        Procesa un fotograma para detectar manos.

        Args:
            frame: Fotograma a procesar (formato BGR)
            dibujar_landmarks: Si es True, dibuja los landmarks de las manos
            usar_prediccion: Si es True, usa predicción en caso de no detectar

        Returns:
            Tupla con el fotograma procesado y la lista de manos detectadas
        """
        if frame is None:
            logger.warning("Frame nulo recibido por el detector")
            return frame, []

        # Convertir imagen a RGB para MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # type: ignore

        # Guardar hora de inicio para métricas
        tiempo_inicio = time.time()

        # Flag para saber si estamos usando predicción
        usando_prediccion = False

        # Realizar detección
        if usar_prediccion and self.ultimos_resultados is not None:
            # Si usar_prediccion es True y tenemos resultados previos, usamos esos
            results = self.ultimos_resultados
            usando_prediccion = True
            self.metricas["predicciones_totales"] += 1
        else:
            # Procesar imagen con MediaPipe
            results = self.hands.process(rgb_frame)
            # Guardar resultados para posible predicción futura
            if results.multi_hand_landmarks:
                self.ultimos_resultados = results

        # Actualizar métricas
        tiempo_fin = time.time()
        self.metricas["tiempo_deteccion_ms"] = (tiempo_fin - tiempo_inicio) * 1000
        self.metricas["ultimo_tiempo_deteccion"] = tiempo_fin

        # Obtener solo las manos que nos interesan
        manos_filtradas = self._filtrar_mano_correcta(results)

        # Dibujar landmarks si es necesario
        if dibujar_landmarks and manos_filtradas:
            frame_copy = frame.copy()
            for hand_landmarks in manos_filtradas:
                # Dibujar landmarks
                if DEBUG_DRAW_LANDMARKS:
                    self.mp_drawing.draw_landmarks(
                        frame_copy,
                        hand_landmarks,
                        self.mp_hands.HAND_CONNECTIONS,
                        self.mp_drawing_styles.get_default_hand_landmarks_style(),
                        self.mp_drawing_styles.get_default_hand_connections_style()
                    )

            # Superponer landmarks con transparencia
            alpha = 0.7
            frame = cv2.addWeighted(frame_copy, alpha, frame, 1 - alpha, 0)  # type: ignore

            # Si estamos usando predicción, añadir indicador visual
            if usando_prediccion:
                cv2.putText(  # type: ignore
                    frame,
                    "PRED",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,  # type: ignore
                    0.7,
                    (0, 0, 255),
                    2
                )

        return frame, manos_filtradas

    def _predecir_posiciones(self) -> Optional[List[Dict[str, Any]]]:
        """
        Predice la posición de las manos basándose en detecciones anteriores.

        Esta función utiliza la velocidad calculada a partir de detecciones
        anteriores para predecir dónde estarán las manos en el fotograma actual.

        Returns:
            Lista con predicciones de posiciones de manos o None si no hay suficiente historial.
        """
        if not self.ultimos_resultados or not self.ultimos_resultados.multi_hand_landmarks:
            return None

        # Obtener la última detección conocida
        ultima_deteccion = self.ultimos_resultados.multi_hand_landmarks

        # Calcular el tiempo transcurrido desde la última detección real
        tiempo_actual = time.time()
        delta_t = tiempo_actual - self.metricas["ultimo_tiempo_deteccion"]

        if delta_t <= 0:
            return ultima_deteccion

        # Crear una copia profunda para la predicción
        manos_predichas = []

        for mano in ultima_deteccion:
            nueva_mano = {"landmarks": [], "timestamp": tiempo_actual}

            # Predecir la posición de cada landmark basado en la velocidad
            for i, landmark in enumerate(mano):
                # Obtener la velocidad del punto u opcionalmente la velocidad promedio
                vx, vy = self.ultima_velocidad

                # Predecir la nueva posición
                nuevo_x = landmark.x + int(vx * delta_t)
                nuevo_y = landmark.y + int(vy * delta_t)

                # Añadir el landmark predicho
                nueva_mano["landmarks"].append({
                    "x": nuevo_x,
                    "y": nuevo_y,
                    "z": landmark.z,
                    "index": i
                })

            manos_predichas.append(nueva_mano)

        return manos_predichas

    def _actualizar_historial(self, manos: Optional[List[Dict[str, Any]]]) -> None:
        """
        Actualiza el historial de detecciones de manos.

        Args:
            manos: Lista de manos detectadas en el fotograma actual.
        """
        self.historial_manos.append(manos)

    def _actualizar_velocidades(self) -> None:
        """
        Calcula la velocidad de movimiento de las manos entre fotogramas.

        Esta función analiza las últimas posiciones conocidas y calcula
        la velocidad de los puntos clave para usarla en predicciones.
        """
        # Necesitamos al menos dos detecciones para calcular velocidad
        if len(self.historial_manos) < 2:
            return

        # Obtener las dos últimas detecciones válidas
        ultima = None
        penultima = None

        # Recorrer el historial desde el final hacia el principio
        for det in reversed(self.historial_manos):
            if det is not None:
                if ultima is None:
                    ultima = det
                elif penultima is None:
                    penultima = det
                    break

        if ultima is None or penultima is None:
            return

        # Asegurarse de que ambas detecciones tengan manos
        if not ultima or not penultima:
            return

        # Para simplificar, usamos la primera mano en cada detección
        if len(ultima) == 0 or len(penultima) == 0:
            return

        ultima_mano = ultima[0]
        penultima_mano = penultima[0]

        # Calcular el delta de tiempo entre las detecciones
        delta_t = ultima_mano["timestamp"] - penultima_mano["timestamp"]
        if delta_t <= 0:
            return

        # Calcular la velocidad promedio de todos los puntos
        # (podríamos refinar esto para usar solo puntos específicos)
        suma_vx = 0.0
        suma_vy = 0.0
        num_puntos = 0

        # Usar solo puntos claves específicos (por ejemplo, la punta del dedo índice)
        # que son más relevantes para el seguimiento
        indices_puntos_clave = [0, 8, 12, 16, 20]  # Base de la mano y puntas de dedos

        for idx in indices_puntos_clave:
            if (idx < len(ultima_mano["landmarks"]) and
                idx < len(penultima_mano["landmarks"])):

                punto_actual = ultima_mano["landmarks"][idx]
                punto_anterior = penultima_mano["landmarks"][idx]

                dx = punto_actual["x"] - punto_anterior["x"]
                dy = punto_actual["y"] - punto_anterior["y"]

                vx = dx / delta_t
                vy = dy / delta_t

                suma_vx += vx
                suma_vy += vy
                num_puntos += 1

        if num_puntos > 0:
            promedio_vx = suma_vx / num_puntos
            promedio_vy = suma_vy / num_puntos

            # Suavizar la velocidad para evitar cambios bruscos
            if not self.historia_velocidades:
                self.ultima_velocidad = (promedio_vx, promedio_vy)
            else:
                # Aplicar filtro de media móvil exponencial
                vx_anterior, vy_anterior = self.ultima_velocidad
                nueva_vx = PREDICTION_SMOOTHING * promedio_vx + (1 - PREDICTION_SMOOTHING) * vx_anterior
                nueva_vy = PREDICTION_SMOOTHING * promedio_vy + (1 - PREDICTION_SMOOTHING) * vy_anterior
                self.ultima_velocidad = (nueva_vx, nueva_vy)

            # Actualizar historial de velocidades
            self.historia_velocidades.append(self.ultima_velocidad)

    def _dibujar_landmarks(self, frame: np.ndarray, hand_landmarks: Any) -> None:
        """
        Dibuja los puntos de referencia (landmarks) de las manos en el fotograma
        usando las utilidades de MediaPipe.

        Args:
            frame: Fotograma donde dibujar los landmarks.
            hand_landmarks: Landmarks de la mano de MediaPipe.
        """
        self.mp_drawing.draw_landmarks(
            frame,
            hand_landmarks,
            self.mp_hands.HAND_CONNECTIONS,
            self.mp_drawing_styles.get_default_hand_landmarks_style(),
            self.mp_drawing_styles.get_default_hand_connections_style()
        )

    def _dibujar_landmarks_manual(self, frame: np.ndarray, landmarks: List[Dict[str, Any]]) -> None:
        """
        Dibuja manualmente los landmarks y conexiones en el frame.

        Args:
            frame: Frame donde dibujar los landmarks.
            landmarks: Lista de landmarks a dibujar.
        """
        # Dibujar las conexiones entre landmarks (simplificado)
        conexiones = [
            (0, 1), (1, 2), (2, 3), (3, 4),  # Pulgar
            (0, 5), (5, 6), (6, 7), (7, 8),  # Índice
            (0, 9), (9, 10), (10, 11), (11, 12),  # Medio
            (0, 13), (13, 14), (14, 15), (15, 16),  # Anular
            (0, 17), (17, 18), (18, 19), (19, 20)  # Meñique
        ]

        # Dibujar las conexiones
        for conexion in conexiones:
            start_idx, end_idx = conexion
            if start_idx < len(landmarks) and end_idx < len(landmarks):
                start_point = (landmarks[start_idx]["x"], landmarks[start_idx]["y"])
                end_point = (landmarks[end_idx]["x"], landmarks[end_idx]["y"])
                cv2.line(frame, start_point, end_point, (0, 255, 0), 2)  # type: ignore

        # Resaltar la punta del dedo índice
        if INDEX_FINGER_TIP < len(landmarks):
            point = (landmarks[INDEX_FINGER_TIP]["x"], landmarks[INDEX_FINGER_TIP]["y"])
            cv2.circle(frame, point, 8, (255, 0, 0), -1)  # type: ignore

    def obtener_punta_indice(self, mano) -> Optional[Tuple[int, int]]:
        """
        Obtiene las coordenadas de la punta del dedo índice.

        Args:
            mano: Objeto NormalizedLandmarkList de MediaPipe con los landmarks de la mano.

        Returns:
            Tupla con las coordenadas (x, y) o None si no está disponible.
        """
        return self.obtener_coordenada_punto(mano, INDEX_FINGER_TIP)

    def obtener_coordenada_punto(self, mano, indice: int) -> Optional[Tuple[int, int]]:
        """
        Obtiene las coordenadas de un punto específico de la mano.

        Args:
            mano: Objeto NormalizedLandmarkList de MediaPipe con los landmarks de la mano.
            indice: Índice del punto a obtener.

        Returns:
            Tupla con las coordenadas (x, y) del punto o None si no está disponible.
        """
        try:
            if mano is None:
                return None

            # Verificar si tenemos suficientes landmarks
            if indice >= len(mano.landmark):
                logger.warning(f"Índice de landmark {indice} fuera de rango (max: {len(mano.landmark)-1})")
                return None

            # Obtener el landmark normalizado
            landmark = mano.landmark[indice]

            # Convertir coordenadas normalizadas a píxeles absolutos
            # Nota: Estas coordenadas están en el rango [0,1], por lo que necesitamos
            # multiplicarlas por las dimensiones del frame para obtener los píxeles
            # Para simplificar, asumimos resolución de cámara
            h, w = CAMERA_HEIGHT, CAMERA_WIDTH
            x = int(landmark.x * w)
            y = int(landmark.y * h)

            return (x, y)

        except (AttributeError, IndexError) as e:
            logger.debug(f"Error al obtener coordenada de punto {indice}: {e}")
        return None

    def es_indice_extendido(self, mano) -> bool:
        """
        Determina si el dedo índice está extendido.

        Args:
            mano: Objeto NormalizedLandmarkList de MediaPipe con los landmarks de la mano.

        Returns:
            True si el dedo índice está extendido, False en caso contrario.
        """
        try:
            if mano is None or not hasattr(mano, 'landmark'):
                return False

            # Verificar si tenemos suficientes landmarks
            if len(mano.landmark) < 9:  # Necesitamos hasta el punto 8 (punta del índice)
                logger.debug("DEBUG-DEDO: No hay suficientes landmarks para determinar estado del índice")
                return False

            # Obtener puntos clave normalizados
            punta_indice = mano.landmark[8]  # Punta del dedo índice
            medio_indice = mano.landmark[7]  # Articulación media del dedo índice
            base_indice = mano.landmark[5]   # Base del dedo índice

            # DEBUG: Registrar coordenadas
            logger.debug(f"DEBUG-DEDO: Punta índice (8): ({punta_indice.x:.4f}, {punta_indice.y:.4f})")
            logger.debug(f"DEBUG-DEDO: Medio índice (7): ({medio_indice.x:.4f}, {medio_indice.y:.4f})")
            logger.debug(f"DEBUG-DEDO: Base índice (5): ({base_indice.x:.4f}, {base_indice.y:.4f})")

            # Calcular si el dedo está extendido
            # Criterio: El dedo está extendido si la punta está más arriba (y menor) que la base
            esta_extendido = punta_indice.y < base_indice.y

            # MEJORA: Añadir criterio adicional - distancia vertical significativa
            distancia_y = abs(punta_indice.y - base_indice.y)
            umbral_distancia = 0.1  # Coordenadas normalizadas (0-1)

            distancia_suficiente = distancia_y > umbral_distancia

            # DEBUG: Registrar cálculos y resultado
            logger.debug(f"DEBUG-DEDO: Criterio y: {punta_indice.y:.4f} < {base_indice.y:.4f} = {punta_indice.y < base_indice.y}")
            logger.debug(f"DEBUG-DEDO: Distancia Y: {distancia_y:.4f}, Umbral: {umbral_distancia:.4f}, Suficiente: {distancia_suficiente}")
            logger.debug(f"DEBUG-DEDO: Resultado final: {esta_extendido and distancia_suficiente}")

            # Devolver el resultado considerando ambos criterios
            return esta_extendido and distancia_suficiente

        except (AttributeError, IndexError) as e:
            logger.debug(f"Error al determinar si el índice está extendido: {e}")
            return False

    def es_menique_extendido(self, mano) -> bool:
        """
        Determina si el dedo meñique está extendido.

        Args:
            mano: Objeto NormalizedLandmarkList de MediaPipe con los landmarks de la mano.

        Returns:
            True si el dedo meñique está extendido, False en caso contrario.
        """
        try:
            if mano is None or not hasattr(mano, 'landmark'):
                return False

            # Verificar si tenemos suficientes landmarks
            if len(mano.landmark) < 21:  # Necesitamos hasta el punto 20 (punta del meñique)
                logger.debug("DEBUG-DEDO: No hay suficientes landmarks para determinar estado del meñique")
                return False

            # Obtener puntos clave normalizados
            punta_menique = mano.landmark[20]  # Punta del dedo meñique
            medio_menique = mano.landmark[19]  # Articulación media del dedo meñique
            base_menique = mano.landmark[17]   # Base del dedo meñique

            # DEBUG: Registrar coordenadas
            logger.debug(f"DEBUG-MENIQUE: Punta meñique (20): ({punta_menique.x:.4f}, {punta_menique.y:.4f})")
            logger.debug(f"DEBUG-MENIQUE: Medio meñique (19): ({medio_menique.x:.4f}, {medio_menique.y:.4f})")
            logger.debug(f"DEBUG-MENIQUE: Base meñique (17): ({base_menique.x:.4f}, {base_menique.y:.4f})")

            # Calcular si el dedo está extendido
            # Criterio: El dedo está extendido si la punta está más arriba (y menor) que la base
            esta_extendido = punta_menique.y < base_menique.y

            # MEJORA: Añadir criterio adicional - distancia vertical significativa
            distancia_y = abs(punta_menique.y - base_menique.y)
            umbral_distancia = 0.1  # Coordenadas normalizadas (0-1)

            distancia_suficiente = distancia_y > umbral_distancia

            # Verificación adicional: Los otros dedos (índice, medio, anular) no deben estar extendidos
            # Esto hace el gesto más específico y reduce falsos positivos
            indice_retraido = mano.landmark[8].y > mano.landmark[5].y  # Punta índice más baja que base
            medio_retraido = mano.landmark[12].y > mano.landmark[9].y  # Punta medio más baja que base
            anular_retraido = mano.landmark[16].y > mano.landmark[13].y  # Punta anular más baja que base

            gesto_especifico = indice_retraido and medio_retraido and anular_retraido

            # DEBUG: Registrar cálculos y resultado
            logger.debug(f"DEBUG-MENIQUE: Criterio y: {punta_menique.y:.4f} < {base_menique.y:.4f} = {punta_menique.y < base_menique.y}")
            logger.debug(f"DEBUG-MENIQUE: Distancia Y: {distancia_y:.4f}, Umbral: {umbral_distancia:.4f}, Suficiente: {distancia_suficiente}")
            logger.debug(f"DEBUG-MENIQUE: Otros dedos retraídos: {gesto_especifico}")
            logger.debug(f"DEBUG-MENIQUE: Resultado final: {esta_extendido and distancia_suficiente and gesto_especifico}")

            # Devolver el resultado considerando todos los criterios
            return esta_extendido and distancia_suficiente and gesto_especifico

        except (AttributeError, IndexError) as e:
            logger.debug(f"Error al determinar si el meñique está extendido: {e}")
            return False

    def es_gesto_pinza(self, mano) -> bool:
        """
        Determina si los dedos pulgar e índice están formando un gesto de pinza.

        Este gesto se define como la cercanía entre las puntas del pulgar y el dedo índice,
        con una distancia menor que un umbral proporcional al tamaño de la mano.

        Args:
            mano: Objeto NormalizedLandmarkList de MediaPipe con los landmarks de la mano.

        Returns:
            True si el gesto es de pinza, False en caso contrario.
        """
        try:
            if mano is None or not hasattr(mano, 'landmark'):
                return False

            # Verificar si tenemos suficientes landmarks
            if len(mano.landmark) < max(THUMB_TIP, INDEX_FINGER_TIP) + 1:
                logger.debug("No hay suficientes landmarks para detectar gesto de pinza")
                return False

            # Obtener coordenadas de las puntas de pulgar e índice (normalizadas)
            thumb_tip = mano.landmark[THUMB_TIP]  # Pulgar
            index_tip = mano.landmark[INDEX_FINGER_TIP]  # Índice

            # Log para debugging
            logger.debug(f"Punta pulgar: ({thumb_tip.x:.4f}, {thumb_tip.y:.4f}), Punta índice: ({index_tip.x:.4f}, {index_tip.y:.4f})")

            # Calcular distancia entre puntas (en coordenadas normalizadas)
            dx = thumb_tip.x - index_tip.x
            dy = thumb_tip.y - index_tip.y
            distancia = np.sqrt(dx*dx + dy*dy)  # Distancia euclidiana

            # Método mejorado para estimar el tamaño de la mano (en coordenadas normalizadas)
            # Usamos múltiples medidas para obtener una estimación más robusta

            # 1. Distancia entre la base de la mano y base del dedo medio
            base_mano = mano.landmark[0]  # Muñeca
            base_medio = mano.landmark[9]  # Base del dedo medio
            d1 = np.sqrt((base_mano.x - base_medio.x)**2 + (base_mano.y - base_medio.y)**2)

            # 2. Distancia entre la base y punta del dedo medio
            base_medio = mano.landmark[9]
            punta_medio = mano.landmark[12]
            d2 = np.sqrt((base_medio.x - punta_medio.x)**2 + (base_medio.y - punta_medio.y)**2)

            # 3. Ancho de la palma (distancia entre base del índice y meñique)
            base_indice = mano.landmark[5]
            base_menique = mano.landmark[17]
            d3 = np.sqrt((base_indice.x - base_menique.x)**2 + (base_indice.y - base_menique.y)**2)

            # Combinamos las medidas para obtener una estimación más robusta
            tamano_mano = (d1 + d2 + d3) / 3

            # Log para debugging
            logger.debug(f"Tamaño mano (normalizado): {tamano_mano:.4f} (d1={d1:.4f}, d2={d2:.4f}, d3={d3:.4f})")

            # Calcular umbral relativo al tamaño de la mano
            umbral_relativo = tamano_mano * PINCH_DISTANCE_THRESHOLD

            # Convertir umbral mínimo de píxeles a coordenadas normalizadas (aproximado)
            umbral_min_normalizado = MIN_PINCH_DISTANCE_PX / max(CAMERA_WIDTH, CAMERA_HEIGHT)

            # Usar el mayor de los dos umbrales
            umbral = max(umbral_relativo, umbral_min_normalizado)

            # Determinar si es gesto de pinza (distancia menor que umbral)
            es_pinza = distancia < umbral

            # Log detallado para debugging
            logger.debug(f"PINZA: {es_pinza} (distancia: {distancia:.4f}, umbral: {umbral:.4f}, umbral_rel: {umbral_relativo:.4f})")

            # Verificaciones adicionales para que el gesto sea más específico
            if es_pinza:
                # Las puntas de los dedos pulgar e índice deben estar más altas que la muñeca
                pulgar_mas_alto = thumb_tip.y < base_mano.y
                indice_mas_alto = index_tip.y < base_mano.y

                # Las puntas deben estar en una posición razonable (no demasiado bajas)
                # Esto ayuda a distinguir cuando la mano está cerrada vs. gesto de pinza
                if not (pulgar_mas_alto and indice_mas_alto):
                    logger.debug("PINZA descartada: pulgar e índice deben estar más altos que la muñeca")
                    es_pinza = False

            return es_pinza

        except (AttributeError, IndexError) as e:
            logger.debug(f"Error al determinar gesto de pinza: {e}")
            return False

    def obtener_metricas(self) -> Dict[str, Any]:
        """
        Obtiene métricas de rendimiento del detector.

        Returns:
            Diccionario con métricas de rendimiento.
        """
        tiempo_total = time.time() - self.tiempo_inicio_metricas
        porcentaje_predicciones = 0
        tiempo_promedio = 0

        total_detecciones = self.num_detecciones_completas + self.num_detecciones_predichas

        if total_detecciones > 0:
            porcentaje_predicciones = (self.num_detecciones_predichas / total_detecciones) * 100
            tiempo_promedio = self.tiempo_total_deteccion / total_detecciones

        return {
            "tiempo_deteccion_ms": self.tiempo_total_deteccion * 1000,
            "detecciones_completas": self.num_detecciones_completas,
            "predicciones_totales": self.num_detecciones_predichas,
            "porcentaje_predicciones": porcentaje_predicciones,
            "tiempo_promedio_ms": tiempo_promedio * 1000,
            "frames_sin_deteccion": self.frames_sin_deteccion,
            "velocidad_actual": self.ultima_velocidad
        }

    def liberar_recursos(self) -> None:
        """
        Libera los recursos utilizados por el detector de manos.
        """
        self.hands.close()
        logger.info("Recursos del detector de manos liberados")
