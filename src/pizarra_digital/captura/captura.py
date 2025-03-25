"""
Módulo para la captura de video mediante OpenCV.

Este módulo proporciona funciones para inicializar la cámara web,
capturar fotogramas y gestionar los recursos de la cámara.
"""
import logging
import cv2
import numpy as np
import threading
import time
import queue
from typing import Tuple, Optional, Dict, Any, List

from ..config import CAMERA_INDEX, CAMERA_WIDTH, CAMERA_HEIGHT

# Configuración del logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CameraError(Exception):
    """Excepción personalizada para errores relacionados con la cámara."""
    pass

class CapturaAsincrona:
    """
    Clase para capturar frames de la cámara en un hilo separado.

    Esta implementación permite que la captura de la cámara se ejecute en segundo plano,
    evitando bloqueos en el hilo principal y proporcionando el frame más reciente
    cuando sea necesario.
    """

    def __init__(self,
                camera_index: int = CAMERA_INDEX,
                buffer_size: int = 1,
                target_fps: int = 30,
                max_queue_size: int = 2) -> None:
        """
        Inicializa el sistema de captura asíncrona de la cámara.

        Args:
            camera_index: Índice de la cámara a utilizar.
            buffer_size: Tamaño del buffer de la cámara.
            target_fps: FPS objetivo para la captura.
            max_queue_size: Tamaño máximo de la cola de frames.
        """
        self.camera_index = camera_index
        self.buffer_size = buffer_size
        self.target_fps = target_fps

        # Cola para almacenar los frames capturados
        self.frames_queue = queue.Queue(maxsize=max_queue_size)

        # Variables de control
        self.running = False
        self.cap = None
        self.thread = None

        # Estadísticas de rendimiento
        self.fps_actual = 0.0
        self.frames_procesados = 0
        self.tiempo_ultimo_calculo_fps = time.time()

        logger.info("Inicializando sistema de captura asíncrona...")

    def iniciar(self) -> bool:
        """
        Inicia el hilo de captura de la cámara.

        Returns:
            bool: True si se inició correctamente, False en caso contrario.
        """
        try:
            # Intentar abrir la cámara
            self.cap = self._inicializar_camara()
            if self.cap is None:
                return False

            # Configurar y empezar el hilo
            self.running = True
            self.thread = threading.Thread(target=self._capture_loop, daemon=True)
            self.thread.start()
            logger.info("Hilo de captura de cámara iniciado correctamente")
            return True

        except Exception as e:
            logger.error(f"Error al iniciar la captura asíncrona: {e}")
            self.detener()
            return False

    def _inicializar_camara(self) -> Optional[cv2.VideoCapture]:
        """
        Inicializa la cámara probando diferentes índices.

        Returns:
            VideoCapture: Objeto de captura de video configurado o None si falla.
        """
        # Lista de índices de cámara a intentar, empezando por el configurado
        indices_camara = [self.camera_index, 0, 1, 2]
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
                    cap.set(cv2.CAP_PROP_FPS, self.target_fps)
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, self.buffer_size)
                    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))  # MJPG suele ser más rápido

                    # Leer un fotograma de prueba para asegurarse de que funciona
                    ret, _ = cap.read()
                    if not ret:
                        logger.warning(f"Se pudo abrir la cámara {indice} pero no se pudo leer un fotograma")
                        cap.release()
                        continue

                    # Registrar que la cámara se ha inicializado correctamente
                    actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                    actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                    self.camera_index = indice  # Actualizar el índice que funcionó
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
        return None

    def _capture_loop(self) -> None:
        """
        Bucle principal del hilo de captura que constantemente lee frames de la cámara
        y actualiza la cola con el frame más reciente.
        """
        ultima_captura = time.time()
        periodo_frame = 1.0 / self.target_fps

        while self.running and self.cap is not None:
            try:
                # Control de frecuencia para no saturar el CPU
                tiempo_actual = time.time()
                tiempo_transcurrido = tiempo_actual - ultima_captura

                if tiempo_transcurrido < periodo_frame:
                    # Esperar para mantener la frecuencia objetivo
                    time.sleep(periodo_frame - tiempo_transcurrido)

                # Capturar frame
                ret, frame = self.cap.read()
                ultima_captura = time.time()

                if not ret or frame is None:
                    logger.warning("No se pudo leer el fotograma de la cámara en el hilo de captura")
                    time.sleep(0.01)  # Pequeña pausa antes de reintentar
                    continue

                # Voltear horizontalmente (efecto espejo)
                frame = cv2.flip(frame, 1)

                # Actualizar estadísticas
                self.frames_procesados += 1
                ahora = time.time()
                if ahora - self.tiempo_ultimo_calculo_fps >= 1.0:
                    self.fps_actual = self.frames_procesados / (ahora - self.tiempo_ultimo_calculo_fps)
                    self.frames_procesados = 0
                    self.tiempo_ultimo_calculo_fps = ahora

                # Si la cola está llena, sacar el frame más antiguo
                if self.frames_queue.full():
                    try:
                        self.frames_queue.get_nowait()
                    except queue.Empty:
                        pass  # La cola ya no está llena

                # Poner el nuevo frame en la cola
                self.frames_queue.put(frame, block=False)

            except queue.Full:
                # Si la cola está llena, simplemente continuamos
                pass
            except Exception as e:
                logger.error(f"Error en el bucle de captura: {e}")
                time.sleep(0.1)  # Pausa antes de reintentar

    def obtener_frame(self, escala: float = 1.0) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Obtiene el frame más reciente de la cola.

        Args:
            escala: Factor de escala para redimensionar el frame (1.0 = tamaño original).
                   Valores menores mejoran el rendimiento.

        Returns:
            Tupla con (éxito, frame). Si no hay frames disponibles, éxito será False.
        """
        if not self.running or self.cap is None:
            return False, None

        try:
            # Intentar obtener el frame más reciente sin bloquear
            frame = self.frames_queue.get_nowait()

            # Redimensionar si se solicita
            if escala != 1.0 and frame is not None:
                ancho = int(frame.shape[1] * escala)
                alto = int(frame.shape[0] * escala)
                frame = cv2.resize(frame, (ancho, alto), interpolation=cv2.INTER_AREA)

            return True, frame

        except queue.Empty:
            # No hay frames disponibles
            return False, None
        except Exception as e:
            logger.error(f"Error al obtener frame: {e}")
            return False, None

    def obtener_fps(self) -> float:
        """
        Obtiene los FPS actuales de captura.

        Returns:
            float: Frames por segundo actuales.
        """
        return self.fps_actual

    def ajustar_parametros(self, parametros: Dict[str, Any]) -> bool:
        """
        Ajusta los parámetros de la cámara.

        Args:
            parametros: Diccionario con los parámetros a ajustar.

        Returns:
            bool: True si se aplicó al menos un parámetro correctamente.
        """
        if not self.running or self.cap is None:
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
                result = self.cap.set(prop, valor)
                if result:
                    logger.info(f"Parámetro '{nombre}' configurado a {valor}")
                    exito = True
                    # Actualizar el valor interno si es un parámetro que almacenamos
                    if nombre == 'fps':
                        self.target_fps = valor
                    elif nombre == 'buffer':
                        self.buffer_size = valor
                else:
                    logger.warning(f"No se pudo configurar el parámetro '{nombre}'")

        return exito

    def detener(self) -> None:
        """
        Detiene el hilo de captura y libera los recursos de la cámara.
        """
        self.running = False

        # Esperar a que termine el hilo
        if self.thread is not None and self.thread.is_alive():
            self.thread.join(timeout=1.0)

        # Liberar la cámara
        if self.cap is not None:
            self.cap.release()
            self.cap = None

        logger.info("Sistema de captura asíncrona detenido correctamente")

# Mantener las funciones originales para compatibilidad
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
