"""
Módulo para la captura de video desde la cámara.

Este módulo proporciona clases y funciones para capturar video
desde la cámara del sistema, con soporte para captura síncrona
y asíncrona para mejorar el rendimiento.
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

class CapturaVideo:
    """Clase para la captura de video síncrona desde la cámara."""

    def __init__(self, camara_index: int = 0, width: int = 640, height: int = 480) -> None:
        """
        Inicializa la captura de video con la cámara especificada.

        Args:
            camara_index: Índice de la cámara (0 para la cámara predeterminada).
            width: Ancho deseado para los frames capturados.
            height: Alto deseado para los frames capturados.
        """
        self.camara_index = camara_index
        self.width = width
        self.height = height
        self.cap = None
        self.inicializar()

        # Métricas de rendimiento
        self.frames_capturados = 0
        self.tiempo_ultimo_frame = 0.0
        self.tiempo_inicio = time.time()

        logger.info(f"CapturaVideo inicializada: cámara={camara_index}, "
                  f"resolución={width}x{height}")

    def inicializar(self) -> None:
        """Inicializa la captura de video con la configuración especificada."""
        try:
            self.cap = cv2.VideoCapture(self.camara_index)

            # Configurar propiedades de la cámara
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

            if not self.cap.isOpened():
                logger.error(f"No se pudo abrir la cámara {self.camara_index}")
                raise RuntimeError(f"No se pudo abrir la cámara {self.camara_index}")

            logger.info("Captura de video inicializada correctamente")
        except Exception as e:
            logger.error(f"Error al inicializar la cámara: {e}")
            raise

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Lee un frame de la cámara.

        Returns:
            Tupla con (éxito, frame):
            - éxito: True si se pudo leer el frame correctamente.
            - frame: Array de NumPy con el frame capturado o None si hubo error.
        """
        if self.cap is None or not self.cap.isOpened():
            logger.warning("Se intentó leer de una cámara no inicializada")
            return False, None

        # Capturar frame
        ret, frame = self.cap.read()

        # Actualizar métricas
        if ret:
            self.frames_capturados += 1
            self.tiempo_ultimo_frame = time.time()

        return ret, frame

    def release(self) -> None:
        """Libera los recursos de la cámara."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
            logger.info("Recursos de la cámara liberados")

    def obtener_metricas(self) -> Dict[str, Any]:
        """
        Obtiene métricas de rendimiento de la captura.

        Returns:
            Diccionario con métricas de rendimiento.
        """
        tiempo_total = time.time() - self.tiempo_inicio
        fps = self.frames_capturados / max(0.001, tiempo_total)

        return {
            "frames_capturados": self.frames_capturados,
            "tiempo_total": tiempo_total,
            "fps_promedio": fps,
            "camara_index": self.camara_index,
            "resolucion": f"{self.width}x{self.height}"
        }

    def __del__(self) -> None:
        """Destructor para asegurar que se liberen los recursos."""
        self.release()


class CapturaVideoAsync(CapturaVideo):
    """
    Clase para captura de video asíncrona (en un hilo separado).

    Esta implementación mejora el rendimiento al capturar frames
    continuamente en un hilo separado, evitando el bloqueo
    del hilo principal mientras se espera por un nuevo frame.
    """

    def __init__(self, camara_index: int = 0, width: int = 640,
                height: int = 480, fps: int = 30,
                buffer_size: int = 3) -> None:
        """
        Inicializa la captura de video asíncrona.

        Args:
            camara_index: Índice de la cámara a utilizar.
            width: Ancho deseado para los frames.
            height: Alto deseado para los frames.
            fps: Frames por segundo objetivo para la captura.
            buffer_size: Tamaño del buffer de frames.
        """
        super().__init__(camara_index, width, height)

        # Configuración específica para captura asíncrona
        self.fps_objetivo = fps
        self.buffer_size = buffer_size

        # Cola para almacenar frames capturados
        self.queue = queue.Queue(maxsize=buffer_size)

        # Variables para el hilo de captura
        self.thread = None
        self.stopped = False

        # Métricas adicionales para captura asíncrona
        self.frames_descartados = 0
        self.tiempo_espera_promedio = 0.0

        # Iniciar el hilo de captura
        self.start()

        logger.info(f"CapturaVideoAsync inicializada: cámara={camara_index}, "
                  f"resolución={width}x{height}, fps={fps}, buffer={buffer_size}")

    def start(self) -> None:
        """Inicia el hilo de captura asíncrona."""
        self.stopped = False
        self.thread = threading.Thread(target=self._capturar, daemon=True)
        self.thread.start()
        logger.info("Hilo de captura asíncrona iniciado")

    def _capturar(self) -> None:
        """
        Método de ejecución del hilo que captura frames continuamente.

        Este método corre en un hilo separado y captura frames
        constantemente para mantener el buffer actualizado.
        """
        intervalo_objetivo = 1.0 / self.fps_objetivo

        while not self.stopped:
            # Capturar frame
            try:
                if not self.cap.isOpened():
                    logger.error("La cámara se cerró durante la captura asíncrona")
                    self.stopped = True
                    break

                ret, frame = self.cap.read()

                if not ret:
                    logger.warning("Error al leer frame en captura asíncrona")
                    time.sleep(0.1)  # Pequeña pausa antes de reintentar
                    continue

                # Actualizar métricas
                self.frames_capturados += 1

                # Si la cola está llena, quitar un frame antiguo para hacer espacio
                if self.queue.full():
                    try:
                        self.queue.get_nowait()
                        self.frames_descartados += 1
                    except queue.Empty:
                        pass

                # Agregar el frame a la cola
                try:
                    self.queue.put(frame, block=False)
                except queue.Full:
                    self.frames_descartados += 1

                # Intentar mantener la tasa de captura cercana al objetivo
                # durmiendo un tiempo apropiado
                tiempo_procesamiento = time.time() - self.tiempo_ultimo_frame
                self.tiempo_ultimo_frame = time.time()

                # Calcular tiempo de espera necesario para mantener FPS objetivo
                tiempo_espera = max(0, intervalo_objetivo - tiempo_procesamiento)

                if tiempo_espera > 0:
                    time.sleep(tiempo_espera)

            except Exception as e:
                logger.error(f"Error en hilo de captura: {e}")
                time.sleep(0.1)  # Pausa antes de reintentar

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Lee un frame del buffer de captura.

        Returns:
            Tupla con (éxito, frame).
        """
        if self.stopped or not self.thread.is_alive():
            return False, None

        try:
            # Intentar obtener el frame más reciente
            frame = self.queue.get(timeout=1.0)
            return True, frame
        except queue.Empty:
            logger.warning("Timeout al esperar por un frame en la cola")
            return False, None
        except Exception as e:
            logger.error(f"Error al leer frame de la cola: {e}")
            return False, None

    def stop(self) -> None:
        """Detiene el hilo de captura asíncrona."""
        self.stopped = True
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)
        logger.info("Hilo de captura asíncrona detenido")

    def release(self) -> None:
        """Libera los recursos (cámara y hilo)."""
        self.stop()
        super().release()

        # Limpiar la cola
        while not self.queue.empty():
            try:
                self.queue.get_nowait()
            except queue.Empty:
                break

    def obtener_metricas(self) -> Dict[str, Any]:
        """
        Obtiene métricas de rendimiento de la captura asíncrona.

        Returns:
            Diccionario con métricas de rendimiento extendidas.
        """
        metricas_base = super().obtener_metricas()

        # Añadir métricas específicas de captura asíncrona
        metricas_adicionales = {
            "frames_descartados": self.frames_descartados,
            "tamaño_cola_actual": self.queue.qsize(),
            "tamaño_buffer": self.buffer_size,
            "fps_objetivo": self.fps_objetivo,
            "es_asincrono": True
        }

        # Combinar métricas
        metricas_base.update(metricas_adicionales)
        return metricas_base

    def __del__(self) -> None:
        """Destructor para asegurar la liberación de recursos."""
        self.release()

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
