"""
Módulo para el lienzo de dibujo de la pizarra digital.

Este módulo proporciona la clase Lienzo que gestiona la representación
visual de la pizarra digital y las operaciones de dibujo.
"""
import logging
import cv2
import numpy as np
import time
from collections import deque
from typing import Tuple, List, Dict, Any, Optional, Deque

from ..config import (
    CANVAS_WIDTH,
    CANVAS_HEIGHT,
    CANVAS_BACKGROUND_COLOR,
    CANVAS_LINE_THICKNESS,
    DEFAULT_COLOR,
    OPTIMIZATION_BUFFER_SIZE,
    OPTIMIZATION_MEMORY_LIMIT_MB,
    DRAWING_SPEED_SMOOTH_FACTOR,
    DRAWING_MIN_THICKNESS,
    DRAWING_MAX_THICKNESS,
    DRAWING_SPEED_THRESHOLD_LOW,
    DRAWING_SPEED_THRESHOLD_HIGH
)

# Configuración del logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Buffer para operaciones de dibujo
BUFFER_SIZE: int = OPTIMIZATION_BUFFER_SIZE
MEMORY_LIMIT_MB: int = OPTIMIZATION_MEMORY_LIMIT_MB

# Ajustes para el grosor adaptativo basado en velocidad
SMOOTH_FACTOR: float = DRAWING_SPEED_SMOOTH_FACTOR
MIN_THICKNESS: int = DRAWING_MIN_THICKNESS
MAX_THICKNESS: int = DRAWING_MAX_THICKNESS
SPEED_THRESHOLD_LOW: float = DRAWING_SPEED_THRESHOLD_LOW
SPEED_THRESHOLD_HIGH: float = DRAWING_SPEED_THRESHOLD_HIGH

class Lienzo:
    """
    Clase que gestiona el lienzo de dibujo y las operaciones de pintura.

    El lienzo está implementado como una imagen en memoria que se puede
    superponer sobre los fotogramas de la cámara.
    """

    def __init__(self,
                 width: int = CANVAS_WIDTH,
                 height: int = CANVAS_HEIGHT,
                 background_color: Tuple[int, int, int] = CANVAS_BACKGROUND_COLOR,
                 enable_optimizations: bool = True) -> None:
        """
        Inicializa un nuevo lienzo de dibujo.

        Args:
            width: Ancho del lienzo en píxeles.
            height: Alto del lienzo en píxeles.
            background_color: Color de fondo del lienzo (BGR).
            enable_optimizations: Si se deben habilitar las optimizaciones de rendimiento.
        """
        self.ancho = width
        self.alto = height
        self.color_fondo = background_color
        self.color_actual = DEFAULT_COLOR
        self.grosor = CANVAS_LINE_THICKNESS
        self.enable_optimizations = enable_optimizations

        # Memoria principal: lienzo y máscara
        self.lienzo = np.ones((height, width, 3), dtype=np.uint8) * background_color[0]
        # Convertir de BGR a valores escalares individuales
        self.lienzo[:, :, 0] = background_color[0]  # B
        self.lienzo[:, :, 1] = background_color[1]  # G
        self.lienzo[:, :, 2] = background_color[2]  # R

        # Máscara: 0 donde no hay dibujo, 255 donde hay dibujo
        self.mascara = np.zeros((height, width), dtype=np.uint8)

        # Variables para implementación del buffer de operaciones
        self.ultima_posicion = None
        self.buffer_operaciones: Deque[Dict[str, Any]] = deque(maxlen=BUFFER_SIZE)
        self.procesando_buffer = False

        # Velocidad de dibujo para grosor adaptativo
        self.ultima_pos = None
        self.ultimo_tiempo = None
        self.velocidad_actual = 0.0

        # Métricas de rendimiento
        self._metricas = {
            'tiempo_ultimo_frame': time.time(),
            'operaciones_buffer': 0,
            'operaciones_procesadas': 0,
            'tiempo_procesamiento': 0.0
        }

        logger.info(f"Lienzo inicializado con tamaño {width}x{height}")
        if enable_optimizations:
            logger.info(f"Optimizaciones habilitadas: buffer={BUFFER_SIZE}, memoria={MEMORY_LIMIT_MB}MB")

    def calcular_grosor_adaptativo(self, posicion: Tuple[int, int]) -> int:
        """
        Calcula el grosor de la línea basado en la velocidad de movimiento.
        Movimientos más rápidos producen líneas más delgadas.

        Args:
            posicion: La posición actual (x, y) para el dibujo.

        Returns:
            El grosor calculado para la línea.
        """
        tiempo_actual = time.time()

        # Si no hay posición anterior, usar el grosor predeterminado
        if self.ultima_pos is None or self.ultimo_tiempo is None:
            self.ultima_pos = posicion
            self.ultimo_tiempo = tiempo_actual
            return self.grosor

        # Calcular la distancia y el tiempo transcurrido
        dx = posicion[0] - self.ultima_pos[0]
        dy = posicion[1] - self.ultima_pos[1]
        distancia = np.sqrt(dx*dx + dy*dy)
        dt = max(0.001, tiempo_actual - self.ultimo_tiempo)  # Evitar división por cero

        # Calcular la velocidad (píxeles por segundo)
        velocidad = distancia / dt

        # Suavizar la velocidad para evitar cambios bruscos
        self.velocidad_actual = (SMOOTH_FACTOR * velocidad +
                               (1 - SMOOTH_FACTOR) * self.velocidad_actual)

        # Actualizar última posición y tiempo
        self.ultima_pos = posicion
        self.ultimo_tiempo = tiempo_actual

        # Mapeo de velocidad a grosor (velocidad alta = grosor bajo)
        if self.velocidad_actual < SPEED_THRESHOLD_LOW:
            # Velocidad baja, grosor máximo
            return MAX_THICKNESS
        elif self.velocidad_actual > SPEED_THRESHOLD_HIGH:
            # Velocidad alta, grosor mínimo
            return MIN_THICKNESS
        else:
            # Interpolación lineal entre los umbrales
            factor = (self.velocidad_actual - SPEED_THRESHOLD_LOW) / (SPEED_THRESHOLD_HIGH - SPEED_THRESHOLD_LOW)
            return int(MAX_THICKNESS - factor * (MAX_THICKNESS - MIN_THICKNESS))

    def cambiar_color(self, color: Tuple[int, int, int]) -> None:
        """
        Cambia el color actual de dibujo.

        Args:
            color: Nuevo color en formato BGR.
        """
        self.color_actual = color
        logger.debug(f"Cambiado color de dibujo a {color}")

    def cambiar_grosor(self, grosor: int) -> None:
        """
        Cambia el grosor de las líneas de dibujo.

        Args:
            grosor: Nuevo grosor de línea en píxeles.
        """
        self.grosor = grosor
        logger.debug(f"Cambiado grosor de línea a {grosor}px")

    def limpiar(self) -> None:
        """
        Limpia el lienzo, eliminando todos los dibujos.
        """
        # Limpiar búfer de operaciones primero
        self.buffer_operaciones.clear()
        self.ultima_posicion = None

        # Limpiar el lienzo
        self.lienzo[:, :] = self.color_fondo
        self.mascara[:, :] = 0

        # Resetear variables de velocidad
        self.ultima_pos = None
        self.ultimo_tiempo = None
        self.velocidad_actual = 0.0

        logger.info("Lienzo limpiado")

    def dibujar_punto(self, x: int, y: int, color: Optional[Tuple[int, int, int]] = None) -> None:
        """
        Dibuja un punto en las coordenadas especificadas.

        Args:
            x: Coordenada x del punto.
            y: Coordenada y del punto.
            color: Color del punto (BGR). Si es None, se usa el color actual.
        """
        if color is None:
            color = self.color_actual

        # Dibujar un círculo pequeño para representar un punto
        radio = max(CANVAS_LINE_THICKNESS, 1)  # Al menos radio 1 para que sea visible
        cv2.circle(
            self.lienzo,
            (x, y),
            radio,
            color,
            -1  # Rellenar el círculo
        )

        # Dibujar el mismo punto en la máscara (todos los canales)
        if self.mascara is not None:
            cv2.circle(
                self.mascara,
                (x, y),
                radio,
                (255, 255, 255),
                -1
            )

        self.modificado = True
        logger.debug(f"Punto dibujado en ({x}, {y}) con color {color}")

    def dibujar_linea(self, x1: int, y1: int, x2: int, y2: int, color: Optional[Tuple[int, int, int]] = None) -> None:
        """
        Dibuja una línea entre dos puntos.

        Args:
            x1: Coordenada x del primer punto.
            y1: Coordenada y del primer punto.
            x2: Coordenada x del segundo punto.
            y2: Coordenada y del segundo punto.
            color: Color de la línea (BGR). Si es None, se usa el color actual.
        """
        if color is None:
            color = self.color_actual

        # Dibujar la línea en la imagen principal
        grosor = CANVAS_LINE_THICKNESS
        cv2.line(
            self.lienzo,
            (x1, y1),
            (x2, y2),
            color,
            thickness=grosor,
            lineType=cv2.LINE_AA
        )

        # Dibujar la misma línea en la máscara (todos los canales)
        if self.mascara is not None:
            cv2.line(
                self.mascara,
                (x1, y1),
                (x2, y2),
                (255, 255, 255),
                thickness=grosor,
                lineType=cv2.LINE_AA
            )

        self.modificado = True
        logger.debug(f"Línea dibujada desde ({x1}, {y1}) a ({x2}, {y2}) con color {color}")

    def dibujar_desde_posicion(self, x: int, y: int, dibujando: bool) -> None:
        """
        Dibuja desde la última posición guardada hasta la nueva posición.

        Este método se utiliza para dibujar continuamente mientras se mueve el dedo.

        Args:
            x: Coordenada x de la nueva posición.
            y: Coordenada y de la nueva posición.
            dibujando: Indica si se debe dibujar (True) o solo actualizar la posición (False).
        """
        # Asegurarse de que las coordenadas estén dentro del lienzo
        x = max(0, min(x, self.ancho - 1))
        y = max(0, min(y, self.alto - 1))

        if dibujando:
            if self.ultima_posicion is not None:
                # Dibujar línea desde la última posición
                ult_x, ult_y = self.ultima_posicion
                self.dibujar_linea(ult_x, ult_y, x, y)
            else:
                # Primera posición, dibujar un punto
                self.dibujar_punto(x, y)

        # Actualizar la última posición
        self.ultima_posicion = (x, y)

    def forzar_procesar_buffer(self) -> None:
        """
        Fuerza el procesamiento inmediato de todas las operaciones en el buffer.
        Útil antes de mostrar el lienzo para asegurar que todos los cambios sean visibles.
        """
        if self.buffer_operaciones:
            self._procesar_buffer_operaciones()

    def actualizar(self) -> None:
        """
        Actualiza el estado del lienzo procesando cualquier operación pendiente.
        Debe llamarse en cada fotograma si se usa el buffer.
        """
        self.forzar_procesar_buffer()

        # Actualizar métricas
        self._metricas['tiempo_ultimo_frame'] = time.time()

    def superponer_en_fotograma(self, frame: np.ndarray) -> np.ndarray:
        """
        Superpone el lienzo en un fotograma de la cámara.

        Args:
            frame: Fotograma de entrada donde superponer el lienzo.

        Returns:
            Fotograma con el lienzo superpuesto.
        """
        # Asegurarse de que todas las operaciones estén procesadas
        self.forzar_procesar_buffer()

        # Redimensionar el frame si es necesario
        if frame.shape[:2] != (self.alto, self.ancho):
            frame_redimensionado = cv2.resize(frame, (self.ancho, self.alto))
        else:
            frame_redimensionado = frame

        # Crear una copia para no modificar el original
        resultado = frame_redimensionado.copy()

        # Superponer el lienzo utilizando la máscara
        resultado = cv2.bitwise_and(resultado, resultado, mask=cv2.bitwise_not(self.mascara))
        temp = cv2.bitwise_and(self.lienzo, self.lienzo, mask=self.mascara)
        resultado = cv2.add(resultado, temp)

        return resultado

    def obtener_imagen(self) -> np.ndarray:
        """
        Obtiene la imagen actual del lienzo.

        Returns:
            Imagen del lienzo (BGR).
        """
        # Asegurarse de que todas las operaciones estén procesadas
        self.forzar_procesar_buffer()
        return self.lienzo.copy()

    def obtener_metricas(self) -> Dict[str, Any]:
        """
        Obtiene métricas de rendimiento del lienzo.

        Returns:
            Diccionario con métricas de rendimiento.
        """
        return {
            "operaciones_buffer": self._metricas['operaciones_buffer'],
            "operaciones_procesadas": self._metricas['operaciones_procesadas'],
            "tiempo_procesamiento": self._metricas['tiempo_procesamiento'] * 1000,  # ms
            "velocidad_dibujo": self.velocidad_actual
        }
