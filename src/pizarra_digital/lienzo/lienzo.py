"""
Módulo para gestionar el lienzo de dibujo.

Este módulo proporciona una clase para crear y manipular un lienzo
donde se realizarán los dibujos.
"""
import logging
import numpy as np
import cv2
import time
from typing import Tuple, Optional, List, Dict, Any, Deque
from collections import deque

from ..config import (
    CANVAS_WIDTH,
    CANVAS_HEIGHT,
    CANVAS_BACKGROUND,
    DEFAULT_DRAWING_COLOR,
    DRAWING_THICKNESS
)

# Configuración del logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuración para optimizar el rendimiento del lienzo
BUFFER_LINEAS_TAMANO: int = 5  # Tamaño del buffer para acumular líneas (dibujar varias de una vez)
OPTIMIZAR_MEMORIA: bool = True  # Usar técnicas de optimización de memoria
GROSOR_PREDICCION_ADAPTATIVO: bool = True  # Ajustar grosor según velocidad de movimiento

class Lienzo:
    """Clase para gestionar el lienzo de dibujo."""

    def __init__(self,
                ancho: int = CANVAS_WIDTH,
                alto: int = CANVAS_HEIGHT,
                color_fondo: Tuple[int, int, int] = CANVAS_BACKGROUND) -> None:
        """
        Inicializa un nuevo lienzo.

        Args:
            ancho: Ancho del lienzo en píxeles.
            alto: Alto del lienzo en píxeles.
            color_fondo: Color de fondo del lienzo en formato BGR.
        """
        self.ancho = ancho
        self.alto = alto
        self.color_fondo = color_fondo

        # Crear el lienzo vacío - usar uint8 para optimizar memoria
        self.lienzo = np.ones((alto, ancho, 3), dtype=np.uint8)
        self.lienzo[:] = color_fondo

        # Crear una máscara para seguir los píxeles dibujados
        self.mascara_dibujo = np.zeros((alto, ancho), dtype=np.uint8)

        # Color y grosor actuales para dibujar
        self.color_dibujo = DEFAULT_DRAWING_COLOR
        self.grosor_dibujo = DRAWING_THICKNESS

        # Caché para operaciones frecuentes
        self._frame_resultado = np.zeros((alto, ancho, 3), dtype=np.uint8)

        # Buffer de líneas para acumular operaciones de dibujo
        self.buffer_lineas: List[Dict[str, Any]] = []

        # Último punto registrado para trazos continuos
        self.ultimo_punto: Optional[Tuple[int, int]] = None

        # Buffer para puntos intermedios
        self.buffer_puntos: Deque[Tuple[int, int]] = deque(maxlen=5)

        # Métricas de rendimiento
        self.tiempo_ultimo_dibujo = time.time()
        self.operaciones_dibujo = 0
        self.tiempo_total_dibujo = 0.0

        # Último grosor utilizado para adaptación
        self.ultimo_grosor_adaptado = self.grosor_dibujo

        logger.info(f"Lienzo optimizado inicializado con dimensiones {ancho}x{alto}")

    def limpiar(self) -> None:
        """
        Limpia el lienzo, restaurándolo a su color de fondo original.
        """
        # Vaciar buffer de operaciones pendientes
        self._procesar_buffer_lineas()

        # Usar operaciones vectorizadas para mayor velocidad
        self.lienzo[:] = self.color_fondo
        self.mascara_dibujo[:] = 0  # Limpiar también la máscara

        # Reiniciar variables de estado
        self.ultimo_punto = None
        self.buffer_puntos.clear()

        logger.info("Lienzo limpiado")

    def dibujar_punto(self,
                     punto: Tuple[int, int],
                     color: Optional[Tuple[int, int, int]] = None,
                     grosor: Optional[int] = None) -> None:
        """
        Dibuja un punto (círculo pequeño) en el lienzo.

        Args:
            punto: Coordenadas (x, y) del punto a dibujar.
            color: Color del punto en formato BGR. Si es None, usa el color actual.
            grosor: Grosor (radio) del punto. Si es None, usa el grosor actual.
        """
        if not self._es_punto_valido(punto):
            return

        color_usado = color if color is not None else self.color_dibujo
        grosor_usado = grosor if grosor is not None else self.grosor_dibujo

        # Usar buffer para acumular operaciones
        if OPTIMIZAR_MEMORIA:
            # Añadir al buffer
            self.buffer_lineas.append({
                'tipo': 'punto',
                'punto': punto,
                'color': color_usado,
                'grosor': grosor_usado
            })

            # Procesar buffer si alcanza el tamaño máximo
            if len(self.buffer_lineas) >= BUFFER_LINEAS_TAMANO:
                self._procesar_buffer_lineas()
        else:
            # Dibujar directamente
            t_inicio = time.time()
            cv2.circle(self.lienzo, punto, grosor_usado // 2, color_usado, -1)
            cv2.circle(self.mascara_dibujo, punto, grosor_usado // 2, 255, -1)
            self._actualizar_metricas_dibujo(time.time() - t_inicio)

        # Actualizar último punto
        self.ultimo_punto = punto

        # Añadir al buffer de puntos
        self._actualizar_buffer_puntos(punto)

    def dibujar_linea(self,
                     punto_inicio: Tuple[int, int],
                     punto_fin: Tuple[int, int],
                     color: Optional[Tuple[int, int, int]] = None,
                     grosor: Optional[int] = None) -> None:
        """
        Dibuja una línea entre dos puntos en el lienzo.

        Args:
            punto_inicio: Coordenadas (x, y) del punto de inicio.
            punto_fin: Coordenadas (x, y) del punto de fin.
            color: Color de la línea en formato BGR. Si es None, usa el color actual.
            grosor: Grosor de la línea. Si es None, usa el grosor actual.
        """
        if not (self._es_punto_valido(punto_inicio) and self._es_punto_valido(punto_fin)):
            return

        color_usado = color if color is not None else self.color_dibujo
        grosor_usado = grosor if grosor is not None else self.grosor_dibujo

        # Calcular la distancia entre los puntos
        dx = punto_fin[0] - punto_inicio[0]
        dy = punto_fin[1] - punto_inicio[1]
        distancia = int(np.sqrt(dx*dx + dy*dy))

        # Ajustar grosor según velocidad si está activado
        if GROSOR_PREDICCION_ADAPTATIVO:
            # Calcular velocidad (píxeles por tiempo)
            tiempo_actual = time.time()
            dt = tiempo_actual - self.tiempo_ultimo_dibujo
            if dt > 0:
                velocidad = distancia / dt
                # Reducir grosor para movimientos rápidos (más suaves)
                if velocidad > 1000:  # Muy rápido
                    grosor_adaptado = max(1, grosor_usado // 3)
                elif velocidad > 500:  # Rápido
                    grosor_adaptado = max(1, grosor_usado // 2)
                else:
                    # Normal o lento, usar grosor completo
                    grosor_adaptado = grosor_usado

                # Suavizar transición entre grosores
                alpha = 0.3  # Factor de suavizado (0-1)
                grosor_usado = int(alpha * grosor_adaptado + (1 - alpha) * self.ultimo_grosor_adaptado)
                self.ultimo_grosor_adaptado = grosor_usado

            self.tiempo_ultimo_dibujo = tiempo_actual

        # Usar buffer para acumular operaciones
        if OPTIMIZAR_MEMORIA:
            # Añadir al buffer
            self.buffer_lineas.append({
                'tipo': 'linea',
                'punto_inicio': punto_inicio,
                'punto_fin': punto_fin,
                'color': color_usado,
                'grosor': grosor_usado
            })

            # Procesar buffer si alcanza el tamaño máximo
            if len(self.buffer_lineas) >= BUFFER_LINEAS_TAMANO:
                self._procesar_buffer_lineas()
        else:
            # Dibujar directamente
            t_inicio = time.time()
            cv2.line(self.lienzo, punto_inicio, punto_fin, color_usado, grosor_usado)
            cv2.line(self.mascara_dibujo, punto_inicio, punto_fin, 255, grosor_usado)
            self._actualizar_metricas_dibujo(time.time() - t_inicio)

        # Actualizar último punto
        self.ultimo_punto = punto_fin

        # Añadir al buffer de puntos
        self._actualizar_buffer_puntos(punto_fin)

    def _procesar_buffer_lineas(self) -> None:
        """
        Procesa todas las operaciones de dibujo acumuladas en el buffer.
        Esto mejora el rendimiento al realizar menos llamadas a OpenCV.
        """
        if not self.buffer_lineas:
            return

        t_inicio = time.time()

        # Procesar cada operación en el buffer
        for operacion in self.buffer_lineas:
            if operacion['tipo'] == 'punto':
                cv2.circle(
                    self.lienzo,
                    operacion['punto'],
                    operacion['grosor'] // 2,
                    operacion['color'],
                    -1
                )
                cv2.circle(
                    self.mascara_dibujo,
                    operacion['punto'],
                    operacion['grosor'] // 2,
                    255,
                    -1
                )
            elif operacion['tipo'] == 'linea':
                cv2.line(
                    self.lienzo,
                    operacion['punto_inicio'],
                    operacion['punto_fin'],
                    operacion['color'],
                    operacion['grosor']
                )
                cv2.line(
                    self.mascara_dibujo,
                    operacion['punto_inicio'],
                    operacion['punto_fin'],
                    255,
                    operacion['grosor']
                )

        # Vaciar el buffer
        self.buffer_lineas.clear()

        # Actualizar métricas
        self._actualizar_metricas_dibujo(time.time() - t_inicio)

    def _actualizar_metricas_dibujo(self, tiempo_operacion: float) -> None:
        """
        Actualiza las métricas de rendimiento del dibujo.

        Args:
            tiempo_operacion: Tiempo que tomó la operación de dibujo.
        """
        self.operaciones_dibujo += 1
        self.tiempo_total_dibujo += tiempo_operacion

    def _actualizar_buffer_puntos(self, punto: Tuple[int, int]) -> None:
        """
        Actualiza el buffer de puntos para suavizado y manejo de trazos rápidos.

        Args:
            punto: Nuevo punto a añadir al buffer
        """
        self.buffer_puntos.append(punto)

    def cambiar_color(self, color: Tuple[int, int, int]) -> None:
        """
        Cambia el color de dibujo actual.

        Args:
            color: Nuevo color en formato BGR.
        """
        # Procesar operaciones pendientes antes de cambiar el color
        self._procesar_buffer_lineas()

        self.color_dibujo = color
        logger.info(f"Color de dibujo cambiado a {color}")

    def cambiar_grosor(self, grosor: int) -> None:
        """
        Cambia el grosor de dibujo actual.

        Args:
            grosor: Nuevo grosor para el dibujo.
        """
        # Procesar operaciones pendientes antes de cambiar el grosor
        self._procesar_buffer_lineas()

        self.grosor_dibujo = max(1, grosor)  # Asegurar que sea al menos 1
        self.ultimo_grosor_adaptado = self.grosor_dibujo  # Actualizar también el grosor adaptado
        logger.info(f"Grosor de dibujo cambiado a {self.grosor_dibujo}")

    def superponer_en_fotograma(self,
                               fotograma: np.ndarray,
                               alpha: float = 0.7) -> np.ndarray:
        """
        Superpone el lienzo sobre un fotograma.

        Args:
            fotograma: Fotograma sobre el que superponer el lienzo.
            alpha: Valor de transparencia (0.0 a 1.0) donde 1.0 significa lienzo opaco.

        Returns:
            Fotograma con el lienzo superpuesto.
        """
        # Procesar operaciones pendientes antes de superponer el lienzo
        self._procesar_buffer_lineas()

        if fotograma is None:
            logger.warning("Se intentó superponer en un fotograma nulo")
            return self.lienzo.copy()

        # Verificar que las dimensiones coincidan
        if fotograma.shape[0] != self.alto or fotograma.shape[1] != self.ancho:
            logger.warning(f"Las dimensiones del fotograma ({fotograma.shape[1]}x{fotograma.shape[0]}) "
                         f"no coinciden con las del lienzo ({self.ancho}x{self.alto})")
            # Redimensionar el fotograma para que coincida con el lienzo
            fotograma = cv2.resize(fotograma, (self.ancho, self.alto))

        # Optimización: reutilizar el buffer preasignado para evitar nuevas asignaciones
        np.copyto(self._frame_resultado, fotograma)

        # Usar operaciones vectorizadas para mayor velocidad
        idx = self.mascara_dibujo > 0
        if np.any(idx):
            # Mezclar solo los píxeles dibujados
            self._frame_resultado[idx] = (
                fotograma[idx] * (1 - alpha) +
                self.lienzo[idx] * alpha
            ).astype(np.uint8)

        return self._frame_resultado

    def _es_punto_valido(self, punto: Tuple[int, int]) -> bool:
        """
        Verifica si un punto está dentro de los límites del lienzo.

        Args:
            punto: Coordenadas (x, y) del punto a verificar.

        Returns:
            True si el punto está dentro del lienzo, False en caso contrario.
        """
        x, y = punto
        return 0 <= x < self.ancho and 0 <= y < self.alto

    def obtener_metricas(self) -> Dict[str, Any]:
        """
        Obtiene métricas de rendimiento del lienzo.

        Returns:
            Diccionario con métricas de rendimiento.
        """
        tiempo_medio = 0.0
        if self.operaciones_dibujo > 0:
            tiempo_medio = self.tiempo_total_dibujo / self.operaciones_dibujo

        return {
            "operaciones_dibujo": self.operaciones_dibujo,
            "tiempo_total_dibujo_ms": self.tiempo_total_dibujo * 1000,
            "tiempo_medio_dibujo_ms": tiempo_medio * 1000,
            "operaciones_pendientes": len(self.buffer_lineas)
        }
