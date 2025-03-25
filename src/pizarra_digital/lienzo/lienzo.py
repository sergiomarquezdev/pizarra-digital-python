"""
Módulo para gestionar el lienzo de dibujo.

Este módulo proporciona una clase para crear y manipular un lienzo
donde se realizarán los dibujos.
"""
import logging
import numpy as np
import cv2
from typing import Tuple, Optional, List

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

        # Último punto registrado para trazos continuos
        self.ultimo_punto: Optional[Tuple[int, int]] = None

        # Buffer para puntos intermedios
        self.buffer_puntos: List[Tuple[int, int]] = []
        self.max_buffer_puntos = 5

        logger.info(f"Lienzo inicializado con dimensiones {ancho}x{alto}")

    def limpiar(self) -> None:
        """
        Limpia el lienzo, restaurándolo a su color de fondo original.
        """
        self.lienzo[:] = self.color_fondo
        self.mascara_dibujo[:] = 0  # Limpiar también la máscara
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

        cv2.circle(self.lienzo, punto, grosor_usado // 2, color_usado, -1)
        # Marcar los píxeles dibujados en la máscara
        cv2.circle(self.mascara_dibujo, punto, grosor_usado // 2, 255, -1)

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

        # Si la distancia es grande, interpolar puntos para evitar espacios en trazos rápidos
        if distancia > 2 * grosor_usado:
            # Interpolar puntos para movimientos rápidos
            puntos_interpolados = self._interpolar_puntos(punto_inicio, punto_fin, distancia // grosor_usado)

            # Dibujar líneas entre todos los puntos interpolados
            punto_anterior = punto_inicio
            for punto in puntos_interpolados:
                cv2.line(self.lienzo, punto_anterior, punto, color_usado, grosor_usado)
                cv2.line(self.mascara_dibujo, punto_anterior, punto, 255, grosor_usado)
                punto_anterior = punto
        else:
            # Para distancias cortas, dibujar una línea directa
            cv2.line(self.lienzo, punto_inicio, punto_fin, color_usado, grosor_usado)
            cv2.line(self.mascara_dibujo, punto_inicio, punto_fin, 255, grosor_usado)

        # Actualizar último punto
        self.ultimo_punto = punto_fin

        # Añadir al buffer de puntos
        self._actualizar_buffer_puntos(punto_fin)

    def _interpolar_puntos(self,
                          punto_inicio: Tuple[int, int],
                          punto_fin: Tuple[int, int],
                          num_puntos: int) -> List[Tuple[int, int]]:
        """
        Interpola puntos entre dos coordenadas para movimientos rápidos.

        Args:
            punto_inicio: Punto inicial (x, y)
            punto_fin: Punto final (x, y)
            num_puntos: Número de puntos a interpolar

        Returns:
            Lista de puntos intermedios
        """
        if num_puntos < 2:
            return [punto_fin]

        puntos = []
        for i in range(1, num_puntos + 1):
            t = i / (num_puntos + 1)
            x = int(punto_inicio[0] * (1 - t) + punto_fin[0] * t)
            y = int(punto_inicio[1] * (1 - t) + punto_fin[1] * t)
            puntos.append((x, y))
        return puntos

    def _actualizar_buffer_puntos(self, punto: Tuple[int, int]) -> None:
        """
        Actualiza el buffer de puntos para suavizado y manejo de trazos rápidos.

        Args:
            punto: Nuevo punto a añadir al buffer
        """
        self.buffer_puntos.append(punto)
        if len(self.buffer_puntos) > self.max_buffer_puntos:
            self.buffer_puntos.pop(0)

    def cambiar_color(self, color: Tuple[int, int, int]) -> None:
        """
        Cambia el color de dibujo actual.

        Args:
            color: Nuevo color en formato BGR.
        """
        self.color_dibujo = color
        logger.info(f"Color de dibujo cambiado a {color}")

    def cambiar_grosor(self, grosor: int) -> None:
        """
        Cambia el grosor de dibujo actual.

        Args:
            grosor: Nuevo grosor para el dibujo.
        """
        self.grosor_dibujo = max(1, grosor)  # Asegurar que sea al menos 1
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
        if fotograma is None:
            logger.warning("Se intentó superponer en un fotograma nulo")
            return self.lienzo.copy()

        # Verificar que las dimensiones coincidan
        if fotograma.shape[0] != self.alto or fotograma.shape[1] != self.ancho:
            logger.warning(f"Las dimensiones del fotograma ({fotograma.shape[1]}x{fotograma.shape[0]}) "
                         f"no coinciden con las del lienzo ({self.ancho}x{self.alto})")
            # Redimensionar el fotograma para que coincida con el lienzo
            fotograma = cv2.resize(fotograma, (self.ancho, self.alto))

        # Usar la máscara de dibujo
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
            True si el punto está dentro de los límites, False en caso contrario.
        """
        x, y = punto
        return 0 <= x < self.ancho and 0 <= y < self.alto
