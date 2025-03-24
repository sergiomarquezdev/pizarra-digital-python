"""
Módulo para gestionar el lienzo de dibujo.

Este módulo proporciona una clase para crear y manipular un lienzo
donde se realizarán los dibujos.
"""
import logging
import numpy as np
import cv2
from typing import Tuple, Optional

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

        # Crear el lienzo vacío
        self.lienzo = np.ones((alto, ancho, 3), dtype=np.uint8)
        self.lienzo[:] = color_fondo

        # Color y grosor actuales para dibujar
        self.color_dibujo = DEFAULT_DRAWING_COLOR
        self.grosor_dibujo = DRAWING_THICKNESS

        logger.info(f"Lienzo inicializado con dimensiones {ancho}x{alto}")

    def limpiar(self) -> None:
        """
        Limpia el lienzo, restaurándolo a su color de fondo original.
        """
        self.lienzo[:] = self.color_fondo
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

        cv2.line(self.lienzo, punto_inicio, punto_fin, color_usado, grosor_usado)

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

        # Crear una máscara para identificar los píxeles dibujados en el lienzo
        mascara = cv2.inRange(self.lienzo, self.color_fondo, self.color_fondo)
        mascara = cv2.bitwise_not(mascara)

        # Combinar el fotograma y el lienzo usando la máscara
        fotograma_resultado = fotograma.copy()

        # Para los píxeles donde la máscara es no cero (píxeles dibujados)
        idx = (mascara > 0)
        fotograma_resultado[idx] = cv2.addWeighted(fotograma, 1-alpha, self.lienzo, alpha, 0)[idx]

        return fotograma_resultado

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
