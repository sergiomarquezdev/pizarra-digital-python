"""
Módulo para la lógica de dibujo basada en gestos de manos.

Este módulo proporciona funciones para interpretar los gestos de las manos
y traducirlos en acciones de dibujo en el lienzo.
"""
import logging
import numpy as np
from typing import Tuple, Dict, List, Any, Optional

from ..lienzo.lienzo import Lienzo

# Configuración del logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DibujoMano:
    """Clase para controlar el dibujo mediante gestos de manos."""

    def __init__(self, lienzo: Lienzo) -> None:
        """
        Inicializa el controlador de dibujo con un lienzo.

        Args:
            lienzo: Instancia de la clase Lienzo donde se realizarán los dibujos.
        """
        self.lienzo = lienzo

        # Posición anterior de la punta del dedo índice
        self.posicion_anterior: Optional[Tuple[int, int]] = None

        # Historial de posiciones para suavizado (opcional)
        self.historial_posiciones: List[Tuple[int, int]] = []
        self.max_historial: int = 3  # Número máximo de posiciones a almacenar

        # Estado del dibujo
        self.dibujando: bool = False

        logger.info("Controlador de dibujo inicializado")

    def procesar_mano(self,
                     mano: Optional[Dict[str, Any]],
                     detector_manos: Any) -> None:
        """
        Procesa la información de una mano y realiza acciones de dibujo según el gesto.

        Args:
            mano: Diccionario con la información de la mano detectada.
            detector_manos: Instancia del detector de manos para utilizar sus métodos.
        """
        if mano is None:
            # Si no se detecta ninguna mano, dejar de dibujar
            self.dibujando = False
            self.posicion_anterior = None
            self.historial_posiciones = []
            return

        # Comprobar si el dedo índice está extendido
        indice_extendido = detector_manos.es_indice_extendido(mano)

        # Obtener la posición de la punta del dedo índice
        punto_actual = detector_manos.obtener_punta_indice(mano)

        if punto_actual is None:
            # Si no se puede obtener la posición del dedo índice, no hacer nada
            return

        # Actualizar el historial de posiciones
        self._actualizar_historial(punto_actual)

        # Obtener la posición suavizada
        punto_suavizado = self._suavizar_posicion()

        if indice_extendido:
            # Si el dedo índice está extendido, dibujar
            if not self.dibujando:
                # Si es el primer punto al comenzar a dibujar
                self.dibujando = True
                self.posicion_anterior = punto_suavizado
                self.lienzo.dibujar_punto(punto_suavizado)
            else:
                # Si ya estábamos dibujando, trazar una línea desde el punto anterior
                if self.posicion_anterior is not None:
                    self.lienzo.dibujar_linea(self.posicion_anterior, punto_suavizado)
                self.posicion_anterior = punto_suavizado
        else:
            # Si el dedo índice no está extendido, dejar de dibujar
            self.dibujando = False
            self.posicion_anterior = None

    def _actualizar_historial(self, punto: Tuple[int, int]) -> None:
        """
        Actualiza el historial de posiciones del dedo índice.

        Args:
            punto: Coordenadas (x, y) de la posición actual.
        """
        self.historial_posiciones.append(punto)

        # Mantener solo las últimas max_historial posiciones
        if len(self.historial_posiciones) > self.max_historial:
            self.historial_posiciones.pop(0)

    def _suavizar_posicion(self) -> Tuple[int, int]:
        """
        Calcula una posición suavizada basada en el historial de posiciones.

        Returns:
            Coordenadas (x, y) de la posición suavizada.
        """
        if not self.historial_posiciones:
            return (0, 0)

        # Si solo hay un punto en el historial, devolverlo directamente
        if len(self.historial_posiciones) == 1:
            return self.historial_posiciones[0]

        # Calcular el promedio de las posiciones en el historial
        x_total = sum(punto[0] for punto in self.historial_posiciones)
        y_total = sum(punto[1] for punto in self.historial_posiciones)

        x_promedio = int(x_total / len(self.historial_posiciones))
        y_promedio = int(y_total / len(self.historial_posiciones))

        return (x_promedio, y_promedio)
