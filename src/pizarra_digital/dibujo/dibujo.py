"""
Módulo para la lógica de dibujo basada en gestos de manos.

Este módulo proporciona funciones para interpretar los gestos de las manos
y traducirlos en acciones de dibujo en el lienzo.
"""
import logging
import numpy as np
import time
from typing import Tuple, Dict, List, Any, Optional, Deque
from collections import deque

from ..lienzo.lienzo import Lienzo

# Configuración del logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Parámetros para mejorar el rendimiento del dibujo
# Mayor valor significa más interpolación y mayor suavizado para movimientos rápidos
INTERPOLACION_ADAPTATIVA: bool = True  # Ajustar la interpolación según la velocidad
UMBRAL_VELOCIDAD_ALTA: float = 50.0  # Píxeles por frame (umbral para considerar movimiento rápido)
MAX_PUNTOS_INTERPOLADOS: int = 20  # Máximo de puntos a interpolar en movimientos muy rápidos
TAMANO_FILTRO_SUAVIZADO: int = 3  # Tamaño del filtro de suavizado
PESO_PUNTO_ACTUAL: float = 2.0  # Peso del punto actual en el filtro (mayor = menos suavizado)

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

        # Historial de posiciones para suavizado (cola FIFO)
        self.historial_posiciones: Deque[Tuple[int, int]] = deque(maxlen=TAMANO_FILTRO_SUAVIZADO)

        # Historial de tiempos para calcular velocidad
        self.historial_tiempos: Deque[float] = deque(maxlen=TAMANO_FILTRO_SUAVIZADO)

        # Velocidad del movimiento (en píxeles por frame)
        self.velocidad_actual: float = 0.0

        # Última vez que se registró un punto
        self.ultimo_tiempo_punto: float = 0.0

        # Estado del dibujo
        self.dibujando: bool = False

        # Métricas de rendimiento
        self.puntos_interpolados_total: int = 0
        self.frames_procesados: int = 0
        self.tiempo_inicio_metricas: float = time.time()

        logger.info("Controlador de dibujo inicializado con optimizaciones para movimientos rápidos")

    def procesar_mano(self,
                     mano: Optional[Dict[str, Any]],
                     detector_manos: Any) -> None:
        """
        Procesa la información de una mano y realiza acciones de dibujo según el gesto.

        Args:
            mano: Diccionario con la información de la mano detectada.
            detector_manos: Instancia del detector de manos para utilizar sus métodos.
        """
        self.frames_procesados += 1
        tiempo_actual = time.time()

        if mano is None:
            # Si no se detecta ninguna mano, dejar de dibujar
            self.dibujando = False
            self.posicion_anterior = None
            self.historial_posiciones.clear()
            self.historial_tiempos.clear()
            self.velocidad_actual = 0.0
            return

        # Comprobar si el dedo índice está extendido
        indice_extendido = detector_manos.es_indice_extendido(mano)

        # Obtener la posición de la punta del dedo índice
        punto_actual = detector_manos.obtener_punta_indice(mano)

        if punto_actual is None:
            # Si no se puede obtener la posición del dedo índice, no hacer nada
            return

        # Calcular velocidad del movimiento si tenemos una posición anterior
        if self.posicion_anterior is not None:
            dx = punto_actual[0] - self.posicion_anterior[0]
            dy = punto_actual[1] - self.posicion_anterior[1]
            distancia = np.sqrt(dx*dx + dy*dy)
            dt = tiempo_actual - self.ultimo_tiempo_punto
            if dt > 0:
                velocidad = distancia / dt  # píxeles por segundo
            else:
                velocidad = 0

            # Suavizar la velocidad con un filtro de media móvil exponencial
            alpha = 0.3  # Factor de suavizado (0-1)
            self.velocidad_actual = alpha * velocidad + (1 - alpha) * self.velocidad_actual

        # Actualizar tiempo del último punto
        self.ultimo_tiempo_punto = tiempo_actual

        # Actualizar el historial de posiciones y tiempos
        self._actualizar_historial(punto_actual, tiempo_actual)

        # Obtener la posición suavizada para reducir el ruido
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
                    # Determinar puntos a interpolar según la velocidad
                    num_puntos_interpolar = self._calcular_puntos_interpolar()

                    if num_puntos_interpolar > 1:
                        # Interpolar puntos para movimientos rápidos usando el optimizador
                        puntos_interpolados = self._interpolar_puntos(
                            self.posicion_anterior,
                            punto_suavizado,
                            num_puntos_interpolar
                        )

                        # Dibujar líneas entre todos los puntos interpolados
                        punto_anterior = self.posicion_anterior
                        for punto in puntos_interpolados:
                            self.lienzo.dibujar_linea(punto_anterior, punto)
                            punto_anterior = punto

                        # Actualizar métricas
                        self.puntos_interpolados_total += len(puntos_interpolados)
                    else:
                        # Para movimientos lentos, dibujo directo sin interpolación
                        self.lienzo.dibujar_linea(self.posicion_anterior, punto_suavizado)

                # Actualizar la posición anterior para el próximo frame
                self.posicion_anterior = punto_suavizado
        else:
            # Si el dedo índice no está extendido, dejar de dibujar
            self.dibujando = False
            self.posicion_anterior = None

    def _calcular_puntos_interpolar(self) -> int:
        """
        Calcula cuántos puntos interpolar según la velocidad del movimiento.

        Para movimientos rápidos generamos más puntos intermedios,
        mejorando la calidad del trazo.

        Returns:
            Número de puntos a interpolar.
        """
        if not INTERPOLACION_ADAPTATIVA:
            return 1  # Sin interpolación adaptativa

        # Convertir velocidad a píxeles por frame
        fps_estimado = 30.0  # Estimación conservadora de FPS
        velocidad_por_frame = self.velocidad_actual / fps_estimado

        if velocidad_por_frame <= 5.0:
            # Movimiento lento, no interpolamos
            return 1
        elif velocidad_por_frame <= UMBRAL_VELOCIDAD_ALTA:
            # Movimiento medio, interpolación ligera
            return int(velocidad_por_frame / 5)
        else:
            # Movimiento rápido, interpolación agresiva
            puntos = int(velocidad_por_frame / 2.5)
            return min(puntos, MAX_PUNTOS_INTERPOLADOS)  # Limitar para evitar exceso

    def _actualizar_historial(self, punto: Tuple[int, int], tiempo: float) -> None:
        """
        Actualiza el historial de posiciones del dedo índice y sus tiempos.

        Args:
            punto: Coordenadas (x, y) de la posición actual.
            tiempo: Tiempo en que se registró el punto.
        """
        self.historial_posiciones.append(punto)
        self.historial_tiempos.append(tiempo)

    def _suavizar_posicion(self) -> Tuple[int, int]:
        """
        Calcula una posición suavizada basada en el historial de posiciones.

        Este método usa un filtro de media ponderada que da más importancia
        a los puntos más recientes, reduciendo el retraso en el seguimiento.

        Returns:
            Coordenadas (x, y) de la posición suavizada.
        """
        if not self.historial_posiciones:
            return (0, 0)

        # Si solo hay un punto en el historial, devolverlo directamente
        if len(self.historial_posiciones) == 1:
            return self.historial_posiciones[0]

        # Aplicar filtro de media ponderada (más peso a puntos recientes)
        total_peso = 0
        x_ponderado = 0
        y_ponderado = 0

        # Último punto (más reciente) tiene peso adicional
        ultimo_punto = self.historial_posiciones[-1]
        x_ponderado += ultimo_punto[0] * PESO_PUNTO_ACTUAL
        y_ponderado += ultimo_punto[1] * PESO_PUNTO_ACTUAL
        total_peso += PESO_PUNTO_ACTUAL

        # Resto de puntos con pesos decrecientes
        for i, punto in enumerate(self.historial_posiciones):
            if i == len(self.historial_posiciones) - 1:
                continue  # Saltamos el último punto que ya procesamos

            # Peso decrece linealmente con la antigüedad
            peso = 1.0 - (i / len(self.historial_posiciones))

            x_ponderado += punto[0] * peso
            y_ponderado += punto[1] * peso
            total_peso += peso

        # Calcular promedio ponderado
        x_suavizado = int(x_ponderado / total_peso)
        y_suavizado = int(y_ponderado / total_peso)

        return (x_suavizado, y_suavizado)

    def _interpolar_puntos(self,
                         punto_inicio: Tuple[int, int],
                         punto_fin: Tuple[int, int],
                         num_puntos: int) -> List[Tuple[int, int]]:
        """
        Interpola puntos entre dos coordenadas para movimientos rápidos.

        Esta función genera una curva suave entre dos puntos para
        mejorar la calidad del trazo en movimientos rápidos.

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

        # Usar interpolación de Catmull-Rom para puntos en historial
        if len(self.historial_posiciones) >= 4:
            # Obtener puntos de control para la curva
            p0 = self.historial_posiciones[-4] if len(self.historial_posiciones) >= 4 else punto_inicio
            p1 = punto_inicio
            p2 = punto_fin
            p3 = self._predecir_punto_futuro(punto_inicio, punto_fin)

            # Generar puntos usando Catmull-Rom
            for i in range(num_puntos):
                t = (i + 1) / (num_puntos + 1)
                punto = self._interpolar_catmull_rom(p0, p1, p2, p3, t)
                puntos.append(punto)
        else:
            # Si no hay suficientes puntos para Catmull-Rom, usar interpolación lineal
            for i in range(1, num_puntos + 1):
                t = i / (num_puntos + 1)
                x = int(punto_inicio[0] * (1 - t) + punto_fin[0] * t)
                y = int(punto_inicio[1] * (1 - t) + punto_fin[1] * t)
                puntos.append((x, y))

        return puntos

    def _predecir_punto_futuro(self, p1: Tuple[int, int], p2: Tuple[int, int]) -> Tuple[int, int]:
        """
        Predice un punto futuro basado en la dirección del movimiento actual.

        Esto ayuda a crear curvas más naturales al interpolar.

        Args:
            p1: Primer punto (anterior)
            p2: Segundo punto (actual)

        Returns:
            Punto predicho (futuro)
        """
        # Calcular vector de dirección
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]

        # Extrapolar en la misma dirección
        x3 = p2[0] + dx
        y3 = p2[1] + dy

        return (int(x3), int(y3))

    def _interpolar_catmull_rom(self,
                               p0: Tuple[int, int],
                               p1: Tuple[int, int],
                               p2: Tuple[int, int],
                               p3: Tuple[int, int],
                               t: float) -> Tuple[int, int]:
        """
        Interpola un punto usando el algoritmo Catmull-Rom.

        Esta función genera curvas suaves que pasan por los puntos de control.

        Args:
            p0, p1, p2, p3: Puntos de control para la curva
            t: Parámetro de interpolación (0-1)

        Returns:
            Punto interpolado
        """
        # Implementación de Catmull-Rom
        t2 = t * t
        t3 = t2 * t

        # Calcular componente x
        x = 0.5 * (
            (2 * p1[0]) +
            (-p0[0] + p2[0]) * t +
            (2*p0[0] - 5*p1[0] + 4*p2[0] - p3[0]) * t2 +
            (-p0[0] + 3*p1[0] - 3*p2[0] + p3[0]) * t3
        )

        # Calcular componente y
        y = 0.5 * (
            (2 * p1[1]) +
            (-p0[1] + p2[1]) * t +
            (2*p0[1] - 5*p1[1] + 4*p2[1] - p3[1]) * t2 +
            (-p0[1] + 3*p1[1] - 3*p2[1] + p3[1]) * t3
        )

        return (int(x), int(y))

    def obtener_metricas(self) -> Dict[str, Any]:
        """
        Obtiene métricas de rendimiento del controlador de dibujo.

        Returns:
            Diccionario con métricas de rendimiento.
        """
        tiempo_total = time.time() - self.tiempo_inicio_metricas
        fps_promedio = self.frames_procesados / max(0.001, tiempo_total)

        return {
            "frames_procesados": self.frames_procesados,
            "puntos_interpolados": self.puntos_interpolados_total,
            "promedio_puntos_por_frame": self.puntos_interpolados_total / max(1, self.frames_procesados),
            "velocidad_actual": self.velocidad_actual,
            "fps_promedio": fps_promedio,
            "tamano_historial": len(self.historial_posiciones)
        }
