"""
Módulo para la interfaz de usuario de la pizarra digital.

Este módulo proporciona clases para crear y gestionar la interfaz de usuario,
incluyendo botones, visualización y eventos de mouse.
"""
import logging
import cv2
import numpy as np
import time
from typing import Tuple, List, Dict, Callable, Optional, Any

from ..config import (
    BUTTON_WIDTH,
    BUTTON_HEIGHT,
    BUTTON_MARGIN,
    BUTTON_COLOR,
    BUTTON_TEXT_COLOR,
    BUTTON_FONT_SCALE,
    COLOR_PALETTE
)

# Configuración del logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constantes para la interacción táctil
TIEMPO_ACTIVACION_BOTON: float = 0.6  # Tiempo en segundos para activar un botón con el dedo
RADIO_INDICADOR_DEDO: int = 10  # Radio del círculo que muestra dónde está el dedo índice

class Boton:
    """Clase para representar un botón en la interfaz."""

    def __init__(self,
                x: int,
                y: int,
                ancho: int,
                alto: int,
                etiqueta: str,
                color: Tuple[int, int, int] = BUTTON_COLOR,
                color_texto: Tuple[int, int, int] = BUTTON_TEXT_COLOR,
                accion: Optional[Callable[[], None]] = None) -> None:
        """
        Inicializa un botón.

        Args:
            x: Posición x del botón.
            y: Posición y del botón.
            ancho: Ancho del botón.
            alto: Alto del botón.
            etiqueta: Texto a mostrar en el botón.
            color: Color de fondo del botón (BGR).
            color_texto: Color del texto del botón (BGR).
            accion: Función a ejecutar cuando se hace clic en el botón.
        """
        self.x = x
        self.y = y
        self.ancho = ancho
        self.alto = alto
        self.etiqueta = etiqueta
        self.color = color
        self.color_texto = color_texto
        self.accion = accion
        self.activo = False

        # Variables para el gesto de "pulsación"
        self.tiempo_inicio_hover: Optional[float] = None
        self.en_hover = False
        self.progreso_activacion = 0.0  # Progreso de activación (0.0 a 1.0)

    def contiene_punto(self, punto: Tuple[int, int]) -> bool:
        """
        Verifica si un punto está dentro del botón.

        Args:
            punto: Coordenadas (x, y) del punto a verificar.

        Returns:
            True si el punto está dentro del botón, False en caso contrario.
        """
        x, y = punto
        return (self.x <= x <= self.x + self.ancho and
                self.y <= y <= self.y + self.alto)

    def dibujar(self, frame: np.ndarray) -> None:
        """
        Dibuja el botón en un fotograma.

        Args:
            frame: Fotograma donde dibujar el botón.
        """
        # Dibujar el rectángulo del botón
        color_borde = (255, 255, 255) if self.activo else (0, 0, 0)
        grosor_borde = 2 if self.activo else 1

        cv2.rectangle(frame,
                     (self.x, self.y),
                     (self.x + self.ancho, self.y + self.alto),
                     self.color,
                     -1)  # Rellenar

        cv2.rectangle(frame,
                     (self.x, self.y),
                     (self.x + self.ancho, self.y + self.alto),
                     color_borde,
                     grosor_borde)  # Borde

        # Si el botón está en hover con el dedo, mostrar indicador de progreso
        if self.en_hover and self.progreso_activacion > 0:
            # Calcular las dimensiones de la barra de progreso
            altura_barra = 4
            ancho_progreso = int(self.ancho * self.progreso_activacion)

            # Dibujar la barra de progreso en la parte inferior del botón
            cv2.rectangle(frame,
                         (self.x, self.y + self.alto - altura_barra),
                         (self.x + ancho_progreso, self.y + self.alto),
                         (0, 255, 255),  # Color amarillo
                         -1)  # Rellenar

        # Dibujar el texto del botón
        # Calcular la posición del texto para centrarlo en el botón
        (ancho_texto, alto_texto), _ = cv2.getTextSize(
            self.etiqueta, cv2.FONT_HERSHEY_SIMPLEX, BUTTON_FONT_SCALE, 1)

        pos_texto_x = self.x + (self.ancho - ancho_texto) // 2
        pos_texto_y = self.y + (self.alto + alto_texto) // 2

        cv2.putText(frame,
                   self.etiqueta,
                   (pos_texto_x, pos_texto_y),
                   cv2.FONT_HERSHEY_SIMPLEX,
                   BUTTON_FONT_SCALE,
                   self.color_texto,
                   1)

    def actualizar_estado_hover(self, punto: Optional[Tuple[int, int]]) -> bool:
        """
        Actualiza el estado de hover del botón y gestiona el tiempo para la activación.

        Args:
            punto: Coordenadas (x, y) del punto del dedo, o None si no hay dedo.

        Returns:
            True si el botón ha sido activado en esta actualización, False en caso contrario.
        """
        activado = False

        if punto and self.contiene_punto(punto):
            # El dedo está sobre el botón
            if not self.en_hover:
                # Acaba de empezar el hover
                self.en_hover = True
                self.tiempo_inicio_hover = time.time()
                self.progreso_activacion = 0.0
            else:
                # Continúa el hover, actualizar progreso
                tiempo_actual = time.time()
                tiempo_transcurrido = tiempo_actual - (self.tiempo_inicio_hover or tiempo_actual)
                self.progreso_activacion = min(1.0, tiempo_transcurrido / TIEMPO_ACTIVACION_BOTON)

                # Verificar si se cumple el tiempo para activar
                if self.progreso_activacion >= 1.0:
                    activado = True
                    self.tiempo_inicio_hover = None  # Reiniciar el tiempo
        else:
            # El dedo no está sobre el botón
            self.en_hover = False
            self.tiempo_inicio_hover = None
            self.progreso_activacion = 0.0

        return activado

    def activar(self) -> None:
        """Activa el botón (cambia su apariencia visual)."""
        self.activo = True

    def desactivar(self) -> None:
        """Desactiva el botón (restaura su apariencia normal)."""
        self.activo = False

    def ejecutar_accion(self) -> None:
        """Ejecuta la acción asociada al botón si existe."""
        if self.accion:
            self.accion()

class InterfazUsuario:
    """Clase para gestionar la interfaz de usuario de la pizarra digital."""

    def __init__(self, ancho_ventana: int, alto_ventana: int) -> None:
        """
        Inicializa la interfaz de usuario.

        Args:
            ancho_ventana: Ancho de la ventana de la aplicación.
            alto_ventana: Alto de la ventana de la aplicación.
        """
        self.ancho_ventana = ancho_ventana
        self.alto_ventana = alto_ventana

        # Lista de botones
        self.botones: List[Boton] = []

        # Botón actualmente seleccionado
        self.boton_seleccionado: Optional[Boton] = None

        # Eventos de mouse
        self.ultimo_clic: Optional[Tuple[int, int]] = None

        # Variables para interacción táctil
        self.ultima_pos_dedo: Optional[Tuple[int, int]] = None
        self.indice_extendido: bool = False
        self.ultimo_tiempo_interaccion = 0.0

        # Buffer para la interfaz - evita crear nuevos arrays en cada frame
        self._buffer_ui = None

        # Diccionario para rendimiento de interacción
        self._metricas = {
            'tiempo_ultimo_frame': time.time(),
            'tiempo_procesamiento': 0.0,
            'frames_procesados': 0,
            'botones_activados': 0
        }

        logger.info(f"Interfaz de usuario inicializada con dimensiones {ancho_ventana}x{alto_ventana}")

    def agregar_boton(self, boton: Boton) -> None:
        """
        Agrega un botón a la interfaz.

        Args:
            boton: Botón a agregar.
        """
        self.botones.append(boton)

    def crear_botones_colores(self, accion_cambiar_color: Callable[[Tuple[int, int, int]], None]) -> None:
        """
        Crea botones para la selección de colores.

        Args:
            accion_cambiar_color: Función a ejecutar cuando se selecciona un color.
        """
        x_inicial = BUTTON_MARGIN
        y_inicial = BUTTON_MARGIN

        # Crear botones para cada color en la paleta
        for nombre_color, color_bgr in COLOR_PALETTE.items():
            # Crear una función de cierre para capturar el valor actual de color_bgr
            def crear_accion_color(color: Tuple[int, int, int]) -> Callable[[], None]:
                return lambda: accion_cambiar_color(color)

            boton = Boton(
                x=x_inicial,
                y=y_inicial,
                ancho=BUTTON_WIDTH,
                alto=BUTTON_HEIGHT,
                etiqueta=nombre_color,
                color=color_bgr,  # Usar el color como fondo del botón
                color_texto=(0, 0, 0) if sum(color_bgr) > 380 else (255, 255, 255),  # Texto oscuro en fondos claros
                accion=crear_accion_color(color_bgr)
            )

            self.agregar_boton(boton)

            # Mover a la siguiente posición
            x_inicial += BUTTON_WIDTH + BUTTON_MARGIN

            # Si se llega al borde, saltar a la siguiente fila
            if x_inicial + BUTTON_WIDTH > self.ancho_ventana // 2:
                x_inicial = BUTTON_MARGIN
                y_inicial += BUTTON_HEIGHT + BUTTON_MARGIN

    def crear_boton_limpiar(self,
                          accion_limpiar: Callable[[], None],
                          x: Optional[int] = None,
                          y: Optional[int] = None) -> None:
        """
        Crea un botón para limpiar el lienzo.

        Args:
            accion_limpiar: Función a ejecutar cuando se hace clic en el botón.
            x: Posición x del botón (opcional).
            y: Posición y del botón (opcional).
        """
        if x is None:
            # Posicionar en la esquina superior derecha
            x = self.ancho_ventana - BUTTON_WIDTH - BUTTON_MARGIN

        if y is None:
            y = BUTTON_MARGIN

        boton_limpiar = Boton(
            x=x,
            y=y,
            ancho=BUTTON_WIDTH,
            alto=BUTTON_HEIGHT,
            etiqueta="Borrar",
            color=(50, 50, 200),  # Rojo en BGR
            color_texto=(255, 255, 255),
            accion=accion_limpiar
        )

        self.agregar_boton(boton_limpiar)

    def dibujar_interfaz(self, frame: np.ndarray) -> np.ndarray:
        """
        Dibuja la interfaz de usuario en un fotograma sin procesar interacción.

        Esta versión optimizada evita crear un nuevo array en cada frame.

        Args:
            frame: Fotograma donde dibujar la interfaz.

        Returns:
            Fotograma con la interfaz dibujada.
        """
        # Crear o redimensionar el buffer si es necesario
        if self._buffer_ui is None or self._buffer_ui.shape != frame.shape:
            self._buffer_ui = np.zeros_like(frame)

        # Copiar el fotograma para no modificar el original
        np.copyto(self._buffer_ui, frame)

        # Dibujar todos los botones
        for boton in self.botones:
            boton.dibujar(self._buffer_ui)

        # Dibujar indicador de posición del dedo si está presente
        if self.ultima_pos_dedo:
            x, y = self.ultima_pos_dedo
            cv2.circle(self._buffer_ui, (x, y), RADIO_INDICADOR_DEDO, (0, 255, 255), 2)

            # Agregar un pequeño punto en el centro para mejor precisión
            cv2.circle(self._buffer_ui, (x, y), 2, (0, 255, 255), -1)

        return self._buffer_ui

    def procesar_interaccion(self,
                           frame: np.ndarray,
                           posicion_dedo: Optional[Tuple[int, int]],
                           indice_extendido: bool) -> np.ndarray:
        """
        Procesa la interacción táctil y dibuja la interfaz en un fotograma.

        Este método combina la actualización del estado de interacción y el dibujo
        de la interfaz en una sola llamada para mayor eficiencia.

        Args:
            frame: Fotograma donde dibujar la interfaz.
            posicion_dedo: Coordenadas (x, y) de la punta del dedo índice, o None si no hay dedo.
            indice_extendido: Indica si el dedo índice está extendido.

        Returns:
            Fotograma con la interfaz dibujada y la interacción procesada.
        """
        # Actualizar estado de interacción
        self.ultima_pos_dedo = posicion_dedo
        self.indice_extendido = indice_extendido

        # Medir tiempo de procesamiento
        inicio_procesamiento = time.time()

        # Procesar interacción si hay un dedo y no está extendido (modo interacción, no dibujo)
        if posicion_dedo is not None and not indice_extendido:
            # Verificar si el dedo está sobre algún botón
            boton_activado = None
            for boton in self.botones:
                if boton.actualizar_estado_hover(posicion_dedo):
                    boton_activado = boton
                    break

            # Si se ha activado un botón, desactivar el botón anterior y ejecutar la acción
            if boton_activado:
                # Desactivar el botón previamente seleccionado
                if self.boton_seleccionado and self.boton_seleccionado != boton_activado:
                    self.boton_seleccionado.desactivar()

                # Activar el nuevo botón y ejecutar su acción
                boton_activado.activar()
                boton_activado.ejecutar_accion()

                # Actualizar el botón seleccionado
                self.boton_seleccionado = boton_activado
                self._metricas['botones_activados'] += 1

                logger.info(f"Botón '{boton_activado.etiqueta}' seleccionado mediante gesto táctil")
        elif posicion_dedo is None or indice_extendido:
            # Si no hay posición del dedo o está dibujando, reiniciar el estado de los botones
            for boton in self.botones:
                boton.en_hover = False
                boton.tiempo_inicio_hover = None
                boton.progreso_activacion = 0.0

        # Dibujar la interfaz
        resultado = self.dibujar_interfaz(frame)

        # Actualizar métricas de rendimiento
        ahora = time.time()
        self._metricas['tiempo_procesamiento'] = ahora - inicio_procesamiento
        self._metricas['frames_procesados'] += 1
        fps_ui = 1.0 / max(0.001, ahora - self._metricas['tiempo_ultimo_frame'])
        self._metricas['tiempo_ultimo_frame'] = ahora

        # Opcionalmente, mostrar métricas de rendimiento en la UI
        # cv2.putText(resultado, f"UI FPS: {fps_ui:.1f}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

        return resultado

    def dibujar(self, frame: np.ndarray) -> np.ndarray:
        """
        Método original para dibujar la interfaz (mantenido por compatibilidad).

        Utilizar dibujar_interfaz() o procesar_interaccion() para mejor rendimiento.

        Args:
            frame: Fotograma donde dibujar la interfaz.

        Returns:
            Fotograma con la interfaz dibujada.
        """
        return self.dibujar_interfaz(frame)

    def procesar_evento_mouse(self, evento: int, x: int, y: int, *args) -> None:
        """
        Procesa un evento de mouse.

        Args:
            evento: Tipo de evento (cv2.EVENT_*).
            x: Coordenada x del evento.
            y: Coordenada y del evento.
            args: Argumentos adicionales del evento.
        """
        if evento == cv2.EVENT_LBUTTONDOWN:
            # Guardar la posición del clic
            self.ultimo_clic = (x, y)

            # Verificar si se hizo clic en algún botón
            for boton in self.botones:
                if boton.contiene_punto((x, y)):
                    # Desactivar el botón previamente seleccionado
                    if self.boton_seleccionado:
                        self.boton_seleccionado.desactivar()

                    # Activar el nuevo botón y ejecutar su acción
                    boton.activar()
                    boton.ejecutar_accion()

                    # Actualizar el botón seleccionado
                    self.boton_seleccionado = boton

                    logger.info(f"Botón '{boton.etiqueta}' seleccionado")
                    return  # Salir después de encontrar un botón

    def procesar_posicion_dedo(self, posicion: Optional[Tuple[int, int]], dibujando: bool) -> None:
        """
        Procesa la posición del dedo índice para interacción táctil.

        Método original mantenido por compatibilidad. Usar procesar_interaccion()
        para mayor eficiencia.

        Args:
            posicion: Coordenadas (x, y) de la punta del dedo índice, o None si no hay dedo.
            dibujando: Indica si el dedo está en modo dibujo (extendido).
        """
        self.ultima_pos_dedo = posicion
        self.indice_extendido = dibujando

        # Si no hay posición del dedo o está dibujando, reiniciar todos los botones
        if posicion is None or dibujando:
            for boton in self.botones:
                boton.en_hover = False
                boton.tiempo_inicio_hover = None
                boton.progreso_activacion = 0.0
            return

        # Verificar si el dedo está sobre algún botón
        boton_activado = None

        for boton in self.botones:
            if boton.actualizar_estado_hover(posicion):
                boton_activado = boton
                break

        # Si se ha activado un botón, desactivar el botón anterior y ejecutar la acción
        if boton_activado:
            # Desactivar el botón previamente seleccionado
            if self.boton_seleccionado and self.boton_seleccionado != boton_activado:
                self.boton_seleccionado.desactivar()

            # Activar el nuevo botón y ejecutar su acción
            boton_activado.activar()
            boton_activado.ejecutar_accion()

            # Actualizar el botón seleccionado
            self.boton_seleccionado = boton_activado

            logger.info(f"Botón '{boton_activado.etiqueta}' seleccionado mediante gesto táctil")

    def registrar_eventos_mouse(self, nombre_ventana: str) -> None:
        """
        Registra la función de callback para eventos de mouse.

        Args:
            nombre_ventana: Nombre de la ventana de OpenCV.
        """
        # Crear una función de cierre para capturar self
        def callback_mouse(evento: int, x: int, y: int, flags: int, param: Any) -> None:
            self.procesar_evento_mouse(evento, x, y, flags, param)

        cv2.setMouseCallback(nombre_ventana, callback_mouse)
