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
    UI_BUTTON_SIZE,
    UI_BUTTON_SPACING,
    UI_BUTTON_RADIUS,
    UI_PANEL_HEIGHT,
    COLORS,
    DEFAULT_COLOR
)

# Configuración del logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constantes para la interacción táctil
TIEMPO_ACTIVACION_BOTON: float = 0.2  # Reducido: Tiempo en segundos para activar un botón con el dedo
RADIO_INDICADOR_DEDO: int = 5  # Reducido de 10 a 5: Radio del círculo que muestra dónde está el dedo índice
BUTTON_FONT_SCALE: float = 0.35  # Reducido de 0.5 a 0.35: Escala del texto en los botones
BUTTON_COLOR: Tuple[int, int, int] = (100, 100, 100)  # Color de fondo de botones
BUTTON_TEXT_COLOR: Tuple[int, int, int] = (255, 255, 255)  # Color de texto de botones
BUTTON_WIDTH: int = UI_BUTTON_SIZE  # Ancho de botones
BUTTON_HEIGHT: int = UI_BUTTON_SIZE  # Alto de botones
BUTTON_MARGIN: int = UI_BUTTON_SPACING  # Margen entre botones

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
        color_borde = (255, 255, 255) if self.activo else (50, 50, 50)
        grosor_borde = 2 if self.activo else 1  # Reducido de 3/2 a 2/1

        # Primero dibujar un borde exterior para mayor visibilidad (esp. para botones de color)
        cv2.rectangle(frame,
                     (self.x - 1, self.y - 1),
                     (self.x + self.ancho + 1, self.y + self.alto + 1),
                     (0, 0, 0),
                     1)  # Reducido de 2 a 1: Borde negro exterior

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
            altura_barra = 2  # Reducido de 4 a 2
            ancho_progreso = int(self.ancho * self.progreso_activacion)

            # Dibujar la barra de progreso en la parte inferior del botón
            cv2.rectangle(frame,
                         (self.x, self.y + self.alto - altura_barra),
                         (self.x + ancho_progreso, self.y + self.alto),
                         (0, 255, 255),  # Color amarillo
                         -1)  # Rellenar

        # Dibujar el texto del botón si tiene
        if self.etiqueta:
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
                       1)  # Grosor fijo de 1 para el texto

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

    def __init__(self, lienzo) -> None:
        """
        Inicializa la interfaz de usuario.

        Args:
            lienzo: Instancia del lienzo de dibujo.
        """
        self.lienzo = lienzo
        self.ancho_ventana = lienzo.ancho
        self.alto_ventana = lienzo.alto

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

        # Inicializar los botones
        self._crear_botones()

        logger.info(f"Interfaz de usuario inicializada con dimensiones {self.ancho_ventana}x{self.alto_ventana}")

    def _crear_botones(self) -> None:
        """
        Crea todos los botones de la interfaz.
        """
        # Crear botones de colores
        self.crear_botones_colores(self.lienzo.cambiar_color)

        # Crear botón de limpieza
        self.crear_boton_limpiar(self.lienzo.limpiar)

    def agregar_boton(self, boton: Boton) -> None:
        """
        Agrega un botón a la interfaz.

        Args:
            boton: Botón a agregar.
        """
        self.botones.append(boton)

    def crear_botones_colores(self, accion_cambiar_color: Callable[[Tuple[int, int, int]], None]) -> None:
        """
        Crea botones para la selección de colores en una sola fila.

        Args:
            accion_cambiar_color: Función a ejecutar cuando se selecciona un color.
        """
        x_inicial = BUTTON_MARGIN
        y_inicial = BUTTON_MARGIN

        # Calcular el ancho total disponible
        ancho_disponible = self.ancho_ventana - (2 * BUTTON_MARGIN)

        # Determinar cuántos colores tenemos
        num_colores = len(COLORS)

        # Calcular el ancho de cada botón de color para que quepan todos en una fila
        # dejando un poco de espacio para el botón de limpiar
        ancho_boton_color = min(
            BUTTON_WIDTH,  # No exceder el ancho máximo
            max(30, (ancho_disponible - BUTTON_WIDTH - BUTTON_MARGIN) // num_colores - BUTTON_MARGIN)  # Asegurar un mínimo de 30px
        )

        # Ajustar altura si es necesario mantener proporciones
        alto_boton_color = min(BUTTON_HEIGHT, ancho_boton_color)
        alto_boton_color = max(30, alto_boton_color)  # Asegurar un mínimo de 30px de altura

        # Crear botones para cada color en la paleta en una sola fila
        for nombre_color, color_bgr in COLORS.items():
            # Crear una función de cierre para capturar el valor actual de color_bgr
            def crear_accion_color(color: Tuple[int, int, int]) -> Callable[[], None]:
                return lambda: accion_cambiar_color(color)

            boton = Boton(
                x=x_inicial,
                y=y_inicial,
                ancho=ancho_boton_color,
                alto=alto_boton_color,
                etiqueta="",  # Eliminar etiqueta para botones más pequeños
                color=color_bgr,  # Usar el color como fondo del botón
                color_texto=(0, 0, 0) if sum(color_bgr) > 380 else (255, 255, 255),  # Texto oscuro en fondos claros
                accion=crear_accion_color(color_bgr)
            )

            self.agregar_boton(boton)

            # Mover a la siguiente posición horizontalmente
            x_inicial += ancho_boton_color + BUTTON_MARGIN

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

    def superponer_lienzo(self, frame: np.ndarray) -> np.ndarray:
        """
        Superpone el lienzo en el frame y dibuja la interfaz.

        Args:
            frame: Frame donde dibujar la interfaz.

        Returns:
            Frame con el lienzo superpuesto y la interfaz dibujada.
        """
        # Superponer el lienzo
        frame_con_lienzo = self.lienzo.superponer_en_fotograma(frame)

        # Dibujar la interfaz
        return self.dibujar_interfaz(frame_con_lienzo)

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

    def procesar_eventos(self) -> None:
        """
        Procesa los eventos de la interfaz.

        Este método debe ser llamado periódicamente para actualizar
        el estado de la interfaz.
        """
        # Por ahora no hay eventos para procesar automáticamente
        pass

    def procesar_interaccion(self,
                           posicion_dedo: Optional[Tuple[int, int]],
                           indice_extendido: bool) -> None:
        """
        Procesa la interacción táctil para actualizar el estado de la interfaz.

        Args:
            posicion_dedo: Coordenadas (x, y) de la punta del dedo índice, o None si no hay dedo.
            indice_extendido: Indica si el dedo índice está extendido.
        """
        # Actualizar estado de interacción
        self.ultima_pos_dedo = posicion_dedo
        self.indice_extendido = indice_extendido

        # DEBUG: Log de inicio de procesamiento de interacción
        logger.debug(f"DEBUG-UI: Procesando interacción - pos={posicion_dedo}, extendido={indice_extendido}")

        # Medir tiempo de procesamiento
        inicio_procesamiento = time.time()

        # CAMBIO: Simplificado - Procesar interacción si hay una mano detectada
        # No dependemos tanto del estado extendido para mayor fiabilidad
        if posicion_dedo is not None:
            # DEBUG: Log de interacción activa
            logger.debug(f"DEBUG-UI: Interacción activa en posición {posicion_dedo}")

            # Verificar si el dedo está sobre algún botón
            boton_activado = None
            for i, boton in enumerate(self.botones):
                # DEBUG: Log de verificación de botón
                contiene = boton.contiene_punto(posicion_dedo)
                logger.debug(f"DEBUG-UI: Botón {i} en ({boton.x},{boton.y},{boton.ancho},{boton.alto}) - ¿Contiene punto? {contiene}")

                # DEBUG: Estado de hover del botón antes de actualizar
                hover_previo = boton.en_hover
                tiempo_hover_previo = boton.tiempo_inicio_hover
                progreso_previo = boton.progreso_activacion

                # Actualizar estado de hover y verificar si se activó
                activado = boton.actualizar_estado_hover(posicion_dedo)

                # DEBUG: Log después de actualizar hover
                logger.debug(f"DEBUG-UI: Botón {i} - Hover: {hover_previo}->{boton.en_hover}, " +
                             f"Tiempo: {tiempo_hover_previo}->{boton.tiempo_inicio_hover}, " +
                             f"Progreso: {progreso_previo:.2f}->{boton.progreso_activacion:.2f}, " +
                             f"Activado: {activado}")

                if activado:
                    boton_activado = boton
                    break

            # Si se ha activado un botón, desactivar el botón anterior y ejecutar la acción
            if boton_activado:
                # DEBUG: Log de activación de botón
                logger.debug(f"DEBUG-UI: ¡BOTÓN ACTIVADO! - {boton_activado.etiqueta}")

                # Desactivar el botón previamente seleccionado
                if self.boton_seleccionado and self.boton_seleccionado != boton_activado:
                    self.boton_seleccionado.desactivar()
                    logger.debug(f"DEBUG-UI: Botón previo desactivado - {self.boton_seleccionado.etiqueta}")

                # Activar el nuevo botón y ejecutar su acción
                boton_activado.activar()
                logger.debug(f"DEBUG-UI: Ejecutando acción del botón")
                boton_activado.ejecutar_accion()

                # Actualizar el botón seleccionado
                self.boton_seleccionado = boton_activado
                self._metricas['botones_activados'] += 1

                logger.info(f"Botón '{boton_activado.etiqueta}' seleccionado mediante gesto táctil")
        else:
            # Si no hay posición del dedo, reiniciar el estado de los botones
            logger.debug(f"DEBUG-UI: Reiniciando estado de hover de todos los botones")
            for i, boton in enumerate(self.botones):
                hover_previo = boton.en_hover
                tiempo_previo = boton.tiempo_inicio_hover
                progreso_previo = boton.progreso_activacion

                boton.en_hover = False
                boton.tiempo_inicio_hover = None
                boton.progreso_activacion = 0.0

                if hover_previo:
                    logger.debug(f"DEBUG-UI: Botón {i} reiniciado - Hover: {hover_previo}->False, " +
                                f"Tiempo: {tiempo_previo}->None, Progreso: {progreso_previo:.2f}->0.0")

        # Actualizar métricas de rendimiento
        ahora = time.time()
        self._metricas['tiempo_procesamiento'] = ahora - inicio_procesamiento
        self._metricas['frames_procesados'] += 1
        self._metricas['tiempo_ultimo_frame'] = ahora

    def obtener_metricas(self) -> Dict[str, Any]:
        """
        Obtiene métricas de rendimiento de la interfaz.

        Returns:
            Diccionario con métricas de rendimiento.
        """
        return {
            "tiempo_procesamiento_ms": self._metricas['tiempo_procesamiento'] * 1000,
            "frames_procesados": self._metricas['frames_procesados'],
            "botones_activados": self._metricas['botones_activados']
        }

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
