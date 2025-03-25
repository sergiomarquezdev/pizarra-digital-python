"""
Módulo principal de la aplicación de pizarra digital.

Este módulo integra todos los componentes para crear la aplicación completa:
- Captura de video
- Detección de manos
- Lienzo de dibujo
- Interfaz de usuario
"""
import logging
import time
import cv2
import numpy as np
from typing import Optional, Tuple, Dict, Any

from .config import (
    CAMERA_WIDTH,
    CAMERA_HEIGHT,
    CANVAS_WIDTH,
    CANVAS_HEIGHT,
    CANVAS_BACKGROUND,
    APP_NAME,
    EXIT_KEY,
    DEFAULT_DRAWING_COLOR
)
from .captura.captura import inicializar_camara, leer_fotograma, liberar_camara, ajustar_parametros_camara, CameraError
from .deteccion.deteccion import DetectorManos
from .lienzo.lienzo import Lienzo
from .dibujo.dibujo import DibujoMano
from .interfaz.interfaz import InterfazUsuario

# Configuración del logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Parámetros de rendimiento
ESCALA_CAMARA: float = 1.0  # Factor de escala para la cámara (1.0 = tamaño completo)
MOSTRAR_FPS: bool = True  # Mostrar contador de FPS
OPTIMIZAR_RENDIMIENTO: bool = True  # Aplicar optimizaciones automáticas

def calcular_fps(tiempo_inicio: float, num_frames: int = 10) -> float:
    """
    Calcula los fotogramas por segundo actuales.

    Args:
        tiempo_inicio: Tiempo de inicio para el cálculo.
        num_frames: Número de fotogramas para promediar.

    Returns:
        FPS calculados.
    """
    return num_frames / (time.time() - tiempo_inicio)

def mostrar_fps(frame: np.ndarray, fps: float) -> np.ndarray:
    """
    Muestra los FPS en un fotograma.

    Args:
        frame: Fotograma donde mostrar los FPS.
        fps: Valor de FPS a mostrar.

    Returns:
        Fotograma con los FPS mostrados.
    """
    # Posición en la esquina inferior izquierda
    posicion = (10, frame.shape[0] - 10)

    # Crear texto
    texto = f"FPS: {fps:.1f}"

    # Dibujar texto con fondo
    cv2.putText(frame, texto, posicion, cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (0, 0, 0), 2)  # Sombra
    cv2.putText(frame, texto, posicion, cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (255, 255, 255), 1)  # Texto

    return frame

def optimizar_rendimiento_automatico(fps: float, detector: DetectorManos, escala_actual: float) -> Tuple[bool, float]:
    """
    Ajusta dinámicamente los parámetros para optimizar el rendimiento.

    Args:
        fps: FPS actuales
        detector: Instancia del detector de manos
        escala_actual: Factor de escala actual de la cámara

    Returns:
        Tupla con (cambios_aplicados, nueva_escala)
    """
    cambios_aplicados = False
    nueva_escala = escala_actual

    # Si los FPS bajan demasiado, reducir la resolución
    if fps < 15 and escala_actual > 0.5:
        nueva_escala = max(0.5, escala_actual - 0.1)
        logger.info(f"FPS bajos ({fps:.1f}), reduciendo escala a {nueva_escala:.1f}")
        cambios_aplicados = True

    # Si los FPS son buenos, aumentar gradualmente la resolución
    elif fps > 25 and escala_actual < 1.0:
        nueva_escala = min(1.0, escala_actual + 0.05)
        logger.info(f"FPS buenos ({fps:.1f}), aumentando escala a {nueva_escala:.1f}")
        cambios_aplicados = True

    return cambios_aplicados, nueva_escala

def inicializar_app() -> Tuple[Optional[cv2.VideoCapture], DetectorManos, Lienzo, DibujoMano, InterfazUsuario]:
    """
    Inicializa todos los componentes de la aplicación.

    Returns:
        Tupla con los componentes principales inicializados.
    """
    try:
        # Inicializar la cámara
        logger.info("Inicializando la cámara...")
        camara = inicializar_camara()

        # Configurar parámetros para mejor rendimiento
        if OPTIMIZAR_RENDIMIENTO:
            parametros_optimizados = {
                'fps': 30,
                'buffer': 1,
                'exposicion': -3,  # Menor exposición puede mejorar la velocidad
            }
            ajustar_parametros_camara(camara, parametros_optimizados)

        # Inicializar el detector de manos
        logger.info("Inicializando el detector de manos...")
        detector = DetectorManos()

        # Inicializar el lienzo
        logger.info("Inicializando el lienzo...")
        lienzo = Lienzo(CANVAS_WIDTH, CANVAS_HEIGHT, CANVAS_BACKGROUND)

        # Inicializar el controlador de dibujo
        logger.info("Inicializando el controlador de dibujo...")
        dibujo = DibujoMano(lienzo)

        # Inicializar la interfaz de usuario
        logger.info("Inicializando la interfaz de usuario...")
        interfaz = InterfazUsuario(CAMERA_WIDTH, CAMERA_HEIGHT)

        # Configurar las acciones de los botones
        def accion_cambiar_color(color: Tuple[int, int, int]) -> None:
            lienzo.cambiar_color(color)
            logger.info(f"Color cambiado a {color}")

        def accion_limpiar_lienzo() -> None:
            lienzo.limpiar()
            logger.info("Lienzo limpiado")

        # Crear los botones
        interfaz.crear_botones_colores(accion_cambiar_color)
        interfaz.crear_boton_limpiar(accion_limpiar_lienzo)

        return camara, detector, lienzo, dibujo, interfaz

    except CameraError as e:
        logger.error(f"Error al inicializar la cámara: {e}")
        return None, DetectorManos(), Lienzo(), DibujoMano(Lienzo()), InterfazUsuario(CAMERA_WIDTH, CAMERA_HEIGHT)

    except Exception as e:
        logger.error(f"Error inesperado durante la inicialización: {e}")
        return None, DetectorManos(), Lienzo(), DibujoMano(Lienzo()), InterfazUsuario(CAMERA_WIDTH, CAMERA_HEIGHT)

def ejecutar_app() -> None:
    """
    Ejecuta la aplicación principal de pizarra digital.
    """
    logger.info("Iniciando la aplicación de pizarra digital...")

    # Inicializar componentes
    camara, detector, lienzo, dibujo, interfaz = inicializar_app()

    if camara is None:
        logger.error("No se pudo inicializar la cámara. Saliendo...")
        return

    # Crear ventana y configurar eventos de mouse
    cv2.namedWindow(APP_NAME)
    interfaz.registrar_eventos_mouse(APP_NAME)

    # Variables para el cálculo de FPS
    contador_frames = 0
    tiempo_inicio_fps = time.time()
    fps = 0.0

    # Control de escala para ajuste de rendimiento
    escala_actual = ESCALA_CAMARA
    tiempo_ultimo_ajuste = time.time()

    # Medición de tiempos para diagnóstico
    tiempos_operaciones: Dict[str, float] = {
        'camara': 0.0,
        'detector': 0.0,
        'dibujo': 0.0,
        'lienzo': 0.0,
        'interfaz': 0.0
    }

    try:
        # Bucle principal
        logger.info("Iniciando bucle principal de la aplicación...")
        while True:
            t_inicio_total = time.time()

            # Capturar fotograma
            t_inicio = time.time()
            ret, frame = leer_fotograma(camara, escala=escala_actual)
            tiempos_operaciones['camara'] = time.time() - t_inicio

            if not ret or frame is None:
                logger.warning("No se pudo leer el fotograma de la cámara")
                break

            # Actualizar contador de FPS
            contador_frames += 1
            if contador_frames >= 10:
                fps = calcular_fps(tiempo_inicio_fps, contador_frames)
                contador_frames = 0
                tiempo_inicio_fps = time.time()

                # Optimizar rendimiento si está habilitado
                if OPTIMIZAR_RENDIMIENTO and time.time() - tiempo_ultimo_ajuste > 2.0:
                    cambios, nueva_escala = optimizar_rendimiento_automatico(fps, detector, escala_actual)
                    if cambios:
                        escala_actual = nueva_escala
                        tiempo_ultimo_ajuste = time.time()

            # Procesar fotograma para detectar manos
            t_inicio = time.time()
            frame_procesado, manos_detectadas = detector.procesar_fotograma(frame, dibujar_landmarks=True)
            tiempos_operaciones['detector'] = time.time() - t_inicio

            # Procesar la primera mano detectada (si hay alguna)
            t_inicio = time.time()
            if manos_detectadas:
                dibujo.procesar_mano(manos_detectadas[0], detector)
            tiempos_operaciones['dibujo'] = time.time() - t_inicio

            # Superponer el lienzo en el fotograma
            t_inicio = time.time()
            frame_con_lienzo = lienzo.superponer_en_fotograma(frame_procesado)
            tiempos_operaciones['lienzo'] = time.time() - t_inicio

            # Dibujar la interfaz de usuario
            t_inicio = time.time()
            frame_final = interfaz.dibujar(frame_con_lienzo)
            tiempos_operaciones['interfaz'] = time.time() - t_inicio

            # Mostrar FPS
            if MOSTRAR_FPS:
                mostrar_fps(frame_final, fps)

                # Para diagnóstico, registrar ocasionalmente los tiempos de las operaciones
                if contador_frames == 0:
                    logger.debug(f"Tiempos (ms): "
                                f"cámara={tiempos_operaciones['camara']*1000:.1f}, "
                                f"detector={tiempos_operaciones['detector']*1000:.1f}, "
                                f"dibujo={tiempos_operaciones['dibujo']*1000:.1f}, "
                                f"lienzo={tiempos_operaciones['lienzo']*1000:.1f}, "
                                f"interfaz={tiempos_operaciones['interfaz']*1000:.1f}")

            # Mostrar el fotograma resultante
            cv2.imshow(APP_NAME, frame_final)

            # Salir si se presiona la tecla configurada
            if cv2.waitKey(1) & 0xFF == ord(EXIT_KEY):
                logger.info(f"Tecla '{EXIT_KEY}' presionada. Saliendo...")
                break

    except Exception as e:
        logger.error(f"Error durante la ejecución: {e}")

    finally:
        # Liberar recursos
        logger.info("Liberando recursos...")
        liberar_camara(camara)
        detector.liberar_recursos()
        cv2.destroyAllWindows()
        logger.info("Aplicación finalizada")

if __name__ == "__main__":
    ejecutar_app()
