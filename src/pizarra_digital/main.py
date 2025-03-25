"""
Módulo principal de la aplicación Pizarra Digital.

Este script inicializa los componentes principales y ejecuta el bucle
principal de la aplicación.
"""
import time
import logging
import numpy as np
import cv2
from datetime import datetime
from typing import Dict, Any, Optional, Tuple, List

from .captura.captura import CapturaVideo, CapturaVideoAsync
from .deteccion.deteccion import DetectorManos
from .dibujo.dibujo import DibujoMano
from .lienzo.lienzo import Lienzo
from .interfaz.interfaz import InterfazUsuario
from .config import (
    CAMERA_WIDTH,
    CAMERA_HEIGHT,
    CAMERA_INDEX,
    CAMERA_FPS,
    MEDIAPIPE_MAX_HANDS,
    MEDIAPIPE_DETECTION_CONFIDENCE,
    MEDIAPIPE_TRACKING_CONFIDENCE,
    CANVAS_WIDTH,
    CANVAS_HEIGHT,
    CANVAS_BACKGROUND_COLOR,
    DEFAULT_COLOR,
    OPTIMIZATION_RESIZE_FACTOR,
    OPTIMIZATION_SKIP_FRAMES,
    OPTIMIZATION_USE_ASYNC_CAPTURE,
    OPTIMIZATION_SHOW_METRICS,
    OPTIMIZATION_QUALITY,
    OPTIMIZATION_SOLO_MANO_DERECHA,
    UI_FOOTER_HEIGHT,
    CAMERA_MIRROR_MODE,
    MEDIAPIPE_MANO_IZQUIERDA
)

# Configuración del sistema de logging
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Límites para optimización de rendimiento
MIN_FPS_OBJETIVO = 20
MAX_FPS_OBJETIVO = 60

def calcular_fps(tiempos_frame: List[float], max_frames: int = 30) -> float:
    """
    Calcula los FPS basados en los tiempos de frame recientes.

    Args:
        tiempos_frame: Lista de tiempos de procesamiento de frames
        max_frames: Número máximo de frames a considerar

    Returns:
        FPS calculados
    """
    if len(tiempos_frame) < 2:
        return 0.0

    # Usar solo los últimos max_frames para el cálculo
    tiempos_recientes = tiempos_frame[-max_frames:]

    # Calcular la diferencia de tiempo promedio entre frames
    diferencias = [tiempos_recientes[i] - tiempos_recientes[i-1]
                  for i in range(1, len(tiempos_recientes))]

    if not diferencias:
        return 0.0

    tiempo_promedio = sum(diferencias) / len(diferencias)

    # Evitar división por cero
    if tiempo_promedio <= 0:
        return 0.0

    return 1.0 / tiempo_promedio

def optimizar_rendimiento(fps_actual: float, factor_calidad: float) -> Dict[str, Any]:
    """
    Ajusta parámetros de rendimiento basados en FPS actuales.

    Args:
        fps_actual: FPS actuales de la aplicación
        factor_calidad: Factor de calidad (0-1, donde 1 es máxima calidad)

    Returns:
        Diccionario con parámetros optimizados
    """
    # Valores por defecto
    parametros = {
        "skip_frames": 0,
        "resize_factor": 1.0,
        "use_prediction": True,
        "interpolation_points": 5,
        "buffer_operations": True
    }

    # Ajustar según FPS
    if fps_actual < MIN_FPS_OBJETIVO:
        # Bajo rendimiento, priorizar velocidad
        calidad_ajustada = factor_calidad * 0.5  # Reducir calidad
        parametros["skip_frames"] = max(1, int(3 * (1 - calidad_ajustada)))
        parametros["resize_factor"] = max(0.5, 1.0 - (0.5 * (1 - calidad_ajustada)))
        parametros["interpolation_points"] = max(2, int(5 * calidad_ajustada))
    elif fps_actual > MAX_FPS_OBJETIVO:
        # Alto rendimiento, priorizar calidad
        calidad_ajustada = min(1.0, factor_calidad * 1.2)  # Aumentar calidad
        parametros["skip_frames"] = 0
        parametros["resize_factor"] = 1.0
        parametros["interpolation_points"] = max(5, int(10 * calidad_ajustada))
    else:
        # Rendimiento equilibrado
        parametros["skip_frames"] = max(0, int(2 * (1 - factor_calidad)))
        parametros["resize_factor"] = max(0.6, 1.0 - (0.4 * (1 - factor_calidad)))
        parametros["interpolation_points"] = max(3, int(8 * factor_calidad))

    return parametros

def dibujar_footer_metricas(frame: np.ndarray,
                          metricas: Dict[str, Any],
                          fps: float,
                          estado_dibujo: str,
                          dibujo_habilitado: bool,
                          solo_mano_derecha: bool = False,
                          solo_mano_izquierda: bool = False) -> np.ndarray:
    """
    Dibuja un footer en la parte inferior de la pantalla con las métricas.

    Args:
        frame: Frame donde dibujar el footer
        metricas: Diccionario con métricas a mostrar
        fps: Frames por segundo actuales
        estado_dibujo: Estado del dibujo ("ACTIVADO"/"DESACTIVADO")
        dibujo_habilitado: Si el dibujo está habilitado
        solo_mano_derecha: Si es True, solo se detecta la mano derecha
        solo_mano_izquierda: Si es True, solo se detecta la mano izquierda

    Returns:
        Frame con el footer de métricas dibujado
    """
    # Crear copia para no modificar el original
    result = frame.copy()

    # Calcular posición y tamaño del footer
    altura_footer = UI_FOOTER_HEIGHT
    y_footer = frame.shape[0] - altura_footer

    # Dibujar fondo para el footer
    overlay = result.copy()
    cv2.rectangle(overlay,  # type: ignore
                 (0, y_footer),
                 (frame.shape[1], frame.shape[0]),
                 (30, 30, 30),
                 cv2.FILLED)  # type: ignore

    # Aplicar transparencia
    alpha = 0.7
    cv2.addWeighted(overlay, alpha, result, 1 - alpha, 0, result)  # type: ignore

    # Preparar texto para el footer
    texto_footer = f"FPS: {int(fps)} | "

    # Añadir estado del dibujo
    color_estado = (0, 255, 0) if dibujo_habilitado else (0, 0, 255)

    # Añadir información relevante
    if metricas:
        # Seleccionar métricas importantes a mostrar
        tiempo_deteccion = metricas.get("Tiempo deteccion (ms)", 0)
        vel_movimiento = metricas.get("Vel. movimiento", 0)

        texto_footer += f"Detección: {tiempo_deteccion:.1f}ms | Velocidad: {vel_movimiento:.1f} | "

    # Añadir texto sobre gesto de pinza al final
    texto_footer += "Pinza (pulgar+índice): activar/desactivar dibujo"

    # Mostrar qué mano se está detectando
    if solo_mano_derecha:
        texto_mano = "MANO DERECHA"
    elif solo_mano_izquierda:
        texto_mano = "MANO IZQUIERDA"
    else:
        texto_mano = "AMBAS MANOS"

    # Mostrar texto del footer
    cv2.putText(result,  # type: ignore
               texto_footer,
               (10, y_footer + altura_footer - 7),
               cv2.FONT_HERSHEY_SIMPLEX,  # type: ignore
               0.5,
               (200, 200, 200),
               1,
               cv2.LINE_AA)  # type: ignore

    # Mostrar estado de dibujo y mano detectada en el lado derecho
    cv2.putText(result,  # type: ignore
               f"{estado_dibujo} | {texto_mano}",
               (frame.shape[1] - 300, y_footer + altura_footer - 7),
               cv2.FONT_HERSHEY_SIMPLEX,  # type: ignore
               0.5,
               color_estado,
               1,
               cv2.LINE_AA)  # type: ignore

    return result

def inicializar_app(use_async_capture: bool = OPTIMIZATION_USE_ASYNC_CAPTURE,
                   show_metrics: bool = OPTIMIZATION_SHOW_METRICS,
                   quality_factor: float = OPTIMIZATION_QUALITY,
                   solo_mano_derecha: bool = OPTIMIZATION_SOLO_MANO_DERECHA,
                   solo_mano_izquierda: bool = MEDIAPIPE_MANO_IZQUIERDA,
                   mirror_mode: bool = CAMERA_MIRROR_MODE) -> Dict[str, Any]:
    """
    Inicializa los componentes principales de la aplicación.

    Args:
        use_async_capture: Si es True, usa captura asíncrona de video
        show_metrics: Si es True, muestra métricas de rendimiento en pantalla
        quality_factor: Factor de calidad (0-1) que afecta el rendimiento
        solo_mano_derecha: Si es True, solo detecta la mano derecha
        solo_mano_izquierda: Si es True, solo detecta la mano izquierda
        mirror_mode: Si es True, muestra la cámara en modo espejo

    Returns:
        Diccionario con componentes inicializados
    """
    logger.info(f"Inicializando aplicación con calidad={quality_factor}, "
               f"captura_asincrona={use_async_capture}, "
               f"mostrar_metricas={show_metrics}, "
               f"solo_mano_derecha={solo_mano_derecha}, "
               f"solo_mano_izquierda={solo_mano_izquierda}, "
               f"modo_espejo={mirror_mode}")

    # Inicializar captura de video
    if use_async_capture:
        captura = CapturaVideoAsync(
            camara_index=CAMERA_INDEX,
            width=CAMERA_WIDTH,
            height=CAMERA_HEIGHT,
            fps=CAMERA_FPS
        )
    else:
        captura = CapturaVideo(
            camara_index=CAMERA_INDEX,
            width=CAMERA_WIDTH,
            height=CAMERA_HEIGHT
        )

    # Inicializar detector de manos
    detector = DetectorManos(
        max_num_hands=MEDIAPIPE_MAX_HANDS,
        min_detection_confidence=MEDIAPIPE_DETECTION_CONFIDENCE,
        min_tracking_confidence=MEDIAPIPE_TRACKING_CONFIDENCE,
        solo_mano_derecha=solo_mano_derecha,
        mano_izquierda=solo_mano_izquierda,
        enable_optimizations=True,
        mirror_mode=mirror_mode
    )

    # Inicializar lienzo
    lienzo = Lienzo(
        width=CANVAS_WIDTH,
        height=CANVAS_HEIGHT,
        background_color=CANVAS_BACKGROUND_COLOR,
        enable_optimizations=True
    )

    logger.info(f"Lienzo inicializado con dimensiones {CANVAS_WIDTH}x{CANVAS_HEIGHT}")

    # Inicializar controlador de dibujo
    controlador_dibujo = DibujoMano(lienzo)

    # Inicializar interfaz
    interfaz = InterfazUsuario(lienzo)

    return {
        "captura": captura,
        "detector": detector,
        "lienzo": lienzo,
        "controlador_dibujo": controlador_dibujo,
        "interfaz": interfaz,
        "show_metrics": show_metrics,
        "quality_factor": quality_factor
    }

def ejecutar_app(componentes: Dict[str, Any]) -> None:
    """
    Ejecuta el bucle principal de la aplicación.

    Args:
        componentes: Diccionario con los componentes inicializados
    """
    captura = componentes["captura"]
    detector = componentes["detector"]
    lienzo = componentes["lienzo"]
    controlador_dibujo = componentes["controlador_dibujo"]
    interfaz = componentes["interfaz"]
    show_metrics = componentes["show_metrics"]
    quality_factor = componentes["quality_factor"]
    mirror_mode = componentes.get("mirror_mode", CAMERA_MIRROR_MODE)
    solo_mano_derecha = detector.solo_mano_derecha
    solo_mano_izquierda = detector.mano_izquierda

    # Fijar resoluciones para debug - AÑADIDO PARA DEPURACIÓN
    factor_resize_fijo = 1.0  # Usar factor 1.0 para evitar transformación de coordenadas

    # Imprimir resoluciones para depuración
    logger.debug(f"DEPURACIÓN - Resolución cámara: {CAMERA_WIDTH}x{CAMERA_HEIGHT}")
    logger.debug(f"DEPURACIÓN - Resolución lienzo: {CANVAS_WIDTH}x{CANVAS_HEIGHT}")
    logger.debug(f"DEPURACIÓN - Resoluciones iguales: {CAMERA_WIDTH == CANVAS_WIDTH and CAMERA_HEIGHT == CANVAS_HEIGHT}")
    logger.debug(f"DEPURACIÓN - Modo espejo: {mirror_mode}")

    # Inicializar variables para cálculo de FPS y rendimiento
    tiempos_frame = []
    contador_frames = 0
    tiempo_inicio = time.time()
    ultimo_tiempo_optimizacion = tiempo_inicio
    parametros_optimizacion = optimizar_rendimiento(30.0, quality_factor)
    # Forzar factor_resize para depuración
    parametros_optimizacion["resize_factor"] = factor_resize_fijo

    logger.debug(f"DEPURACIÓN - Parámetros iniciales: {parametros_optimizacion}")

    # Nombre de la ventana
    nombre_ventana = "Pizarra Digital"
    cv2.namedWindow(nombre_ventana, cv2.WINDOW_NORMAL)  # type: ignore

    # Variables para control de gestos
    gestos_pinza = []  # Lista para almacenar detecciones recientes de gestos de pinza
    ultimo_toggle = 0.0  # Timestamp del último toggle para evitar activaciones múltiples

    logger.info("Iniciando bucle principal de la aplicación")

    try:
        while True:
            tiempo_frame_inicio = time.time()

            # Capturar frame
            ret, frame = captura.read()
            if not ret:
                logger.warning("Error al capturar frame")
                break

            # Aplicar efecto espejo si está activado
            if mirror_mode:
                frame = cv2.flip(frame, 1)  # type: ignore # Voltear horizontalmente (efecto espejo)

            # LOG de dimensiones del frame capturado
            logger.debug(f"DEPURACIÓN - Frame capturado: {frame.shape[1]}x{frame.shape[0]}")

            # Redimensionar frame si es necesario para rendimiento - DESHABILITADO para evitar problemas de coordenadas
            factor_resize = 1.0  # Siempre usar factor 1.0
            frame_procesamiento = frame.copy()
            ancho_orig, alto_orig = frame.shape[1], frame.shape[0]
            logger.debug(f"DEPURACIÓN - Frame sin redimensionar: {ancho_orig}x{alto_orig}")

            # Procesar frame con detector de manos (cada N frames)
            saltar_frame = contador_frames % max(1, parametros_optimizacion["skip_frames"] + 1) != 0
            frame_procesado, manos = detector.procesar_fotograma(
                frame_procesamiento,
                dibujar_landmarks=True,
                usar_prediccion=parametros_optimizacion["use_prediction"] and saltar_frame
            )

            # Actualizar interfaz de usuario
            interfaz.procesar_eventos()

            # Si se detectan manos, procesar interacción con la interfaz
            if manos and len(manos) > 0:
                # Usar solo la primera mano detectada
                mano = manos[0]

                # Obtener la punta del dedo índice
                punta_indice = detector.obtener_punta_indice(mano) if mano else None

                # Obtener si el dedo índice está extendido o no
                indice_extendido = detector.es_indice_extendido(mano) if mano else False

                # Obtener si se está haciendo gesto de pinza
                gesto_pinza = detector.es_gesto_pinza(mano) if mano else False

                # Loguear resultados para debug
                if mano:
                    logger.debug(f"DEPURACIÓN - Mano detectada: punta_indice={punta_indice}, "
                               f"extendido={indice_extendido}, gesto_pinza={gesto_pinza}")
                else:
                    logger.debug("DEPURACIÓN - No se detectaron manos")

                # Verificar si hay cambio en el gesto de pinza para toggle de dibujo
                if mano and gesto_pinza:
                    # Almacenar estado de gestos recientes para estabilidad
                    gestos_pinza.append(True)
                    if len(gestos_pinza) > 3:  # Mantener solo los últimos 3
                        gestos_pinza.pop(0)

                    # Activar toggle solo si llevamos 3 frames seguidos con gesto de pinza
                    if len(gestos_pinza) == 3 and all(gestos_pinza) and time.time() - ultimo_toggle > 1.0:
                        # Toggle del estado de dibujo (activado/desactivado)
                        controlador_dibujo.toggle_dibujo_habilitado()
                        ultimo_toggle = time.time()  # Actualizar timestamp para evitar toggles múltiples

                        # Log y feedback visual sobre el cambio de estado
                        nuevo_estado = "ACTIVADO" if controlador_dibujo.es_dibujo_habilitado() else "DESACTIVADO"
                        logger.info(f"Estado de dibujo cambiado a: {nuevo_estado}")
                else:
                    # Reiniciar contador de gestos si no hay gesto de pinza
                    gestos_pinza.clear()

                # Procesar el dibujo si está habilitado
                if controlador_dibujo.es_dibujo_habilitado() and mano and punta_indice:
                    # Ajustar por factor de escala si se redimensionó el frame
                    x_ajustado, y_ajustado = punta_indice
                    if factor_resize < 1.0:
                        x_ajustado = int(x_ajustado / factor_resize)
                        y_ajustado = int(y_ajustado / factor_resize)

                    # Verificar si el índice está extendido para dibujar
                    if indice_extendido:
                        controlador_dibujo.actualizar_posicion(x_ajustado, y_ajustado)
                    else:
                        # Verificar si el método existe antes de llamarlo
                        if hasattr(controlador_dibujo, 'actualizar_posicion_sin_dibujar'):
                            controlador_dibujo.actualizar_posicion_sin_dibujar(x_ajustado, y_ajustado)
                        else:
                            # Usar método alternativo para actualizar sin dibujar
                            logger.warning("Método 'actualizar_posicion_sin_dibujar' no encontrado. Usando alternativa.")
                            # Guardar estado temporal del dibujo
                            dibujo_habilitado_temp = controlador_dibujo.es_dibujo_habilitado()
                            # Deshabilitar dibujo temporalmente
                            controlador_dibujo.dibujo_habilitado = False
                            # Actualizar posición sin dibujar
                            controlador_dibujo.actualizar_posicion(x_ajustado, y_ajustado)
                            # Restaurar estado original
                            controlador_dibujo.dibujo_habilitado = dibujo_habilitado_temp

                # Procesar interacciones con la interfaz
                if mano and punta_indice:
                    # Ajustar coordenadas si se redimensionó
                    x_ui, y_ui = punta_indice
                    if factor_resize < 1.0:
                        x_ui = int(x_ui / factor_resize)
                        y_ui = int(y_ui / factor_resize)

                    # En la UI siempre procesamos interacciones independientemente
                    # del estado de dibujo habilitado/deshabilitado
                    interfaz.procesar_interaccion((x_ui, y_ui), indice_extendido)
                else:
                    # Si no hay mano, reiniciar estado de hover
                    interfaz.procesar_interaccion(None, False)
            else:
                # No hay manos detectadas, actualizar interfaz con valores nulos
                interfaz.procesar_interaccion(None, False)
                logger.debug("DEPURACIÓN - No se detectaron manos")

            # Superponer lienzo en el frame
            frame_con_lienzo = interfaz.superponer_lienzo(frame_procesado)

            # Calcular y mostrar métricas
            tiempo_frame_actual = time.time()
            tiempos_frame.append(tiempo_frame_actual)

            # Mantener solo los últimos 100 tiempos para el cálculo de FPS
            if len(tiempos_frame) > 100:
                tiempos_frame.pop(0)

            fps_actual = calcular_fps(tiempos_frame)

            # Estado del dibujo
            estado_dibujo = "DIBUJO ACTIVADO" if controlador_dibujo.es_dibujo_habilitado() else "DIBUJO DESACTIVADO"

            # Recolectar métricas para el footer
            metricas_footer = None
            if show_metrics:
                metricas_footer = {
                    "FPS": fps_actual,
                    "Manos": len(manos) if manos else 0,
                    "Factor resize": factor_resize,
                    "Skip frames": parametros_optimizacion["skip_frames"],
                    "Prediccion": parametros_optimizacion["use_prediction"]
                }

                # Obtener métricas adicionales
                metricas_detector = detector.obtener_metricas()
                metricas_dibujo = controlador_dibujo.obtener_metricas()

                # Añadir métricas específicas
                metricas_footer["Tiempo deteccion (ms)"] = metricas_detector.get("tiempo_deteccion_ms", 0)
                metricas_footer["Predicciones"] = metricas_detector.get("predicciones_totales", 0)
                metricas_footer["Vel. movimiento"] = metricas_dibujo.get("velocidad_actual", 0)

            # Dibujar footer con métricas
            frame_con_lienzo = dibujar_footer_metricas(
                frame_con_lienzo,
                metricas_footer,
                fps_actual,
                estado_dibujo,
                controlador_dibujo.es_dibujo_habilitado(),
                solo_mano_derecha=solo_mano_derecha,
                solo_mano_izquierda=solo_mano_izquierda
            )

            # Mostrar frame final
            cv2.imshow(nombre_ventana, frame_con_lienzo)  # type: ignore

            # Optimizar rendimiento cada 3 segundos
            tiempo_actual = time.time()
            if tiempo_actual - ultimo_tiempo_optimizacion > 3.0:
                parametros_optimizacion = optimizar_rendimiento(fps_actual, quality_factor)
                ultimo_tiempo_optimizacion = tiempo_actual
                logger.debug(f"Parámetros optimizados: {parametros_optimizacion}")

            # Salir si se presiona 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):  # type: ignore
                break

            contador_frames += 1

    except KeyboardInterrupt:
        logger.info("Aplicación interrumpida por el usuario")
    except Exception as e:
        logger.error(f"Error en el bucle principal: {e}", exc_info=True)
    finally:
        # Liberar recursos
        captura.release()
        cv2.destroyAllWindows()  # type: ignore
        logger.info("Aplicación finalizada correctamente")

def main(use_async_capture: bool = OPTIMIZATION_USE_ASYNC_CAPTURE,
         show_metrics: bool = OPTIMIZATION_SHOW_METRICS,
         quality_factor: float = OPTIMIZATION_QUALITY,
         solo_mano_derecha: bool = OPTIMIZATION_SOLO_MANO_DERECHA,
         solo_mano_izquierda: bool = MEDIAPIPE_MANO_IZQUIERDA,
         mirror_mode: bool = CAMERA_MIRROR_MODE) -> None:
    """
    Función principal de la aplicación.

    Args:
        use_async_capture: Si es True, usa captura asíncrona de video
        show_metrics: Si es True, muestra métricas de rendimiento
        quality_factor: Factor de calidad (0-1)
        solo_mano_derecha: Si es True, solo detecta la mano derecha
        solo_mano_izquierda: Si es True, solo detecta la mano izquierda
        mirror_mode: Si es True, muestra la cámara en modo espejo
    """
    try:
        logger.info("Iniciando aplicación Pizarra Digital")

        # Inicializar componentes
        componentes = inicializar_app(
            use_async_capture=use_async_capture,
            show_metrics=show_metrics,
            quality_factor=quality_factor,
            solo_mano_derecha=solo_mano_derecha,
            solo_mano_izquierda=solo_mano_izquierda,
            mirror_mode=mirror_mode
        )

        # Añadir configuración de modo espejo
        componentes["mirror_mode"] = mirror_mode

        # Ejecutar aplicación
        ejecutar_app(componentes)

    except Exception as e:
        logger.error(f"Error al iniciar la aplicación: {e}", exc_info=True)

if __name__ == "__main__":
    main()
