"""
Módulo principal de la aplicación de pizarra digital.

Este módulo integra todos los componentes y ejecuta la aplicación principal.
"""
import sys
import time
import logging
import cv2
import numpy as np
from typing import Dict, Any, Optional, Tuple, List

from .captura.captura import inicializar_camara, leer_fotograma, liberar_camara, CapturaAsincrona
from .deteccion.deteccion import DetectorManos
from .lienzo.lienzo import Lienzo
from .dibujo.dibujo import DibujoMano
from .interfaz.interfaz import InterfazUsuario
from .config import (
    CAMERA_INDEX,
    CAMERA_WIDTH,
    CAMERA_HEIGHT,
    OPTIMIZATION_RESIZE_FACTOR,
    APP_NAME,
    MEDIAPIPE_MAX_HANDS,
    MEDIAPIPE_DETECTION_CONFIDENCE,
    MEDIAPIPE_TRACKING_CONFIDENCE,
    FPS_HISTORY_SIZE
)

# Configuración del logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuración de rendimiento
USAR_CAPTURA_ASINCRONA: bool = True  # Usar captura asíncrona para mejorar FPS
USAR_PREDICCION_MANOS: bool = True  # Usar predicción de posición para manos
USAR_BUFFER_LIENZO: bool = True  # Usar buffer para operaciones de dibujo
MOSTRAR_METRICAS: bool = True  # Mostrar métricas de rendimiento en pantalla
UMBRAL_FPS_BAJO: float = 15.0  # Umbral para considerar FPS bajos
UMBRAL_FPS_ADVERTENCIA: float = 25.0  # Umbral para advertencia de FPS

def calcular_fps(tiempos_fotograma: List[float]) -> float:
    """
    Calcula los FPS (frames por segundo) basado en los tiempos de fotogramas.

    Args:
        tiempos_fotograma: Lista de tiempos de procesamiento de fotogramas.

    Returns:
        FPS promedio calculado.
    """
    if len(tiempos_fotograma) < 2:
        return 0.0
    # Calcular el promedio de tiempo entre fotogramas
    delta_tiempos = [tiempos_fotograma[i] - tiempos_fotograma[i-1]
                     for i in range(1, len(tiempos_fotograma))]
    tiempo_promedio = sum(delta_tiempos) / len(delta_tiempos)
    if tiempo_promedio <= 0:
        return 0.0
    return 1.0 / tiempo_promedio

def optimizar_rendimiento(fps_actual: float, metricas: Dict[str, Any]) -> Dict[str, Any]:
    """
    Ajusta parámetros de rendimiento basados en los FPS actuales.

    Args:
        fps_actual: FPS actuales de la aplicación.
        metricas: Métricas actuales de rendimiento.

    Returns:
        Diccionario con parámetros de optimización actualizados.
    """
    optimizaciones = {
        'resize_factor': OPTIMIZATION_RESIZE_FACTOR,
        'prediccion_manos': USAR_PREDICCION_MANOS,
        'captura_asincrona': USAR_CAPTURA_ASINCRONA,
        'umbral_salto_frames': 0,  # Cuántos frames saltar en procesamiento
        'modo_ahorro_energia': False  # Modo de ahorro de energía
    }

    # Ajustar optimizaciones según FPS
    if fps_actual < UMBRAL_FPS_BAJO:
        # Si los FPS son muy bajos, aumentar el factor de escala
        optimizaciones['resize_factor'] = max(0.3, OPTIMIZATION_RESIZE_FACTOR - 0.1)
        optimizaciones['umbral_salto_frames'] = 1  # Saltar un frame de cada dos

    elif fps_actual < UMBRAL_FPS_ADVERTENCIA:
        optimizaciones['resize_factor'] = OPTIMIZATION_RESIZE_FACTOR
        optimizaciones['umbral_salto_frames'] = 0  # No saltar frames

    # Actualizar métricas con optimizaciones
    metricas.update({
        'optimizaciones': optimizaciones
    })

    return optimizaciones

def dibujar_metricas(frame: np.ndarray, metricas: Dict[str, Any]) -> np.ndarray:
    """
    Dibuja métricas de rendimiento en el fotograma.

    Args:
        frame: Fotograma donde dibujar las métricas.
        metricas: Diccionario con métricas de rendimiento.

    Returns:
        Fotograma con las métricas dibujadas.
    """
    if not MOSTRAR_METRICAS or frame is None:
        return frame

    # Crear una copia del frame para no modificar el original
    result = frame.copy()

    # Zona para las métricas (rectángulo semitransparente)
    altura, anchura = result.shape[:2]
    panel_altura = 120
    panel_inicio_y = 10

    # Dibujar panel semitransparente para métricas
    overlay = result.copy()
    cv2.rectangle(overlay, (10, panel_inicio_y), (240, panel_inicio_y + panel_altura),
                 (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.7, result, 0.3, 0, result)

    # Obtener métricas
    fps = metricas.get('fps', 0)
    tiempo_total = metricas.get('tiempo_total_ms', 0)
    tiempo_captura = metricas.get('tiempo_captura_ms', 0)
    tiempo_deteccion = metricas.get('tiempo_deteccion_ms', 0)
    tiempo_dibujo = metricas.get('tiempo_dibujo_ms', 0)

    # Color según FPS
    color_fps = (0, 255, 0)  # Verde por defecto
    if fps < UMBRAL_FPS_BAJO:
        color_fps = (0, 0, 255)  # Rojo para FPS bajos
    elif fps < UMBRAL_FPS_ADVERTENCIA:
        color_fps = (0, 165, 255)  # Naranja para FPS en advertencia

    # Dibujar textos de métricas
    y_offset = panel_inicio_y + 25
    cv2.putText(result, f"FPS: {fps:.1f}", (20, y_offset),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_fps, 2)

    y_offset += 20
    cv2.putText(result, f"Total: {tiempo_total:.1f} ms", (20, y_offset),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    y_offset += 20
    cv2.putText(result, f"Captura: {tiempo_captura:.1f} ms", (20, y_offset),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    y_offset += 20
    cv2.putText(result, f"Detección: {tiempo_deteccion:.1f} ms", (20, y_offset),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    y_offset += 20
    cv2.putText(result, f"Dibujo: {tiempo_dibujo:.1f} ms", (20, y_offset),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    return result

def inicializar_app() -> Tuple[Any, DetectorManos, Lienzo, DibujoMano, InterfazUsuario]:
    """
    Inicializa todos los componentes necesarios para la aplicación.

    Returns:
        Tupla con los componentes inicializados:
        - Captura de cámara
        - Detector de manos
        - Lienzo de dibujo
        - Controlador de dibujo
        - Interfaz de usuario
    """
    logger.info("Inicializando la aplicación")

    # Inicializar la cámara (modo asíncrono o tradicional)
    if USAR_CAPTURA_ASINCRONA:
        logger.info("Inicializando captura asíncrona")
        camara = CapturaAsincrona(
            indice_camara=CAMERA_INDEX,
            ancho_captura=CAMERA_WIDTH,
            alto_captura=CAMERA_HEIGHT
        )
        camara.iniciar()
    else:
        logger.info("Inicializando captura tradicional")
        camara = inicializar_camara(
            indice_camara=CAMERA_INDEX,
            ancho_captura=CAMERA_WIDTH,
            alto_captura=CAMERA_HEIGHT
        )

    # Inicializar el detector de manos con las optimizaciones
    detector_manos = DetectorManos(
        max_manos=MEDIAPIPE_MAX_HANDS,
        min_confianza_deteccion=MEDIAPIPE_DETECTION_CONFIDENCE,
        min_confianza_seguimiento=MEDIAPIPE_TRACKING_CONFIDENCE,
        enable_optimizations=USAR_PREDICCION_MANOS
    )

    # Inicializar el lienzo
    lienzo = Lienzo()

    # Inicializar el controlador de dibujo
    dibujo_mano = DibujoMano(lienzo)

    # Inicializar la interfaz de usuario
    interfaz_usuario = InterfazUsuario(lienzo)

    return camara, detector_manos, lienzo, dibujo_mano, interfaz_usuario

def ejecutar_app() -> None:
    """
    Ejecuta la aplicación principal.
    """
    # Inicializar componentes
    camara, detector_manos, lienzo, dibujo_mano, interfaz_usuario = inicializar_app()

    # Variables para el cálculo de FPS
    tiempos_fotograma = []
    conteo_fotogramas = 0
    tiempo_inicio_total = time.time()
    fps_actual = 0.0

    # Diccionario para métricas
    metricas = {
        'fps': 0.0,
        'tiempo_total_ms': 0.0,
        'tiempo_captura_ms': 0.0,
        'tiempo_deteccion_ms': 0.0,
        'tiempo_dibujo_ms': 0.0,
        'num_manos_detectadas': 0
    }

    # Configuración de optimización
    optimizaciones = optimizar_rendimiento(0.0, metricas)
    ultimo_tiempo_optimizacion = time.time()
    contador_saltado = 0

    try:
        logger.info("Iniciando bucle principal")
        cv2.namedWindow(APP_NAME, cv2.WINDOW_NORMAL)

        # Bucle principal
        while True:
            tiempo_inicio_frame = time.time()

            # 1. Capturar fotograma
            tiempo_inicio_captura = time.time()
            if USAR_CAPTURA_ASINCRONA:
                # En modo asíncrono, simplemente obtenemos el último fotograma
                fotograma = camara.obtener_ultimo_fotograma(scale_factor=optimizaciones['resize_factor'])
            else:
                # En modo tradicional, leemos el fotograma directamente
                fotograma = leer_fotograma(
                    camara,
                    scale_factor=optimizaciones['resize_factor']
                )
            tiempo_captura = time.time() - tiempo_inicio_captura

            # Verificar si la captura fue exitosa
            if fotograma is None:
                logger.warning("No se pudo capturar fotograma")
                break

            # Saltear procesamiento de algunos frames si es necesario para mantener rendimiento
            contador_saltado += 1
            if optimizaciones['umbral_salto_frames'] > 0 and contador_saltado % (optimizaciones['umbral_salto_frames'] + 1) != 0:
                if contador_saltado > 100:  # Reiniciar contador para evitar overflow
                    contador_saltado = 0
                continue

            # 2. Procesar fotograma para detección de manos
            tiempo_inicio_deteccion = time.time()
            fotograma_procesado, resultado_manos = detector_manos.procesar_fotograma(fotograma)
            tiempo_deteccion = time.time() - tiempo_inicio_deteccion

            # 3. Procesar gestos y acciones
            tiempo_inicio_dibujo = time.time()
            if resultado_manos:
                # Actualizar estado de dibujo basado en posición de manos
                dibujo_mano.procesar_mano(resultado_manos)

                # Actualizar la interfaz de usuario con la posición del dedo
                punto_indice = detector_manos.obtener_coordenada_punto(resultado_manos, 8)  # Punta del índice
                indice_extendido = detector_manos.indice_extendido(resultado_manos)

                # Procesar la interacción con la interfaz
                interfaz_usuario.procesar_interaccion(punto_indice, indice_extendido)

            # 4. Superponer lienzo sobre el fotograma
            fotograma_con_lienzo = lienzo.superponer_en_fotograma(fotograma_procesado)

            # 5. Dibujar interfaz sobre el fotograma
            fotograma_final = interfaz_usuario.dibujar_interfaz(fotograma_con_lienzo)

            tiempo_dibujo = time.time() - tiempo_inicio_dibujo

            # 6. Calcular métricas de rendimiento
            tiempo_total_frame = time.time() - tiempo_inicio_frame
            tiempos_fotograma.append(tiempo_inicio_frame)
            if len(tiempos_fotograma) > FPS_HISTORY_SIZE:
                tiempos_fotograma.pop(0)

            conteo_fotogramas += 1
            if conteo_fotogramas % 10 == 0:  # Actualizar FPS cada 10 fotogramas
                fps_actual = calcular_fps(tiempos_fotograma)

                # Actualizar métricas generales
                metricas.update({
                    'fps': fps_actual,
                    'tiempo_total_ms': tiempo_total_frame * 1000,
                    'tiempo_captura_ms': tiempo_captura * 1000,
                    'tiempo_deteccion_ms': tiempo_deteccion * 1000,
                    'tiempo_dibujo_ms': tiempo_dibujo * 1000,
                    'num_manos_detectadas': len(resultado_manos) if resultado_manos else 0
                })

                # Optimizar cada segundo
                tiempo_actual = time.time()
                if tiempo_actual - ultimo_tiempo_optimizacion > 1.0:
                    optimizaciones = optimizar_rendimiento(fps_actual, metricas)
                    ultimo_tiempo_optimizacion = tiempo_actual

            # 7. Dibujar métricas de rendimiento si está habilitado
            if MOSTRAR_METRICAS:
                fotograma_final = dibujar_metricas(fotograma_final, metricas)

            # 8. Mostrar el fotograma procesado
            cv2.imshow(APP_NAME, fotograma_final)

            # 9. Verificar si se debe salir
            tecla = cv2.waitKey(1) & 0xFF
            if tecla == 27 or tecla == ord('q'):  # Esc o 'q' para salir
                logger.info("Saliendo de la aplicación (tecla de salida presionada)")
                break

            # Limitar el uso de CPU en el modo de ahorro de energía
            if optimizaciones.get('modo_ahorro_energia', False):
                time.sleep(0.01)  # Pequeña pausa para reducir el uso de CPU

    except Exception as e:
        logger.exception(f"Error en la ejecución de la aplicación: {e}")

    finally:
        # Liberar recursos y cerrar ventanas
        logger.info("Liberando recursos")
        if USAR_CAPTURA_ASINCRONA:
            camara.detener()
        else:
            liberar_camara(camara)
        cv2.destroyAllWindows()

        # Mostrar resumen de rendimiento
        tiempo_total = time.time() - tiempo_inicio_total
        if conteo_fotogramas > 0 and tiempo_total > 0:
            fps_promedio = conteo_fotogramas / tiempo_total
            logger.info(f"Rendimiento final: {fps_promedio:.2f} FPS promedio, "
                      f"{conteo_fotogramas} fotogramas procesados en {tiempo_total:.2f} segundos")

if __name__ == "__main__":
    logger.info("Iniciando aplicación de pizarra digital")
    ejecutar_app()
