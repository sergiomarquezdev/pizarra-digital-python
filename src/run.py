#!/usr/bin/env python3
"""
Punto de entrada principal para la aplicación de pizarra digital.

Este script inicia la aplicación y configura el logging.
"""
import os
import sys
import logging
import argparse
from typing import Dict, Any

# Añadir directorio raíz al path para importaciones relativas
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Configuración inicial del logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_args() -> Dict[str, Any]:
    """
    Procesa los argumentos de línea de comandos.

    Returns:
        Diccionario con los argumentos procesados.
    """
    parser = argparse.ArgumentParser(description='Pizarra Digital con detección de gestos')

    parser.add_argument('--camera', type=int, default=0,
                      help='Índice de la cámara a utilizar (default: 0)')
    parser.add_argument('--debug', action='store_true',
                      help='Activar modo de depuración')
    parser.add_argument('--no-async', action='store_true',
                      help='Desactivar captura asíncrona (menor rendimiento pero mayor compatibilidad)')
    parser.add_argument('--no-metrics', action='store_true',
                      help='Ocultar métricas de rendimiento')
    parser.add_argument('--quality', choices=['low', 'medium', 'high'], default='medium',
                      help='Calidad de procesamiento (afecta rendimiento)')

    return vars(parser.parse_args())

def configurar_app(args: Dict[str, Any]) -> None:
    """
    Configura la aplicación según los argumentos recibidos.

    Args:
        args: Diccionario con argumentos de configuración.
    """
    from pizarra_digital.config import (
        logger as config_logger,
        CAMERA_INDEX,
    )

    # Actualizar nivel de logging si estamos en modo debug
    if args.get('debug', False):
        logging.getLogger().setLevel(logging.DEBUG)
        config_logger.setLevel(logging.DEBUG)
        logger.debug("Modo de depuración activado")

    # Configurar índice de cámara
    camera_index = args.get('camera', CAMERA_INDEX)
    if camera_index != CAMERA_INDEX:
        # Modificar configuración global
        import pizarra_digital.config as config
        config.CAMERA_INDEX = camera_index
        logger.info(f"Usando cámara con índice: {camera_index}")

    # Configurar calidad/rendimiento
    if 'quality' in args:
        import pizarra_digital.config as config
        import pizarra_digital.main as main

        quality = args['quality']
        if quality == 'low':
            # Baja calidad, alto rendimiento
            config.OPTIMIZATION_RESIZE_FACTOR = 0.5
            main.USAR_PREDICCION_MANOS = True
            logger.info("Configuración de rendimiento: ALTA (calidad baja)")
        elif quality == 'high':
            # Alta calidad, menor rendimiento
            config.OPTIMIZATION_RESIZE_FACTOR = 1.0
            main.USAR_PREDICCION_MANOS = False
            logger.info("Configuración de rendimiento: BAJA (calidad alta)")
        else:
            # Configuración media (predeterminada)
            config.OPTIMIZATION_RESIZE_FACTOR = 0.75
            main.USAR_PREDICCION_MANOS = True
            logger.info("Configuración de rendimiento: MEDIA (equilibrada)")

    # Configurar captura asíncrona
    if args.get('no_async', False):
        import pizarra_digital.main as main
        main.USAR_CAPTURA_ASINCRONA = False
        logger.info("Captura asíncrona desactivada")

    # Configurar visualización de métricas
    if args.get('no_metrics', False):
        import pizarra_digital.main as main
        main.MOSTRAR_METRICAS = False
        logger.info("Visualización de métricas desactivada")

def main() -> None:
    """
    Función principal que inicia la aplicación.
    """
    logger.info("Iniciando aplicación de Pizarra Digital")

    try:
        # Procesar argumentos de línea de comandos
        args = parse_args()

        # Configurar la aplicación
        configurar_app(args)

        # Importar e iniciar la aplicación principal
        from pizarra_digital.main import ejecutar_app
        ejecutar_app()

    except ImportError as e:
        logger.error(f"Error al importar módulos: {e}")
        logger.error("Asegúrate de haber instalado todas las dependencias")
        sys.exit(1)
    except KeyboardInterrupt:
        logger.info("Aplicación interrumpida por el usuario")
    except Exception as e:
        logger.exception(f"Error inesperado: {e}")
        sys.exit(1)

    logger.info("Aplicación finalizada")

if __name__ == "__main__":
    main()
