#!/usr/bin/env python3
"""
Script principal para ejecutar la aplicación Pizarra Digital.

Este script es el punto de entrada principal para la aplicación,
procesa los argumentos de línea de comandos y configura el entorno
antes de iniciar la aplicación.
"""
import os
import sys
import logging
import argparse
from typing import Dict, Any, Optional

# Agregar directorio raíz al path para importar módulos
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Importar módulos de la aplicación
from pizarra_digital.main import main
from pizarra_digital.config import (
    OPTIMIZATION_USE_ASYNC_CAPTURE,
    OPTIMIZATION_SHOW_METRICS,
    OPTIMIZATION_QUALITY,
    OPTIMIZATION_SOLO_MANO_DERECHA,
    CAMERA_MIRROR_MODE
)

# Configuración de logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_args() -> argparse.Namespace:
    """
    Procesa los argumentos de línea de comandos.

    Returns:
        Namespace con los argumentos procesados.
    """
    parser = argparse.ArgumentParser(
        description="Pizarra Digital - Aplicación de dibujo con gestos de manos"
    )

    # Argumentos de configuración
    parser.add_argument(
        "--camera",
        type=int,
        default=None,
        help="Índice de la cámara a utilizar (default: 0)"
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Activar modo de depuración con información adicional"
    )

    parser.add_argument(
        "--no-async",
        action="store_true",
        help="Desactivar captura asíncrona de video"
    )

    parser.add_argument(
        "--no-metrics",
        action="store_true",
        help="Ocultar métricas de rendimiento en pantalla"
    )

    parser.add_argument(
        "--quality",
        choices=["low", "medium", "high"],
        default="medium",
        help="Calidad de procesamiento (low = máximo rendimiento, high = máxima calidad)"
    )

    parser.add_argument(
        "--resolution",
        choices=["low", "medium", "high"],
        default="low",
        help="Resolución de la cámara (low=320x240, medium=640x480, high=1280x720)"
    )

    parser.add_argument(
        "--no-mirror",
        action="store_true",
        help="Desactivar el modo espejo de la cámara (por defecto está activado)"
    )

    # Grupo exclusivo para selección de mano
    grupo_mano = parser.add_mutually_exclusive_group()
    grupo_mano.add_argument(
        "--mano-derecha",
        action="store_true",
        help="Detectar la mano derecha del usuario (la que está en tu lado derecho)"
    )
    grupo_mano.add_argument(
        "--mano-izquierda",
        action="store_true",
        help="Detectar la mano izquierda del usuario (la que está en tu lado izquierdo) - Opción predeterminada"
    )
    grupo_mano.add_argument(
        "--ambas-manos",
        action="store_true",
        help="Detectar ambas manos del usuario"
    )

    return parser.parse_args()

def configurar_app(args: argparse.Namespace) -> Dict[str, Any]:
    """
    Configura la aplicación basado en los argumentos de línea de comandos.

    Args:
        args: Argumentos de línea de comandos procesados.

    Returns:
        Diccionario con la configuración de la aplicación.
    """
    # Configurar nivel de logging
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Modo de depuración activado")

    # Mapear calidad a valor numérico
    quality_map = {
        "low": 0.5,      # Prioriza rendimiento sobre calidad
        "medium": 0.8,   # Equilibrado
        "high": 1.0      # Máxima calidad
    }

    # Configuración de manos
    # Por defecto usar mano izquierda, a menos que se especifique mano derecha o ambas manos
    solo_mano_izquierda = args.mano_izquierda or (not args.mano_derecha and not args.ambas_manos)
    solo_mano_derecha = args.mano_derecha

    # Crear diccionario de configuración
    config = {
        "camera_index": args.camera,
        "use_async_capture": not args.no_async,
        "show_metrics": not args.no_metrics,
        "quality_factor": quality_map[args.quality],
        "solo_mano_derecha": solo_mano_derecha and not solo_mano_izquierda,
        "solo_mano_izquierda": solo_mano_izquierda,
        "mirror_mode": not args.no_mirror
    }

    # Registrar configuración
    logger.info(f"Configuración: cámara={config['camera_index']}, "
               f"async={config['use_async_capture']}, "
               f"métricas={config['show_metrics']}, "
               f"calidad={args.quality}, "
               f"solo_mano_derecha={config['solo_mano_derecha']}, "
               f"solo_mano_izquierda={config['solo_mano_izquierda']}, "
               f"ambas_manos={args.ambas_manos}, "
               f"resolución={args.resolution}, "
               f"modo_espejo={config['mirror_mode']}")

    return config

def main_wrapper() -> None:
    """
    Función principal que inicializa y ejecuta la aplicación.
    """
    try:
        # Procesar argumentos
        args = parse_args()

        # Configurar aplicación
        config = configurar_app(args)

        # Iniciar aplicación con la configuración
        logger.info("Iniciando Pizarra Digital...")
        main(
            use_async_capture=config["use_async_capture"],
            show_metrics=config["show_metrics"],
            quality_factor=config["quality_factor"],
            solo_mano_derecha=config["solo_mano_derecha"],
            solo_mano_izquierda=config["solo_mano_izquierda"],
            mirror_mode=config["mirror_mode"]
        )

        logger.info("Aplicación terminada correctamente")

    except KeyboardInterrupt:
        logger.info("Aplicación interrumpida por el usuario")
    except Exception as e:
        logger.error(f"Error al ejecutar la aplicación: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main_wrapper()
