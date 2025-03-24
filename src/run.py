#!/usr/bin/env python
"""
Script de punto de entrada para la aplicación de pizarra digital.

Ejecutar este script iniciará la aplicación de pizarra digital.
"""
import logging
import sys
import os

# Añadir el directorio src al path para que las importaciones funcionen
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pizarra_digital.main import ejecutar_app

if __name__ == "__main__":
    # Configurar logging para la consola
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Ejecutar la aplicación
    ejecutar_app()
