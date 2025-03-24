#!/usr/bin/env python
"""
Script de punto de entrada para la aplicación de pizarra digital.

Ejecutar este script iniciará la aplicación de pizarra digital.
"""
import logging
from pizarra_digital.main import ejecutar_app

if __name__ == "__main__":
    # Configurar logging para la consola
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Ejecutar la aplicación
    ejecutar_app()
