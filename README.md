# Pizarra Digital Controlada por Gestos de Manos

Una aplicación Python que permite dibujar en una pizarra digital utilizando gestos de manos capturados a través de la cámara web.

## Características

- Dibujo usando el dedo índice como lápiz
- Soporte para múltiples colores (blanco, negro, azul, rojo, verde)
- Función de borrado de la pizarra
- Interfaz de usuario intuitiva con botones
- Visualización de FPS en tiempo real

## Requisitos de Sistema

- Python 3.10 o superior
- Cámara web funcional
- Espacio suficiente para ejecutar procesamiento de video en tiempo real

## Instalación

1. Clona este repositorio:
   ```
   git clone https://github.com/[tu-usuario]/pizarra-digital-python.git
   cd pizarra-digital-python
   ```

2. Crea un entorno virtual (recomendado):
   ```
   python -m venv venv
   source venv/bin/activate  # En Windows: venv\Scripts\activate
   ```

3. Instala las dependencias requeridas:
   ```
   pip install -r requirements.txt
   ```

## Uso

1. Ejecuta la aplicación:
   ```
   python src/run.py
   ```

2. Controles:
   - Extiende tu dedo índice frente a la cámara para dibujar
   - Haz clic en los botones de colores para cambiar el color del trazo
   - Haz clic en el botón "Borrar" para limpiar la pizarra
   - Presiona 'q' para salir de la aplicación

## Estructura del Proyecto

- `src/pizarra_digital/`: Paquete principal
  - `captura/`: Módulo para la captura de video
  - `deteccion/`: Módulo para la detección de manos
  - `lienzo/`: Módulo para el lienzo de dibujo
  - `dibujo/`: Módulo para la lógica de dibujo
  - `interfaz/`: Módulo para la interfaz de usuario
  - `config.py`: Configuración global
  - `main.py`: Módulo principal

## Licencia

Este proyecto está licenciado bajo los términos de la licencia MIT.
