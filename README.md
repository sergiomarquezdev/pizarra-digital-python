# Pizarra Digital Controlada por Gestos de Manos

![Estado](https://img.shields.io/badge/Estado-En%20Desarrollo-brightgreen)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![OpenCV](https://img.shields.io/badge/OpenCV-4.5.0%2B-green)
![MediaPipe](https://img.shields.io/badge/MediaPipe-0.8.10%2B-orange)

Una aplicación Python que permite dibujar en una pizarra digital utilizando gestos de manos capturados a través de la cámara web. Este proyecto combina visión por computadora y seguimiento de manos en tiempo real para crear una experiencia de dibujo interactiva sin contacto físico.

## 📋 Características

- **Dibujo Intuitivo:** Usa tu dedo índice como lápiz digital para dibujar sobre un lienzo virtual
- **Gesto de Pinza:** Activa/desactiva el modo de dibujo juntando el pulgar y el índice
- **Paleta de Colores:** Incluye múltiples colores para dibujar (rojo, verde, azul, amarillo, magenta, cian, negro, blanco)
- **Borrado Rápido:** Botón para limpiar la pizarra completamente
- **Modo Espejo:** Visión como en un espejo para una experiencia más natural
- **Selección de Manos:** Configura la app para detectar la mano derecha, izquierda o ambas
- **Interpolación Adaptativa:** Líneas suaves incluso en movimientos rápidos
- **Grosor Adaptativo:** Varía el grosor de las líneas según la velocidad del movimiento
- **Interfaz Minimalista:** Panel de controles simple e intuitivo
- **Métricas en Tiempo Real:** Visualización de FPS y otros datos de rendimiento
- **Optimizaciones de Rendimiento:** Ajustes automáticos para mantener una experiencia fluida

## 🖥️ Demostración

_[Imagen de ejemplo del proyecto en funcionamiento - Sugerido añadir una cuando esté disponible]_

## 🔧 Requisitos

### Software
- Python 3.10 o superior
- Dependencias Python (ver `requirements.txt`):
  - opencv-python (≥4.5.0)
  - mediapipe (≥0.8.10)
  - numpy (≥1.20.0)
  - Herramientas de desarrollo: ruff, pytest

### Hardware
- Cámara web funcional (integrada o externa)
- Procesador: Recomendado Intel Core i5 o superior (o equivalente)
- RAM: 4GB mínimo, 8GB recomendado
- Espacio libre en disco: 500MB

## 🚀 Instalación

1. **Clona el repositorio:**
   ```bash
   git clone https://github.com/[tu-usuario]/pizarra-digital-python.git
   cd pizarra-digital-python
   ```

2. **Crea un entorno virtual (recomendado):**
   ```bash
   # Windows
   python -m venv venv
   venv\Scripts\activate

   # macOS/Linux
   python -m venv venv
   source venv/bin/activate
   ```

3. **Instala las dependencias:**
   ```bash
   pip install -r requirements.txt
   ```

## 📝 Uso

### Ejecutar la Aplicación

```bash
python src/run.py
```

Con opciones adicionales:
```bash
# Modo de depuración con procesamiento síncrono y solo mano derecha
python src/run.py --debug --no-async --mano-derecha

# Calidad alta, resolución media y sin métricas
python src/run.py --quality high --resolution medium --no-metrics
```

### Opciones de Línea de Comandos

* `--camera ÍNDICE`: Especifica el índice de la cámara a utilizar
* `--debug`: Activa el modo de depuración con logs detallados
* `--no-async`: Desactiva la captura asíncrona de video
* `--no-metrics`: Oculta las métricas de rendimiento en pantalla
* `--quality {low,medium,high}`: Establece la calidad de procesamiento
* `--resolution {low,medium,high}`: Establece la resolución de la cámara
* `--no-mirror`: Desactiva el modo espejo de la cámara
* `--mano-derecha`: Detecta solo la mano derecha (la del lado derecho)
* `--mano-izquierda`: Detecta solo la mano izquierda (la del lado izquierdo) - Opción predeterminada
* `--ambas-manos`: Detecta ambas manos del usuario

### Controles:

- **Dibujar:** Extiende tu dedo índice frente a la cámara
- **Activar/Desactivar Dibujo:** Junta tu pulgar e índice en gesto de pinza
- **Cambiar color:** Haz clic en los botones de colores
- **Borrar todo:** Haz clic en el botón "Borrar"
- **Salir:** Presiona 'q' en cualquier momento

## 🏗️ Estructura del Proyecto

```
pizarra-digital-python/
├── src/
│   ├── pizarra_digital/         # Paquete principal
│   │   ├── captura/             # Gestión de entrada de cámara
│   │   ├── deteccion/           # Detección de manos con MediaPipe
│   │   ├── dibujo/              # Lógica de dibujo y gestos
│   │   ├── interfaz/            # Elementos de la interfaz de usuario
│   │   ├── lienzo/              # Manejo del lienzo de dibujo
│   │   ├── utils/               # Utilidades varias
│   │   ├── config.py            # Configuración global
│   │   └── main.py              # Punto de entrada principal
│   └── run.py                   # Script para ejecutar la aplicación
├── tests/                       # Pruebas unitarias
├── project_docs/                # Documentación técnica
├── requirements.txt             # Dependencias del proyecto
├── README.md                    # Este archivo
└── .gitignore                   # Patrones de archivos a ignorar
```

## ⚠️ Solución de Problemas

### Problemas con la Cámara

Si la aplicación no detecta tu cámara correctamente:

1. **Intenta varios índices:** Usa `--camera 1`, `--camera 2`, etc. para probar diferentes cámaras
2. **Verifica los permisos:** Asegúrate de que tu sistema operativo permita el acceso a la cámara
3. **Cierra otras aplicaciones:** Otras aplicaciones podrían estar usando la cámara (Zoom, Teams, navegadores)
4. **Ajusta la resolución:** Usa `--resolution low` para diagnosticar problemas

### Problemas de Rendimiento

1. **Reduce la calidad:** Usa `--quality low` para mejorar los FPS
2. **Desactiva la asyncronía:** Prueba con `--no-async` si experimentas problemas
3. **Reduce la resolución:** Usa `--resolution low` para mejor rendimiento
4. **Cierra aplicaciones en segundo plano:** Libera recursos del sistema

### Problemas de Detección de Manos

1. **Mejora la iluminación:** Asegúrate de que tu mano esté bien iluminada
2. **Ajusta la distancia:** Mantén tu mano a una distancia moderada de la cámara (20-50 cm)
3. **Cambia la configuración:** Prueba con `--mano-derecha` o `--mano-izquierda` según corresponda
4. **Modo espejo:** Prueba con o sin `--no-mirror` para ver qué funciona mejor

## 🔄 Cambios Recientes

- **Mano predeterminada:** Cambiada a mano izquierda para mayor comodidad para la mayoría de usuarios
- **Gesto de pinza:** Implementado para activar/desactivar el dibujo juntando pulgar e índice
- **Modo espejo:** Añadida opción para ver la cámara en modo espejo para experiencia más natural
- **Opciones de línea de comandos:** Mejoradas para mayor flexibilidad y control
- **Optimizaciones de interpolación:** Líneas más suaves durante movimientos rápidos
- **Mejoras en la detección:** Mayor precisión en la detección de gestos y posición de dedos
- **Selección de manos:** Soporte para seleccionar qué mano detectar (derecha, izquierda o ambas)
- **Panel de métricas:** Visualización mejorada de datos de rendimiento en tiempo real

## 🛣️ Desarrollo Futuro

- Guardar y cargar dibujos
- Herramientas adicionales (formas, texto)
- Reconocimiento de gestos adicionales
- Modo multijugador para colaboración
- Exportación a formatos estándar
- Optimizaciones para dispositivos de bajo rendimiento

## 📄 Licencia

Este proyecto está licenciado bajo los términos de la licencia MIT. Ver el archivo LICENSE para más detalles.

## 👥 Contribuir

Las contribuciones son bienvenidas. Por favor, siente libre de enviar pull requests o abrir issues para mejorar el proyecto.

---

Desarrollado como parte de un proyecto personal para explorar las capacidades de OpenCV y MediaPipe en la creación de interfaces naturales de usuario.
