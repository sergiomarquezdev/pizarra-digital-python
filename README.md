# Pizarra Digital Controlada por Gestos de Manos

![Estado](https://img.shields.io/badge/Estado-En%20Desarrollo-brightgreen)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![OpenCV](https://img.shields.io/badge/OpenCV-4.5.0%2B-green)
![MediaPipe](https://img.shields.io/badge/MediaPipe-0.8.10%2B-orange)

Una aplicación Python que permite dibujar en una pizarra digital utilizando gestos de manos capturados a través de la cámara web. Este proyecto combina visión por computadora y seguimiento de manos en tiempo real para crear una experiencia de dibujo interactiva sin contacto físico.

## 📋 Características

- **Dibujo Intuitivo:** Usa tu dedo índice como lápiz digital
- **Paleta de Colores:** Incluye cinco colores predefinidos (blanco, negro, azul, rojo, verde)
- **Borrado Rápido:** Botón para limpiar la pizarra completamente
- **Interfaz Minimalista:** Panel de controles simple e intuitivo
- **Experiencia Fluida:** Optimizado para rendimiento en tiempo real con visualización de FPS
- **Detector Robusto:** Seguimiento preciso de la mano con MediaPipe Hands

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

### Diagnóstico de Cámara

Antes de ejecutar la aplicación principal, puedes verificar que tu cámara funciona correctamente:

```bash
python tests/test_camera.py
```

Este script probará diferentes índices de cámara y te indicará cuál está disponible.

### Ejecutar la Aplicación

```bash
python src/run.py
```

### Controles:

- **Dibujar:** Extiende tu dedo índice frente a la cámara
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
│   │   ├── dibujo/              # Lógica de dibujo basada en gestos
│   │   ├── interfaz/            # Elementos de la interfaz de usuario
│   │   ├── lienzo/              # Manejo del lienzo de dibujo
│   │   ├── utils/               # Utilidades varias
│   │   ├── config.py            # Configuración global
│   │   └── main.py              # Punto de entrada principal
│   ├── run.py                   # Script para ejecutar la aplicación
│   └── test_camera.py           # Herramienta de diagnóstico de cámara
├── tests/                       # Pruebas unitarias
├── project_docs/                # Documentación técnica
├── requirements.txt             # Dependencias del proyecto
├── README.md                    # Este archivo
└── .gitignore                   # Patrones de archivos a ignorar
```

## ⚠️ Solución de Problemas

### Problemas con la Cámara

Si la aplicación no detecta tu cámara correctamente:

1. **Verifica los permisos:** Asegúrate de que tu sistema operativo permita el acceso a la cámara para aplicaciones Python.
2. **Cierra otras aplicaciones:** Otras aplicaciones podrían estar usando la cámara (Zoom, Teams, navegadores).
3. **Prueba diferentes índices:** Ejecuta `python src/test_camera.py` para identificar el índice correcto.
4. **Actualiza la configuración:** Modifica `CAMERA_INDEX` en `src/pizarra_digital/config.py` con el índice correcto.

### Problemas de Detección de Manos

1. **Mejora la iluminación:** Asegúrate de que tu mano esté bien iluminada.
2. **Ajusta la distancia:** Mantén tu mano a una distancia moderada de la cámara (20-50 cm).
3. **Evita fondos complejos:** Un fondo simple mejora la precisión de la detección.

## 🔄 Cambios Recientes

- **Mejora en la detección de cámaras:** Implementada detección automática de cámaras disponibles.
- **Ajustes en MediaPipe:** Optimizada la configuración para mejorar la precisión de detección.
- **Nueva herramienta de diagnóstico:** Añadido script para verificar y configurar la cámara.
- **Correcciones de importación:** Solucionados problemas con las rutas de módulos.

## 🛣️ Desarrollo Futuro

Ver [project_docs/roadmap.md](project_docs/roadmap.md) para detalles sobre las próximas funcionalidades planificadas.

## 📄 Licencia

Este proyecto está licenciado bajo los términos de la licencia MIT. Ver el archivo LICENSE para más detalles.

## 👥 Contribuir

Las contribuciones son bienvenidas. Por favor, siente libre de enviar pull requests o abrir issues para mejorar el proyecto.

---

Desarrollado como parte de un proyecto personal para explorar las capacidades de OpenCV y MediaPipe en la creación de interfaces naturales de usuario.
