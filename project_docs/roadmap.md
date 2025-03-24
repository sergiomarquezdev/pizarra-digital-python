# Roadmap de Desarrollo: Pizarra Digital Controlada por Gestos

## 1. Configuración Inicial del Proyecto
- [x] Crear estructura básica de directorios (src, tests, docs)
- [x] Configurar entorno virtual (venv/conda)
- [x] Crear archivo requirements.txt con dependencias iniciales (opencv-python, mediapipe, numpy)
- [x] Configurar herramientas de desarrollo (ruff para formateo de código)
- [x] Crear archivo config.py para configuraciones globales
- [x] Inicializar repositorio git

## 2. Implementación de la Captura de Video con OpenCV
- [x] Crear módulo para la captura de video (captura.py)
- [x] Implementar función para inicializar la cámara web
- [x] Añadir manejo de errores para cuando la cámara no está disponible
- [x] Implementar función para leer fotogramas de la cámara
- [x] Añadir función para liberar recursos de la cámara
- [x] Crear bucle básico de captura de video
- [x] Añadir test para verificar la disponibilidad de la cámara

## 3. Integración de MediaPipe para el Seguimiento de Manos
- [x] Crear módulo para el seguimiento de manos (deteccion.py)
- [x] Implementar inicialización de MediaPipe Hands
- [x] Añadir función para procesar fotogramas y detectar manos
- [x] Extraer coordenadas de los landmarks de las manos
- [x] Implementar la detección específica de la punta del dedo índice
- [x] Añadir visualización de landmarks para depuración
- [ ] Crear test para verificar la detección de manos

## 4. Implementación del Lienzo de Dibujo
- [x] Crear módulo para el lienzo de dibujo (lienzo.py)
- [x] Implementar clase para gestionar el lienzo (tamaño, color de fondo)
- [x] Añadir funciones para dibujar puntos/líneas en el lienzo
- [x] Implementar función para limpiar el lienzo
- [x] Crear función para superponer el lienzo sobre el fotograma de video
- [ ] Añadir test para verificar las operaciones básicas del lienzo

## 5. Desarrollo de la Funcionalidad de Dibujo con el Dedo Índice
- [x] Crear módulo para la lógica de dibujo (dibujo.py)
- [x] Implementar seguimiento de la posición del dedo índice
- [x] Añadir lógica para determinar cuándo se debe dibujar (dedo índice extendido)
- [x] Implementar dibujo de líneas entre posiciones consecutivas del dedo índice
- [x] Añadir ajustes para la sensibilidad del dibujo
- [ ] Crear test para verificar la detección del gesto de dibujo

## 6. Creación de la Interfaz de Usuario con Botones
- [x] Crear módulo para la interfaz de usuario (interfaz.py)
- [x] Diseñar layout básico de la interfaz
- [x] Implementar clase para gestionar botones en la interfaz
- [x] Añadir función para renderizar botones sobre el fotograma
- [x] Implementar detección de clics en los botones (por posición del mouse)
- [ ] Añadir test para verificar la funcionalidad de los botones

## 7. Implementación de la Funcionalidad de Cambio de Colores
- [x] Añadir paleta de colores predefinidos (blanco, negro, azul, rojo, verde)
- [x] Implementar botones de selección de color en la interfaz
- [x] Añadir lógica para cambiar el color actual del dibujo
- [x] Implementar indicador visual del color seleccionado
- [ ] Crear test para verificar el cambio de colores

## 8. Implementación del Borrado de Pizarra
- [x] Añadir botón "Borrar Pizarra" en la interfaz
- [x] Implementar función para borrar completamente el lienzo
- [ ] Añadir confirmación visual de borrado (opcional)
- [ ] Crear test para verificar la funcionalidad de borrado

## 9. Integración de Componentes y Aplicación Principal
- [x] Crear módulo principal (main.py)
- [x] Integrar todos los componentes (captura, detección, lienzo, dibujo, interfaz)
- [x] Implementar bucle principal de la aplicación
- [x] Añadir manejo de eventos de teclado (salir con 'q')
- [x] Implementar visualización de FPS
- [x] Añadir mensajes de estado en la interfaz
- [ ] Crear test de integración para la aplicación completa

## 10. Optimización del Rendimiento
- [x] Medir y documentar el rendimiento inicial (FPS)
- [ ] Optimizar el bucle principal de procesamiento
- [ ] Evaluar y ajustar la resolución de la cámara para un rendimiento óptimo
- [ ] Minimizar las copias de arrays y optimizar operaciones con NumPy
- [ ] Implementar técnicas para reducir el uso de CPU
- [ ] Medir y documentar el rendimiento después de las optimizaciones

## 11. Pruebas y Depuración
- [ ] Realizar pruebas manuales extensivas
- [ ] Identificar y corregir problemas de detección de manos
- [ ] Mejorar la precisión del dibujo
- [ ] Refinar la interfaz de usuario según resultados de pruebas
- [ ] Documentar problemas conocidos y limitaciones

## 12. Documentación y Finalización
- [x] Añadir docstrings completos a todas las funciones y clases
- [x] Crear README.md con instrucciones de instalación y uso
- [x] Documentar requisitos de hardware y software
- [x] Añadir ejemplos de uso
- [ ] Revisar y corregir el formato del código (usando ruff)
- [ ] Realizar revisión final del código
