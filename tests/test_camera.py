"""
Script para probar el acceso a la cámara.
Este script intenta abrir la cámara usando directamente OpenCV
sin depender del resto de la aplicación.
"""
import cv2
import time
import sys

def test_camera():
    # Intentar con diferentes índices de cámara
    for camera_index in range(3):  # Prueba índices 0, 1, 2
        print(f"Intentando abrir cámara con índice {camera_index}...")

        # Intentar abrir la cámara
        cap = cv2.VideoCapture(camera_index)

        if not cap.isOpened():
            print(f"  No se pudo abrir la cámara con índice {camera_index}")
            continue

        # Configurar resolución
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        # Leer un fotograma
        ret, frame = cap.read()

        if not ret:
            print(f"  Se pudo abrir la cámara {camera_index} pero no leer un fotograma")
            cap.release()
            continue

        # Si llegamos aquí, la cámara funciona
        print(f"  Cámara {camera_index} funciona correctamente!")
        print(f"  Resolución: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")

        # Mostrar el fotograma
        print("  Mostrando fotograma de prueba (presiona 'q' para salir)...")

        # Bucle para mostrar algunos fotogramas
        start_time = time.time()
        frame_count = 0

        while time.time() - start_time < 5:  # 5 segundos de prueba
            ret, frame = cap.read()
            if not ret:
                print("  Error al leer fotograma")
                break

            frame_count += 1

            # Voltear horizontalmente
            frame = cv2.flip(frame, 1)

            # Mostrar información
            cv2.putText(frame, f"Camera: {camera_index}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Mostrar fotograma
            cv2.imshow("Camera Test", frame)

            # Salir con 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        fps = frame_count / (time.time() - start_time)
        print(f"  FPS promedio: {fps:.2f}")

        # Liberar recursos
        cap.release()
        cv2.destroyAllWindows()

        return camera_index  # Devuelve el índice de la cámara que funciona

    # Si llegamos aquí, ninguna cámara funcionó
    print("No se pudo abrir ninguna cámara. Verifica que tu cámara esté conectada y no esté siendo utilizada por otra aplicación.")
    return -1

if __name__ == "__main__":
    working_camera = test_camera()

    if working_camera >= 0:
        print(f"\nInformación para configurar tu proyecto:\n")
        print(f"Edita el archivo 'src/pizarra_digital/config.py' y establece:")
        print(f"CAMERA_INDEX: int = {working_camera}")

        # Sugerir ejecutar la aplicación
        print(f"\nLuego ejecuta:")
        print(f"python src/run.py")
    else:
        sys.exit(1)
