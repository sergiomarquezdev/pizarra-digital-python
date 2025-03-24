"""
Tests para el módulo de captura de video.
"""
import pytest
import cv2
import numpy as np
from unittest.mock import patch, MagicMock

from src.pizarra_digital.captura.captura import (
    inicializar_camara,
    leer_fotograma,
    liberar_camara,
    CameraError
)

def test_inicializar_camara_error():
    """Test para verificar que se lanza una excepción cuando la cámara no está disponible."""
    # Mockear VideoCapture para simular que la cámara no está disponible
    with patch('cv2.VideoCapture') as mock_cap:
        # Configurar el mock para que isOpened() devuelva False
        mock_instance = MagicMock()
        mock_instance.isOpened.return_value = False
        mock_cap.return_value = mock_instance

        # Verificar que se lanza la excepción esperada
        with pytest.raises(CameraError):
            inicializar_camara()

def test_leer_fotograma_exito():
    """Test para verificar la lectura exitosa de un fotograma."""
    # Crear un fotograma de prueba
    test_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    # Mockear VideoCapture para simular una lectura exitosa
    mock_cap = MagicMock()
    mock_cap.read.return_value = (True, test_frame)

    # Mockear cv2.flip para que devuelva el mismo fotograma
    with patch('cv2.flip', return_value=test_frame):
        # Llamar a la función y verificar el resultado
        ret, frame = leer_fotograma(mock_cap)

        assert ret == True
        assert frame is not None
        assert frame.shape == (480, 640, 3)

def test_leer_fotograma_fallo():
    """Test para verificar el manejo del fallo en la lectura de un fotograma."""
    # Mockear VideoCapture para simular una lectura fallida
    mock_cap = MagicMock()
    mock_cap.read.return_value = (False, None)

    # Llamar a la función y verificar el resultado
    ret, frame = leer_fotograma(mock_cap)

    assert ret == False
    assert frame is None

def test_liberar_camara():
    """Test para verificar la liberación de recursos de la cámara."""
    # Crear un mock para VideoCapture
    mock_cap = MagicMock()

    # Llamar a la función
    liberar_camara(mock_cap)

    # Verificar que se llamó al método release()
    mock_cap.release.assert_called_once()
