import matplotlib.pyplot as plt
from muestras import muestras as samples
import numpy as np


if __name__ == "__main__":
    # 1. Resolución espectral.
    fs = 200  # Hz
    N = len(samples)
    spectral_resolution = fs / N
    print(f"La resolución espectral es de {spectral_resolution} Hz")

    # 2. Espectro de la frecuencia de la señal.
    fft = np.abs(1 / N * np.fft.fft(samples)) ** 2
    frequency = spectral_resolution * np.linspace(
        start=0, stop=N, num=N, endpoint=False
    )
    plt.plot(frequency, fft)
    plt.show()

    # 3. A simple inspección que frecuencia(s) distingue.
    print("A simple inspección las frecuencias son 50 y 150 Hz")

    # 4. Aplique alguna técnica que le permita mejorar la resolución espectral y tome nuevamente el espectro.
    zero_padding = np.zeros((4 * N))
    zero_padding[:N] = samples
    fft = np.abs(1 / N * np.fft.fft(zero_padding)) ** 2
    frequency = spectral_resolution * np.linspace(
        start=0, stop=N, num=4 * N, endpoint=False
    )
    td = np.linspace(start=0, stop=N, num=4 * N, endpoint=False)
    plt.figure(figsize=(16, 4))
    plt.subplot(1, 2, 1)
    plt.plot(td, zero_padding)
    plt.subplot(1, 2, 2)
    plt.plot(frequency, fft)
    plt.show()

    # 5. Indique si ahora los resultados difieren del punto 3 y argumente su respuesta. 
    print("La resolución espectral producto de 'zero padding' es 4 veces menor")