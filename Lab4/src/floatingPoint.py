import numpy as np
import matplotlib.pyplot as plt
from legacy.DSPtools import resp_freq, eyediagram

# Parameters for floating point representation

T = 1.0/100.0e6  # 100 MHz
OS = 4
N_BAUDS = 8
BETA = 0.5  # roll-off
N_SYMB = 2**9 - 1  # 511 symbols

SEED_I = 0x1AA
SEED_Q = 0x1FE

# Functions


def PRBS9_generator(n_bits, seed=0x1FF):
    """
    Generates a PRBS9 sequence of length n_bits.

    @param n_bits: Number of bits to generate
    @param seed: Initial state of the PRBS9 generator (default is 0x1FF)
    @return: List of generated bits (0s and 1s)
    """
    prbs9 = []
    state = seed

    for _ in range(n_bits):
        prbs9.append(state & 1)

        new_bit = ((state >> 3) ^ (state >> 8)) & 1
        state = (state >> 1) | (new_bit << 8)

    return prbs9


def qpsk_symbols(bits):
    """
    Maps bits to QPSK symbols: 0 -> -1, 1 -> +1.

    @param bits: List of bits (0s and 1s)
    @return: List of QPSK symbols (+1/-1)
    """
    return np.array(bits) * 2 - 1


def interpolate_symbols(symbols):
    """
    Interpolates the QPSK symbols by a factor of os.

    @param symbols: List of QPSK symbols
    @return: List of interpolated symbols
    """
    interpolated = []

    for symbol in symbols:
        interpolated.append(symbol)
        for _ in range(OS - 1):
            interpolated.append(0)  # Insert zeros for interpolation

    return interpolated


def tx_filter(symbols):
    """
    Filtra los símbolos con un pulso Raised Cosine.

    @param symbols: Array de símbolos (+1/-1)
    @return: Señal filtrada
    """
    from legacy.DSPtools import rcosine

    (t, rc) = rcosine(BETA, T, OS, N_BAUDS, Norm=False)

    interpolated = interpolate_symbols(symbols)

    filtered = np.convolve(rc, interpolated, mode='same')

    return filtered, rc, t


def BER(bits_tx, bits_rx):
    """
    Computes the Bit Error Rate (BER) between transmitted and received bits.

    @param bits_tx: List of transmitted bits
    @param bits_rx: List of received bits
    @return: BER value
    """
    
    n = min(len(bits_tx), len(bits_rx))
    errors = np.sum(np.array(bits_tx[:n]) != np.array(bits_rx[:n]))
    ber = errors / n
    
    return ber, errors, n


if __name__ == "__main__":
    bits_i = PRBS9_generator(N_SYMB, SEED_I)
    bits_q = PRBS9_generator(N_SYMB, SEED_Q)

    symbols_i = qpsk_symbols(bits_i)
    symbols_q = qpsk_symbols(bits_q)

    signal_i, h, t = tx_filter(symbols_i)
    signal_q, _, _ = tx_filter(symbols_q)

    # -------- 1. BITS TRANSMITIDOS --------

    plt.figure(figsize=[14, 6])
    plt.subplot(2, 1, 1)
    plt.stem(bits_i[:50], linefmt='b-', markerfmt='bo', basefmt='k-')
    plt.title('Bits Transmitidos - Canal I (PRBS9 seed %03X)' % SEED_I)
    plt.xlabel('Índice de bit')
    plt.ylabel('Valor')
    plt.ylim(-0.2, 1.2)
    plt.grid(True)

    plt.subplot(2, 1, 2)
    plt.stem(bits_q[:50], linefmt='r-', markerfmt='ro', basefmt='k-')
    plt.title('Bits Transmitidos - Canal Q (PRBS9 seed %03X)' % SEED_Q)
    plt.xlabel('Índice de bit')
    plt.ylabel('Valor')
    plt.ylim(-0.2, 1.2)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('../images/bits_transmitidos.png')

    # -------- 2. RESPUESTA AL IMPULSO Y FRECUENCIA --------

    Ts = T / OS
    Nfreqs = 256
    [H, _, F] = resp_freq(h, Ts, Nfreqs)

    # Impulso
    plt.figure(figsize=[14, 6])
    plt.plot(t, h, 'bo-', linewidth=2.0, label=r'$\beta=%.2f$' % BETA)
    plt.xlabel('Tiempo')
    plt.ylabel('Amplitud')
    plt.title('Respuesta al impulso del filtro')
    plt.legend()
    plt.grid(True)
    plt.savefig('../images/impulso.png')

    # Frecuencia (convertir a MHz para legibilidad)
    F_MHz = np.array(F) / 1e6
    f_baud_half = (1.0 / T) / 2.0 / 1e6    # Fbaud/2 en MHz
    f_s_half = (1.0 / Ts) / 2.0 / 1e6      # Fs/2 en MHz
    f_low = f_baud_half * (1 - BETA)       # Inicio banda de transición
    f_high = f_baud_half * (1 + BETA)      # Fin banda de transición

    plt.figure(figsize=[14, 6])
    plt.semilogx(F_MHz, 20*np.log10(H), 'b', linewidth=2.0,
                 label=r'$\beta=%.2f$' % BETA)

    # Región sombreada de banda de transición
    plt.axvspan(f_low, f_high, alpha=0.15, color='orange',
                label=r'Banda transición (%.0f–%.0f MHz)' % (f_low, f_high))

    plt.axvline(x=f_baud_half, color='gray', linewidth=1.5,
                linestyle=':', label=r'$F_{baud}/2 = %.0f$ MHz' % f_baud_half)
    plt.axvline(x=f_s_half, color='k', linewidth=1.5,
                linestyle='--', label=r'$F_s/2 = %.0f$ MHz' % f_s_half)
    plt.axhline(y=20*np.log10(0.5), color='gray', linewidth=1.2,
                linestyle='-.', label='20$\\log_{10}$(0.5) = -6 dB')

    plt.legend(loc=3, fontsize=10)
    plt.grid(True, which='both', alpha=0.3)
    plt.title('Respuesta en frecuencia del filtro Raised Cosine')
    plt.xlim(F_MHz[1], F_MHz[-1])
    plt.xlabel('Frecuencia [MHz]')
    plt.ylabel('Magnitud [dB]')
    plt.tight_layout()
    plt.savefig('../images/frecuencia.png')

    # -------- 3. SALIDA Y DIAGRAMA DE OJO --------

    plt.figure(figsize=[14, 6])
    plt.subplot(2, 1, 1)
    plt.plot(signal_i, 'b-', linewidth=2.0, label='Canal I')
    plt.xlim(1000, 1250)
    plt.grid(True)
    plt.legend()
    plt.xlabel('Muestras')
    plt.ylabel('Magnitud')
    plt.title('Salida del filtro Tx')

    plt.subplot(2, 1, 2)
    plt.plot(signal_q, 'r-', linewidth=2.0, label='Canal Q')
    plt.xlim(1000, 1250)
    plt.grid(True)
    plt.legend()
    plt.xlabel('Muestras')
    plt.ylabel('Magnitud')
    plt.savefig('../images/salida_filtro.png')

    # Diagrama de ojo
    n_bauds = 8
    offset = 1  # Fase óptima

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=[14, 5])
    eyediagram(signal_i, n_bauds, offset, T, ax=ax1, color='b')
    ax1.set_title('Canal I')
    ax1.set_xlabel('Tiempo [s]')
    eyediagram(signal_q, n_bauds, offset, T, ax=ax2, color='r')
    ax2.set_title('Canal Q')
    ax2.set_xlabel('Tiempo [s]')
    ax2.set_ylabel('Magnitud')
    fig.suptitle(f'Diagramas de Ojo - Offset = {offset}')
    plt.tight_layout()
    plt.savefig('../images/diagrama_ojo.png')

    # -------- 4. CONSTELACIÓN POR FASE --------

    offsets = [0, 1, 2, 3]
    colors = ['orange', 'brown', 'green', 'magenta']
    offset_labels = ['$offset=0$', '$offset=1$',
                     '$offset=2$', '$offset=3$']

    plt.figure(figsize=[18, 5])
    for idx, (off, label) in enumerate(zip(offsets, offset_labels)):
        plt.subplot(1, 4, idx + 1)
        plt.plot(signal_i[off::OS], signal_q[off::OS],
                 '.', color=colors[idx], markersize=3)
        plt.title('Constelación - %s' % label)
        plt.xlabel('I')
        plt.ylabel('Q')
        plt.grid(True)
        plt.xlim(-1.5, 1.5)
        plt.ylim(-1.5, 1.5)
        plt.gca().set_aspect('equal', adjustable='box')

    plt.tight_layout()
    plt.savefig('../images/constelacion.png')

    # -------- 5. BER (Bit Error Rate) --------

    offset_opt = 3 # Fase óptima de muestreo
    
    sampled_i = signal_i[offset_opt::OS]
    sampled_q = signal_q[offset_opt::OS]
    
    discreted_i = (sampled_i > 0).astype(int)
    discreted_q = (sampled_q > 0).astype(int)
    
    ber, errors_i, n_i = BER(bits_i, discreted_i)
    ber, errors_q, n_q = BER(bits_q, discreted_q)
    
    print("Bit Error Rate (BER) Results: (offset = %d)" % offset_opt)
    print(f"BER Canal I: {ber:.2f} % (Errores: {errors_i}/{n_i})")
    print(f"BER Canal Q: {ber:.2f} % (Errores: {errors_q}/{n_q})")


