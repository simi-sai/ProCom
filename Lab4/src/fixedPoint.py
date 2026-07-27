import numpy as np
import matplotlib.pyplot as plt
from legacy.DSPtools import rcosine, resp_freq, eyediagram
from legacy._fixedInt import arrayFixedInt
from floatingPoint import PRBS9_generator, qpsk_symbols, interpolate_symbols, BER

# Parameters

T = 1.0/100.0e6  # 100 MHz
OS = 4
N_BAUDS = 8
BETA = 0.5  # roll-off
N_SYMB = 2**9 - 1  # 511 symbols
OFFSET_OPT = 1

SEED_I = 0x1AA
SEED_Q = 0x1FE

# Fixed-point configurations

coeff_config = [
    ("S(1,7)", 1, 7),
    ("S(1,4)", 1, 4),
]

filter_config = [
    ("S(2,7)", 2, 7),
    ("S(2,4)", 2, 4),
]


# Functions

def quantize_filter(h, int_width, fract_width, round_mode='trunc'):
    """
    Quantizes the filter coefficients to fixed-point representation.

    @param h: Array of filter coefficients
    @param int_width: Integer width of the fixed-point representation
    @param fract_width: Fractional width of the fixed-point representation
    @param round_mode: Rounding mode ('trunc', 'round', 'ceil', 'floor')
    @return: Array of quantized filter coefficients
    """

    h_q = arrayFixedInt(intWidth=int_width, fractWidth=fract_width, N=h,
                        signedMode='S', roundMode=round_mode,
                        saturateMode='saturate')

    return np.array([c.fValue for c in h_q])


if __name__ == "__main__":
    # -------- Generación de señales (igual que floating point) --------

    bits_i = PRBS9_generator(N_SYMB, SEED_I)
    bits_q = PRBS9_generator(N_SYMB, SEED_Q)

    symbols_i = qpsk_symbols(bits_i)
    symbols_q = qpsk_symbols(bits_q)

    interpolated_i = interpolate_symbols(symbols_i)
    interpolated_q = interpolate_symbols(symbols_q)

    (t, h) = rcosine(BETA, T, OS, N_BAUDS, Norm=False)

    # -------- 1. RESPUESTA AL IMPULSO (comparación) --------

    plt.figure(figsize=[14, 6])
    plt.plot(t, h, 'b-o', linewidth=2.0, label='Original (flotante)')

    for name, int_w, frac_w in coeff_config:
        h_q = quantize_filter(h, int_w, frac_w, 'trunc')
        plt.plot(t, h_q, '--', linewidth=1.5, label=name)

    plt.xlabel('Tiempo')
    plt.ylabel('Amplitud')
    plt.title('Respuesta al impulso - Comparación punto fijo vs flotante')
    plt.legend()
    plt.grid(True)
    plt.savefig('../images/fixed_point/impulso_comparacion.png')
    plt.close()

    # -------- 2. RESPUESTA EN FRECUENCIA (comparación) --------

    Ts = T / OS
    Nfreqs = 256
    [H, _, F] = resp_freq(h, Ts, Nfreqs)
    F_MHz = np.array(F) / 1e6

    plt.figure(figsize=[14, 6])
    plt.semilogx(F_MHz, 20*np.log10(H), 'b', linewidth=2.0, label='Original')

    for name, int_w, frac_w in coeff_config:
        h_q = quantize_filter(h, int_w, frac_w, 'trunc')
        [H_q, _, _] = resp_freq(h_q, Ts, Nfreqs)
        plt.semilogx(F_MHz, 20*np.log10(H_q), '--', linewidth=1.5, label=name)

    plt.axvline(x=(1.0/T)/2.0/1e6, color='gray', linewidth=1.5,
                linestyle=':', label=r'$F_{baud}/2$')
    plt.axhline(y=20*np.log10(0.5), color='gray', linewidth=1.2,
                linestyle='-.', label='-6 dB')
    plt.legend(loc=3)
    plt.grid(True, which='both', alpha=0.3)
    plt.title('Respuesta en frecuencia - Comparación punto fijo vs flotante')
    plt.xlim(F_MHz[1], F_MHz[-1])
    plt.xlabel('Frecuencia [MHz]')
    plt.ylabel('Magnitud [dB]')
    plt.tight_layout()
    plt.savefig('../images/fixed_point/frecuencia_comparacion.png')
    plt.close()

    # -------- 3. SALIDA FILTRADA Y DIAGRAMA DE OJO --------

    for name, int_w, frac_w in filter_config:
        h_q = quantize_filter(
            h, coeff_config[0][1], coeff_config[0][2], 'trunc')

        filtered_i = np.convolve(h_q, interpolated_i, mode='same')
        filtered_q = np.convolve(h_q, interpolated_q, mode='same')

        # Salida filtrada
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=[14, 6])
        ax1.plot(filtered_i, 'b-', linewidth=2.0)
        ax1.set_xlim(1000, 1250)
        ax1.set_xlabel('Muestras')
        ax1.set_ylabel('Magnitud')
        ax1.set_title(f'Salida filtro Tx - {name} (Canal I)')
        ax1.grid(True)

        ax2.plot(filtered_q, 'r-', linewidth=2.0)
        ax2.set_xlim(1000, 1250)
        ax2.set_xlabel('Muestras')
        ax2.set_ylabel('Magnitud')
        ax2.set_title(f'Salida filtro Tx - {name} (Canal Q)')
        ax2.grid(True)

        plt.tight_layout()
        plt.savefig('../images/fixed_point/salida_%s.png' % name.replace('(',
                    '').replace(')', '').replace(',', '-').replace(' ', ''))
        plt.close()

        # Diagrama de ojo
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=[14, 5])
        eyediagram(filtered_i, 8, OFFSET_OPT, T, ax=ax1, color='b')
        ax1.set_title('Canal I')
        eyediagram(filtered_q, 8, OFFSET_OPT, T, ax=ax2, color='r')
        ax2.set_title('Canal Q')
        fig.suptitle(f'Diagramas de Ojo - {name}')
        plt.tight_layout()
        plt.savefig('../images/fixed_point/ojo_%s.png' % name.replace('(',
                    '').replace(')', '').replace(',', '-').replace(' ', ''))
        plt.close()

    # -------- 4. CONSTELACIÓN POR FASE --------

    offsets = [0, 1, 2, 3]
    offset_labels = ['offset=0', 'offset=1', 'offset=2', 'offset=3']

    for name, int_w, frac_w in filter_config:
        h_q = quantize_filter(
            h, coeff_config[0][1], coeff_config[0][2], 'trunc')

        filtered_i = np.convolve(h_q, interpolated_i, mode='same')
        filtered_q = np.convolve(h_q, interpolated_q, mode='same')

        fig, axes = plt.subplots(1, 4, figsize=[18, 5])
        for idx, (off, label) in enumerate(zip(offsets, offset_labels)):
            axes[idx].plot(filtered_i[off::OS], filtered_q[off::OS],
                           '.', color='black', markersize=3)
            axes[idx].set_title('Constelación - %s' % label)
            axes[idx].set_xlabel('I')
            axes[idx].set_ylabel('Q')
            axes[idx].grid(True)
            axes[idx].set_xlim(-1.5, 1.5)
            axes[idx].set_ylim(-1.5, 1.5)
            axes[idx].set_aspect('equal', adjustable='box')

        fig.suptitle(f'Constelación - {name}')
        plt.tight_layout()
        plt.savefig('../images/fixed_point/constelacion_%s.png' % name.replace('(',
                    '').replace(')', '').replace(',', '-').replace(' ', ''))
        plt.close()

    # -------- 5. BER --------

    for name, int_w, frac_w in filter_config:
        h_q = quantize_filter(
            h, coeff_config[0][1], coeff_config[0][2], 'trunc')

        filtered_i = np.convolve(h_q, interpolated_i, mode='same')
        filtered_q = np.convolve(h_q, interpolated_q, mode='same')

        sampled_i = filtered_i[OFFSET_OPT::OS]
        sampled_q = filtered_q[OFFSET_OPT::OS]

        bits_rx_i = (sampled_i > 0).astype(int)
        bits_rx_q = (sampled_q > 0).astype(int)

        ber_i, err_i, n_i = BER(bits_i, bits_rx_i)
        ber_q, err_q, n_q = BER(bits_q, bits_rx_q)

        print(f"BER - {name}")
        print(f"  Canal I: {ber_i:.4f} ({err_i}/{n_i})")
        print(f"  Canal Q: {ber_q:.4f} ({err_q}/{n_q})")

    # -------- 6. ARCHIVOS DE ESTÍMULOS PARA VM --------

    # PRBS9 bits - ciclo a ciclo (como tb_vector_gen.py)
    file_bits_i = open('../vm/stim_bits_i.out', 'w')
    file_bits_q = open('../vm/stim_bits_q.out', 'w')

    for ptr in range(N_SYMB):
        file_bits_i.write('%d\n' % bits_i[ptr])
        file_bits_q.write('%d\n' % bits_q[ptr])

    file_bits_i.close()
    file_bits_q.close()

    # Símbolos QPSK - ciclo a ciclo
    file_sym_i = open('../vm/stim_symbols_i.out', 'w')
    file_sym_q = open('../vm/stim_symbols_q.out', 'w')

    for ptr in range(N_SYMB):
        file_sym_i.write('%d\n' % int(symbols_i[ptr]))
        file_sym_q.write('%d\n' % int(symbols_q[ptr]))

    file_sym_i.close()
    file_sym_q.close()

    # Coeficientes del filtro cuantizados (mejor config)
    h_q_best = quantize_filter(h, coeff_config[0][1], coeff_config[0][2], 'trunc')
    file_coeff = open('../vm/stim_filter_coeff.out', 'w')

    for ptr in range(len(h_q_best)):
        file_coeff.write('%.6f\n' % h_q_best[ptr])

    file_coeff.close()

    print("\nArchivos de estímulos generados en ../vm/")
