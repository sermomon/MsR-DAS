# %% Libs:
import numpy as np
import matplotlib.pyplot as plt
import scipy as scp

# %% SigFilt_HP function: 
def SigFilt_HP(signal, fs, n_filt, fp, plotter=False):
    '''
    Filter a signal using a high-pass Butterworth filter.

    Extended description:
    This function applies a high-pass Butterworth filter to the input signal to remove low-frequency components. 
    The order of the filter and the cutoff frequency are specified by the user. Optionally, it can plot the original 
    and filtered signals for visualization.

    Parameters:
    - signal (np.ndarray): Signal to be filtered.
    - fs (float): Sampling frequency of the signal [Hz].
    - n_filt (int): Order of the Butterworth filter.
    - fp (float): Cutoff frequency of the high-pass filter [Hz].
    - plotter (bool, optional): Flag to plot the signal data. Default is False.

    Returns:
    - signal_filt (np.ndarray): Filtered signal.

    Raises:
    - TypeError: If n_filt is not an integer.

    Application:
    signal_filtered = SigFilt_HP(signal, fs, n_filt, fp, plotter)
    
    Created/Last modified: 2024-04-05
    '''
    # Function implementation goes here
    if not isinstance(n_filt, int):
        raise TypeError("n_filt must be an integer")

    b_hp, a_hp = scp.signal.butter(n_filt, fp, fs=fs, btype='high', analog=False)
    signal_filt = scp.signal.filtfilt(b_hp, a_hp, signal)
    
    if np.any(np.isnan(signal_filt)):
        print("Signal contains NaN values in SigFilt_HP app!")
        signal_filt = scp.signal.lfilter(b_hp, a_hp, signal)
    
    if plotter:
        signal_t = np.arange(0, len(signal) / fs, 1 / fs)
        fig, axs = plt.subplots(1, 2, figsize=(12, 5))
        # Plot the signal
        axs[0].plot(signal_t,signal, color='grey', linewidth=0.5, label='Original')
        axs[0].plot(signal_t,signal_filt, color='blue', label='Filtered')
        axs[0].legend(ncol=2)
        axs[0].set_ylabel('Amplitude [a.u.]')
        axs[0].set_xlabel('Time [s]')
        axs[0].set_title('SigFilt_HP()') 
        axs[0].set_xlim(0,signal_t[-1])
        # Filter response plot:
        FNyquist = fs/2
        w, h = scp.signal.freqz(b_hp, a_hp)  
        f_response = (w/max(w))*FNyquist
        h_dB = 20*np.log10(abs(h)) 
        idx_fdecay = (np.abs(f_response - fp)).argmin()
        if fp==f_response[idx_fdecay]:
            decay_dB = h_dB[idx_fdecay]
        else:
            func_interp = scp.interpolate.interp1d(f_response,h_dB)
            decay_dB = func_interp(fp)
            
        if fp<=5e3:
            axs[1].plot(f_response,h_dB,color='blue')
            axs[1].axvline(x=fp, color='black', linestyle='-')
            axs[1].set_xlabel('Frequency [Hz]')
            axs[1].set_xlim(0,np.max(f_response))
            axs[1].set_title('Butterworth filter frequency response\nN$_{filt}$: %i ; F$_{p}$: %.1f Hz (decay: %.1f dB)' %(n_filt,fp,decay_dB))
        else:
            axs[1].plot(f_response*1e-3,h_dB,color='blue')
            axs[1].axvline(x=fp*1e-3, color='black', linestyle='-')
            axs[1].set_xlabel('Frequency [kHz]')
            axs[1].set_xlim(0,np.max(f_response)*1e-3)    
            axs[1].set_title('Butterworth filter frequency response\nN$_{filt}$: %i ; F$_{p}$: %.1f kHz (decay: %.1f dB)' %(n_filt,fp*1e-3,decay_dB))
        axs[1].set_ylabel('Magnitude [dB]')
        axs[1].set_ylim(-12,1)
        # axs[1].set_xscale('log')
        axs[1].grid()
        
        plt.tight_layout()
        plt.show()

    return signal_filt

# %% SigFilt_LP function: 
def SigFilt_LP(signal, fs, n_filt, fc, plotter=False):
    '''
    Filter a signal using a low-pass Butterworth filter.

    Extended description:
    This function applies a low-pass Butterworth filter to the input signal to remove high-frequency components.
    The order of the filter and the cutoff frequency are specified by the user. Optionally, it can plot the original
    and filtered signals for visualization.

    Parameters:
    - signal (np.ndarray): Signal to be filtered.
    - fs (float): Sampling frequency of the signal [Hz].
    - n_filt (int): Order of the Butterworth filter.
    - fc (float): Cutoff frequency of the low-pass filter [Hz].
    - plotter (bool, optional): Flag to plot the signal data. Default is False.

    Returns:
    - signal_filt (np.ndarray): Filtered signal.

    Raises:
    - TypeError: If n_filt is not an integer.

    Application:
    signal_filtered = SigFilt_LP(signal, fs, n_filt, fc, plotter)
    
    Created/Last modified: 2024-04-05
    '''
    # Function implementation goes here
    if not isinstance(n_filt, int):
        raise TypeError("n_filt must be an integer")

    b_lp, a_lp = scp.signal.butter(n_filt, fc, fs=fs, btype='low', analog=False)
    signal_filt = scp.signal.filtfilt(b_lp, a_lp, signal)
    
    if plotter:
        signal_t = np.arange(0, len(signal) / fs, 1 / fs)
        fig, axs = plt.subplots(1, 2, figsize=(12, 5))
        # Plot the signal
        axs[0].plot(signal_t,signal, color='grey', linewidth=0.5, label='Original')
        axs[0].plot(signal_t,signal_filt, color='red', label='Filtered')
        axs[0].legend(ncol=2)
        axs[0].set_ylabel('Amplitude [a.u.]')
        axs[0].set_xlabel('Time [s]')
        axs[0].set_title('SigFilt_LP()') 
        axs[0].set_xlim(0,signal_t[-1])
        # Filter response plot:
        FNyquist = fs/2
        w, h = scp.signal.freqz(b_lp, a_lp)  
        f_response = (w/max(w))*FNyquist
        h_dB = 20*np.log10(abs(h)) 
        idx_fdecay = (np.abs(f_response - fc)).argmin()
        if fc==f_response[idx_fdecay]:
            decay_dB = h_dB[idx_fdecay]
        else:
            func_interp = scp.interpolate.interp1d(f_response,h_dB)
            decay_dB = func_interp(fc)
            
        if fc<=5e3:
            axs[1].plot(f_response,h_dB, color = 'red')
            axs[1].axvline(x=fc, color='black', linestyle='-')
            axs[1].set_xlabel('Frequency [Hz]')
            axs[1].set_xlim(0,np.max(f_response))
            axs[1].set_xlim(0,np.max(f_response[h_dB>=-15]))  
            axs[1].set_title('Butterworth filter frequency response\nN$_{filt}$: %i ; F$_{c}$: %.1f Hz (decay: %.1f dB)' %(n_filt,fc,decay_dB))
        else:
            axs[1].plot(f_response*1e-3,h_dB, color = 'red')
            axs[1].axvline(x=fc*1e-3, color='black', linestyle='-')
            axs[1].set_xlabel('Frequency [kHz]')
            axs[1].set_xlim(0,np.max(f_response[h_dB>=-15])*1e-3)    
            axs[1].set_title('Butterworth filter frequency response\nN$_{filt}$: %i ; F$_{c}$: %.1f kHz (decay: %.1f dB)' %(n_filt,fc*1e-3,decay_dB))
        axs[1].set_ylabel('Magnitude [dB]')
        axs[1].set_ylim(-12,1)
        # axs[1].set_xscale('log')
        axs[1].grid()
        
        plt.tight_layout()
        plt.show()

    return signal_filt