# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.sans-serif'] = ['SimHei']


class DataPrePlotter:
    @staticmethod
    def plot_cwt(cwt_matr, freqs, time):
        """
        绘制2D CWT时频图和3D CWT时频图，测试用
        """
        plt.figure(figsize=(10, 6))
        plt.imshow(np.abs(cwt_matr), extent=[0, time[-1], freqs[-1], freqs[0]], cmap='jet', aspect='auto')
        plt.colorbar(label='振幅')
        plt.xlabel('时间 (秒)')
        plt.ylabel('频率 (Hz)')
        plt.title('2D CWT时频图')
        plt.show()

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        t, f = np.meshgrid(time, freqs)
        ax.plot_surface(t, f, np.abs(cwt_matr), cmap='jet')
        ax.set_xlabel('时间 (秒)')
        ax.set_ylabel('频率 (Hz)')
        ax.set_zlabel('振幅')
        plt.title('3D CWT时频图')
        plt.show()

    @staticmethod
    def plot_frequency_spectrum_as_bar(spectrum, freq_axis, fs):
        # 绘制频谱柱状图
        plt.figure()
        plt.bar(freq_axis[:len(spectrum) // 2], np.abs(spectrum)[:len(spectrum) // 2], width=fs / len(spectrum))
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Magnitude')
        plt.title('Frequency Spectrum Bar Plot')
        plt.grid(True)
        plt.show()

    @staticmethod
    def plot_cepstrum_as_bar(cepstrum, fs):
        # 获取频率轴
        freq_axis = np.fft.fftfreq(len(cepstrum), 1 / fs)

        # 绘制倒频谱柱状图
        plt.figure()
        plt.bar(freq_axis[:len(cepstrum) // 2], cepstrum[:len(cepstrum) // 2], width=fs / len(cepstrum))
        plt.xlabel('index')
        plt.ylabel('Cepstrum Value')
        plt.title('Cepstrum Bar Plot')
        plt.grid(True)
        plt.show()

    @staticmethod
    def plot_test(cwtmatr_flitered, freqs, time, signal):
        plt.figure(figsize=(10, 6))
        plt.subplot(211)  # 第一整行
        plt.plot(time, signal)
        plt.xlabel(u'time(s)')
        plt.title(u'Time spectrum')
        plt.subplot(212)  # 第二整行

        plt.contourf(time, freqs, abs(cwtmatr_flitered))
        plt.ylabel(u'freq(Hz)')
        plt.xlabel(u'time(s)')
        plt.subplots_adjust(hspace=0.4)
        plt.show()
