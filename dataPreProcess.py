# -*- coding: utf-8 -*-
import os
import pywt
import torch
import numpy as np
import pandas as pd
from typing import Tuple, Dict
from dataPrePlot import DataPrePlotter
from dataPreMath import DataPreMath
from scipy.fftpack import fft, ifft
from scipy.signal.windows import hamming

RELATED_PATH_TO_CSV = "D://Working//DataSet//lixiuhong//test//"


def folder_read(folder_path, suffix):
    """
    读取文件夹中所有同类型文件，
    Args:
    folder_path: 文件夹路径
    suffix: 文件后缀
    Return:
        (file_frames, files):同类型的所有文件内容和对应文件名
    """
    file_names = os.listdir(folder_path)
    files = [file_name for file_name in file_names if file_name.endswith(suffix)]

    file_frames = []

    for file_name in files:
        file_path = os.path.join(folder_path, file_name)
        file_frames.append(pd.read_csv(file_path))
    return file_frames, files


class DataPreprocess:
    @staticmethod
    def cwt_and_denoised(original_signal, fs, wavelet_name, total_scales, frequency_threshold):
        """
        对信号执行CWT连续小波变换，默认使用“morlet”小波基函数，并且会对CWT结果进行去噪。
            去噪：分为频率阈值和小波系数阈值
                频率阈值：默认将50Hz以下的小波系数全部设为0
                小波系数阈值：在频率阈值过滤后，再检查小波系数矩阵，将所有小于0.1 * 最大值的小波系数设为0，Touchpass上的阈值推测是0.2 * 最大值，但情况很抽象，酌情选用。
        :param fs: 采样率
        :param frequency_threshold: 频率去噪阈值，默认为50Hz
        :param total_scales: CWT尺度，默认为256，不方便调整，如果采样率发生非常巨大的波动才需要修改，比如从500Hz跃升到1000Hz这种极端情况
        :param wavelet_name: 默认“morlet”
        :param original_signal: 信号振动部分，不支持传入dataframe
        :return: 返回去噪后的CWT结果，二维矩阵形式，大小为 (total_scales-1) * len(original_signal)
                返回CWT得到的频率分量，当采样率对于500Hz的数量级相对稳定时，频率分量也基本保持不变，频率分量是一个 (total_scales-1,) 的ndarray
                    但出于 debug 时的作图需要，所以仍然返回，只在data_pre_entry返回时单独把cwtmatr_filtered拿出来
        """
        fc = pywt.central_frequency(wavelet_name)  # 计算小波函数的中心频率
        cparam = 2 * fc * total_scales  # 常数c
        scales = cparam / np.arange(1, total_scales, +1)
        cwtmatr, freqs = pywt.cwt(original_signal, scales, wavelet_name, sampling_period=1 / fs)

        # 频率阈值过滤
        cwtmatr_filtered = np.where(freqs[:, None] < frequency_threshold, 0, cwtmatr)
        # 小波系数阈值过滤
        # cwtmatr_filtered[cwtmatr_filtered < (0.1 * np.max(cwtmatr_filtered))] = 0

        data_cwt = (cwtmatr_filtered, freqs)
        return data_cwt

    @staticmethod
    def get_fft_cepstrum(signal, fs):
        """
        对通过butter的50Hz高通滤波器后的信号进行倒谱处理，在倒谱前先对整个信号加汉明窗，以防止信号边缘可能的频谱泄露（一般只在信号周期不完整时出现）
        一般来说，fft后，spectrum的结果长度是传入信号的长度，即采样点个数，但由于传入的稳态信号一般都较短，spectrum结果较短，分辨率不高，
            故对fft结果长度作延拓，拓展为fs / 2长度，采样率的一半，即奈奎斯特频率，再大不仅没有意义，还可能导致失真。
            fft方法会自动将spectrum的变化细分到整个横轴上，以增强分辨率。
        :param signal: 振动信号部分，不支持传入dataframe
        :param fs: 采样率
        :return: 返回倒谱结果cepstrum，和spectrum长度保持一致，都是 (fs / 2,) 大小的ndarray
        """
        windowed_signal = signal * hamming(len(signal))
        spectrum = fft(windowed_signal, int(fs / 2))
        cepstrum = np.abs(ifft(np.log(np.abs(spectrum))))

        return cepstrum

    @staticmethod
    def data_pre_entry(signal: pd.DataFrame, pattern: str = 'default', expect_label: str = 'acc_x',
                       plot_choice: str = 'None',
                       frequency_threshold: int = 50,
                       window_time: int = 300, steady_window_time: int = 50,
                       wavelet_name: str = 'morl', total_scales: int = 256) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, torch.Tensor], str]:
        """
        数据处理的入口函数。对信号进行CWT和倒谱处理，标签列只要求信号的xyz三轴加速度和对应时间戳部分。
        一些注意事项：
            vibrate_window_signal 是一个存放xyz轴振动信号部分的 dataframe
                find_signal_window找到信号主轴的振动部分的起始索引和结束索引，每轴都按照主轴索引进行切分。
            window_size & steady_window_size & fs 都根据传入参数确定。
            该类中的 cutCSV方法 相比 myUtil类 的作了一定调整，以适配方差窗口的结果，vibrate / stop_time_period若需要作为参数，自己加。
            对原始信号进行去均值处理 / 高通滤波器处理，分别得到meanless_signal / filtered_signal，分别投入 CWT 和 倒谱处理。
        两个pattern：
            default时，默认将signal的xyz轴都切分成三段，即9个部分，且进行CWT和倒谱处理，直接返回结果，不作图。
            debug时，通过expect_label传入希望看到的信号对应轴，支持作图功能，通过plot_choice选择作图类型。
        :param signal: 原始振动信号，要求具有xyz三轴加速度和对应时间戳四个标签列
        :param pattern: 处理模式，分为 default 和 debug 模式
        :param expect_label: debug模式下，调试的目标标签列，需要是xyz任一一轴的加速度标签列
        :param plot_choice: 作图选择
        :param frequency_threshold: CWT处理中，进行频率过滤的阈值参数，默认为50Hz
        :param window_time: 振动信号窗口大小，即采样点个数，决定振动段整体长度
        :param steady_window_time: 稳态振动信号窗口大小，即采样点个数，决定稳态振动段整体长度
        :param wavelet_name: CWT使用的小波基函数，默认为“morlet”小波基，不建议修改
        :param total_scales: CWT变换尺度，默认256，详见 cwt_and_denoised 方法
        :return: 返回字典 data_pre_result ，标签为 {label}_cwt / cepstrum_{i} ，每个键值对存储的都是 Tensor 张量
                    label表示该键值对的振动轴xyz，i从 0 ~ 2 ，表示振动信号切分后的第 1 ~ 3 段，
                    信号主轴存放在 main_axis 键值对中
        """
        vibrate_window_signal = pd.DataFrame()
        fs = len(signal) / (signal.loc[len(signal) - 1, "timestamp"] - signal.loc[0, "timestamp"]) * 1000000000
        window_size = int(window_time / 1000 * fs)
        steady_window_size = int(steady_window_time / 1000 * fs)
        all_label = ['acc_x', 'acc_y', 'acc_z']

        main_axis = DataPreMath.find_main_axis(signal, all_label)
        main_start_idx, main_end_idx = DataPreMath.find_signal_window(signal[main_axis], window_size, False)
        for label in all_label:
            vibrate_window_signal[label] = signal[label][main_start_idx:main_end_idx].values

        cut_idx = DataPreMath.cutCSV(90000000, 10000000, signal['timestamp'][main_start_idx:main_end_idx].values)

        if pattern == 'default':
            data_pre_result = []
            for i in range(3):
                cwt_tensor_list = []
                cepstrum_tensor_list = []
                for label in all_label:
                    original_signal = vibrate_window_signal[label][cut_idx[2 * i]: cut_idx[2 * i + 1]]
                    steady_period_signal = DataPreMath.find_signal_window(original_signal, steady_window_size)

                    meanless_signal = original_signal - np.mean(original_signal)
                    filtered_signal = DataPreMath.highpass_filter(steady_period_signal, fs)

                    data_cwt = DataPreprocess.cwt_and_denoised(meanless_signal, fs, wavelet_name, total_scales,
                                                               frequency_threshold)
                    data_cepstrum = DataPreprocess.get_fft_cepstrum(filtered_signal.ravel(), fs)

                    cwt_tensor_list.append(torch.from_numpy(data_cwt[0][50:, :]))
                    cepstrum_tensor_list.append(torch.from_numpy(data_cepstrum[:100]))

                cwt_tensor = torch.stack(cwt_tensor_list, dim=0)
                cepstrum_tensor = torch.stack(cepstrum_tensor_list, dim=0)
                data_pre_result.append({'cwt_tensor': cwt_tensor, 'cepstrum_tensor': cepstrum_tensor})

            if len(data_pre_result) != 3:
                print('data pre result len error')
                return

            return data_pre_result[0], data_pre_result[1], data_pre_result[2], main_axis

        if pattern == 'debug':
            data_cwt = []
            data_cepstrum = []
            for i in range(3):
                original_signal = vibrate_window_signal[expect_label][cut_idx[2 * i]: cut_idx[2 * i + 1]]
                steady_period_signal = DataPreMath.find_signal_window(original_signal, steady_window_size)
                meanless_signal = original_signal - np.mean(original_signal)
                filtered_signal = DataPreMath.highpass_filter(steady_period_signal, fs)

                # 时间向量
                data_length = len(meanless_signal)
                time_data = np.linspace(0, data_length / fs, data_length, endpoint=False)

                data_cwt.append(
                    DataPreprocess.cwt_and_denoised(meanless_signal, fs, wavelet_name, total_scales,
                                                    frequency_threshold))
                data_cepstrum.append(
                    DataPreprocess.get_fft_cepstrum(filtered_signal.ravel(), fs))
                # 测试用，作图区
                if plot_choice == 'None':
                    continue
                if plot_choice == 'cwt_plot_2&3D':
                    DataPrePlotter.plot_cwt(data_cwt[i][0], data_cwt[i][1], time_data)
                if plot_choice == 'cwt_plot_test':
                    DataPrePlotter.plot_test(data_cwt[i][0], data_cwt[i][1], time_data, meanless_signal)
                if plot_choice == 'cwt_plot_cepstrum':
                    DataPrePlotter.plot_cepstrum_as_bar(data_cepstrum[i], fs)
                    windowed_signal = signal * hamming(len(filtered_signal.ravel()))
                    spectrum = fft(windowed_signal, int(fs / 2))
                    freq_axis = np.fft.fftfreq(len(spectrum), 1 / fs)
                    DataPrePlotter.plot_frequency_spectrum_as_bar(spectrum, freq_axis, fs)


if __name__ == '__main__':
    data_frames, titles = folder_read(RELATED_PATH_TO_CSV, suffix='.csv')

    for acceleration in data_frames:
        DataPreprocess.data_pre_entry(acceleration, pattern='default', expect_label='acc_x',
                                      plot_choice='cwt_plot_cepstrum')
