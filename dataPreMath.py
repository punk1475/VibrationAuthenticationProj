# -*- coding: utf-8 -*-
import numpy as np
from scipy.signal import butter, filtfilt


class DataPreMath:
    @staticmethod
    def cut_cwt(cwtmatr_filtered):
        # 找到非零元素的索引
        nonzero_indices = np.where(cwtmatr_filtered != 0)

        # 提取非零元素所在的行和列索引
        nonzero_rows = nonzero_indices[0]

        # 找到非零元素所在的最小和最大行索引
        min_row = np.min(nonzero_rows)

        # 截取非零元素所在的部分
        cwtmatr_filtered_trimmed = cwtmatr_filtered[min_row:, :]

        # 返回截取后的结果
        return cwtmatr_filtered_trimmed

    @staticmethod
    def find_main_axis(signal, all_label):
        label_total_variance = []
        for label in all_label:
            if label not in signal:
                print('error:{}'.format(label))
            label_total_variance.append(np.var(signal[label]))

        main_axis_index = np.argmax(label_total_variance)
        return all_label[main_axis_index]

    @staticmethod
    def find_signal_window(signal, window_size, segment_return=True):
        """
        在信号主体上产生一个滑动窗口，统计窗口内的方差 ：
            当传入timestamp时，认为是需要将信号的振动主体部分分离出来，因为此时需要将信号的dataframe中携带的timestamp标签列一并切割出来，
                寻找信号上方差最大的窗口，返回振动信号部分和每个采样点对应的时间戳
            当不传入timestamp时，认为是需要将信号的稳态部分分离出来，此时寻找信号上方差最小的窗口，仅返回信号稳态部分
        :param segment_return:
        :param signal: 信号部分，不能传入dataframe
        :param window_size: 期望分离出的信号窗口大小，即窗口内包含的采样点个数
        :return: 详情见上述描述，segment 和 timestamp都作为 series 返回
        """
        # 计算滑动窗口方差
        variances = []
        for i in range(len(signal) - window_size + 1):
            window = signal[i:i + window_size]
            variances.append(np.var(window))

        if segment_return:
            # 寻找方差 最小 的窗口作为信号段起始点
            start_idx = np.argmin(variances)
            end_idx = start_idx + window_size

            segment = signal[start_idx:end_idx]
            return segment

        # 寻找方差 最大 的窗口作为信号段起始点
        start_idx = np.argmax(variances)
        end_idx = start_idx + window_size

        return start_idx, end_idx

    @staticmethod
    def highpass_filter(data, fs, cutoff_freq=50, order=4):
        """
        高通滤波器，注意：滤波后的信号，50Hz以下的部分并不是就不存在了
        :param data: 需要滤波的信号，只能传入信号主体部分，不能携带timestamp等标签列
        :param fs: 信号采样率
        :param cutoff_freq: 高通滤波阈值，此处为50Hz，
        :param order:
        :return: 以 ndarray 形式，返回滤波后的信号
        """
        nyquist_freq = 0.5 * fs
        normal_cutoff = cutoff_freq / nyquist_freq
        b, a = butter(order, normal_cutoff, btype='high', analog=False)
        filtered_data = filtfilt(b, a, data)
        return filtered_data

    @staticmethod
    def cutCSV(vibrate_time_period, stop_time_period, timestamp):
        """
        按照时间戳，负责对振动窗口内的信号进行三段分割
        :param timestamp: 时间戳序列
        :param vibrate_time_period: 振动开启的持续时间
        :param stop_time_period: 振动停止的持续时间
        :return: index_list作为 数组 返回，存储三段振动信号各自的开始索引和结束索引
        """
        start_time = timestamp[0]
        reference_node_list = [0, 0, 0, 0, 0, 0]
        reference_node_list[0] = start_time
        reference_node_list[1] = reference_node_list[0] + vibrate_time_period
        reference_node_list[2] = reference_node_list[1] + stop_time_period
        reference_node_list[3] = reference_node_list[2] + vibrate_time_period
        reference_node_list[4] = reference_node_list[3] + stop_time_period
        reference_node_list[5] = reference_node_list[4] + vibrate_time_period
        index_list = [0, 0, 0, 0, 0, 0]
        i = 0
        for index in range(len(timestamp)):
            if timestamp[index] >= reference_node_list[i]:
                index_list[i] = index
                i += 1
            if i > 5:
                adjust_index_list = DataPreMath.adjust_slices(index_list)
                if adjust_index_list[len(index_list) - 1] > len(timestamp):
                    print('the end item of index_list overflows because of the adjustment')
                    return
                return adjust_index_list

    @staticmethod
    def adjust_slices(index_list):
        # 计算差值列表
        differences = [index_list[i + 1] - index_list[i] for i in range(0, len(index_list), 2)]

        # 找到最大差值和最小差值
        max_diff = max(differences)
        min_diff = min(differences)

        # 如果所有差值都相同，则不需要调整
        if max_diff == min_diff:
            return index_list

        # 如果差值都不同，则选择差值位于中间的那个
        if len(set(differences)) == 3:
            middle_diff_index = differences.index(sorted(differences)[1])
            target_diff = differences[middle_diff_index]
        else:
            # 计算每个差值的数量
            diff_counts = {diff: differences.count(diff) for diff in set(differences)}

            # 找到数量最多的差值
            target_diff = max(diff_counts, key=diff_counts.get)

        # 调整索引列表
        adjusted_index_list = index_list.copy()

        # 调整差值对应的结束索引，使其与目标差值相等
        for i in range(0, len(index_list), 2):
            if differences[i // 2] != target_diff:
                diff_adjustment = target_diff - differences[i // 2]
                adjusted_index_list[i + 1] += diff_adjustment

        return adjusted_index_list
