#该程序为手动调整alpha值来获取最优结果，如果不知道调整逻辑请使用Auto E-VMD，若想缩短调整时间可使用此程序
#This program manually adjusts the alpha value to obtain the optimal result. If you do not know the adjustment logic,
# please use auto E-VMD. If you want to shorten the adjustment time, you can use this program
# 导入必要的库
import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QPushButton, QLabel, QLineEdit, \
    QFileDialog, QHBoxLayout, QSplitter, QComboBox, QAction, QMessageBox
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from scipy.integrate import simpson
from vmdpy import VMD

# 设置字体为 Times New Roman
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 12  # 设置字体大小
plt.rcParams['text.antialiased'] = True
from PyQt5.QtCore import QThread, pyqtSignal, Qt


class ExcelLoaderThread(QThread):
    finished = pyqtSignal(pd.DataFrame, list, int)
    error = pyqtSignal(str)

    def __init__(self, file_path):
        super().__init__()
        self.file_path = file_path

    def run(self):
        try:
            df = pd.read_excel(self.file_path, header=0, engine='openpyxl')
            valid_columns = []
            wave_data_dict = {}

            for group_start in range(0, len(df.columns), 4):
                if group_start + 3 >= len(df.columns):
                    break

                time_col_idx = group_start
                acc_col_idx = group_start + 1
                vel_col_idx = group_start + 2
                disp_col_idx = group_start + 3

                time_col_name = str(df.columns[time_col_idx]).strip()

                if not time_col_name or 'Unnamed' in time_col_name or ' ' in time_col_name:
                    continue

                valid_columns.append(time_col_name)

                wave_data_dict[time_col_name] = {
                    'time': df.iloc[:, time_col_idx].values,
                    'acc': df.iloc[:, acc_col_idx].values,
                    'vel': df.iloc[:, vel_col_idx].values,
                    'disp': df.iloc[:, disp_col_idx].values
                }

            df.attrs['wave_data'] = wave_data_dict
            self.finished.emit(df, valid_columns, len(valid_columns))
        except Exception as e:
            self.error.emit(f"文件读取失败:\n{str(e)}")
# 创建绘图窗口类

class PlotWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.init_ui()
        self.setWindowTitle("E-VMD, author:Yongbo Xiang")
        self.setFixedSize(1500, 1000)
        splitter = QSplitter()   # 创建一个分割器，用于分割主窗口的绘图部分和控制部分
        # 设置样式表
        self.setStyleSheet("""
            QLabel { 
                font-size: 22px;
                font-weight: bold;
            }
            QLineEdit {
                padding: 6px;
                font-family: "Times New Roman";
                font-size: 22px;
            }
        """)
        self.mode_time = None
        self.filtered_signal = None
        self.filtered_result = None
        self.result_time = None
        self.final_mode = None
        self.t = None  # 原始时间序列
        self.vel = None  # 原始速度数据

        # 绘图部分
        self.plot_widget = QWidget()
        plot_layout = QVBoxLayout()  # 使用垂直布局

        self.figure = plt.figure()  # 创建figure对象
        self.canvas = FigureCanvas(plt.figure())  # 创建一个用于绘图的画布
        plot_layout.addWidget(self.canvas)  # 将画布添加到布局中

        self.plot_widget.setLayout(plot_layout)
        # 控制部分
        self.control_widget = QWidget()
        control_layout = QVBoxLayout()  # 使用垂直布局


        # 添加地震波下拉框的代码
        self.earthquake_wave_label = QLabel("Select earthquake wave:")
        self.earthquake_wave_combo = QComboBox()
        combo_style = """
                QComboBox::item {
                    color: #FF5733; /* 设置项目的字体颜色为深橙色 */
                }
                QComboBox::item:selected {
                    background-color: #FF8C33; /* 设置被选中的项目的背景颜色为浅橙色 */
                    color: white; /* 设置被选中的项目的字体颜色为白色 */
                }
                QComboBox::item {

                    font-size: 18px; /* 设置项目的字体大小为14像素 */
                }
                """
        self.earthquake_wave_combo.setStyleSheet(combo_style)
        self.earthquake_wave_combo.addItems(['无'])
        control_layout.addWidget(self.earthquake_wave_label)
        control_layout.addWidget(self.earthquake_wave_combo)

        # 创建文本标签和文本框，并设置默认值
        self.a1_label = QLabel("alpha1:")  # 创建一个文本标签，显示为 "alpha1:"
        self.a1_line_edit = QLineEdit()  # 创建一个文本框
        self.a1_line_edit.setText('300')  # 设置文本框的默认值为 '300'
        control_layout.addWidget(self.a1_label)  # 将文本标签添加到控制部分的布局中
        control_layout.addWidget(self.a1_line_edit)  # 将文本框添加到控制部分的布局中

        self.a2_label = QLabel("alpha2:")  # 创建一个文本标签，显示为 "alpha2:"
        self.a2_line_edit = QLineEdit()  # 创建一个文本框
        self.a2_line_edit.setText('300')  # 设置文本框的默认值为 '300'
        control_layout.addWidget(self.a2_label)  # 将文本标签添加到控制部分的布局中
        control_layout.addWidget(self.a2_line_edit)  # 将文本框添加到控制部分的布局中

        self.b_label = QLabel("tau:")  # 创建一个文本标签，显示为 "tau:"
        self.b_line_edit = QLineEdit()  # 创建一个文本框
        self.b_line_edit.setText('0')  # 设置文本框的默认值为 '0'
        control_layout.addWidget(self.b_label)  # 将文本标签添加到控制部分的布局中
        control_layout.addWidget(self.b_line_edit)  # 将文本框添加到控制部分的布局中

        self.c_label = QLabel("K:")  # 创建一个文本标签，显示为 "K:"
        self.c_line_edit = QLineEdit()  # 创建一个文本框
        self.c_line_edit.setText('9')  # 设置文本框的默认值为 '9'
        control_layout.addWidget(self.c_label)  # 将文本标签添加到控制部分的布局中
        control_layout.addWidget(self.c_line_edit)

        self.d_label = QLabel("DC:")  # 创建一个文本标签，显示为 "DC:"
        self.d_line_edit = QLineEdit()  # 创建一个文本框
        self.d_line_edit.setText('0')  # 设置文本框的默认值为 '0'
        control_layout.addWidget(self.d_label)  # 将文本标签添加到控制部分的布局中
        control_layout.addWidget(self.d_line_edit)  # 将文本框添加到控制部分的布局中

        self.e_label = QLabel("init:")  # 创建一个文本标签，显示为 "init:"
        self.e_line_edit = QLineEdit()  # 创建一个文本框
        self.e_line_edit.setText('1')  # 设置文本框的默认值为 '2'
        control_layout.addWidget(self.e_label)  # 将文本标签添加到控制部分的布局中
        control_layout.addWidget(self.e_line_edit)  # 将文本框添加到控制部分的布局中

        self.start_label = QLabel("tol:")  # 创建一个文本标签，显示为 "tol:"
        self.start_line_edit = QLineEdit()  # 创建一个文本框
        self.start_line_edit.setText('1e-8')  # 设置文本框的默认值为 '0'
        control_layout.addWidget(self.start_label)  # 将文本标签添加到控制部分的布局中
        control_layout.addWidget(self.start_line_edit)  # 将文本框添加到控制部分的布局中


        # 添加按钮
        button_style = """
                QPushButton {
                    background-color: #4CAF50; /* 设置按钮背景色 */
                    border-style: outset;
                    border-width: 2px;
                    border-radius: 10px; /* 设置按钮边框圆角 */
                    border-color: beige;
                    font: bold 22px; /* 设置按钮字体加粗和大小 */
                    color: white; /* 设置按钮文字颜色 */
                    min-width: 10em;
                    padding: 6px;
                }

                QPushButton:hover {
                    background-color: #45a049; /* 鼠标悬停时按钮背景色 */
                }

                QPushButton:pressed {
                    background-color:  #C933FF; /* 按下按钮时的背景色 */
                }
                """

        # 创建并设置"Load the TXT"按钮
        self.load_txt_button = QPushButton("Load the Excel")
        self.load_txt_button.clicked.connect(self.load_excel_file)  # 点击按钮时连接load_txt_file函数
        self.load_txt_button.setStyleSheet(button_style)  # 设置按钮样式
        control_layout.addWidget(self.load_txt_button)  # 将按钮添加到控件布局中

        # 创建并设置"Drawing"按钮
        self.plot_button = QPushButton("Drawing")
        self.plot_button.clicked.connect(self.load_data)  # 点击按钮时连接load_data函数
        self.plot_button.setStyleSheet(button_style)  # 设置按钮样式
        control_layout.addWidget(self.plot_button)  # 将按钮添加到控件布局中

        # 创建并设置"Export data"按钮
        self.export_button = QPushButton("Export data")
        self.export_button.clicked.connect(self.export_data)  # 点击按钮时连接export_data函数
        self.export_button.setStyleSheet(button_style)  # 设置按钮样式
        control_layout.addWidget(self.export_button)  # 将按钮添加到控件布局中

        self.control_widget.setLayout(control_layout)  # 将控件布局应用到控件窗口

        # 将绘图部分和控制部分添加到分割器中
        splitter.addWidget(self.plot_widget)
        splitter.addWidget(self.control_widget)
        splitter.setSizes([1500, 1])
        self.setCentralWidget(splitter)
        self.data = None
        self.txt_data = None

    # 初始化界面
    def init_ui(self):
        menubar = self.menuBar()
        file_menu = menubar.addMenu('Options')
        show_text_action = QAction('Parameter description', self)
        show_text_action.triggered.connect(self.show_text_dialog)
        file_menu.addAction(show_text_action)
        self.loading_dialog = None  # 加载提示对话框


    # 显示参数描述对话框
    def show_text_dialog(self):
        msg_box = QMessageBox()
        msg_box.setWindowTitle("Parameter description")
        msg_box.setText("在这里写您想要显示的文本。")
        text = "<h3 style='color:blue; font-weight:bold;'>参数取值说明</h3>" \
               "<p style='font-size:18px; color:red;'>alpha1：惩罚系数，控制脉冲周期长短(Punishment coefficient, controlling the length of pulse period)，</p>" \
               "<p style='font-size:18px; color:red;'>alpha2：惩罚系数，控制脉冲信号与原始信号的相似度(Punishment coefficient, controlling the similarity between the pulse signal and the original signal)，</p>" \
               "<p style='font-size:18px; color:red;'>tau：噪声容忍度，0表示无噪声(Noise tolerance, 0 indicates no noise)，</p>" \
               "<p style='font-size:18px; color:red;'>K：最大分解模态数(Maximum number of decomposition modes)，</p>" \
               "<p style='font-size:18px; color:red;'>DC：是否分解直流成分(Does it decompose the DC component)，</p>" \
               "<p style='font-size:18px; color:red;'>init：初始化模式 1：随机，0：全零(Initialization mode 1: Random, 0: All zeros)，</p>" \
               "<p style='font-size:18px; color:red;'>tol：收敛容忍度(Convergence tolerance)，</p>"\
               "<p style='font-size:18px; color:red;'>版权声明：该程序由向永博制作，仅供学习交流使用，引用请标明出处，未经授权，禁止用于商业用途。未经许可的商业使用将追究法律责任。(Copyright Notice: This program is produced by Yongbo Xiang and is only for learning and communication purposes. Please indicate the source when quoting. Unauthorized use for commercial purposes is prohibited. Unauthorized commercial use will be held legally responsible.)，</p>"
        msg_box.setTextFormat(1)
        msg_box.setText(text)
        msg_box.addButton(QMessageBox.Ok)
        msg_box.addButton(QMessageBox.Cancel)
        msg_box.exec_()

    def load_excel_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择Excel文件", "", "Excel文件 (*.xlsx)"
        )
        if file_path:
            self.show_loading_dialog("正在加载Excel文件...")
            self.loader_thread = ExcelLoaderThread(file_path)
            self.loader_thread.finished.connect(self.handle_excel_loaded)
            self.loader_thread.error.connect(self.handle_excel_error)
            self.loader_thread.finished.connect(self.close_loading_dialog)
            self.loader_thread.error.connect(self.close_loading_dialog)
            self.loader_thread.start()
            self.load_txt_button.setEnabled(False)

    def show_loading_dialog(self, message):
        """显示加载提示对话框"""
        if self.loading_dialog is None:  # 确保只有一个对话框实例
            self.loading_dialog = QMessageBox(self)
            self.loading_dialog.setWindowTitle("请稍候")
            self.loading_dialog.setText(message)
            self.loading_dialog.setStandardButtons(QMessageBox.NoButton)
            self.loading_dialog.show()

    def close_loading_dialog(self):
        """关闭加载提示"""
        if self.loading_dialog is not None:
            self.loading_dialog.accept()  # 关闭对话框
            self.loading_dialog = None  # 清空对话框实例

    def handle_excel_loaded(self, df, valid_columns, count):
        self.excel_data = df
        self.load_txt_button.setEnabled(True)
        self.earthquake_wave_combo.clear()
        self.earthquake_wave_combo.addItem("无")
        self.earthquake_wave_combo.addItems(valid_columns)

        QMessageBox.information(
            self,
            "加载结果",
            f"有效地震波：{count} 个",
            QMessageBox.Ok
        )

    def handle_excel_error(self, error_msg):
        self.load_txt_button.setEnabled(True)
        QMessageBox.critical(
            self,
            "错误",
            error_msg,
            QMessageBox.Ok
        )
        self.earthquake_wave_combo.addItem("无")

    def get_wave_data(self, wave_name):
        if not hasattr(self.excel_data, 'attrs'):
            return None

        wave_data = self.excel_data.attrs.get('wave_data', {}).get(wave_name)
        if wave_data:
            # 使用时处理NaN
            return {
                'time': pd.Series(wave_data['time']).dropna().values,
                'acc': pd.Series(wave_data['acc']).dropna().values,
                'vel': pd.Series(wave_data['vel']).dropna().values,
                'disp': pd.Series(wave_data['disp']).dropna().values
            }
        return None

    def export_data(self):
        """导出图片和分解结果数据"""
        # 获取当前地震波名称
        wave_name = self.earthquake_wave_combo.currentText()
        if not wave_name or wave_name == "无":
            QMessageBox.warning(self, "警告", "请先选择地震波")
            return

        # 手动选择保存目录
        save_dir = QFileDialog.getExistingDirectory(self, "选择保存文件夹", "")
        if not save_dir:  # 用户取消选择
            return

        try:
            # ================== 保存图片 ==================
            # 生成安全文件名（替换特殊字符）
            safe_name = wave_name.replace(" ", "_").replace("/", "-").replace("\\", "-")

            # 保存ax1对应的图片
            fig1 = plt.figure(figsize=(8, 4))
            ax1_new = fig1.add_subplot(111)

            # 重新绘制ax1内容
            pulse_pairs = self.filtered_result['pulse_pairs']
            energy_ratios = self.filtered_result['energy_ratios']
            for i, (start, end) in enumerate(pulse_pairs):
                seg_time = self.mode_time[start:end + 1]
                seg_signal = self.filtered_signal[start:end + 1]
                color = 'red' if energy_ratios[i] >= 0.1 else 'black'
                ax1_new.plot(seg_time, seg_signal,
                             color=color,
                             linewidth=1,
                             alpha=0.8)

            # 设置ax1样式
            ax1_new.set_xlim(self.t.min(), self.t.max())
            ax1_new.set_title(f"{safe_name}",
                              fontsize=10, y=-0.25,  # 控制垂直位置 (负值表示下方)
                              pad=20,  # 标题与坐标轴的间距
                              loc='center')
            ax1_new.grid(False)

            # 保存图片
            img1_path = os.path.join(save_dir, f"{safe_name}_Characteristic curve of seismic wave acceleration extraction.png")
            fig1.savefig(img1_path, dpi=300, bbox_inches='tight')
            plt.close(fig1)

            # 保存ax2对应的图片
            has_final_mode = self.final_mode is not None and len(self.final_mode) > 0
            # 保存ax2对应的图片
            if has_final_mode:
                fig2 = plt.figure(figsize=(8, 4))
                ax2_new = fig2.add_subplot(111)

                # 重新绘制ax2内容
                min_len = min(len(self.result_time), len(self.final_mode))
                ax2_new.plot(self.t, self.vel,
                             color='gray',
                             linewidth=1,
                             alpha=0.5,
                             label='Original')
                ax2_new.plot(self.result_time[:min_len],
                             self.final_mode[:min_len],
                             color='#FF5733',
                             linewidth=1,
                             label=f'Stable mode (iterations: {self.vmd_iterations})')

                # 设置ax2样式
                ax2_new.set_xlim(self.t.min(), self.t.max())
                # 原代码
                ax2_new.set_title("E-VMD Decomposition results", fontsize=10)
                # 修改后代码 (添加位置参数)
                ax2_new.set_title(f"{safe_name}",
                                  fontsize=10,
                                  y=-0.25,  # 控制垂直位置 (负值表示下方)
                                  pad=20,  # 标题与坐标轴的间距
                                  loc='center')  # 水平居中
                ax2_new.legend()
                ax2_new.grid(False)

                # 保存图片
                img2_path = os.path.join(save_dir, f"{safe_name}_Decomposition_result.png")
                fig2.savefig(img2_path, dpi=300, bbox_inches='tight')
                plt.close(fig2)

            # ================== 保存Excel数据 ==================
            if has_final_mode:
                # 添加数组有效性检查
                if len(self.result_time) == 0 or len(self.final_mode) == 0:
                    QMessageBox.warning(self, "警告", "稳定模态数据为空")
                    return

                # 创建数据表格
                min_len = min(len(self.result_time), len(self.final_mode))
                df = pd.DataFrame({
                    f'{safe_name}_Time(s)': self.result_time[:min_len],
                    f'{safe_name}_StableMode': self.final_mode[:min_len]
                })

                # 保存Excel文件
                excel_path = os.path.join(save_dir, f"{safe_name}_StableMode.xlsx")
                df.to_excel(excel_path, index=False)

            success_msg = f"数据已保存至:\n{img1_path}"
            if has_final_mode:
                success_msg += f"\n{img2_path}\n{excel_path}"
            QMessageBox.information(self, "导出成功", success_msg)
        except Exception as e:
            QMessageBox.critical(self, "导出错误", f"导出失败: {str(e)}")

    def load_data(self):
        """处理数据加载与最短脉冲保存"""
        wave_name = self.earthquake_wave_combo.currentText()
        if not wave_name or wave_name == "无":
            QMessageBox.warning(self, "警告", "请先选择地震波")
            return
        # 改为通过get_wave_data获取数据（关键修改点）
        wave_data = self.get_wave_data(wave_name)
        if not wave_data:
            QMessageBox.critical(self, "错误", f"无法加载地震波数据: {wave_name}")
            return

        # 添加数据校验
        if 'vel' not in wave_data or len(wave_data['vel']) == 0:
            QMessageBox.critical(self, "错误", f"速度数据为空或无效: {wave_name}")
            return
        try:
            self.t = wave_data['time']  # <--- 新增保存原始时间
            self.vel = wave_data['vel']  # <--- 新增保存原始速度

            # 添加数据长度校验
            if len(self.t) != len(self.vel):
                QMessageBox.critical(self, "错误", "时间序列与速度数据长度不匹配")
                return

        except KeyError as e:
            QMessageBox.critical(self, "错误", f"数据字段缺失: {str(e)}")
            return

        # 执行核心计算
        result = self._calculate_half_pulse(self.t, self.vel)

        print("显著脉冲数:", len(result['significant_pulses']))

        # 处理显著脉冲
        sig_pulses = result['significant_pulses']
        if not sig_pulses:
            QMessageBox.information(self, "提示", "未检测到显著半脉冲")
            return

        total_energy_ratio = sum(p['energy_ratio'] for p in sig_pulses)  # 所有显著脉冲总能量占比
        pulse_count = len(sig_pulses)
        # 找到最短持续时间的脉冲
        min_pulse = min(sig_pulses, key=lambda x: x['duration'])

        # 保存结果到实例变量
        self.min_half_pulse = {
            'duration': min_pulse['duration'],
            'time_range': (
                result['time'][min_pulse['start_idx']],
                result['time'][min_pulse['end_idx']]
            ),
            'velocity': result['velocity'][min_pulse['start_idx']:min_pulse['end_idx'] + 1],
            'energy_ratio': min_pulse['energy_ratio'],
            'pulse_type': self._judge_pulse_type(pulse_count, total_energy_ratio)
        }

        print(f"已保存最短脉冲: {min_pulse['duration']:.4f}s")
        print(f"脉冲类型: {self.min_half_pulse['pulse_type']}")

        # --- 执行VMD分析 ---
        try:
            # 获取界面参数
            params = (
                self.a1_line_edit.text(),  # alpha
                self.b_line_edit.text(),  # tau
                self.c_line_edit.text(),  # K
                self.d_line_edit.text(),  # DC
                self.e_line_edit.text(),  # init
                self.start_line_edit.text()  # tol
            )
            t = wave_data['time']  # 原始时间序列
            acc = wave_data['acc']
            # 执行分解
            u, omega = self.perform_vmd(
                acc,  # 将信号作为位置参数传递
                *params
            )
            min_length = min(len(t), len(u[0]))
            # 计算能量比例
            total_energy = self.calculate_energy(acc)
            first_mode_energy = self.calculate_energy(u[0])
            energy_ratio = first_mode_energy / total_energy

            self.first_mode = {
                'time': t[:min_length],  # 使用经过清理和长度修剪后的时间序列
                'signal': u[0][:min_length],
                'frequency': omega[0][0],
                'energy_ratio': energy_ratio,
                'valid': energy_ratio >= 0.3
            }

        except ValueError as e:
            QMessageBox.critical(self, "参数错误", str(e))
            return
        except Exception as e:
            QMessageBox.critical(self, "分解错误", f"VMD执行失败: {str(e)}")
            return
        # 输出能量信息
        print(f"加速度第一模态能量占比: {energy_ratio:.2%}")

        # --- 新增滤波处理模块 ---
        try:
            # 获取最短持续时间
            min_duration = self.min_half_pulse['duration']
            t = wave_data['time']
            vel = wave_data['vel']
            print(f"开始滤波处理，最小持续时间阈值: {min_duration:.2f}s")

            # 执行频域滤波
            def filter_signal_by_period(time, signal, min_period):
                """傅里叶域滤波实现"""
                dt = np.mean(np.diff(time))
                if dt <= 0:
                    raise ValueError("时间序列步长异常，可能包含重复时间点")
                n = len(signal)
                if n < 4:
                    raise ValueError("信号长度过短，无法进行FFT")
                # 计算傅里叶变换
                freqs = np.fft.fftfreq(n, d=dt)
                fft_values = np.fft.fft(signal)
                # 设置截止频率（保留周期>=min_period的分量）
                cutoff_frequency = 1 / min_period
                fft_values[np.abs(freqs) > cutoff_frequency] = 0
                # 逆变换重构信号
                filtered = np.real(np.fft.ifft(fft_values))
                return filtered

            # 获取模态信号
            mode_time = self.first_mode['time']
            mode_signal = self.first_mode['signal']
            # 执行滤波（注意：这里使用持续时间作为最小周期阈值）
            filtered_signal = filter_signal_by_period(
                mode_time,
                mode_signal,
                min_period=min_duration
            )
            # 新增保存：滤波后的核心数据
            self.mode_time = mode_time  # <--- 新增保存时间序列
            self.filtered_signal = filtered_signal  # <--- 新增保存滤波信号
            # 二次脉冲分析
            filtered_result = self._calculate_half_pulse(mode_time, filtered_signal)
            duration = sum(p['duration'] for p in result['significant_pulses'])
            print(f"总的脉冲周期（Total pulse period）:{duration}")
            self.filtered_result = filtered_result  # <--- 关键保存点
            print(f"滤波后显著脉冲数: {len(filtered_result['significant_pulses'])}")

            # 初始化模态相关变量
            self.final_mode = None
            self.vmd_iterations = 0

            # ===================================================================
            # 修改后的功能4：分组处理连续脉冲并叠加结果
            # ===================================================================
            try:
                # 使用滤波后的分析结果
                sig_pulses = filtered_result['significant_pulses']
                if not sig_pulses:
                    QMessageBox.information(self, "提示", "无显著脉冲可处理")
                    return
                # ===================================================================
                # 第一步：将显著脉冲分组（连续脉冲为一组）
                # ===================================================================
                # 按起始时间排序脉冲
                sorted_pulses = sorted(sig_pulses, key=lambda x: x['start_idx'])
                time_array = mode_time  # 使用滤波后的时间序列
                # 动态计算采样间隔
                time_step = np.mean(np.diff(time_array)) if len(time_array) > 1 else 0
                tolerance = time_step * 1.1 + 1e-6  # 自适应容差
                # 分组连续脉冲
                pulse_groups = []
                current_group = [sorted_pulses[0]] if sorted_pulses else []
                for pulse in sorted_pulses[1:]:
                    # 获取前一个脉冲的结束时间和当前脉冲的开始时间
                    prev_end = time_array[current_group[-1]['end_idx']]
                    curr_start = time_array[pulse['start_idx']]

                    # 判断连续性
                    if np.isclose(curr_start, prev_end, atol=tolerance):
                        current_group.append(pulse)
                    else:
                        pulse_groups.append(current_group)
                        current_group = [pulse]

                if current_group:
                    pulse_groups.append(current_group)
                # ===================================================================
                # 第二步：处理每个脉冲组并收集结果
                # ===================================================================
                final_modes = []
                total_iterations = 0
                for group_idx, pulse_group in enumerate(pulse_groups):
                    try:
                        # 获取当前组的起始和结束索引
                        group_start = min(p['start_idx'] for p in pulse_group)
                        group_end = max(p['end_idx'] for p in pulse_group)
                        # 创建掩码
                        mask = np.zeros(len(vel), dtype=bool)
                        valid_start = max(group_start, 0)
                        valid_end = min(group_end + 1, len(mask))
                        if valid_start < len(mask) and valid_end <= len(mask):
                            mask[valid_start:valid_end] = True
                        # 生成当前组的补零信号
                        filtered_velocity = np.zeros_like(vel)
                        filtered_velocity[mask] = vel[mask]
                        # ===================================================================
                        # 第三步：连续VMD分解当前组
                        # ===================================================================
                        MAX_ITERATIONS = 15
                        current_signal = filtered_velocity.copy()
                        group_final_mode = None

                        for iteration in range(MAX_ITERATIONS):
                            try:
                                u, omega = self.perform_vmd(
                                    current_signal,
                                    self.a2_line_edit.text(),
                                    self.b_line_edit.text(),
                                    self.c_line_edit.text(),
                                    self.d_line_edit.text(),
                                    self.e_line_edit.text(),
                                    self.start_line_edit.text()
                                )
                            except Exception as e:
                                QMessageBox.warning(self, "分解错误",
                                                    f"第{group_idx + 1}组第{iteration + 1}次VMD失败: {str(e)}")
                                break

                            if len(u) == 0:
                                QMessageBox.warning(self, "错误",
                                                    f"第{group_idx + 1}组未获得有效模态分量")
                                break

                            first_mode = u[0]
                            if not self.has_multiple_extrema_between_zeros(first_mode):
                                group_final_mode = first_mode
                                total_iterations += iteration + 1
                                break
                            else:
                                current_signal = first_mode
                        if group_final_mode is not None:
                            final_modes.append(group_final_mode)

                    except Exception as e:
                        QMessageBox.warning(self, "组处理错误",
                                            f"第{group_idx + 1}组处理失败: {str(e)}")
                # ===================================================================
                # 第四步：叠加所有组的最终模态
                # ===================================================================
                if final_modes:
                    # 确保所有模态长度一致
                    min_length = min(len(mode) for mode in final_modes)
                    trimmed_modes = [mode[:min_length] for mode in final_modes]
                    self.final_mode = np.sum(trimmed_modes, axis=0)
                    all_significant_indices = [
                        slice(g[0]['start_idx'], g[-1]['end_idx'])
                        for g in pulse_groups
                    ]

                    # 合成评分信号
                    try:
                        mode_segment = np.concatenate(
                            [self.final_mode[indices] for indices in all_significant_indices],
                            axis=0
                        )
                        vel_segment = np.concatenate(
                            [self.vel[indices] for indices in all_significant_indices],
                            axis=0
                        )
                        score = self.tanimoto_coeff(mode_segment, vel_segment)

                    except Exception as e:
                        print(f"评分计算失败: {str(e)}")
                        score = 0
                    print(f"相关系数（correlation coefficient）: {score}")
                    self.vmd_iterations = total_iterations
                else:
                    self.final_mode = None
                    self.vmd_iterations = 0
                self.result_time = time_array[:len(self.final_mode)] if self.final_mode is not None else None
            except Exception as e:
                QMessageBox.critical(self, "处理错误", f"功能4执行失败: {str(e)}")




            self.canvas.figure.clf()
            ax1 = self.canvas.figure.add_subplot(211)
            ax2 = self.canvas.figure.add_subplot(212)
            #print(f"[调试] 绘图数据长度验证:")
            #print(f"mode_time 长度: {len(mode_time)}")
            #print(f"filtered_signal 长度: {len(filtered_signal)}")
            if self.final_mode is not None:
                print(f"final_mode 长度: {len(self.final_mode)}")
            else:
                print("final_mode 未定义")

            try:
                # =============== 绘制滤波脉冲结果 ===============


                # 获取所有脉冲区间和能量信息
                pulse_pairs = filtered_result['pulse_pairs']
                energy_ratios = filtered_result['energy_ratios']

                # 循环绘制每个脉冲区间
                for i, (start, end) in enumerate(pulse_pairs):
                    seg_time = mode_time[start:end + 1]
                    seg_signal = filtered_signal[start:end + 1]
                    color = 'red' if energy_ratios[i] >= 0.1 else 'black'
                    ax1.plot(seg_time, seg_signal, color=color, linewidth=1, alpha=0.8)

                # 设置x轴的边界为数据的起止点
                ax1.set_xlim(t.min(), t.max())  # 设置x轴范围为t的最小值和最大值

                # 创建图例句柄
                red_patch = plt.Line2D([], [], color='red', label='Significant')
                black_patch = plt.Line2D([], [], color='black', label='Non-significant')


                # 添加图例，确保所有句柄都在图例中
                ax1.legend(handles=[red_patch, black_patch])
                ax1.set_title("Characteristic curve of seismic wave acceleration extraction", fontsize=12)
                ax1.grid(False)
                # =============== 绘制稳定模态结果 ===============
                if self.final_mode is not None:
                    # 添加长度校验
                    min_len = min(len(self.result_time), len(self.final_mode))
                    # 绘制原始信号
                    ax2.plot(t, vel, color='gray', linewidth=1, alpha=0.5, label='Original')
                    ax2.plot(self.result_time[:min_len],
                             self.final_mode[:min_len],
                             color='#FF5733',
                             linewidth=1,
                             label=f'Stable mode (iteration{self.vmd_iterations}time(s)')
                    # 设置x轴的边界为数据的起止点
                    ax2.set_xlim(t.min(), t.max())  # 设置x轴范围为t的最小值和最大值
                    # 创建图例句柄
                    red_patch = plt.Line2D([], [], color='red', label=f'Stable mode (iteration{self.vmd_iterations}time(s)')
                    gray_patch = plt.Line2D([], [], color='gray', label='Original')  # 添加Original的句柄
                    ax2.legend(handles=[gray_patch, red_patch])
                    ax2.set_title("E-VMD Decomposition results", fontsize=12)
                    ax2.legend()
                else:
                    ax2.text(0.5, 0.5, 'Stable mode not obtained',
                             ha='center', va='center',
                             transform=ax2.transAxes,
                             color='gray', alpha=0.8)
                # 设置公共属性
                ax2.set_xlabel("Time (s)", fontsize=10)
                ax2.grid(False)
                plt.tight_layout()
                self.canvas.draw()
            except Exception as e:
                QMessageBox.critical(self, "绘图错误",
                                     f"图形渲染失败: {str(e)}\n"
                                     f"数据长度详情:\n"
                                     f"- 时间序列长度: {len(mode_time)}\n"
                                     f"- 滤波信号长度: {len(filtered_signal)}\n"
                                     f"- 稳定模态长度: {len(self.final_mode) if self.final_mode else 0}")
        except ValueError as e:
            QMessageBox.critical(self, "滤波参数错误", f"滤波失败: {str(e)}")
        except Exception as e:
            QMessageBox.critical(self, "滤波错误", f"滤波处理异常: {str(e)}\n{type(e).__name__}")
            raise

    def tanimoto_coeff(self, x, y):
        """Tanimoto相似度计算"""
        x_centered = x - np.mean(x)
        y_centered = y - np.mean(y)
        cov = np.dot(x_centered, y_centered)
        norm_x = np.linalg.norm(x_centered)
        norm_y = np.linalg.norm(y_centered)
        denominator = norm_x ** 2 + norm_y ** 2 - cov
        return cov / denominator if denominator != 0 else 0.0

    def calculate_energy(self, signal, t=None):
        """计算信号能量（支持非均匀采样）"""
        if t is not None and len(t) == len(signal):
            return simpson(y=signal ** 2, x=t)  # 精确积分（支持非均匀采样）
        else:
            return simpson(y=signal ** 2)  # 均匀采样默认积分

    def _calculate_half_pulse(self, t, vel, energy_threshold = 0.1):
        """（需确保与独立脚本相同的脉冲检测逻辑）"""
        # 添加与独立脚本相同的零交叉点检测逻辑
        zero_crossings = np.where(np.diff(np.sign(vel)))[0]

        # 生成脉冲对时增加容错
        pulse_pairs = []
        prev = 0
        for zc in zero_crossings:
            if zc > prev:
                # 确保包含完整的半周期
                pulse_pairs.append((prev, zc))
                prev = zc
        # 处理末尾未闭合的脉冲
        if prev < len(vel) - 1:
            pulse_pairs.append((prev, len(vel) - 1))
        # 能量计算
        energies = np.array([simpson(vel[s:e + 1] ** 2, t[s:e + 1]) for s, e in pulse_pairs])
        total_energy = energies.sum()

        # 处理零能量情况
        if total_energy == 0:
            return {'significant_pulses': []}

        # 显著脉冲筛选
        energy_ratios = energies / total_energy
        significant_mask = energy_ratios >= energy_threshold
        sig_indices = np.where(significant_mask)[0]

        # 构造完整的significant_pulses结构（核心修复点）
        significant_pulses = [
            {
                'start_idx': pulse_pairs[i][0],
                'end_idx': pulse_pairs[i][1],
                'duration': t[pulse_pairs[i][1]] - t[pulse_pairs[i][0]],
                'energy_ratio': energy_ratios[i]
            }
            for i in sig_indices
        ]

        return {  # 增加返回pulse_pairs
            'time': t,
            'velocity': vel,
            'significant_pulses': significant_pulses,
            'pulse_pairs': pulse_pairs,  # 新增返回所有脉冲区间
            'total_energy': total_energy,
            'energy_ratios': energy_ratios
        }


    def _judge_pulse_type(self, pulse_count, total_energy_ratio):
        """同步原始程序的复合判断逻辑"""
        type_thresholds = {
            1: 0.30,
            2: 0.42,
            3: 0.50,
            4: 0.57,
            5: 0.73
        }
        # 反向遍历确保优先匹配高类型
        for typ, th in reversed(type_thresholds.items()):
            if pulse_count == typ and total_energy_ratio >= th:
                return typ
        return None

    def perform_vmd(self, signal, alpha, tau, K, DC, init, tol):
        """执行VMD分解并返回模态分量"""
        # 参数有效性校验
        try:
            alpha = float(alpha)
            tau = float(tau)
            K = int(K)
            DC = int(DC)
            init = int(init)
            tol = float(tol)
        except ValueError as e:
            raise ValueError(f"无效参数格式: {str(e)}") from e

        # 信号校验
        if len(signal) < 10:
            raise ValueError("输入信号过短（至少需要10个数据点）")
        if np.all(signal == 0):
            raise ValueError("输入信号全为零值")

        try:
            u, u_hat, omega = VMD(
                signal,
                alpha=alpha,
                tau=tau,
                K=K,
                DC=DC,
                init=init,
                tol=tol
            )
            return u, omega
        except Exception as e:
            raise RuntimeError(f"VMD分解失败: {str(e)}") from e

    def has_multiple_extrema_between_zeros(self, signal):
        """检查相邻零点间是否存在多个极值"""
        # 找到零点交叉索引
        zero_crossings = np.where(np.diff(np.sign(signal)))[0]
        # 遍历所有相邻零点区间
        for i in range(len(zero_crossings) - 1):
            start = zero_crossings[i]
            end = zero_crossings[i + 1]
            segment = signal[start:end]
            # 找到极值点
            extrema = np.where((np.diff(np.sign(np.diff(segment))) != 0))[0]
            if len(extrema) > 1:
                return True
        return False


def main():
    app = QApplication(sys.argv)
    main_window = PlotWindow()
    main_window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()