#该程序为自动调整alpha值来获取最优结果，若缩短运行速度请使用Manual E-VMD2程序，不知道alpha的调整逻辑可使用该程序
#This program automatically adjusts the alpha value to obtain the optimal result. If you want to shorten the running speed,
# please use the Manual E-VMD2 program. If you do not know the adjustment logic of alpha, you can use this program
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
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pandas import DataFrame
from PyQt5.QtCore import QThread, pyqtSignal, Qt

# 设置字体为 Times New Roman
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 12  # 设置字体大小
plt.rcParams['text.antialiased'] = True

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
        self.a_label = QLabel("alpha:")  # 创建一个文本标签，显示为 "Vp:"
        self.a_line_edit = QLineEdit()  # 创建一个文本框
        self.a_line_edit.setText('300')  # 设置文本框的默认值为 '300'
        control_layout.addWidget(self.a_label)  # 将文本标签添加到控制部分的布局中
        control_layout.addWidget(self.a_line_edit)  # 将文本框添加到控制部分的布局中

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
               "<p style='font-size:18px; color:red;'>alpha：惩罚系数，控制模态带宽(Punishment coefficient, controlling modal bandwidth)，</p>" \
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

            # 保存ax2对应的图片
            has_final_mode = self.final_mode is not None and len(self.final_mode) > 0
            # 保存ax2对应的图片
            if has_final_mode:
                fig2 = plt.figure(figsize=(8, 4))
                ax2_new = fig2.add_subplot(111)

                # 重新绘制ax2内容
                min_len = min(len(self.t), len(self.final_mode))
                ax2_new.plot(self.t, self.vel,
                             color='gray',
                             linewidth=1,
                             alpha=0.5,
                             label='Original Velocity')
                ax2_new.plot(self.t[:min_len],
                             self.final_mode[:min_len],
                             color='#FF5733',
                             linewidth=1,
                             label=f'E-VMD')

                # 设置ax2样式
                ax2_new.set_xlim(self.t.min(), self.t.max())
                # 原代码

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
                if len(self.t) == 0 or len(self.final_mode) == 0:
                    QMessageBox.warning(self, "警告", "稳定模态数据为空")
                    return

                # 创建数据表格
                min_len = min(len(self.t), len(self.final_mode))
                df = pd.DataFrame({
                    f'{safe_name}_Time(s)': self.t[:min_len],
                    f'{safe_name}_StableMode': self.final_mode[:min_len]
                })

                # 保存Excel文件
                excel_path = os.path.join(save_dir, f"{safe_name}_StableMode.xlsx")
                df.to_excel(excel_path, index=False)

            success_msg = f"数据已保存至:\n{img2_path}"
            if has_final_mode:
                success_msg += f"\n{img2_path}\n{excel_path}"
            QMessageBox.information(self, "导出成功", success_msg)
        except Exception as e:
            QMessageBox.critical(self, "导出错误", f"导出失败: {str(e)}")

    def load_data(self):
        """主数据处理流程"""
        try:
            # 数据加载与校验
            wave_name = self.earthquake_wave_combo.currentText()
            if not wave_name or wave_name == "无":
                QMessageBox.warning(self, "警告", "请先选择地震波")
                return

            wave_data = self.get_wave_data(wave_name)
            if not wave_data:
                QMessageBox.critical(self, "错误", f"无法加载地震波数据: {wave_name}")
                return

            # 初始化数据
            self.t = wave_data['time']
            self.vel = wave_data['vel']
            if len(self.t) != len(self.vel):
                QMessageBox.critical(self, "错误", "时间序列与速度数据长度不匹配")
                return

            # 初始脉冲分析
            initial_result = self._calculate_half_pulse(self.t, self.vel)
            if not initial_result['significant_pulses']:
                QMessageBox.information(self, "提示", "未检测到显著半脉冲")
                return

            # 获取关键参数
            min_duration = min(p['duration'] for p in initial_result['significant_pulses'])
            self.min_half_pulse = {
                'duration': min_duration,
                'time_range': (
                    self.t[initial_result['significant_pulses'][0]['start_idx']],
                    self.t[initial_result['significant_pulses'][-1]['end_idx']]
                ),
                'velocity': initial_result['velocity'],
                'energy_ratio': sum(p['energy_ratio'] for p in initial_result['significant_pulses']),
                'pulse_type': self._judge_pulse_type(
                    len(initial_result['significant_pulses']),
                    sum(p['energy_ratio'] for p in initial_result['significant_pulses'])
                )
            }

            # 参数优化核心
            best_alpha, best_result = self._optimize_alpha(
                acc_data=wave_data['acc'],
                time_data=self.t,
                min_duration_o=min_duration
            )

            if not best_result:
                QMessageBox.warning(self, "警告", "未找到有效参数组合")
                return

            # 更新系统状态
            self.a_line_edit.setText(str(round(best_alpha, 1)))

            self.filtered_result = best_result  # <--- 关键保存点
            self.total_duration = sum(p['duration'] for p in best_result['significant_pulses'])

            # 执行功能4处理流程
            final_mode = self._process_pulse_groups(
                vel_data=self.vel,
                time_data=self.t,
                filtered_result=best_result
            )

            self.final_mode = final_mode

            # 结果可视化
            self._plot_results(
                result=best_result,
                final_mode=final_mode,
                velocity_data=self.vel,
                time_data=self.t
            )

        except Exception as e:
            QMessageBox.critical(
                self, "系统错误",
                f"异常类型: {type(e).__name__}\n"
                f"错误详情: {str(e)}\n"
                f"追踪信息: {traceback.format_exc()}"
            )

    def _optimize_alpha(self, acc_data, time_data, min_duration_o):
        """多线程优化版alpha参数搜索"""
        base_params = {
            'tau': float(self.b_line_edit.text()),
            'K': int(self.c_line_edit.text()),
            'DC': int(self.d_line_edit.text()),
            'init': int(self.e_line_edit.text()),
            'tol': float(self.start_line_edit.text())
        }

        # 初始alpha值列表
        alphas = np.linspace(0.1, 500, 8).round(1)
        best_alpha = None
        best_result = None
        max_duration = 0
        no_valid_counter = 0
        last_boundaries = None
        adjust_counter = 0

        def _process_alpha(alpha):
            """单个alpha处理任务"""
            try:
                u, _ = self.perform_vmd(acc_data, alpha=alpha, **base_params)
                filtered = self._apply_filter(time_data, u[0], min_duration_o)
                result = self._calculate_half_pulse(time_data, filtered)
                duration = sum(p['duration'] for p in result['significant_pulses'])
                return alpha, duration, result
            except Exception as e:
                print(f"Alpha {alpha} 失败: {str(e)}")
                return alpha, 0, None

        while True:
            current_max = 0
            valid_scores = []
            old_alphas = alphas.copy()
            results = []

            # 使用线程池并行计算
            with ThreadPoolExecutor(max_workers=8) as executor:
                futures = [executor.submit(_process_alpha, alpha) for alpha in alphas]
                for future in as_completed(futures):
                    alpha, duration, result = future.result()
                    results.append((alpha, duration, result))
                    print(f"Alpha: {alpha}, Duration: {duration}")

            results.sort(key=lambda x: x[0])
            valid_scores = [d for _, d, _ in results]

            # 更新最佳结果
            for alpha, duration, result in results:
                if duration > max_duration:
                    max_duration = duration
                    best_alpha = alpha
                    best_result = result

            # 检查终止条件
            if len(valid_scores) == len(alphas) and all(d == 0 for d in valid_scores):
                no_valid_counter += 1
                if no_valid_counter >= 2:
                    self._show_warning("无法找到有效模态函数")
                    break
                # 扩展搜索范围
                alphas = np.array([0.1] + [a + 25 for a in alphas[1:-1]] + [500])
                continue
            else:
                no_valid_counter = 0

            # 提取最大值及其边界
            max_value = max(valid_scores)
            peak_indices = [i for i, d in enumerate(valid_scores) if d == max_value]

            if peak_indices:
                new_left = alphas[max(0, min(peak_indices) - 1)]
                new_right = alphas[min(len(alphas) - 1, max(peak_indices) + 1)]

                # 检查最大值两边的 alpha 差值
                if new_right - new_left <= 1:
                    print("最大显著脉冲时间两边的 alpha 差值不超过1，终止优化")
                    break

                # 处理无法缩小边界的情况
                if (new_left, new_right) == last_boundaries:
                    adjust_counter += 1
                    if adjust_counter >= 2:
                        print("边界连续2次未变化，终止优化")
                        break

                    boundary_diff = new_right - new_left
                    step = boundary_diff * 0.05
                    middle_points = [round(a + step, 1) for a in alphas[1:-1]]
                    middle_points = [min(max(p, new_left), new_right) for p in middle_points]
                    alphas = np.array([new_left] + middle_points + [new_right])
                    print(f"边界未变化，调整中间值 (+{step}): {alphas}")
                else:
                    adjust_counter = 0
                    last_boundaries = (new_left, new_right)

                    # 重新生成新的 alpha 列表
                    alphas = np.linspace(new_left, new_right, 8).round(1)
                    print(f"新边界范围: {new_left}-{new_right}")

            # 如果没有变化，终止
            if np.array_equal(old_alphas, alphas):
                print("无法进一步优化alpha值")
                break

        print(f"Best Alpha: {best_alpha}, Best Total Duration: {max_duration}")
        return best_alpha, best_result

    def _apply_filter(self, time, signal, min_period):
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

    def _process_pulse_groups(self, vel_data, time_data, filtered_result):
        """功能4核心处理"""
        # 脉冲分组
        pulse_groups = []
        current_group = []
        sorted_pulses = sorted(filtered_result['significant_pulses'], key=lambda x: x['start_idx'])
        self.final_mode = np.array([])
        # 分组逻辑添加容错处理
        if sorted_pulses:
            current_group.append(sorted_pulses[0])
            for pulse in sorted_pulses[1:]:
                last_end = time_data[current_group[-1]['end_idx']]
                curr_start = time_data[pulse['start_idx']]
                if np.isclose(curr_start, last_end, rtol=0.05):
                    current_group.append(pulse)
                else:
                    pulse_groups.append(current_group)
                    current_group = [pulse]
            if current_group:
                pulse_groups.append(current_group)

        # 打印各组起止时间（新增部分）
        print(f"\n发现 {len(pulse_groups)} 个脉冲组：")
        for i, group in enumerate(pulse_groups, 1):
            group_start = time_data[group[0]['start_idx']]
            group_end = time_data[group[-1]['end_idx']]
            print(
                f"第{i}组｜开始时间: {group_start:.2f}s ｜结束时间: {group_end:.2f}s ｜持续时间: {group_end - group_start:.2f}s")

        # 分组处理


        best_alpha, final_mode = self._optimize_alpha_4(
            pulse_groups,
            vel_data=self.vel,
            time_data=self.t
        )

        if final_mode is None or final_mode.size == 0:
            QMessageBox.warning(self, "警告", "未找到有效参数组合")
            return

        return  final_mode

    def _optimize_alpha_4(self, pulse_groups, vel_data, time_data):
        """智能alpha参数搜索"""
        base_params = {
            'tau': float(self.b_line_edit.text()),
            'K': int(self.c_line_edit.text()),
            'DC': int(self.d_line_edit.text()),
            'init': int(self.e_line_edit.text()),
            'tol': float(self.start_line_edit.text())
        }

        # 初始alpha值列表（基于用户指定的9个值）
        alphas = np.linspace(0.1, 500, 8).round(1)
        best_alpha = None
        best_final_mode = None
        max_score = 0
        no_valid_counter = 0
        last_boundaries = None
        adjust_counter = 0

        def _process_alpha(alpha):
            """单个alpha处理任务"""
            try:
                final_modes = []
                for group_idx, group in enumerate(pulse_groups, 1):
                    try:
                        group_signal = self._create_group_signal(vel_data, group, time_data)
                        mode, iterations = self._vmd_optimize(group_signal, alpha)
                        if mode is None:
                            print(f"第{group_idx}组未获得有效模态")
                            continue
                        significant_indices = slice(group[0]['start_idx'], group[-1]['end_idx'])
                        trimmed_mode = mode
                        final_modes.append(trimmed_mode)
                    except Exception as e:
                        print(f"第{group_idx}组处理失败: {str(e)}")
                        continue

                # 处理无效模态情况
                if not final_modes:
                    print(f"Alpha {alpha} 无有效模态")
                    return alpha, np.array([]), 0  # 返回空数组和0分

                # 有效模态处理流程
                min_len = min(len(m) for m in final_modes)
                local_final_mode = np.sum([m[:min_len] for m in final_modes], axis=0)

                # 截取显著脉冲区间
                all_significant_indices = [
                    slice(g[0]['start_idx'], g[-1]['end_idx'])
                    for g in pulse_groups
                ]

                # 合成评分信号
                try:
                    mode_segment = np.concatenate(
                        [local_final_mode[indices] for indices in all_significant_indices],
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

                return alpha, local_final_mode, score

            except Exception as e:
                print(f"Alpha {alpha} 处理发生全局异常: {str(e)}")
                return alpha, np.array([]), 0


        for group_idx, group in enumerate(pulse_groups, 1):
            print(f"处理第{group_idx}组脉冲：")
            for pulse in group:
                start_idx = pulse['start_idx']
                end_idx = pulse['end_idx']
                print(f"├ 脉冲开始索引: {start_idx}, 结束索引: {end_idx}")
            group_signal = self._create_group_signal(vel_data, group, time_data)
            print(f"└ 信号长度: {len(group_signal)}")

        while True:
            current_max = 0
            scores = []
            valid_scores = []
            old_alphas = alphas.copy()
            results = []

            # 并行评估当前alpha列表
            with ThreadPoolExecutor(max_workers=8) as executor:
                futures = [executor.submit(_process_alpha, alpha) for alpha in alphas]
                for future in as_completed(futures):
                    alpha, local_final_mode, score = future.result()
                    results.append((alpha, local_final_mode, score))
                    print(f"Alpha: {alpha}, Score: {score}")

            # 处理评估结果（保持原有排序和筛选逻辑）
            results.sort(key=lambda x: x[0])
            valid_scores = [score for _, _, score in results]

            # 更新最佳结果
            for alpha, local_final_mode, score in results:
                if score > max_score:
                    max_score = score
                    best_alpha = alpha
                    best_final_mode = local_final_mode

            if max_score >= 0.98:  # 阈值达到立即终止
                print(f"达到终止阈值0.98，提前退出优化")
                break
            # 终止条件1：相邻有效结果差异<=1
            if len(valid_scores) == len(alphas) and all(d == 0 for d in valid_scores):
                no_valid_counter += 1
                if no_valid_counter >= 2:
                    self._show_warning("无法找到有效模态函数")
                    break
                # 扩展搜索范围
                alphas = np.array([0.1] + [a + 25 for a in alphas[1:-1]] + [500])
                continue
            else:
                no_valid_counter = 0

            # 提取最大值及其边界
            max_value = max(valid_scores)
            peak_indices = [i for i, d in enumerate(valid_scores) if d == max_value]

            if peak_indices:
                new_left = alphas[max(0, min(peak_indices) - 1)]
                new_right = alphas[min(len(alphas) - 1, max(peak_indices) + 1)]

                # 检查最大值两边的 alpha 差值
                if new_right - new_left <= 1:
                    print("最大显著脉冲时间两边的 alpha 差值不超过1，终止优化")
                    break

                # 处理无法缩小边界的情况
                if (new_left, new_right) == last_boundaries:
                    adjust_counter += 1
                    if adjust_counter >= 2:
                        print("边界连续5次未变化，终止优化")
                        break

                    boundary_diff = new_right - new_left
                    step = boundary_diff * 0.05
                    middle_points = [round(a + step, 1) for a in alphas[1:-1]]
                    middle_points = [min(max(p, new_left), new_right) for p in middle_points]
                    alphas = np.array([new_left] + middle_points + [new_right])
                    print(f"边界未变化，调整中间值 (+{step}): {alphas}")
                else:
                    adjust_counter = 0
                    last_boundaries = (new_left, new_right)

                    # 重新生成新的 alpha 列表
                    alphas = np.linspace(new_left, new_right, 8).round(1)
                    print(f"新边界范围: {new_left}-{new_right}")

            # 如果没有变化，终止
            if np.array_equal(old_alphas, alphas):
                print("无法进一步优化alpha值")
                break

        print(f"Best Alpha: {best_alpha}, Max Tanimoto_Coeff: {max_score}")
        return best_alpha, best_final_mode

    def _trim_to_amplitude_threshold(self, mode, group, vel_data, time_data):
        """基于幅值阈值的精确截断处理（含调试打印）"""
        # 获取原始时间序列和脉冲边界索引
        t = self.t
        group_start_idx = min(p['start_idx'] for p in group)
        group_end_idx = max(p['end_idx'] for p in group)

        # 打印原始脉冲时间范围
        print(f"[调试] 原始脉冲组时间范围: {t[group_start_idx]:.2f}s (index={group_start_idx}) "
              f"-> {t[group_end_idx]:.2f}s (index={group_end_idx})")

        # 计算动态阈值
        # 正确获取全量程绝对极值（同时考虑正负峰值）
        abs_mode = np.abs(mode)
        peak_positive = np.max(mode)  # 最大正值
        peak_negative = np.min(mode)  # 最大负值
        peak_abs = max(abs(peak_positive), abs(peak_negative))  # 全量程最大绝对值

        # 验证计算逻辑正确性
        assert np.allclose(peak_abs, np.max(abs_mode)), "峰值计算不一致，请检查数据"

        threshold = 0.0003 * peak_abs
        print(f"[验证] 实际使用峰值: {peak_abs:.4f} (正峰值: {peak_positive:.4f}, 负峰值: {peak_negative:.4f})")

        # 向左扫描寻找截断点
        left_threshold_idx = group_start_idx
        for i in range(group_start_idx - 1, -1, -1):
            if np.abs(mode[i]) <= threshold:
                left_threshold_idx = i
                break

        # 向右扫描寻找截断点
        right_threshold_idx = group_end_idx
        for i in range(group_end_idx + 1, len(mode)):
            if np.abs(mode[i]) <= threshold:
                right_threshold_idx = i
                break

        # 获取边界时间并打印
        t1 = t[left_threshold_idx]
        t2 = t[right_threshold_idx]
        print(f"[调试] 截断后时间范围: {t1:.2f}s (index={left_threshold_idx}) "
              f"-> {t2:.2f}s (index={right_threshold_idx})")
        print(f"[调试] 截断区间长度: {t2 - t1:.2f}s (原始脉冲长度: {t[group_end_idx] - t[group_start_idx]:.2f}s)\n")

        # 后续处理保持不变...
        mode_time = np.linspace(t[0], t[-1], len(mode))
        trimmed_mode = np.zeros_like(mode)
        keep_mask = (mode_time >= t1) & (mode_time <= t2)

        boundary_tol = 1e-6
        keep_mask |= (np.abs(mode_time - t1) < boundary_tol)
        keep_mask |= (np.abs(mode_time - t2) < boundary_tol)

        if np.sum(keep_mask) < 2:
            raise ValueError(f"无效截断区间[{t1:.2f}s, {t2:.2f}s]")

        trimmed_mode[keep_mask] = mode[keep_mask]
        return trimmed_mode, (t1, t2)

    def _create_group_signal(self, vel_data, pulse_group, time_data):
        """生成分组信号"""
        mask = np.zeros_like(vel_data, dtype=bool)
        for pulse in pulse_group:
            start = max(pulse['start_idx'], 0)
            end = min(pulse['end_idx'], len(mask) - 1)
            mask[start:end + 1] = True
        return np.where(mask, vel_data, 0)

    def _vmd_optimize(self, signal, alpha, max_iter=5):
        """迭代优化VMD分解，当相邻零点间存在单个极值或达到最大迭代时停止"""
        current_signal = signal.copy()
        total_iter = 0

        for _ in range(max_iter):
            try:
                # 执行VMD分解
                u, _ = self.perform_vmd(
                    current_signal,
                    alpha,
                    tau=float(self.b_line_edit.text()),
                    K=int(self.c_line_edit.text()),
                    DC=int(self.d_line_edit.text()),
                    init=int(self.e_line_edit.text()),
                    tol=float(self.start_line_edit.text())
                )

                first_mode = u[0]
                total_iter += 1

                # 检查停止条件
                if not self.has_multiple_extrema_between_zeros(first_mode):
                    return first_mode, total_iter

                current_signal = first_mode  # 更新信号继续分解

            except Exception as e:
                print(f"VMD迭代失败: {str(e)}")
                return None, total_iter  # 出现异常直接返回无效

        # 最终检查（达到最大迭代次数后的最后模态）
        if self.has_multiple_extrema_between_zeros(current_signal):
            return None, total_iter
        return current_signal, total_iter

    def tanimoto_coeff(self, x, y):
        """Tanimoto相似度计算"""
        x_centered = x - np.mean(x)
        y_centered = y - np.mean(y)
        cov = np.dot(x_centered, y_centered)
        norm_x = np.linalg.norm(x_centered)
        norm_y = np.linalg.norm(y_centered)
        denominator = norm_x ** 2 + norm_y ** 2 - cov
        return cov / denominator if denominator != 0 else 0.0

    def _plot_results(self, result, final_mode, velocity_data, time_data):
        """增强版绘图方法"""
        self.canvas.figure.clf()

        # 创建双轴布局
        ax1 = self.canvas.figure.add_subplot(211)
        ax2 = self.canvas.figure.add_subplot(212)

        # 绘制脉冲特征（使用result数据）
        if result and 'pulse_pairs' in result:
            for i, (start, end) in enumerate(result['pulse_pairs']):
                seg_time = time_data[start:end + 1]
                seg_velocity = result['velocity'][start:end + 1]
                energy_ratio = result['energy_ratios'][i] if i < len(result['energy_ratios']) else 0
                color = 'red' if energy_ratio >= 0.1 else 'gray'
                ax1.plot(seg_time, seg_velocity, color=color, linewidth=1)

        ax1.set_title("Acceleration Pulse Characteristics", fontsize=12)
        ax1.set_xlim(time_data.min(), time_data.max())
        ax1.legend(
            handles=[
                plt.Line2D([], [], color='red', label='Significant (≥10% energy)'),
                plt.Line2D([], [], color='gray', label='Non-significant')
            ],
            loc='upper right'
        )

        # 绘制VMD分解结果（使用final_mode数据）
        # 确保始终绘制原始速度曲线
        ax2.plot(time_data, velocity_data,
                 color='gray', alpha=0.6, linewidth=1, label='Original Velocity')
        ax2.set_xlim(time_data.min(), time_data.max())  # 设置x轴范围为t的最小值和最大值
        # 只有当final_mode有数据时才绘制稳定模态
        if final_mode.size > 0:
            min_length = min(len(final_mode), len(time_data))
            ax2.plot(time_data[:min_length], final_mode[:min_length],
                     color='#FF5733', linewidth=1,
                     label=f'Stable Mode ')

        ax2.set_xlabel("Time (s)", fontsize=12)
        ax2.set_title("Optimized VMD Decomposition", fontsize=12)
        ax2.legend(loc='upper right')  # 始终显示图例

        plt.tight_layout(pad=2.0)
        self.canvas.draw()


    def filter_by_duration(self, t, signal, min_duration):
        """根据持续时间阈值滤波信号"""
        # 检测显著脉冲
        result = self._calculate_half_pulse(t, signal)

        # 创建掩码
        mask = np.zeros_like(t, dtype=bool)
        for pulse in result['significant_pulses']:
            if pulse['duration'] >= min_duration:
                start = np.searchsorted(t, pulse['start_time'])
                end = np.searchsorted(t, pulse['end_time'])
                mask[start:end + 1] = True

        return np.where(mask, signal, 0)

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