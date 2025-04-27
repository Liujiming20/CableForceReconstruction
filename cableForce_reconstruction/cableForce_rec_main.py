import json
import os
import shutil
import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

from dataRec import rec_abnormal_data


class FileHandler(FileSystemEventHandler):  # watchdog这种方式高效，因为它利用了操作系统的文件系统事件通知机制，不需要频繁地查询目录状态。
    def __init__(self, directory):
        super().__init__()
        self.target_filepath = directory

    def on_any_event(self, event):  # on_any_event 方法会在目录内的任何文件事件（如创建、修改、删除）发生时被触发。如果检测到文件事件，立即执行后续程序并停止监控。
        if not event.is_directory:  # 确保事件是文件相关的
            # print(f"检测到文件事件：{event.event_type} - {event.src_path}")

            if (click_filepath in event.src_path) and ("created" == event.event_type):
                self.execute_subsequent_program()

    def execute_subsequent_program(self):
        file_list = os.listdir(self.target_filepath)
        # 在此处编写你的后续程序逻辑
        if len(file_list) != 2:
            SystemExit("Python监控的文件夹下目标文件数有异，请核查！")

        if target_filename01 in file_list:
            rec_abnormal_data("./config_files/" + target_filename01)
        elif target_filename02 in file_list:
            time.sleep(0.5)
            # 创建一个空的字典
            empty_dict = {}

            # 将空的字典写入 JSON 文件
            target_write_file = "G:/KG_study/reason_main/src/main/resources/abnormalDataRecServiceData/recovered_cable_force.json"
            # 判断文件是否存在
            if os.path.exists(target_write_file):
                # 删除文件
                os.remove(target_write_file)

            with open(target_write_file, 'w') as json_file:
                json.dump(empty_dict, json_file, indent=4)
        else:
            SystemExit("Python监控的文件夹下目标文件名称有异，请核查！")

        clear_directory(self.target_filepath)


def clear_directory(directory):
    # 检查目录是否存在
    if os.path.exists(directory):
        # 遍历目录中的所有文件和子目录
        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            try:
                # 如果是文件，删除文件
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                # 如果是目录，删除目录及其内容
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f'删除 {file_path} 时出错: {e}')
        print(f'{directory} 目录下的所有文件和子目录已被清除。')
    else:
        print(f'目录 {directory} 不存在。')


def monitor_directory(directory, scan_duration):
    event_handler = FileHandler(directory)
    observer = Observer()
    observer.schedule(event_handler, directory, recursive=False)  # 需要监控子目录，可以将 recursive=False 改为 True。

    observer.start()
    print("开始监控目录...")

    # 持续监控一段时间后终止，为了防止java模块出错而一直没有产生重构信号
    try:
        time.sleep(scan_duration)  # 监控持续时间
    except KeyboardInterrupt:
        print("监控被手动中断。")

    observer.stop()  # 停止监控
    observer.join()  # 确保所有线程正确关闭
    print("监控结束。")


def run_periodic_monitoring(directory, interval=30 * 60, scan_duration=10 * 60):
    while True:
        monitor_directory(directory, scan_duration)

        print("等待下一次监控...")
        time.sleep(interval)  # 每半小时等待一次


def main():
    global target_filename01, target_filename02, click_filepath
    target_filename01 = "cableForce_rec_info.json"
    target_filename02 = "no_reconstruction_requirement.json"
    click_filepath = "click.json"

    target_directory = "./config_files"  # 被监控目录
    clear_directory(target_directory)

    run_periodic_monitoring(target_directory)


if __name__ == '__main__':
    main()