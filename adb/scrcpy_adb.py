import sys
from typing import Tuple, Union
from utils.cv_show import CvShow
from adbutils import adb
import scrcpy
import cv2 as cv
import time
from utils.yolov5 import YoloV5s
import queue
import threading


class ScrcpyADB:

    def __init__(self):
        devices = adb.device_list()
        client = scrcpy.Client(device=devices[0])
        adb.connect("127.0.0.1:5555")
        client.add_listener(scrcpy.EVENT_FRAME, self.on_frame)
        client.start(threaded=True)
        self.client = client
        self.yolo = YoloV5s(
            target_size=640,
            prob_threshold=0.25,
            nms_threshold=0.45,
            num_threads=4,
            use_gpu=True,
        )
        self.last_screen = None

        self.frame_queue = queue.Queue(maxsize=10)
        self.stop_event = threading.Event()
        self.cv_show = CvShow()

        # 等待第一帧画面并获取屏幕尺寸
        self.screen_width, self.screen_height = self.get_screen_size()
        print(f"Screen width: {self.screen_width}, Screen height: {self.screen_height}")

    def on_frame(self, frame: cv.Mat):
        """
        把当前帧添加到队列里面
        """
        if frame is not None:
            self.last_screen = frame
            if sys.platform.startswith("darwin"):
                # 每处理一次帧，跳过一帧
                if self.frame_queue.qsize() < 5:  # 队列长度小于5时才添加帧
                    self.frame_queue.put(frame)

    def get_screen_size(self) -> Tuple[int, int]:
        """
        获取屏幕宽度和高度
        :return: (width, height)
        """
        # 等待第一帧画面
        while self.last_screen is None:
            time.sleep(0.1)
        # 获取图像尺寸
        height, width, _ = self.last_screen.shape
        return width, height

    def display_frames(self):
        """
        渲染帧
        :return:
        """
        while not self.stop_event.is_set():
            time.sleep(0.1)
            try:
                frame = self.frame_queue.get(timeout=1)
                if frame is not None:
                    self.cv_show.picture_frame(frame)
            except queue.Empty:
                continue
            except Exception as e:
                print(e)

    def touch_start(self, x: Union[int, float], y: Union[int, float]):
        self.client.control.touch(int(x), int(y), scrcpy.ACTION_DOWN)

    def touch_move(self, x: Union[int, float], y: Union[int, float]):
        self.client.control.touch(int(x), int(y), scrcpy.ACTION_MOVE)

    def touch_end(self, x: Union[int, float] = 0, y: Union[int, float] = 0):
        self.client.control.touch(int(x), int(y), scrcpy.ACTION_UP)

    def tap(
        self,
        x: Union[int, float],
        y: Union[int, float],
        t: Union[int, float] = 0.001,
    ):
        self.touch_start(x, y)
        time.sleep(t)
        self.touch_end(x, y)


if __name__ == "__main__":
    adb = ScrcpyADB()
    adb.display_frames()
    time.sleep(3)
    adb.tap(1225, 29)
