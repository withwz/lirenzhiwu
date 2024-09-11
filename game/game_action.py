from utils.logger import logger
from utils.cv_show import CvShow
from utils.yolov5 import YoloV5s
from adb.scrcpy_adb import ScrcpyADB
import time


class GameAction:
    def __init__(self, adb):
        self.adb = adb
        self.yolo = YoloV5s(
            target_size=640,
            prob_threshold=0.2,
            nms_threshold=0.45,
            num_threads=4,
            use_gpu=True,
        )
        self.cv_show = CvShow()
        self.mid_posi = self.adb.screen_width / 2

    def do_action(self):
        count = 0
        while True:
            time.sleep(0.15)
            screen = self.adb.last_screen
            if screen is None:
                continue

            # self.cv_show.picture_frame(screen)
            results = self.yolo(screen)

            for x in results:
                if x.label == 1:
                    # 变大后，狂点
                    if x.rect.w > 350:
                        logger.info(x.rect.w)
                        for _ in range(666):
                            self.adb.tap(self.mid_posi + 400, 400)
                    continue

            # 遍历检测结果
            for x in results:
                # 判断物体的中心 x 坐标是否接近屏幕中间位置
                if abs(self.mid_posi - (x.rect.x + x.rect.w / 2)) < 150:
                    if x.label == 0:
                        # 移除类别为 0 且位于屏幕中间的物体
                        results.remove(x)  # 将识别为 0 且在屏幕中间的物体从结果中移除

            # 没有怪物
            monsterList = [obj for obj in results if obj.label == 0]
            if len(monsterList) == 0:
                count = count + 1
                if count > 15:
                    self.play_again()
                    return
                continue

            # 找到距离 self.mid_posi 最近的monster x.rect.x
            closest_result = min(
                monsterList,
                key=lambda x: abs(x.rect.x + (x.rect.w / 2) - self.mid_posi),
            )
            # 距离过远不要点击
            if (
                abs(
                    (closest_result.rect.x + (closest_result.rect.w / 2))
                    - self.mid_posi
                )
                > 250
            ):
                continue

            # 判断是在左边还是右边，点击对应侧屏幕
            if closest_result.rect.x > self.mid_posi:
                self.adb.tap(self.mid_posi + 200, 200)
            else:
                self.adb.tap(self.mid_posi - 200, 200)

    def play_again(self):
        """
        再次挑战
        """
        logger.info("再次挑战")
        self.adb.tap(1373, 973)
        self.do_action()


if __name__ == "__main__":
    action = GameAction(ScrcpyADB())
    action.do_action()
