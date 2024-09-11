import os
from utils.path_manager import PathManager
import cv2 as cv
from utils.yolov5 import YoloV5s

screen_scale = 1.69


class CvShow:
    LABLE_LIST = [
        line.strip()
        for line in open(os.path.join(PathManager.MODEL_PATH, "class.txt")).readlines()
    ]

    def __init__(self):
        self.yolo = YoloV5s(
            target_size=640,
            prob_threshold=0.45,
            nms_threshold=0.45,
            num_threads=4,
            use_gpu=True,
        )
        pass

    def picture_frame(self, frame: cv.Mat, show=True):
        """
        在 cv 中画出目标框，并显示标签名称和置信度
        :return:
        """
        font = cv.FONT_HERSHEY_SIMPLEX  # 字体样式
        font_scale = 1  # 字体大小
        thickness = 3  # 文本线条厚

        objs = self.yolo(frame)
        for obj in objs:
            color = (0, 255, 0)
            if obj.label == 0:
                color = (255, 0, 0)
            elif obj.label == 4:
                color = (0, 0, 255)
            cv.rectangle(
                frame,
                (int(obj.rect.x), int(obj.rect.y)),
                (
                    int(obj.rect.x + obj.rect.w),
                    int(obj.rect.y + obj.rect.h),
                ),
                color,
                2,
            )

            # 构造显示的标签文本
            label_text = f"{CvShow.LABLE_LIST[int(obj.label)]}:{obj.prob:.2f}"

            # 计算文本位置
            text_size, _ = cv.getTextSize(label_text, font, font_scale, thickness)
            text_x = int(obj.rect.x)
            text_y = int(obj.rect.y - 5)  # 将文本放置在矩形框上方

            # 如果文本超出边界，则将其放置在矩形框下方
            if text_y < 0:
                text_y = int(obj.rect.y + obj.rect.h + 5)

            # 绘制标签文本
            cv.putText(
                frame,
                label_text,
                (text_x, text_y),
                font,
                font_scale,
                color,
                thickness=thickness,
            )
        if show:
            cv.resize(frame, (240, 108))
            cv.imshow("frame", frame)
            cv.waitKey(33)

        return frame


if __name__ == "__main__":
    pass
