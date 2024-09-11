from adb.scrcpy_adb import ScrcpyADB
import time
import cv2 as cv

if __name__ == "__main__":
    adb = ScrcpyADB()

    index = 0

    while True:
        index += 1
        time.sleep(2)
        screen = adb.last_screen
        print("index: ", index)
        cv.imwrite(f"img/{index}.png", screen)
