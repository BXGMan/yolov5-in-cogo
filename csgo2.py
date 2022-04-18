import torch
import cv2
import numpy as np
from mss import mss
import pyautogui
import time
from threading import Thread
import keyboard

heng = float(0)   # 变量heng为需要移动的X轴像素点
shu = float(0)    # 变量shu为需要移动的Y轴像素点
hengold = float(0)  # 初始化
model = torch.hub.load('./', 'custom', path='csgo.pt', source='local')  # 载入模型

bounding_box = (760, 390, 1160, 690)  # 识别的区域 一定要屏幕正中间
sct = mss()


def mous():
    global hengold
    if hengold != heng:  # 识别有延迟 为了防止鼠标同一个参数移动两次
        pyautogui.moveRel(heng, shu)   # 移动鼠标参数 *不知道为什么heng和shu变量不用加全局就能获取
        if heng < 100 and heng != 0:    # 当横向移动 小于100个像素且不为0时直接开火
            pyautogui.click(button='left')   # 开火
        hengold = heng


def mouse():
    keyboard.add_hotkey('8', mous)  # 监听键盘
    keyboard.wait()


t1 = Thread(target=mouse)  # 启动键盘线程
t1.start()
while True:
    sct_img = sct.grab(bounding_box)
    scr_img = np.array(sct_img)
    scr_img = model(scr_img)   # 载入识别模块
    asd = scr_img.xyxy[0]    # 获取识别率最高的模型
    x_row = np.size(asd, 0)   # 当没有识别物体时x_row会返回0
    # print(str(asd))
    if x_row:
        a1 = float(str(asd[0, 0]).replace('tensor(', '').replace(", device='cuda:0')", ''))
        a2 = float(str(asd[0, 1]).replace('tensor(', '').replace(", device='cuda:0')", ''))
        a3 = float(str(asd[0, 2]).replace('tensor(', '').replace(", device='cuda:0')", ''))
        a4 = float(str(asd[0, 3]).replace('tensor(', '').replace(", device='cuda:0')", ''))   # x1y1x2y2的参数整理
        heng = a3 - a1  # 计算头部位置
        heng = a1 + heng / 2 - 200   # 计算头部位置 *200是识别的区域（1160-760）/2的结果
        shu = a4 - a2   # 计算头部位置
        shu = a2 + shu / 5 - 150   # 计算头部位置   *因为模型的头部就在y轴的上半边 所以除以5 是头的中心点
        heng = heng * 1.35
        shu = shu * 1.35  # 鼠标的倍数
    else:
        heng = float(0)
        shu = float(0)  # 没有识别物体
    cv2.imshow("Screen Realtime", np.array(scr_img.render())[0])
    # print(str(bounding_box))
    if (cv2.waitKey(1) & 0xFF) == ord('q'):
        cv2.destroyAllWindows()
        break
