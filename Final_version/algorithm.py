import math

import cv2
import numpy as np
from PyQt5.QtGui import QPixmap, QImage, qRgb
import HandTrackingModule as ht
import autopy
import time
import tkinter as tk
from tkinter import ttk, Scale
from PIL import Image, ImageTk

import cv2
import numpy as np
import math
import cv2

import imageio
import cv2
import numpy as np
from scipy.interpolate import UnivariateSpline
from typing_extensions import Self
def insert_element_into_image(element, img, output_path='./result.jpg', initial_size=1):
    def change_size(x):
        global size
        size = x

    def mark(event, x, y, flags, param):
        global insert, result, size
        duplication = param[1].copy()
        if event == cv2.EVENT_LBUTTONDOWN:
            insert = True
        if insert == True and event == cv2.EVENT_MOUSEMOVE:
            param[0] = cv2.resize(param[0], None, fx=size, fy=size)
            x_offset = int(x - param[0].shape[1] * 0.5)
            y_offset = int(y - param[0].shape[0] * 0.5)
            x1, x2 = max(x_offset, 0), min(x_offset + param[0].shape[1], param[1].shape[1])
            y1, y2 = max(y_offset, 0), min(y_offset + param[0].shape[0], param[1].shape[0])
            element_x1 = max(0, -x_offset)
            element_x2 = element_x1 + x2 - x1
            element_y1 = max(0, -y_offset)
            element_y2 = element_y1 + y2 - y1

            beta = param[0][element_y1:element_y2, element_x1:element_x2, 3] / 255
            alpha = 1 - beta

            for channel in range(0, 3):
                duplication[y1:y2, x1:x2, channel] = (beta * param[0][element_y1:element_y2, element_x1:element_x2, channel]
                                                      + alpha * param[1][y1:y2, x1:x2, channel])
            result = duplication.copy()

        cv2.imshow('Image', duplication)

    element = cv2.cvtColor(element, cv2.COLOR_BGR2BGRA)
    for strip in element:
        for pixel in strip:
            if pixel[0] == 0 and pixel[1] == 0 and pixel[2] == 0:
                pixel[3] = 0
    element = cv2.resize(element, None, fx=0.25, fy=0.25)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    duplication = img.copy()
    insert = False
    size = initial_size
    cv2.namedWindow('Image')
    cv2.createTrackbar('Size', 'Image', int(size * 10), 50, change_size)
    cv2.setMouseCallback('Image', mark, [element, img])

    while True:
        if cv2.waitKey(1) & 0xFF == 13:
            img_write = cv2.imencode(".jpg", result)[1].tofile(output_path)
            break

    cv2.destroyAllWindows()

    return result


def mouse_Affine(img):
    temp = img.copy()
    height, width, _ = img.shape
    height_edge_1 = height / 2
    height_edge_2 = 3 * height / 2
    width_egde_1 = width / 2
    width_edge_2 = 3 * width / 2

    # Assuming you have the coordinates
    top_left = (0, 0)
    top_right = (width, 0)
    bottom_right = (width, height)
    bottom_left = (0, height)

    p1 = np.float32([top_left, top_right, bottom_right, bottom_left])
    p2 = np.float32([(0, 0), top_right, bottom_right, bottom_left])

    points = []

    def mouse_callback(event, x, y, flags, param):
        nonlocal points, temp

        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))

        elif event == cv2.EVENT_MOUSEMOVE and flags & cv2.EVENT_LBUTTONDOWN:
            if x < width_egde_1 and y < height_edge_1:
                p2[0] = (x, y)

                M = cv2.getPerspectiveTransform(p1, p2)
                img = cv2.warpPerspective(temp, M, (width, height))
                cv2.imshow('Image', img)
            if x > width_egde_1 and y < height_edge_1:
                p2[1] = (x, y)

                M = cv2.getPerspectiveTransform(p1, p2)
                img = cv2.warpPerspective(temp, M, (width, height))
                cv2.imshow('Image', img)
            if x > width_egde_1 and y > height_edge_1:
                p2[2] = (x, y)

                M = cv2.getPerspectiveTransform(p1, p2)
                img = cv2.warpPerspective(temp, M, (width, height))
                cv2.imshow('Image', img)
            if x < width_egde_1 and y > height_edge_1:
                p2[3] = (x, y)

                M = cv2.getPerspectiveTransform(p1, p2)
                img = cv2.warpPerspective(temp, M, (width, height))
                cv2.imshow('Image', img)

    cv2.imshow("Image", img)
    cv2.setMouseCallback("Image", mouse_callback)


    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == 13:  # Enter key
            if len(points) == 4:
                break

    cv2.destroyAllWindows()
    print(p2[0])
    M = cv2.getPerspectiveTransform(p1, p2)
    img = cv2.warpPerspective(temp, M, (width, height))



    transformed_with_alpha = np.zeros((temp.shape[0], temp.shape[1], 4), dtype=np.uint8)
    transformed_with_alpha[:, :, 0:3] = img
    transformed_with_alpha[:, :, 3] = 255
    transformed_with_alpha[np.all(transformed_with_alpha == [0, 0, 0, 255], axis=2)] = [0, 0, 0, 0]

    return transformed_with_alpha


def scan(img):
    def order_points(pts):
        # 一共4个坐标点
        rect = np.zeros((4, 2), dtype="float32")

        # 按顺序找到对应坐标0123分别是 左上，右上，右下，左下
        # 计算左上，右下
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]

        # 计算右上和左下
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]

        return rect

    def four_point_transform(image, pts):
        # 获取输入坐标点
        rect = order_points(pts)
        (tl, tr, br, bl) = rect

        # 计算输入的w和h值
        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        maxWidth = max(int(widthA), int(widthB))

        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        maxHeight = max(int(heightA), int(heightB))

        # 变换后对应坐标位置
        dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]], dtype="float32")

        # 计算变换矩阵
        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

        # 返回变换后结果
        return warped

    def resize(image, width=None, height=None, inter=cv2.INTER_AREA):
        dim = None
        (h, w) = image.shape[:2]
        if width is None and height is None:
            return image
        if width is None:
            r = height / float(h)
            dim = (int(w * r), height)
        else:
            r = width / float(w)
            dim = (width, int(h * r))
        resized = cv2.resize(image, dim, interpolation=inter)
        return resized

    # 读取输入
    image = img
    # 坐标也会相同变化
    ratio = image.shape[0] / 500.0
    orig = image.copy()

    image = resize(orig, height=500)

    # 预处理
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(gray, 75, 200)
    edged_copy = edged.copy()

    # 展示预处理结果
    print("STEP 1: 边缘检测")
    img_stack = np.hstack((image, cv2.cvtColor(edged_copy, cv2.COLOR_GRAY2BGR)))
    cv2.imshow("Image", img_stack)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # 轮廓检测
    cnts, _ = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]

    # 遍历轮廓
    for c in cnts:
        # 计算轮廓近似
        peri = cv2.arcLength(c, True)
        # C表示输入的点集
        # epsilon表示从原始轮廓到近似轮廓的最大距离，它是一个准确度参数
        # True表示封闭的
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        # 4个点的时候就拿出来
        if len(approx) == 4:
            screenCnt = approx
            break

    # 展示结果
    print("STEP 2: 获取轮廓")
    cv2.drawContours(image, [screenCnt], -1, (0, 255, 0), 2)
    cv2.imshow("Outline", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # 透视变换
    warped = four_point_transform(orig, screenCnt.reshape(4, 2) * ratio)

    cv2.imshow("warped", warped)

    # cv2.imshow("Original", resize(orig, height = 650))
    # cv2.imshow("Scanned", resize(ref, height = 650))
    cv2.waitKey(0)
    return warped
def adjust_brightness_and_sharpness(img):
    # 读取输入图片
    original_image =img

    # 创建一个窗口以显示图片
    cv2.namedWindow('Adjust Image')

    # 创建滚动条来控制亮度和锐化度
    brightness = 50
    sharpness = 1

    cv2.createTrackbar('Brightness', 'Adjust Image', brightness, 100, lambda x: None)
    cv2.createTrackbar('Sharpness', 'Adjust Image', sharpness, 5, lambda x: None)

    while True:
        # 获取滚动条的值
        brightness = cv2.getTrackbarPos('Brightness', 'Adjust Image')
        sharpness = cv2.getTrackbarPos('Sharpness', 'Adjust Image')

        # 调整亮度
        adjusted_image = cv2.convertScaleAbs(original_image, alpha=brightness / 50.0, beta=0)

        # 锐化图像
        kernel = np.array([[-1, -1, -1],
                            [-1,  9, -1],
                            [-1, -1, -1]])
        sharpened_image = cv2.filter2D(adjusted_image, -1, kernel * sharpness)

        # 显示调整后的图像
        cv2.imshow('Adjust Image', sharpened_image)

        # 按下 ESC 键退出循环
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break

    # 关闭窗口
    cv2.destroyAllWindows()
def text_augumention(img):



    image = img
    image = cv2.resize(image, None, fx=0.5, fy=0.5)


    # 方法1
    reflect_img = cv2.copyMakeBorder(image, 1, 1, 1, 1, cv2.BORDER_REPLICATE)  # 复制相邻像素来填充边界
    x, y = reflect_img.shape[:2]
    for depth in range(0, 3):
        for row in range(1, x - 1):
            for col in range(1, y - 1):
                HighPass = (reflect_img.item(row, col, depth) << 2) - reflect_img.item(row - 1, col, depth) \
                           - reflect_img.item(row + 1, col, depth) - reflect_img.item(row, col - 1, depth) \
                           - reflect_img.item(row, col + 1, depth)
                Value = image.item(row - 1, col - 1, depth) + 100 * HighPass // 100
                if Value > 255:
                    Value = 255
                elif Value < 0:
                    Value = 0
                image.itemset((row - 1, col - 1, depth), Value)




    b, g, r = cv2.split(image)  # 拆
    bH = cv2.equalizeHist(b)  # 对三个通道图进行直方图均衡化
    gH = cv2.equalizeHist(g)
    rH = cv2.equalizeHist(r)
    result = cv2.merge([bH, gH, rH])  # 合
    res = np.hstack((image, result))


    threshold = 10  # 设立一个阈值
    h, w = image.shape[:2]
    for i in range(0, h):
        for j in range(0, w):
            B = result[i, j, 0]
            G = result[i, j, 1]
            R = result[i, j, 2]
            if B > threshold and G > threshold and R > threshold:  # 如果大于阈值就赋予白色，保证黑色留下
                result[i, j, 0] = 255
                result[i, j, 1] = 255
                result[i, j, 2] = 255


    return result

def text_transform(img):
    # 读取图片
    src = img

    # 获取图像大小
    rows, cols = src.shape[:2]

    # 将源图像高斯模糊
    img = cv2.GaussianBlur(src, (3, 3), 0)
    # 进行灰度化处理
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 边缘检测（检测出图像的边缘信息）
    edges = cv2.Canny(gray, 50, 250, apertureSize=3)

    # 通过霍夫变换得到A4纸边缘
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 50, minLineLength=90, maxLineGap=10)
    x1 = 0
    y1 = 0
    x2 = 0
    y2 = 0
    x3 = 0
    x4 = 0
    y3 = 0
    y4 = 0

    # 下面输出的四个点分别为四个顶点
    for a, b, c, d in lines[0]:
        x1 = a
        y1 = b
        x2 = c
        y2 = d

    for a, b, c, d in lines[1]:
        x3 = a
        y3 = b
        x4 = c
        y4 = d

    # 绘制边缘
    for x1, y1, x2, y2 in lines[0]:
        cv2.line(gray, (x1, y1), (x2, y2), (0, 0, 255), 1)
    distance1 = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    distance2 = math.sqrt((x4 - x3) ** 2 + (y4 - y3) ** 2)
    # 根据四个顶点设置图像透视变换矩阵
    pos1 = np.float32([[x2, y2], [x4, y4], [x1, y1], [x3, y3]])
    pos2 = np.float32([[0, 0], [distance2, 0], [0, distance1], [distance2, distance1]])
    M = cv2.getPerspectiveTransform(pos1, pos2)

    x = int(distance2)
    y = int(distance1)
    # 图像透视变换
    result = cv2.warpPerspective(src, M, (x, y))

    return result


def apply_vignette_filter(img):
    pressed = False
    roi_x, roi_y = img.shape[0] // 2, img.shape[1] // 2
    original_image = img.copy()

    def select_roi(event, x, y, flags, param):
        nonlocal roi_x, roi_y, pressed
        if event == cv2.EVENT_LBUTTONDOWN:
            pressed = True
        elif event == cv2.EVENT_MOUSEMOVE:
            if pressed == True and original_image.shape[1] < x:
                roi_x = x - original_image.shape[1]
                roi_y = y
                apply_vignette()
        elif event == cv2.EVENT_LBUTTONUP:
            pressed = False

    def apply_vignette():
        nonlocal img, original_image, vignette_radius, roi_x, roi_y
        rows, cols = img.shape[:2]
        orginal_img = img
        kernel_x = cv2.getGaussianKernel(cols * 2, vignette_radius * 10)
        kernel_y = cv2.getGaussianKernel(rows * 2, vignette_radius * 10)
        kernel = kernel_y * kernel_x.T
        kernel = 255 * kernel / np.linalg.norm(kernel)
        mask = kernel[rows - roi_y:rows - roi_y + rows,
               cols - roi_x:cols - roi_x + cols]
        output = np.copy(original_image)
        for i in range(3):
            output[:, :, i] = output[:, :, i] * mask

        cv2.imshow('Vignette Filter', np.hstack((img, output)))

    def update_vignette(val):
        nonlocal vignette_radius
        vignette_radius = val
        apply_vignette()

    cv2.namedWindow('Vignette Filter')

    vignette_radius = 10
    cv2.createTrackbar('Vignette Radius', 'Vignette Filter', vignette_radius, 50,
                       update_vignette)

    cv2.imshow('Vignette Filter', np.hstack((original_image, img)))  # 初始显示原图

    cv2.setMouseCallback('Vignette Filter', select_roi)

    while True:
        key = cv2.waitKey(1)
        if key == 27:
            break

    cv2.destroyAllWindows()

    return img
# select foreground by mask

class Foreground_mask_selector:
    def __init__(self: Self, _img: np.ndarray, _win_name: str = 'Select foreground mask') -> None:
        self.winname: str = _win_name
        self.origin_img: np.ndarray = _img.copy()
        self.is_shown: bool = False
        self.fg_choosen: np.ndarray = np.zeros(self.origin_img.shape[:2], np.uint8)

    def select(self: Self) -> np.ndarray:
        self.show_window()
        save_mask: np.ndarray = np.zeros_like(self.fg_choosen)
        while True:
            self.fg_choosen *= 0
            key = cv2.waitKey() & 0xFF
            if key == 13:  # Enter:GrabCut
                self.update()
                save_mask = self.fg_choosen.copy()
                break

        bgdModel = np.zeros((1, 65), np.float64)
        fgdModel = np.zeros((1, 65), np.float64)
        self.fg_choosen, bgdModel, fgdModel = \
            cv2.grabCut(self.origin_img, save_mask, None, bgdModel, fgdModel, cv2.GC_INIT_WITH_MASK)
        final_mask = \
            np.where((save_mask == 2), 0, 1).astype('uint8')
        to_save = self.origin_img * final_mask[:, :, np.newaxis]
        to_save = cv2.cvtColor(to_save, cv2.COLOR_BGR2BGRA)
        to_save[:, :, 3] = np.where((final_mask == 0), 0, 255)

        self.destory_window()
        return to_save

    def show_window(self: Self) -> None:
        if self.is_shown: return

        cv2.namedWindow(self.winname)
        self.update()
        cv2.setMouseCallback(self.winname, self.callback)

        self.is_shown = True
        pass

    def destory_window(self: Self) -> None:
        if not self.is_shown: return

        cv2.destroyWindow(self.winname)

        self.is_shown = False
        pass

    def update(self: Self) -> None:
        _mask = cv2.cvtColor(self.fg_choosen, cv2.COLOR_GRAY2BGR)
        output: np.ndarray = cv2.addWeighted(self.origin_img, 0.8, _mask, 17, 0)
        cv2.imshow(self.winname, output)

    def callback(self: Self, event: int, x: int, y: int, flags: int, param: any) -> None:
        if flags & cv2.EVENT_FLAG_LBUTTON != 0:
            cv2.circle(self.fg_choosen, (x, y), 18, (3), -1)
            _mask = cv2.cvtColor(self.fg_choosen, cv2.COLOR_GRAY2BGR)
            output: np.ndarray = cv2.addWeighted(self.origin_img, 0.8, _mask, 17, 0)
            cv2.imshow(self.winname, output)

def select_background_by_roi(img: np.ndarray) -> np.ndarray:
    image = img.copy()
    roi = cv2.selectROI(image,showCrosshair=False)
    cv2.destroyWindow('ROI selector')
    mask = np.zeros(image.shape[:2], np.uint8)

    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    cv2.grabCut(image, mask, roi, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)

    mask2 = np.where((mask == 3) | (mask == 1), 0, 1).astype('uint8')

    result = image * mask2[:, :, np.newaxis]


    rgba_image = np.zeros((img.shape[0], img.shape[1], 4), dtype=np.uint8)
    rgba_image[:, :, :3] = result

    black_pixels = np.all(rgba_image[:, :, :3] != [0, 0, 0], axis=2)
    rgba_image[black_pixels, 3] = 255

    return rgba_image


def select_foreground_by_roi(img: np.ndarray) -> np.ndarray:
    image = img.copy()
    roi = cv2.selectROI(image,showCrosshair=False)
    cv2.destroyWindow('ROI selector')
    mask = np.zeros(image.shape[:2], np.uint8)

    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    cv2.grabCut(image, mask, roi, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)

    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')

    result = image * mask2[:, :, np.newaxis]


    rgba_image = np.zeros((img.shape[0], img.shape[1], 4), dtype=np.uint8)
    rgba_image[:, :, :3] = result

    black_pixels = np.all(rgba_image[:, :, :3] != [0, 0, 0], axis=2)
    rgba_image[black_pixels, 3] = 255

    return rgba_image


def knife(img):
    while True:
        # 显示输入图像，获取选择区域
        cv2.imshow('image', img)
        [min_x, min_y, w, h] = cv2.selectROI('image', img, showCrosshair=False)
        result = img[min_x:min_x + w, min_y:min_y + h]
        cv2.destroyAllWindows()  # 关闭图像显示窗口

        # 检查是否成功选择了区域
        if w > 0 and h > 0:
            break
    # 获取选定区
    return result
def apply_ctext(img):
    global selected_points  # Declare selected_points as a global variable

    def draw_curve(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            selected_points.append((x, y))
            cv2.circle(canvas, (x, y), 5, (0, 0, 0), -1)

    canvas = np.copy(img)
    cv2.namedWindow('Result')
    cv2.setMouseCallback('Result', draw_curve)

    while True:
        cv2.imshow('Result', canvas)
        key = cv2.waitKey(1) & 0xFF

        if key == 27 and len(selected_points) >= 3:
            selected_points.sort(key=lambda point: point[0])
            points = np.array(selected_points, dtype=np.int32)
            spline = UnivariateSpline(points[:, 0], points[:, 1], s=0, k=2)
            last_point = selected_points[len(selected_points) // 2 - 1:len(selected_points) // 2 + 1][len(selected_points) % 2]
            derivative = spline.derivative()
            tangent_angle = np.arctan2(derivative(last_point[0]), 1)

            font = cv2.FONT_HERSHEY_SIMPLEX
            text = 'OPENCV'
            font_scale = 1
            font_thickness = 3
            text_color = (255, 255, 255)
            (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, font_thickness)
            text_center_x = last_point[0]
            text_center_y = last_point[1]
            x = int(text_center_x - (text_width // 2))
            y = int(text_center_y + (text_height // 2))
            text_image = np.zeros_like(img)
            cv2.putText(text_image, text, (x, y), font, font_scale, text_color, font_thickness)
            rotation_matrix = cv2.getRotationMatrix2D((text_center_x, text_center_y), -np.degrees(tangent_angle), 1)
            rotated_text_image = cv2.warpAffine(text_image, rotation_matrix, (text_image.shape[1], text_image.shape[0]))
            result_image = cv2.addWeighted(canvas, 1, rotated_text_image, 1, 0)

            for x in range(points[0][0], points[-1][0]):
                y = int(spline(x))
                cv2.circle(result_image, (x, y), 1, (0, 0, 0), -1)

            cv2.imshow('Result', result_image)
            cv2.waitKey(0)
            break

    cv2.destroyAllWindows()
    return result_image


def apply_imgchange(img1, img2):
    img2 = cv2.resize(img2, img1.shape[:2][::-1])

    def percent_func_gen(a, b, time, n, mode):
        if mode == "slower":
            a, b = b, a
        delta = abs(a - b)
        sgn = 1 if b - a > 0 else (-1 if b - a < 0 else 0)

        def percent_calc(ti):
            if mode == "slower":
                ti = time - ti
            return sgn * delta / (time ** n) * (ti ** n) + a

        return percent_calc

    def transition_flash_black(img1, img2, d):
        load_f = 20
        tim = d / 1000
        percent_func1 = percent_func_gen(a=1, b=0, time=tim, n=1, mode="null")
        percent_func2 = percent_func_gen(a=0, b=1, time=tim, n=1, mode="null")
        for t in range(int(tim * 1000) // load_f + 1):
            percent = percent_func1(t * load_f / 1000)
            img_show = cv2.multiply(img1, (1, 1, 1, 1), scale=percent)
            cv2.imshow("show", img_show)
            cv2.waitKey(load_f)
        for t in range(int(tim * 1000) // load_f + 1):
            percent = percent_func2(t * load_f / 1000)
            img_show = cv2.multiply(img2, (1, 1, 1, 1), scale=percent)
            cv2.imshow("show", img_show)
            cv2.waitKey(load_f)

    def transition_slide_left(img1, img2, d):
        load_f = 20
        tim = d / 1000
        percent_func = percent_func_gen(a=0, b=1, time=tim, n=2, mode="faster")
        rows, cols = img1.shape[:2]
        img = np.hstack([img1, img2])
        for t in range(int(tim * 1000) // load_f + 1):
            percent = percent_func(t * load_f / 1000)
            x = int(percent * cols)
            M = np.float32([[1, 0, -x], [0, 1, 0]])
            res = cv2.warpAffine(img, M, (rows, cols))
            cv2.imshow("show", res)
            cv2.waitKey(load_f)

    def transition_slide_right(img1, img2, d):
        load_f = 20
        tim = d / 1000
        percent_func = percent_func_gen(a=0, b=1, time=tim, n=2, mode="faster")
        rows, cols = img1.shape[:2]
        img = np.hstack([img1, img2])
        for t in range(int(tim * 1000) // load_f + 1):
            percent = percent_func(t * load_f / 1000)
            x = int(percent * cols)
            M = np.float32([[1, 0, x], [0, 1, 0]])
            res = cv2.warpAffine(img, M, (rows, cols))
            if x > 0:
                res[:, :x] = img1[:, -x:]
            cv2.imshow("show", res)
            cv2.waitKey(load_f)

    def transition_slide_up(img1, img2, d):
        load_f = 20
        tim = d / 1000
        percent_func = percent_func_gen(a=0, b=1, time=tim, n=2, mode="faster")
        rows, cols = img1.shape[:2]
        img = np.vstack([img1, img2])
        for t in range(int(tim * 1000) // load_f + 1):
            percent = percent_func(t * load_f / 1000)
            y = int(percent * rows)
            M = np.float32([[1, 0, 0], [0, 1, -y]])
            res = cv2.warpAffine(img, M, (cols, rows))
            cv2.imshow("show", res)
            cv2.waitKey(load_f)

    def transition_slide_down(img1, img2, d):
        load_f = 20
        tim = d / 1000
        percent_func = percent_func_gen(a=0, b=1, time=tim, n=2, mode="faster")
        rows, cols = img1.shape[:2]
        img = np.vstack([img2, img1])
        for t in range(int(tim * 1000) // load_f + 1):
            percent = percent_func(t * load_f / 1000)
            y = int(percent * rows)
            M = np.float32([[1, 0, 0], [0, 1, y]])
            res = cv2.warpAffine(img, M, (cols, rows))
            if y > 0:
                res[:y, :] = img1[-y:, :]
            cv2.imshow("show", res)
            cv2.waitKey(load_f)

    def transition_erase_down(img1, img2, d):
        load_f = 20
        tim = d / 1000
        percent_func = percent_func_gen(a=0, b=1, time=tim, n=1, mode="null")
        rows, cols = img1.shape[:2]
        for t in range(int(tim * 1000) // load_f + 1):
            percent = percent_func(t * load_f / 1000)
            height = int(percent * rows)
            img1[:height, :] = img2[:height, :]
            cv2.imshow("show", img1)
            cv2.waitKey(load_f)

    def transition_horizontal_blinds(img1, img2, d):
        load_f = 20
        tim = d / 1000
        percent_func = percent_func_gen(a=0, b=0.5, time=tim, n=1, mode="null")
        rows, cols = img1.shape[:2]
        half = int(rows / 2)
        for t in range(int(tim * 1000) // load_f + 1):
            percent = percent_func(t * load_f / 1000)
            width = int(percent * rows)
            ys, ye = half - width, half + width
            img1[:, ys:ye] = img2[:, ys:ye]
            cv2.imshow("show", img1)
            cv2.waitKey(load_f)

    def transition_rotate(img1, img2, d):
        load_f = 20
        tim = d / 1000
        angle_all = 150
        point1 = (img1.shape[1] * 3, img1.shape[0] * 3)
        point2 = (img1.shape[1] * 3, img1.shape[0] * 4)
        percent_func1 = percent_func_gen(a=0, b=1, time=tim, n=4, mode="faster")
        percent_func2 = percent_func_gen(a=1, b=0, time=tim, n=4, mode="slower")
        for t in range(int(tim * 1000) // load_f + 1):
            percent = percent_func1(t * load_f / 1000)
            angle = percent * angle_all
            img1_rotated = rotate_image(img1, angle)
            cv2.imshow("show", img1_rotated)
            cv2.waitKey(load_f)
        for t in range(int(tim * 1000) // load_f + 1):
            percent = percent_func2(t * load_f / 1000)
            angle = -percent * angle_all
            img2_rotated = rotate_image(img2, angle)
            cv2.imshow("show", img2_rotated)
            cv2.waitKey(load_f)

    def rotate_image(image, angle, d):
        center = tuple(np.array(image.shape[1::-1]) / 2)
        rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
        result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
        return result



    def save_as_gif(frames, filename, duration_ms):
        images = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in frames]
        imageio.mimsave(filename, images, duration=duration_ms / 1000.0)

    # 创建一个列表来存储转场帧
    transition_frames = []

    # 创建主窗口
    root = tk.Tk()
    root.title("图片转场选择")

    # 创建标签
    label = tk.Label(root, text="请选择图片转场效果:")
    label.pack()

    # 创建下拉框
    options = ["闪黑", "左移", "右移", "上移", "下移", "向下擦除", "横向拉幕", "旋转"]
    combo = ttk.Combobox(root, values=options)
    combo.pack()

    # 创建滚动条
    scrollbar = Scale(root, from_=100, to=3000, orient="horizontal", label="转场时间(ms)")
    scrollbar.set(1000)  # 设置默认持续时间为1000ms
    scrollbar.pack()

    # 创建按钮
    button = tk.Button(
        root, text="开始转场", command=lambda: perform_transition(combo.get(), int(scrollbar.get()))
    )
    button.pack()

    # 定义执行转场的函数
    def perform_transition(selected_option, duration_ms):

        if selected_option == "闪黑":
            transition_flash_black(img1, img2, duration_ms)
        elif selected_option == "左移":
            transition_slide_left(img1, img2, duration_ms)
        elif selected_option == "右移":
            transition_slide_right(img1, img2, duration_ms)
        elif selected_option == "上移":
            transition_slide_up(img1, img2, duration_ms)
        elif selected_option == "下移":
            transition_slide_down(img1, img2, duration_ms)
        elif selected_option == "向下擦除":
            transition_erase_down(img1, img2, duration_ms)
        elif selected_option == "横向拉幕":
            transition_horizontal_blinds(img1, img2, duration_ms)
        elif selected_option == "旋转":
            transition_rotate(img1, img2, duration_ms)

        # 保存转场过程为动态GIF
        save_as_gif(transition_frames, "output.gif", duration_ms)
        # 清空转场帧列表
        transition_frames.clear()

    # 创建一个Canvas用于显示转场帧
    canvas = tk.Canvas(root, width=img1.shape[1], height=img1.shape[0])
    canvas.pack()

    # 将每一帧添加到Canvas中显示
    def display_frame(frame):
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        photo = ImageTk.PhotoImage(image=image)
        canvas.create_image(0, 0, anchor="nw", image=photo)
        canvas.image = photo

    # 启动主循环
    root.mainloop()
def apply_mosaic_to_roi(img, roi):
    x, y, w, h = roi
    roi_image = img[y:y + h, x:x + w]

    # 缩小图像以创建马赛克效果
    small_roi = cv2.resize(roi_image, (10, 10))
    mosaic_roi = cv2.resize(small_roi, roi_image.shape[:2][::-1], interpolation=cv2.INTER_NEAREST)

    # 将马赛克区域放回原始图像中
    img[y:y + h, x:x + w] = mosaic_roi

    return img

def sector_ring_mapping(image, center, inner_radius, outer_radius, start_angle, end_angle):
    height, width = image.shape[:2]

    # Create a blank canvas to hold the mapped image
    sector_ring_mapped = np.zeros_like(image)

    # Loop through the destination image and map pixels from the source image
    for y in range(height):
        for x in range(width):
            # Calculate polar coordinates (rho and theta) for the destination pixel
            rho = int(np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2))
            theta = np.arctan2(y - center[1], x - center[0])

            # Convert theta to degrees
            theta_degrees = np.degrees(theta)

            # Check if the pixel is within the sector angle and between the inner and outer radius
            if start_angle <= theta_degrees <= end_angle and inner_radius <= rho <= outer_radius:
                # Map the pixel from the source image to the destination image
                source_x = int((theta_degrees - start_angle) / (end_angle - start_angle) * image.shape[1])
                if source_x >= 300:
                    source_x = 299
                sector_ring_mapped[y, x] = image[source_x, y]

    return sector_ring_mapped

def apply_image_xuanwo(img, x_scale=1, y_scale=1, degree=1):
    def transform(input_img, x, y, dg):
        row, col, channel = input_img.shape
        trans_img = input_img.copy()
        img_out = input_img * 1.0
        degree = dg
        center_x = x
        center_y = y
        y_mask, x_mask = np.indices((row, col))
        xx_dif = x_mask - center_x
        yy_dif = center_y - y_mask
        r = np.sqrt(xx_dif * xx_dif + yy_dif * yy_dif)
        theta = np.arctan(yy_dif / xx_dif)
        mask_1 = xx_dif < 0
        theta = theta * (1 - mask_1) + (theta + math.pi) * mask_1
        theta = theta + r / degree
        x_new = r * np.cos(theta) + center_x
        y_new = center_y - r * np.sin(theta)
        x_new = x_new.astype(np.float32)
        y_new = y_new.astype(np.float32)
        dst = cv2.remap(trans_img, x_new, y_new, cv2.INTER_LINEAR)
        return dst

    cv2.namedWindow('Image Transformation')

    cv2.createTrackbar('X Scale', 'Image Transformation', 0, 300, lambda x: None)
    cv2.createTrackbar('Y Scale', 'Image Transformation', 0, 300, lambda x: None)
    cv2.createTrackbar('degree', 'Image Transformation', 0, 200, lambda x: None)

    while True:
        x_scale = cv2.getTrackbarPos('X Scale', 'Image Transformation')
        y_scale = cv2.getTrackbarPos('Y Scale', 'Image Transformation')
        degree = cv2.getTrackbarPos('degree', 'Image Transformation')

        transformed_image = transform(img, x_scale, y_scale, degree)

        stacked_image = np.hstack((img, transformed_image))
        cv2.imshow('Image Transformation', stacked_image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    return transformed_image

def apply_image_transformations(img, x_scale=1, y_scale=1, x_period=0, y_period=0):
    def transform(input_img, x, y, xx, yy):
        row, col, _ = input_img.shape
        trans_image = input_img.copy()
        alpha = x
        beta = y
        degree_x = xx
        degree_y = yy
        center_x = (col - 1) / 2.0
        center_y = (row - 1) / 2.0
        y_mask, x_mask = np.indices((row, col))
        xx_dif = x_mask - center_x
        yy_dif = center_y - y_mask
        x = degree_x * np.sin(2 * math.pi * yy_dif / alpha) + xx_dif
        y = degree_y * np.cos(2 * math.pi * xx_dif / beta) + yy_dif
        x_new = x + center_x
        y_new = center_y - y
        x_new = x_new.astype(np.float32)
        y_new = y_new.astype(np.float32)
        dst = cv2.remap(trans_image, x_new, y_new, cv2.INTER_LINEAR)
        return dst

    cv2.namedWindow('Image Transformations')

    cv2.createTrackbar('X Scale', 'Image Transformations', 0, 200, lambda x: None)
    cv2.createTrackbar('Y Scale', 'Image Transformations', 0, 200, lambda x: None)
    cv2.createTrackbar('X Period', 'Image Transformations', 0, 20, lambda x: None)
    cv2.createTrackbar('Y Period', 'Image Transformations', 0, 20, lambda x: None)

    while True:
        x_scale = cv2.getTrackbarPos('X Scale', 'Image Transformations')
        y_scale = cv2.getTrackbarPos('Y Scale', 'Image Transformations')
        x_period = cv2.getTrackbarPos('X Period', 'Image Transformations')
        y_period = cv2.getTrackbarPos('Y Period', 'Image Transformations')

        transformed_image = transform(img, x_scale, y_scale, x_period, y_period)

        stacked_image = np.hstack((img, transformed_image))
        cv2.imshow('Image Transformations', stacked_image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    return transformed_image


def apply_convex_lens_effect(img):
    def filter_convex_lens(src_img, x, y, radius):
        new_img = src_img.copy()

        i_indices, j_indices = np.indices(src_img.shape[:2])

        distances = (i_indices - x) ** 2 + (j_indices - y) ** 2
        new_dists = np.sqrt(distances)

        mask = (distances <= radius ** 2)

        new_i = np.floor(new_dists * (i_indices - x) / radius + x).astype(int)
        new_j = np.floor(new_dists * (j_indices - y) / radius + y).astype(int)

        new_img[mask] = src_img[new_i[mask], new_j[mask]]
        return new_img

    row, col, _ = img.shape
    x = int(row / 2)
    y = int(col / 2)
    radius = int(math.sqrt(x * x + y * y + (row - x) ** 2 + (col - y) ** 2) / 3)

    def change_center(t):
        nonlocal x, y
        x = cv2.getTrackbarPos('x', 'convex_image')
        y = cv2.getTrackbarPos('y', 'convex_image')

    def change_radius(t):
        nonlocal radius
        radius = cv2.getTrackbarPos('Radius', 'convex_image')

    cv2.namedWindow('convex_image')
    cv2.createTrackbar('x', 'convex_image', 0, row, change_center)
    cv2.createTrackbar('y', 'convex_image', 0, col, change_center)
    cv2.createTrackbar('Radius', 'convex_image', 0, radius * 3, change_radius)
    cv2.setTrackbarPos('x', 'convex_image', 1)
    cv2.setTrackbarPos('y', 'convex_image', 1)
    cv2.setTrackbarPos('Radius', 'convex_image', radius)

    while True:
        new_image = filter_convex_lens(img, x, y, radius)
        show = np.hstack([img, new_image])
        cv2.imshow('convex_image', show)
        if cv2.waitKey(1) & 0xFF == 13:
            break

    cv2.destroyAllWindows()
    return new_image
import cv2
import numpy as np

def apply_concave_lens_effect(img):
    def filter_concave_lens(src, x, y, R):

        height, width = src.shape[:2]
        center = (x, y)
        img2 = np.zeros(src.shape, dtype=np.uint8)

        pos_y, pos_x = np.indices((height, width))
        norm_x, norm_y = [pos_x - center[0], pos_y - center[1]]

        theta = np.arctan2(norm_x, norm_y)
        R2 = (np.sqrt(np.linalg.norm(np.array([norm_x, norm_y]), axis=0)) * R).astype(int)

        new_x = center[0] + (R2 * np.cos(theta)).astype(int)
        new_y = center[1] + (R2 * np.sin(theta)).astype(int)

        new_x = np.clip(new_x, 0, width - 1)
        new_y = np.clip(new_y, 0, height - 1)

        img2 = src[new_x, new_y]
        return img2

    row, col, _ = img.shape
    x = int(row / 2)
    y = int(col / 2)
    R = 8

    def change_center(t):
        nonlocal x, y
        x = cv2.getTrackbarPos('x', 'concave_image')
        y = cv2.getTrackbarPos('y', 'concave_image')

    def change_radius(t):
        nonlocal R
        R = cv2.getTrackbarPos('R', 'concave_image')

    cv2.namedWindow('concave_image')
    cv2.createTrackbar('x', 'concave_image', 0, row, change_center)
    cv2.createTrackbar('y', 'concave_image', 0, col, change_center)
    cv2.createTrackbar('R', 'concave_image', 1, 20, change_radius)
    cv2.setTrackbarPos('x', 'concave_image', 1)
    cv2.setTrackbarPos('y', 'concave_image', 1)
    cv2.setTrackbarPos('R', 'concave_image', 8)

    while True:
        new_image = filter_concave_lens(img, x, y, R)
        show = np.hstack([img, new_image])
        cv2.imshow('concave_image', show)
        if cv2.waitKey(1) & 0xFF == 13:
            break
    cv2.destroyAllWindows()
    return new_image


def perform_perspective_transformation(img):
    temp = img.copy()
    height, width, _ = img.shape

    # Assuming you have the coordinates
    top_left = (0, 0)
    top_right = (width, 0)
    bottom_right = (width, height)
    bottom_left = (0, height)

    p1 = np.float32([top_left, top_right, bottom_right, bottom_left])

    points = []

    def mouse_callback(event, x, y, flags, param):
        nonlocal points, temp

        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))
            cv2.imshow("Image", temp)
            cv2.circle(temp, (x, y), 5, (0, 0, 255), -1)
            if len(points) == 2:
                cv2.line(temp, points[0], points[1], (0, 255, 0), 2)
            elif len(points) == 3:
                cv2.line(temp, points[1], points[2], (0, 255, 0), 2)
            elif len(points) == 4:
                cv2.line(temp, points[2], points[3], (0, 255, 0), 2)
                cv2.line(temp, points[0], points[3], (0, 255, 0), 2)
            cv2.imshow("Image", temp)

    cv2.imshow("Image", temp)
    cv2.setMouseCallback("Image", mouse_callback)

    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == 13:  # Enter key
            if len(points) == 4:
                break

    cv2.destroyAllWindows()

    p2 = np.float32(points)

    M = cv2.getPerspectiveTransform(p1, p2)
    dst = cv2.warpPerspective(img, M, (width, height))

    transformed_with_alpha = np.zeros((img.shape[0], img.shape[1], 4), dtype=np.uint8)
    transformed_with_alpha[:, :, 0:3] = dst
    transformed_with_alpha[:, :, 3] = 255
    transformed_with_alpha[np.all(transformed_with_alpha == [0, 0, 0, 255], axis=2)] = [0, 0, 0, 0]

    return transformed_with_alpha

def perform_affine_transformation(img):
    temp = img.copy()
    height, width, _ = img.shape
    top_left = (0, 0)
    top_right = (width, 0)
    bottom_right = (width, height)

    p1 = np.float32([top_left, top_right, bottom_right])

    def calculate_angle(line1, line2):
        angle = math.degrees(math.atan2(line2[1][1] - line2[0][1], line2[1][0] - line2[0][0]) -
                             math.atan2(line1[1][1] - line1[0][1], line1[1][0] - line1[0][0]))
        return angle

    points = []

    def mouse_callback(event, x, y, flags, param):
        nonlocal points, temp

        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))
            cv2.circle(temp, (x, y), 5, (0, 0, 255), -1)
            if len(points) == 2:
                cv2.line(temp, points[0], points[1], (0, 255, 0), 2)
            elif len(points) == 3:
                cv2.line(temp, points[1], points[2], (0, 255, 0), 2)
                cv2.line(temp, points[0], points[2], (0, 255, 0), 2)
                angle = calculate_angle([points[0], points[1]], [points[0], points[2]])
                cv2.putText(temp, f"Angle: {angle:.2f} degrees", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow("Image", temp)
        elif event == cv2.EVENT_MOUSEMOVE:
            temp_image = img.copy()
            if len(points) == 1:
                cv2.line(temp_image, points[0], (x, y), (255, 0, 0), 1)
            elif len(points) == 2:
                cv2.line(temp_image, points[0], points[1], (0, 255, 0), 2)
                cv2.line(temp_image, points[1], (x, y), (255, 0, 0), 1)
                new_points = (x, y)
                angle = calculate_angle([points[1], points[0]], [points[1], new_points])
                cv2.putText(temp_image, f"Angle: {angle:.2f} degrees", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            elif len(points) == 3:
                cv2.line(temp_image, points[1], points[2], (0, 255, 0), 2)
                cv2.line(temp_image, points[0], points[2], (0, 255, 0), 2)
                angle = calculate_angle([points[0], points[1]], [points[0], points[2]])
                cv2.putText(temp_image, f"Angle: {angle:.2f} degrees", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow("Image", temp_image)

    cv2.imshow("Image", temp)
    cv2.setMouseCallback("Image", mouse_callback)

    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == 13:  # Enter key
            if len(points) == 3:
                break

    cv2.destroyAllWindows()

    p2 = np.float32(points)

    M = cv2.getAffineTransform(p1, p2)

    dst = cv2.warpAffine(img, M, (width, height))

    transformed_with_alpha = np.zeros((img.shape[0], img.shape[1], 4), dtype=np.uint8)
    transformed_with_alpha[:, :, 0:3] = dst
    transformed_with_alpha[:, :, 3] = 255
    transformed_with_alpha[np.all(transformed_with_alpha == [0, 0, 0, 255], axis=2)] = [0, 0, 0, 0]

    return transformed_with_alpha

def gray_picture(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return img_gray
def mean_blur(img,kernel_size):
    img_blur = cv2.blur(img, (kernel_size, kernel_size))

    return img_blur
def median_blur(img,kernel_size):
    img_blur = cv2.medianBlur(img, kernel_size)

    return img_blur
def gaussian_blur(img,kernel_size):

    img_blur = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

    return img_blur

def Thresold(img,op,x,y):

    thre_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    thre_img = cv2.threshold(thre_img, x, y, op)[1]
    thre_img = cv2.cvtColor(thre_img, cv2.COLOR_GRAY2BGR)
    return thre_img

def Edge(img,x,y):
    img = cv2.Canny(img, x, y)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img

def Grad(img,op,ksize,x,y):
    if op=="sobel":
        img_grad = cv2.Sobel(img, -1, x, y, ksize)
    if op=="scharr":
        img_grad = cv2.Scharr(img,-1,x,y)
    if op=="laplacian":
        img_grad = cv2.Laplacian(img,-1)
    return img_grad

def Morph(img,op,kshape,ksize):
    shape=cv2.MORPH_ELLIPSE
    cvop=cv2.MORPH_ERODE
    if kshape=="椭圆形":
        shape=cv2.MORPH_ELLIPSE
    if kshape=="十字形":
        shape=cv2.MORPH_CROSS
    if kshape=="方形":
        shape=cv2.MORPH_RECT
    if op=="腐蚀":
        cvop=cv2.MORPH_ERODE
    if op=="膨胀":
        cvop=cv2.MORPH_DILATE
    if op=="开":
        cvop=cv2.MORPH_OPEN
    if op=="闭":
        cvop=cv2.MORPH_CLOSE
    if op=="梯度":
        cvop=cv2.MORPH_GRADIENT
    if op=="顶帽":
        cvop=cv2.MORPH_TOPHAT
    if op=="黑帽":
        cvop=cv2.MORPH_BLACKHAT
    kernal = cv2.getStructuringElement(shape, (ksize, ksize))

    Morph_img = cv2.morphologyEx(img, cvop, kernal)
    return Morph_img
def Equalize(img, kind):
    b, g, r = cv2.split(img)
    if kind=='B':
        b = cv2.equalizeHist(b)
    if kind=='G':
        g = cv2.equalizeHist(g)
    if kind=='R':
        r = cv2.equalizeHist(r)
    return cv2.merge((b, g, r))

def qimage2mat(qtpixmap):    #qtpixmap转opencv
    qimg = qtpixmap.toImage()
    temp_shape = (qimg.height(), qimg.bytesPerLine() * 8 // qimg.depth())
    temp_shape += (4,)
    ptr = qimg.bits()
    ptr.setsize(qimg.byteCount())
    result = np.array(ptr, dtype=np.uint8).reshape(temp_shape)
    result = result[..., :3]
    return result


def matqimage(cvimg):       #opencv转QImage
    if cvimg.ndim==2:              #单通道
        height, width= cvimg.shape
        cvimg = cv2.cvtColor(cvimg, cv2.COLOR_BGR2RGB)
        cvimg = QImage(cvimg.data, width, height, QImage.Format_RGB888)
        pix = QPixmap.fromImage(cvimg)
        return pix
    else:                          #多个通道
        width = cvimg.shape[1]
        height = cvimg.shape[0]
        pixmap = QPixmap(width, height)  # 根据已知的高度和宽度新建一个空的QPixmap,
        qimg = pixmap.toImage()         # 将pximap转换为QImage类型的qimg
        for row in range(0, height):
            for col in range(0, width):
                b = cvimg[row, col, 0]
                g = cvimg[row, col, 1]
                r = cvimg[row, col, 2]
                pix = qRgb(r, g, b)
                qimg.setPixel(col, row, pix)
                pix = QPixmap.fromImage(qimg)
        return pix

###############图像翻转####################################################
def flip_picture(image,x):    #图像  水平翻转:0   垂直翻转:1  沿xy轴翻转:-1
    image = cv2.flip(image,x)
    return image

def hand_tracking_mouse_control():
    pTime = 0
    width, height = 640, 480
    frameR = 100
    smoothening = 8
    prev_x, prev_y = 0, 0
    curr_x, curr_y = 0, 0

    cap = cv2.VideoCapture(0)
    cap.set(3, width)
    cap.set(4, height)

    detector = ht.handDetector(maxHands=1)
    screen_width, screen_height = autopy.screen.size()

    while True:
        success, img = cap.read()
        img = detector.findHands(img)
        lmlist, bbox = detector.findPosition(img)

        if len(lmlist) != 0:
            x1, y1 = lmlist[8][1:]
            x2, y2 = lmlist[12][1:]

            fingers = detector.fingersUp()
            cv2.rectangle(img, (frameR, frameR), (width - frameR, height - frameR), (255, 0, 255), 2)

            if fingers[1] == 1 and fingers[2] == 0:
                x3 = np.interp(x1, (frameR, width - frameR), (0, screen_width))
                y3 = np.interp(y1, (frameR, height - frameR), (0, screen_height))

                curr_x = prev_x + (x3 - prev_x) / smoothening
                curr_y = prev_y + (y3 - prev_y) / smoothening

                autopy.mouse.move(screen_width - curr_x, curr_y)
                cv2.circle(img, (x1, y1), 7, (255, 0, 255), cv2.FILLED)
                prev_x, prev_y = curr_x, curr_y

            if fingers[1] == 1 and fingers[2] == 1:
                length, img, lineInfo = detector.findDistance(8, 12, img)

                if length < 40:
                    cv2.circle(img, (lineInfo[4], lineInfo[5]), 15, (0, 255, 0), cv2.FILLED)
                    autopy.mouse.click()

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, str(int(fps)), (20, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
        cv2.imshow("Image", img)
        cv2.waitKey(1)
