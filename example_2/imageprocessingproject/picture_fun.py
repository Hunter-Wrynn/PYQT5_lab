import math

import cv2
import numpy as np

####################灰度图和二值化########################################

def gray_picture(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return img_gray

def erzhihua(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rst = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    return rst[1]

##################加噪###########################################################
def add_noisy(image, n=10000):    #椒盐彩色
    result = image.copy()
    w, h = image.shape[:2]
    for i in range(n):
        # 宽和高的范围内生成一个随机值，模拟表x,y坐标
        x = np.random.randint(1, w)
        y = np.random.randint(1, h)
        if np.random.randint(0, 2) == 0:
            # 生成白色噪声（盐噪声）
            result[x, y] = 0
        else:
            # 生成黑色噪声（椒噪声）
            result[x, y] = 255
    return result



####################彩色图像添加高斯噪声#######################################
def add_noise(image,mean,var):
    img = np.array(image / 255, dtype=float)
    noise = np.random.normal(mean, var ** 0.5, img.shape)
    out_img = img + noise
    if out_img.min() < 0:
        low_clip = -1
    else:
        low_clip = 0
        out_img = np.clip(out_img, low_clip, 1.0)
        out_img = out_img * 255
    return out_img


################图像锐化#####################
def lap_9(image):                                               #拉普拉斯变化
    lap_9 = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    # 拉普拉斯9的锐化
    image = cv2.filter2D(image, cv2.CV_8U, lap_9)
    return image

def gama_transfer(img,power1=1.5):                              #伽马变化
    if len(img.shape) == 3:
         img= cv2.cvtColor(img,cv2.CV_8U)
    img = 255*np.power(img/255,power1)
    img = np.around(img)
    img[img>255] = 255
    out_img = img.astype(np.uint8)
    return out_img


def get_imghist(img):                                             #直方图均衡
    # 判断图像是否为三通道；
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.CV_8U)
    # 无 Mask，256个bins,取值范围为[0,255]
    hist = cv2.calcHist([img], [0], None, [256], [0, 255])
    return hist


def cal_equalhist(img):
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.CV_8U)
    h, w = img.shape[:2]
    grathist = get_imghist(img)

    zerosumMoment = np.zeros([256], np.uint32)
    for p in range(256):
        if p == 0:
            zerosumMoment[p] = grathist[0]
        else:
            zerosumMoment[p] = zerosumMoment[p - 1] + grathist[p]

    output_q = np.zeros([256], np.uint8)
    cofficient = 256.0 / (h * w)
    for p in range(256):
        q = cofficient * float(zerosumMoment[p]) - 1
        if q >= 0:
            output_q[p] = math.floor(q)
        else:
            output_q[p] = 0

    equalhistimage = np.zeros(img.shape, np.uint8)
    for i in range(h):
        for j in range(w):
            equalhistimage[i][j] = output_q[img[i][j]]

    return equalhistimage

################图像去噪####################################################
def boxFilterfun(image):       #方波滤波
    image=cv2.boxFilter(image,-1,(2,2),normalize=0)
    return image



def medianBlurfun(image):      #中值滤波
    image=cv2.medianBlur(image,3)
    return image



def bilateralFilterfun(image):   #双边滤波
    image=cv2.bilateralFilter(image,25,100,100)
    return image


def GaussianBlurfun(image):      #高斯滤波
    image=cv2.GaussianBlur(image,(5,5),0,0)
    return image


def blurfun(image):              #均值滤波
    image==cv2.blur(image,(5,5))
    return image


###############图像翻转####################################################
def flipfun(image,x):    #图像  水平翻转:0   垂直翻转:1  沿xy轴翻转:-1
    image = cv2.flip(image,x)
    return image


##############按位取反###################################################
def bitwise_notfun(image):
    image = cv2.bitwise_not(image)
    return image


##############轮廓检测####################################################
def morphologyExfun(image):
    kernel = np.ones((3, 3), dtype=np.uint8)
    image_gradient = cv2.morphologyEx(image, cv2.MORPH_GRADIENT, kernel)
    return image_gradient


#############sift检测#######################################################
def sift_fun(image):
    sift = cv2.SIFT_create()
    kps = sift.detect(image)
    image_sift = cv2.drawKeypoints(image, kps, None, -1, cv2.DrawMatchesFlags_DEFAULT)
    return image_sift


##################平均值池化#################################################
def average_poolingfun(img, G=4):
    out = img.copy()
    H, W, C = img.shape
    Nh = int(H / G)
    Nw = int(W / G)
    for y in range(Nh):
        for x in range(Nw):
            for c in range(C):
                out[G * y:G * (y + 1), G * x:G * (x + 1), c] = np.mean(
                    out[G * y:G * (y + 1), G * x:G * (x + 1), c]).astype(np.int)
    return out

##################################修复####
def xiufu(image):
    _, mask1 = cv2.threshold(image, 245, 255, cv2.THRESH_BINARY)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask1 = cv2.dilate(mask1, k)
    result1 = cv2.inpaint(image, mask1[:, :, -1], 5, cv2.INPAINT_NS)
    return result1

if __name__ == '__main__':
    image = cv2.imread("inpaint2.jpg")
    cv2.imshow(' ',image)
    newimage=xiufu(image)
    cv2.imshow(' ',newimage)
    cv2.waitKey(0)

