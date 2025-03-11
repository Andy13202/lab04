import cv2
import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import interact, fixed

def edge_detection(image_path, method, ksize=3, threshold1=100, threshold2=200):
    # 讀取圖片
    img = cv2.imread(image_path)
    if img is None:
        print(f"無法讀取圖片：{image_path}")
        return

    # 轉換為灰階
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 邊緣檢測
    if method == 'Sobel':
        grad_x = cv2.Sobel(gray, cv2.CV_16S, 1, 0, ksize=ksize)
        grad_y = cv2.Sobel(gray, cv2.CV_16S, 0, 1, ksize=ksize)
        abs_grad_x = cv2.convertScaleAbs(grad_x)
        abs_grad_y = cv2.convertScaleAbs(grad_y)
        edge_detected = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
    elif method == 'Scharr':
        grad_x = cv2.Scharr(gray, cv2.CV_16S, 1, 0)
        grad_y = cv2.Scharr(gray, cv2.CV_16S, 0, 1)
        abs_grad_x = cv2.convertScaleAbs(grad_x)
        abs_grad_y = cv2.convertScaleAbs(grad_y)
        edge_detected = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
    elif method == 'Laplacian':
        laplacian = cv2.Laplacian(gray, cv2.CV_16S, ksize=ksize)
        edge_detected = cv2.convertScaleAbs(laplacian)
    elif method == 'Canny':
        edge_detected = cv2.Canny(gray, threshold1, threshold2)
    else:
        print(f"未知的方法：{method}")
        return

    # 顯示結果
    plt.figure(figsize=(10, 5))
    plt.subplot(121), plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title('原始圖片'), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(edge_detected, cmap='gray')
    plt.title(f'{method} 邊緣檢測'), plt.xticks([]), plt.yticks([])
    plt.show()

# 互動式介面
interact(edge_detection,
         image_path=fixed('請輸入圖片路徑'),
         method=['Sobel', 'Scharr', 'Laplacian', 'Canny'],
         ksize=(1, 31, 2),
         threshold1=(0, 255),
         threshold2=(0, 255))