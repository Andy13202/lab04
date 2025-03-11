import cv2
import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import interact, fixed

def canny_edge_detection(image_path, low_threshold, high_threshold):
    # 讀取圖片
    image = cv2.imread(image_path)
    if image is None:
        print(f"無法讀取圖片：{image_path}")
        return
    
    # 轉換為灰階
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 使用高斯模糊降低噪聲
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 1.4)
    
    # Canny 邊緣檢測
    edges = cv2.Canny(blurred_image, low_threshold, high_threshold)
    
    # 顯示結果
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title('原始圖片')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(edges, cmap='gray')
    plt.title(f'Canny 邊緣檢測\n低閾值: {low_threshold}, 高閾值: {high_threshold}')
    plt.axis('off')
    
    plt.show()

# 互動式介面
interact(canny_edge_detection,
         image_path=fixed('請輸入圖片路徑'),
         low_threshold=(0, 255, 1),
         high_threshold=(0, 255, 1))
