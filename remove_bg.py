import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog

def remove_background(image_path, lower_bound, upper_bound, blur_size, smooth_factor):
    # 讀取圖片
    image = cv2.imread(image_path)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # 建立遮罩
    mask = cv2.inRange(hsv, lower_bound, upper_bound)
    mask = cv2.medianBlur(mask, blur_size)  # 平滑遮罩
    
    # 產生透明背景
    result = cv2.bitwise_and(image, image, mask=~mask)
    result = cv2.cvtColor(result, cv2.COLOR_BGR2BGRA)
    result[:, :, 3] = cv2.bitwise_not(mask) // smooth_factor  # 透明度調整
    
    return result

def choose_file():
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.png;*.jpeg")])
    if file_path:
        process_image(file_path)

def process_image(image_path):
    lower_h = int(lower_hue.get())
    lower_s = int(lower_saturation.get())
    lower_v = int(lower_value.get())
    upper_h = int(upper_hue.get())
    upper_s = int(upper_saturation.get())
    upper_v = int(upper_value.get())
    blur = int(blur_size.get())
    smooth = int(smooth_factor.get())
    
    lower_bound = np.array([lower_h, lower_s, lower_v])
    upper_bound = np.array([upper_h, upper_s, upper_v])
    
    output = remove_background(image_path, lower_bound, upper_bound, blur, smooth)
    cv2.imshow("Background Removed", output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# 建立 GUI 介面
root = tk.Tk()
root.title("去背調整工具")

tk.Label(root, text="Lower Hue").pack()
lower_hue = tk.Scale(root, from_=0, to=179, orient="horizontal")
lower_hue.pack()

tk.Label(root, text="Lower Saturation").pack()
lower_saturation = tk.Scale(root, from_=0, to=255, orient="horizontal")
lower_saturation.pack()

tk.Label(root, text="Lower Value").pack()
lower_value = tk.Scale(root, from_=0, to=255, orient="horizontal")
lower_value.pack()

tk.Label(root, text="Upper Hue").pack()
upper_hue = tk.Scale(root, from_=0, to=179, orient="horizontal")
upper_hue.pack()

tk.Label(root, text="Upper Saturation").pack()
upper_saturation = tk.Scale(root, from_=0, to=255, orient="horizontal")
upper_saturation.pack()

tk.Label(root, text="Upper Value").pack()
upper_value = tk.Scale(root, from_=0, to=255, orient="horizontal")
upper_value.pack()

tk.Label(root, text="Blur Size").pack()
blur_size = tk.Scale(root, from_=1, to=25, orient="horizontal")
blur_size.pack()

tk.Label(root, text="Smooth Factor").pack()
smooth_factor = tk.Scale(root, from_=1, to=10, orient="horizontal")
smooth_factor.pack()

tk.Button(root, text="Mask.jpg", command=choose_file).pack()
root.mainloop()
