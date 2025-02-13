import cv2
import numpy as np

def merge_foreground_background(foreground_path, mask_path, background_path):
    # 讀取前景圖片
    foreground = cv2.imread(foreground_path)
    # 讀取遮罩圖片
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    # 讀取背景圖片
    background = cv2.imread(background_path)
    # 調整背景圖片尺寸以匹配前景圖片
    background = cv2.resize(background, (foreground.shape[1], foreground.shape[0]))

    # 用遮罩合併前景和背景
    fg_img = cv2.bitwise_and(foreground, foreground, mask=mask)
    mask_inv = cv2.bitwise_not(mask)
    bg_img = cv2.bitwise_and(background, background, mask=mask_inv)

    # 合併前景和背景
    final_image = cv2.add(fg_img, bg_img)
    output_path = 'final_image.jpg'
    cv2.imwrite(output_path, final_image)
    print(f"Final image with background saved as {output_path}")

if __name__ == '__main__':
    merge_foreground_background('segmented_image.jpg', 'mask_image.jpg', 'your_background.jpg')