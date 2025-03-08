import numpy as np
import cv2

# 读取输入图像
image_path = 'Mask.jpg'  # 替换为你的图片路径
img = cv2.imread(image_path)

# 确保图像正确加载
if img is None:
    raise ValueError("无法加载图像，请检查路径是否正确！")

# 复制图像，避免直接修改原图
img_copy = img.copy()

# 创建窗口和滑动条
cv2.namedWindow('Adjust Parameters')

# 颜色范围参数（仅保留 H_min, H_max）
def nothing(x):
    pass

cv2.createTrackbar('H_min', 'Adjust Parameters', 0, 180, nothing)
cv2.createTrackbar('H_max', 'Adjust Parameters', 180, 180, nothing)

# 饱和度阈值
cv2.createTrackbar('S_threshold', 'Adjust Parameters', 50, 255, nothing)

# 形态学操作强度
cv2.createTrackbar('Morph_Size', 'Adjust Parameters', 1, 20, nothing)

while True:
    # 获取滑动条的值
    h_min = cv2.getTrackbarPos('H_min', 'Adjust Parameters')
    h_max = cv2.getTrackbarPos('H_max', 'Adjust Parameters')
    s_threshold = cv2.getTrackbarPos('S_threshold', 'Adjust Parameters')
    morph_size = cv2.getTrackbarPos('Morph_Size', 'Adjust Parameters')

    # 复制原始图像
    img_copy = img.copy()

    # 转换为 HSV 颜色空间
    hsv = cv2.cvtColor(img_copy, cv2.COLOR_BGR2HSV)

    # 颜色阈值分割（仅 H 通道）
    lower_bound = np.array([h_min, s_threshold, 50])
    upper_bound = np.array([h_max, 255, 255])
    mask = cv2.inRange(hsv, lower_bound, upper_bound)

    # 反转掩码：保留人物部分
    mask_inv = cv2.bitwise_not(mask)

    # 形态学操作（去除噪点）
    kernel = np.ones((morph_size, morph_size), np.uint8)
    mask_inv = cv2.morphologyEx(mask_inv, cv2.MORPH_CLOSE, kernel)

    # 应用掩码进行去背
    result = cv2.bitwise_and(img_copy, img_copy, mask=mask_inv)

    # 显示实时调节的结果
    cv2.imshow('Result', result)

    # 按 's' 保存图片，按 'q' 退出
    key = cv2.waitKey(1) & 0xFF
    if key == ord('s'):
        cv2.imwrite('segmented_output.png', result)
        print("去背图片已保存： segmented_output.png")
    elif key == ord('q'):
        break

cv2.destroyAllWindows()
