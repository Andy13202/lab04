import numpy as np
import cv2

# 读取输入图像
image_path = 'Mask.jpg'  # 请替换为你的图片路径
img = cv2.imread(image_path)

# 确保图像正确加载
if img is None:
    raise ValueError("无法加载图像，请检查路径是否正确！")

# 复制图像，避免直接修改原图
img_copy = img.copy()

# 转换为 HSV 颜色空间
hsv = cv2.cvtColor(img_copy, cv2.COLOR_BGR2HSV)

# 自动选择颜色范围（适用于蓝色或绿色背景）
lower_bound = np.array([35, 40, 40])   # 绿色背景
upper_bound = np.array([90, 255, 255])

# 生成背景掩码
mask = cv2.inRange(hsv, lower_bound, upper_bound)

# 反转掩码：保留人物部分
mask_inv = cv2.bitwise_not(mask)

# 形态学操作（去除噪点）
kernel = np.ones((5, 5), np.uint8)
mask_inv = cv2.morphologyEx(mask_inv, cv2.MORPH_CLOSE, kernel)

# 应用掩码进行去背
result = cv2.bitwise_and(img_copy, img_copy, mask=mask_inv)

# 显示结果
cv2.imshow('Result', result)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 保存去背结果
cv2.imwrite('segmented_output.png', result)
print("去背图片已保存： segmented_output.png")
