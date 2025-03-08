import numpy as np
import cv2

# 读取输入图像
image_path = 'Mask.jpg'  # 替换为你的图片路径
img = cv2.imread(image_path)

# 确保图像正确加载
if img is None:
    raise ValueError("无法加载图像，请检查路径是否正确！")

# 转换为灰度图
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 使用高斯模糊降低噪点
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# 使用 Canny 边缘检测
edges = cv2.Canny(blurred, 50, 150)

# 找到轮廓
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 创建空白掩码
mask = np.zeros_like(gray)

# 选择最大轮廓（通常是主体）
if contours:
    largest_contour = max(contours, key=cv2.contourArea)
    cv2.drawContours(mask, [largest_contour], -1, (255), thickness=cv2.FILLED)

# 形态学操作，填补轮廓内的小空洞
kernel = np.ones((5, 5), np.uint8)
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

# 应用掩码进行去背
result = cv2.bitwise_and(img, img, mask=mask)

# 显示结果
cv2.imshow('Result', result)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 保存去背结果
cv2.imwrite('segmented_output.png', result)
print("去背图片已保存： segmented_output.png")
