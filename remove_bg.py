import cv2
import numpy as np

# 读取输入图像
image_path = 'Mask.jpg'  # 替换成你的图片路径
img = cv2.imread(image_path)

# 确保图像正确加载
if img is None:
    raise ValueError("无法加载图像，请检查路径是否正确！")

# 复制原图，防止修改原始数据
img_copy = img.copy()

# 创建掩码 (0=背景, 1=前景)
mask_gc = np.zeros(img.shape[:2], np.uint8)

# 设定手动前景和背景区域（避免 grabCut 失败）
rect = (10, 10, img.shape[1] - 20, img.shape[0] - 20)  # (x, y, w, h)

# 创建背景和前景模型
bgd_model = np.zeros((1, 65), np.float64)
fgd_model = np.zeros((1, 65), np.float64)

# 第一次运行 grabCut，使用矩形方式初始化
cv2.grabCut(img_copy, mask_gc, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)

# 调整掩码，将可能的前景和背景进行优化
mask_gc_final = np.where((mask_gc == 2) | (mask_gc == 0), 0, 255).astype(np.uint8)

# 应用掩码进行去背
result = cv2.bitwise_and(img_copy, img_copy, mask=mask_gc_final)

# 显示结果
cv2.imshow('Result', result)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 保存去背结果
cv2.imwrite('segmented_output.png', result)
print("去背图片已保存： segmented_output.png")