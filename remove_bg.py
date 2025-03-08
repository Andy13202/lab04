import numpy as np
import cv2

# 读取输入图像
image_path = 'Mask.jpg'  # 替换为你的图片路径
img = cv2.imread(image_path)

# 确保图像正确加载
if img is None:
    raise ValueError("无法加载图像，请检查路径是否正确！")

# 加载 OpenCV 预训练的人脸检测器（Haar 级联分类器）
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# 转换为灰度图
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 检测人脸（用于自动确定前景区域）
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

# 确保检测到人脸
if len(faces) == 0:
    print("未检测到人脸，使用默认矩形框进行 GrabCut")
    rect = (10, 10, img.shape[1] - 20, img.shape[0] - 20)  # 默认框
else:
    # 获取检测到的最大人脸区域
    x, y, w, h = faces[0]
    rect = (x - 50, y - 50, w + 100, h + 200)  # 扩展框，确保包含整个上半身

# 创建掩码
mask_gc = np.zeros(img.shape[:2], np.uint8)

# 创建背景和前景模型
bgd_model = np.zeros((1, 65), np.float64)
fgd_model = np.zeros((1, 65), np.float64)

# 使用 GrabCut 进行人物分割
cv2.grabCut(img, mask_gc, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)

# 处理掩码
mask_gc_final = np.where((mask_gc == 2) | (mask_gc == 0), 0, 255).astype(np.uint8)

# 应用掩码进行去背
result = cv2.bitwise_and(img, img, mask=mask_gc_final)

# 显示结果
cv2.imshow('Result', result)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 保存去背结果
cv2.imwrite('segmented_output.png', result)
print("去背图片已保存： segmented_output.png")
