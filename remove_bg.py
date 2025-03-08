import numpy as np
import cv2

# 读取输入图像
image_path = 'Mask.jpg'  # 替换为你的图片路径
img = cv2.imread(image_path)

# 确保图像正确加载
if img is None:
    raise ValueError("无法加载图像，请检查路径是否正确！")

# 加载 DNN 人脸检测模型（Caffe 版本）
prototxt_path = "deploy.prototxt"
model_path = "res10_300x300_ssd_iter_140000.caffemodel"
face_net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

# 读取图像并转换为 DNN 输入格式
(h, w) = img.shape[:2]
blob = cv2.dnn.blobFromImage(img, 1.0, (300, 300), (104.0, 177.0, 123.0))

# 运行 DNN 模型进行人脸检测
face_net.setInput(blob)
detections = face_net.forward()

# 解析检测结果
rect = None
for i in range(detections.shape[2]):
    confidence = detections[0, 0, i, 2]
    if confidence > 0.5:  # 置信度超过 50%
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (x, y, x2, y2) = box.astype("int")
        rect = (max(0, x - 50), max(0, y - 50), min(w, x2 - x + 100), min(h, y2 - y + 200))
        break  # 只取第一个检测到的人脸

# 如果未检测到人脸，使用默认矩形框
if rect is None:
    print("未检测到人脸，使用默认矩形框进行 GrabCut")
    rect = (10, 10, img.shape[1] - 20, img.shape[0] - 20)

# 创建 GrabCut 掩码
mask_gc = np.zeros(img.shape[:2], np.uint8)

# 创建背景和前景模型
bgd_model = np.zeros((1, 65), np.float64)
fgd_model = np.zeros((1, 65), np.float64)

# 应用 GrabCut 进行人物去背
cv2.grabCut(img, mask_gc, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)

# 处理掩码
mask_gc_final = np.where((mask_gc == 2) | (mask_gc == 0), 0, 255).astype(np.uint8)

# 应用掩码进行去背
result = cv2.bitwise_and(img, img, mask=mask_gc_final)

# 显示去背结果
cv2.imshow('Result', result)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 保存去背结果
cv2.imwrite('segmented_output.png', result)
print("去背图片已保存： segmented_output.png")
