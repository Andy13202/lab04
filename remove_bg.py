import numpy as np
import cv2
from tflite_runtime.interpreter import Interpreter

# 加载 TensorFlow Lite 模型
model_path = 'mobilenet_v2_1.0_224_quant.tflite'
interpreter = Interpreter(model_path=model_path)
interpreter.allocate_tensors()

# 获取模型的输入和输出张量信息
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# 读取并预处理输入图像
input_image = cv2.imread('input_image.jpg')
input_image = cv2.resize(input_image, (224, 224))
input_image = np.expand_dims(input_image, axis=0).astype(np.float32)

# 将图像输入模型并进行推理
interpreter.set_tensor(input_details[0]['index'], input_image)
interpreter.invoke()

# 获取模型输出并处理
output_data = interpreter.get_tensor(output_details[0]['index'])
segmentation_mask = np.argmax(output_data, axis=-1).squeeze()

# 将分割掩码调整为原始图像大小
segmentation_mask = cv2.resize(segmentation_mask.astype(np.uint8), (input_image.shape[2], input_image.shape[1]))

# 应用掩码进行人物去背
result_image = cv2.bitwise_and(input_image[0], input_image[0], mask=segmentation_mask)

# 保存结果图像
cv2.imwrite('output_image.png', result_image)