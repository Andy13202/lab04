import numpy as np
import cv2
from tflite_runtime.interpreter import Interpreter

def load_image(image_path):
    img = cv2.imread(image_path)
    original_img = img.copy()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (257, 257))
    img = img.astype(np.float32)
    img /= 255.0  # Normalize the image to 0 to 1
    img = np.expand_dims(img, 0)  # Add batch dimension
    return img, original_img

def load_model(model_path):
    interpreter = Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter

def run_inference(interpreter, image):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    interpreter.set_tensor(input_details[0]['index'], image)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    return output_data[0]

def display_and_save_image(original_img, mask, raw_mask):
    # 只保存第一个通道，作为灰度图查看原始模型输出
    first_channel = raw_mask[:, :, 0]
    normalized_mask = first_channel * 255 / np.max(first_channel) if np.max(first_channel) > 0 else first_channel
    normalized_mask = normalized_mask.astype(np.uint8)
    cv2.imwrite('raw_mask.jpg', normalized_mask)

    # 降低阈值尝试找到合适的分割效果
    mask = cv2.resize(mask, (original_img.shape[1], original_img.shape[0]), interpolation=cv2.INTER_NEAREST)
    mask = (mask > 0.3).astype(np.uint8)
    mask = mask * 255  # 确保遮罩为二进制形式（0或255）

    # 确保遮罩是单通道 uint8 类型
    if len(mask.shape) > 2:
        mask = mask[:, :, 0]  # 只取第一个通道
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)

    cv2.imwrite('adjusted_mask.jpg', mask)

    segmented_image = cv2.bitwise_and(original_img, original_img, mask=mask)
    cv2.imwrite('segmented_image.jpg', segmented_image)
    cv2.imshow('Segmented Image', segmented_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main(image_path, model_path):
    interpreter = load_model(model_path)
    image, original_img = load_image(image_path)
    segmentation_mask = run_inference(interpreter, image)
    
    # 将原始模型输出传递给显示函数
    display_and_save_image(original_img, segmentation_mask, segmentation_mask)

if __name__ == '__main__':
    main('Mask.jpg', 'deeplabv3.tflite')