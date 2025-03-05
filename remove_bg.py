import numpy as np
import cv2
from tflite_runtime.interpreter import Interpreter

def load_image(image_path):
    img = cv2.imread(image_path)
    original_img = img.copy()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (257, 257))
    img = img.astype(np.float32)
    img /= 255.0  # 确保数据在 0 到 1 的范围内
    img = np.expand_dims(img, 0)  # 增加批次维度
    return img, original_img


def load_model(model_path):
    interpreter = Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter


def run_inference(interpreter, image):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_shape = input_details[0]['shape']

    # 检查输入数据是否匹配模型的期望输入尺寸
    if image.shape != tuple(input_shape):
        raise ValueError(f"Input dimensions mismatch, expected {tuple(input_shape)}, got {image.shape}")

    interpreter.set_tensor(input_details[0]['index'], image)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    return output_data[0]

def display_and_save_image(original_img, mask):
    mask = mask[:,:,0]  # 选择我们关注的类别通道
    mask = cv2.resize(mask, (original_img.shape[1], original_img.shape[0]), interpolation=cv2.INTER_NEAREST)
    mask = (mask > 0.5).astype(np.uint8)  # 阈值处理，确保遮罩为二值图像

    segmented_image = (original_img * mask[..., None]).astype(np.uint8)  # 应用遮罩
    cv2.imwrite('segmented_image.jpg', segmented_image)
    cv2.imshow('Segmented Image', segmented_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main(image_path, model_path):
    interpreter = load_model(model_path)
    image, original_img = load_image(image_path)  # 正确加载图像
    segmentation_mask = run_inference(interpreter, image)
    mask = (segmentation_mask > 0.5).astype(np.uint8)  # 阈值处理生成遮罩
    display_and_save_image(original_img, mask)  # 传递原始图像数据而非路径

if __name__ == '__main__':
    main('Mask.jpg', 'deeplabv3.tflite')