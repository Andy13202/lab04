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

def display_and_save_image(original_img, mask):
    # 选择第一个通道作为遮罩
    mask = mask[:, :, 0]  # 假设第一个通道是我们感兴趣的类别
    mask = cv2.resize(mask, (original_img.shape[1], original_img.shape[0]), interpolation=cv2.INTER_NEAREST)
    mask = (mask > 0.5).astype(np.uint8) * 255  # 确保遮罩为二进制形式（0或255）

    print("Mask dtype:", mask.dtype)  # 打印遮罩数据类型
    print("Mask shape:", mask.shape)  # 打印遮罩尺寸
    print("Original image dtype:", original_img.dtype)  # 打印原始图像数据类型
    print("Original image shape:", original_img.shape)  # 打印原始图像尺寸

    segmented_image = cv2.bitwise_and(original_img, original_img, mask=mask)
    cv2.imwrite('segmented_image.jpg', segmented_image)
    cv2.imshow('Segmented Image', segmented_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main(image_path, model_path):
    interpreter = load_model(model_path)
    image, original_img = load_image(image_path)
    segmentation_mask = run_inference(interpreter, image)
    display_and_save_image(original_img, segmentation_mask)

if __name__ == '__main__':
    main('Mask.jpg', 'deeplabv3.tflite')