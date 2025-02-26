import numpy as np
import cv2
from tflite_runtime.interpreter import Interpreter

def load_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (513, 513))  # 根据模型要求调整图像尺寸
    img = img.astype(np.float32)
    img = np.expand_dims(img, 0)  # 增加批次维度
    return img

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

def display_and_save_image(original_image_path, mask):
    image = cv2.imread(original_image_path)
    # 将掩模转换为三通道，以便与原图结合
    mask = np.tile(mask, (1, 1, 3))
    segmented_image = (image * mask).astype(np.uint8)
    cv2.imwrite('segmented_image.jpg', segmented_image)
    cv2.imshow('Segmented Image', segmented_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main(image_path, model_path):
    interpreter = load_model(model_path)
    image = load_image(image_path)
    segmentation_mask = run_inference(interpreter, image)
    mask = (segmentation_mask > 0.5).astype(np.uint8)  # 阈值处理生成遮罩
    display_and_save_image(image_path, mask)

if __name__ == '__main__':
    main('path_to_your_image.jpg', '/path/to/downloaded/deeplabv3.tflite')