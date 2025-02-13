import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import cv2

# 函數：加載和預處理圖像
def load_and_preprocess_image(image_path):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [513, 513])
    img = img / 255.0  # 正規化到 [0,1] 範圍
    img = tf.expand_dims(img, 0)  # 增加批次維度
    return img

# 函數：創建遮罩
def create_mask(pred_mask):
    pred_mask = tf.argmax(pred_mask, axis=-1)
    pred_mask = pred_mask[..., tf.newaxis][0]  # 移除批次維度
    return (pred_mask * 255).numpy().astype(np.uint8)

# 函數：保存遮罩圖像
def save_mask_image(mask, save_path='mask_image.jpg'):
    cv2.imwrite(save_path, mask)
    print(f"Mask image saved as {save_path}")

# 函數：顯示並保存圖像
def display_and_save_image(image_path, mask):
    image = cv2.imread(image_path)
    masked_image = cv2.bitwise_and(image, image, mask=mask)
    output_path = 'segmented_image.jpg'  # 定義輸出文件的名稱和路徑
    cv2.imwrite(output_path, masked_image)  # 保存文件
    cv2.imshow('Segmented Image', masked_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# 主程序
def main(image_path):
    model_url = 'https://tfhub.dev/google/deepLabV3/1'
    model = hub.load(model_url)

    print("Model loaded successfully.")

    image = load_and_preprocess_image(image_path)
    print("Image loaded and preprocessed.")

    result = model(image)
    print("Segmentation completed.")

    mask = create_mask(result)
    print("Mask created.")

    # 保存遮罩圖片
    save_mask_image(mask)

    # 顯示並保存去背結果
    display_and_save_image(image_path, mask)

if __name__ == '__main__':
    main('path_to_your_image.jpg')