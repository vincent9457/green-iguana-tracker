import cv2
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

# 載入模型和類別標籤
model_path = r"C:\Users\USER\Downloads\converted_keras"  # 替換為你的 keras_model.h5 路徑
labels_path = r"C:\Users\USER\Downloads\converted_keras"  # 替換為你的 labels.txt 路徑

model = load_model(model_path)

# 讀取標籤
with open(labels_path, "r") as f:
    class_names = [line.strip() for line in f.readlines()]

# 處理影像的函數
def preprocess_image(image, target_size):
    """將影像調整為模型要求的格式"""
    img = Image.fromarray(image)  # 轉換為 PIL 圖像
    img = img.resize(target_size)  # 調整大小
    img_array = np.array(img) / 255.0  # 標準化到 [0, 1]
    return np.expand_dims(img_array, axis=0)

# 使用相機進行即時辨識
cap = cv2.VideoCapture(0)  # 開啟相機

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 模型需要的輸入大小（224x224）
    input_size = (224, 224)
    processed_frame = preprocess_image(frame, input_size)

    # 模型進行預測
    predictions = model.predict(processed_frame)
    class_idx = np.argmax(predictions)  # 找到最高分數的索引
    confidence = predictions[0][class_idx]  # 取得該分類的信心分數

    # 繪製辨識結果
    label = f"{class_names[class_idx]}: {confidence*100:.2f}%"
    cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # 顯示影像
    cv2.imshow("Teachable Machine Detector", frame)

    # 按 'q' 鍵退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
