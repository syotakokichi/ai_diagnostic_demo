# streamlit/app.py

import os
import streamlit as st
import torch
from PIL import Image
import numpy as np
from model import get_model
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt

# デバイスの設定
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# モデルのロード
@st.cache_resource
def load_model():
    model = get_model()
    MODEL_PATH = os.path.join(os.path.dirname(__file__), './best_model.pth')
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()
    return model

model = load_model()

# 前処理関数
def preprocess_image(image):
    transform = A.Compose([
        A.Resize(512, 512),
        A.Normalize(mean=(0.5,), std=(0.5,)),
        ToTensorV2(),
    ])
    image = np.array(image.convert('L'))
    augmented = transform(image=image)
    image = augmented['image']
    return image.unsqueeze(0)

# アプリのタイトル
st.title('歯科X線画像異常検出システム')

# 画像のアップロード
uploaded_file = st.file_uploader("画像をアップロードしてください", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # 画像の表示
    image = Image.open(uploaded_file)
    st.image(image, caption='アップロードされた画像。', use_column_width=True)

    # 前処理
    input_image = preprocess_image(image).to(device, dtype=torch.float32)

    # 推論
    with torch.no_grad():
        output = model(input_image)
        output = output.squeeze(0).squeeze(0).cpu().numpy()
        output = (output > 0.5).astype(np.uint8)

    # 結果の表示
    fig, ax = plt.subplots(1, 2, figsize=(10,5))
    ax[0].imshow(np.array(image.convert('L')), cmap='gray')
    ax[0].set_title('Original Image')
    ax[0].axis('off')

    ax[1].imshow(output, cmap='gray')
    ax[1].set_title('Predicted Mask')
    ax[1].axis('off')

    st.pyplot(fig)

    # 結果の保存
    result_image = Image.fromarray((output * 255).astype(np.uint8))
    result_image.save('predicted_mask.png')

    # ダウンロードボタンの追加
    with open('predicted_mask.png', 'rb') as file:
        btn = st.download_button(
            label="予測結果をダウンロード",
            data=file,
            file_name="predicted_mask.png",
            mime="image/png"
        )