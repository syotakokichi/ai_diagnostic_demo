# src/model.py

import segmentation_models_pytorch as smp

def get_model():
    num_classes = 1  # バイナリセグメンテーション

    model = smp.Unet(
        encoder_name='resnet34',        # エンコーダの種類
        encoder_weights='imagenet',     # 学習済み重みを使用
        in_channels=1,                  # 入力チャンネル数（グレースケール画像の場合は1）
        classes=num_classes,            # 出力クラス数
        activation=None                 # 出力にシグモイドを適用するため、Noneに設定
    )

    return model