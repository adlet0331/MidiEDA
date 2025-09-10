# https://github.com/PRamoneda/rubricnet/blob/master/rubricnet/rubricnet.py

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .evaluation import inference_from_pred

class RubricNet(nn.Module):
    """
    "Towards Explainable and Interpretable Musical Difficulty Estimation: 
    A parameter-efficient approach" 논문에 제안된 RubricNet 모델의 PyTorch 구현.
    """
    def __init__(self, num_features=14, scaler_parameter=None, performance_top=9, threshold=0.5, dropout_rate=0.5):
        """
        모델의 구성 요소를 초기화합니다.
        Args:
            num_features (int): 입력으로 사용될 음악적 특징의 수.
        """
        super(RubricNet, self).__init__()
        self.num_features = num_features
        if scaler_parameter is not None and len(scaler_parameter[0]) == num_features and len(scaler_parameter[1]) == num_features:
            # num_features: 입력으로 사용될 음악적 특징의 수
            self.register_buffer('scaler_means', torch.tensor(scaler_parameter[0], dtype=torch.float32))
            self.register_buffer('scaler_stds', torch.tensor(scaler_parameter[1], dtype=torch.float32))
        else:
            self.register_buffer('scaler_means', torch.zeros(num_features, dtype=torch.float32))
            self.register_buffer('scaler_stds', torch.ones(num_features, dtype=torch.float32))

        self.performance_top = performance_top
        self.threshold = threshold
        self.dropout_rate = dropout_rate

        # 1. 각 특징을 독립적으로 처리하는 선형 레이어(단일 뉴런)
        self.descriptor_layers = nn.ModuleList([
            nn.Linear(1, 1) for _ in range(num_features)
        ])

        # 2. 종합 난이도 결정 (Final Difficulty Estimation)
        # 논문에서는 개별 평가 점수를 합산한 후, 다시 N개의 독립적인 선형레이어를 통과시켜 시그모이드를 적용합니다.
        # 여기서 Threshold (0.5) 이상 중, 가장 높은 점수를 선택합니다.
        self.final_layer = nn.Linear(
            in_features=1,
            out_features=performance_top,
            bias=True,
        )
        self.sigmoid = nn.Sigmoid()

    def set_scaler_parameter_manually(self, means, stds):
        if len(means) == self.num_features and len(stds) == self.num_features:
            self.scaler_means = torch.tensor(means, dtype=torch.float32)
            self.scaler_stds = torch.tensor(stds, dtype=torch.float32)
        else:
            raise ValueError("Length of means and stds must match num_features")

    def forward(self, x):
        # (batch_size, num_features) 크기의 입력을 받았다고 가정
        x = torch.tensor(x, dtype=torch.float32)
        if x.ndim == 1:
            x = x.unsqueeze(0)
        x = (x - self.scaler_means) / self.scaler_stds

        if self.training:
            x = F.dropout(x, p=self.dropout_rate)  # 드롭아웃 적용

        scores = [torch.tanh(layer(x[:, idx].unsqueeze(-1))) for idx, layer in enumerate(self.descriptor_layers)]
        score_summed = torch.sum(torch.stack(scores), dim=0)
        logits = self.final_layer(score_summed)  # (batch_size, num_features, 1)
        if self.training:
            return logits
        
        probabilities = self.sigmoid(logits)  # Sigmoid 활성화 함수 적용
        return probabilities
    
    def get_descriptive_scores(self, x):
        if x.ndim == 1:
            x = x.unsqueeze(0)
        x = (x - self.scaler_means) / self.scaler_stds
        scores = [torch.tanh(layer(x[:, idx].unsqueeze(-1))).detach().numpy() for idx, layer in enumerate(self.descriptor_layers)]
        return scores
    
    def predict(self, x):
        self.eval()
        output = self.forward(x)
        # print(output)
        return inference_from_pred(output, threshold=self.threshold)

    def get_scaler_infos(self):
        return self.scaler_means, self.scaler_stds