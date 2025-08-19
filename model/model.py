# https://github.com/PRamoneda/rubricnet/blob/master/rubricnet/rubricnet.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class RubricNet(nn.Module):
    """
    "Towards Explainable and Interpretable Musical Difficulty Estimation: 
    A parameter-efficient approach" 논문에 제안된 RubricNet 모델의 PyTorch 구현.
    """
    def __init__(self, num_features=18, performance_top=9, threshold=0.5, dropout_rate=0.5):
        """
        모델의 구성 요소를 초기화합니다.
        Args:
            num_features (int): 입력으로 사용될 음악적 특징의 수.
        """
        super(RubricNet, self).__init__()
        # num_features: 입력으로 사용될 음악적 특징의 수
        self.num_features = num_features
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
        # Sigmoid 활성화 함수
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, training=True):
        """
        모델의 순전파 연산을 정의합니다.
        Args:
            x (torch.Tensor): 모델의 입력 텐서. 
                               크기는 (batch_size, num_features)여야 합니다.        
        Returns:
            torch.Tensor: 각 특징에 대한 최종 난이도 기여도 점수. 
                          크기는 (batch_size, num_features)입니다.
        """
        # (batch_size, num_features) 크기의 입력을 받았다고 가정

        if self.training:
            x = F.dropout(x, p=self.dropout_rate)  # 드롭아웃 적용

        scores = [torch.tanh(layer(x[:, idx].unsqueeze(-1))) for idx, layer in enumerate(self.descriptor_layers)]
        score_summed = torch.sum(torch.stack(scores), dim=0)
        logits = self.final_layer(score_summed)  # (batch_size, num_features, 1)
        if self.training:
            return logits
        
        probabilities = self.sigmoid(logits)  # Sigmoid 활성화 함수 적용
        return probabilities

    def get_descriptor_scores(self, descriptors):
        if self.training:
            descriptors = F.dropout(descriptors, p=self.dropout)
        # Process each descriptor through its layer, apply tanh, and scale output
        scores = [torch.tanh(layer(descriptors[:, idx].unsqueeze(-1))).detach().cpu().squeeze() for idx, layer in
                  enumerate(self.descriptor_layers)]
        return scores