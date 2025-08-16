import torch
import torch.nn as nn

class RubricNet(nn.Module):
    """
    "Towards Explainable and Interpretable Musical Difficulty Estimation: 
    A parameter-efficient approach" 논문에 제안된 RubricNet 모델의 PyTorch 구현.
    """
    def __init__(self, num_features=18, performance_top=9, threshold=0.5):
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

        # 1. 각 특징을 독립적으로 처리하는 선형 레이어(단일 뉴런)
        self.feature_assessors = nn.Conv1d(
            in_channels=num_features,
            out_channels=num_features,
            kernel_size=1,  # 1x1 컨볼루션을 사용하여 각 특징을 독립적으로 처리
            bias=True,
            dtype=torch.float32,  # dtype을 float32로 설정
        )
        # Tanh 활성화 함수
        self.tanh = nn.Tanh()

        # 2. 종합 난이도 결정 (Final Difficulty Estimation)
        # 논문에서는 개별 평가 점수를 합산한 후, 다시 N개의 독립적인 선형레이어를 통과시켜 시그모이드를 적용합니다.
        # 여기서 Threshold (0.5) 이상 중, 가장 높은 점수를 선택합니다.
        self.difficulty_estimator = nn.Linear(
            in_features=1,
            out_features=performance_top,
            bias=True,
        )
        # Sigmoid 활성화 함수
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
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
        
        x = x.unsqueeze(-1)  # (batch_size, num_features, 1)로 변환
        x = self.feature_assessors(x)  # (batch_size, num_features, 1)
        x = self.tanh(x)  # Tanh 활성화 적용
        x = torch.sum(x, dim=1)  # 각 특징에 대한 점수 합산 (batch_size, 1)
            
        output = self.difficulty_estimator(x)  # (batch_size, performance_top)
        
        # x = self.sigmoid(x)  # Sigmoid 활성화 적용
        
        # Threshold 이상인 점수 중 최대값을 선택
        # output = torch.where(
        #     (m := (x >= self.threshold)),
        #     torch.arange(x.size(1), device=x.device).expand_as(x),
        #     torch.full_like(x, 0, dtype=torch.float32)
        # ).amax(dim=1) + 1
        
        return output