import torch
import torch.nn as nn

class RubricNet(nn.Module):
    """
    "Towards Explainable and Interpretable Musical Difficulty Estimation: 
    A parameter-efficient approach" 논문에 제안된 RubricNet 모델의 PyTorch 구현.
    """
    def __init__(self, num_features=18):
        """
        모델의 구성 요소를 초기화합니다.
        Args:
            num_features (int): 입력으로 사용될 음악적 특징의 수.
        """
        super(RubricNet, self).__init__()
        self.num_features = num_features

        # 1. 각 특징을 독립적으로 처리하는 18개의 선형 레이어(단일 뉴런)
        self.feature_assessors = nn.ModuleList([nn.Linear(1, 1) for _ in range(num_features)])
        # Tanh 활성화 함수
        self.tanh = nn.Tanh()

        # 2. 종합 난이도 결정 (Final Difficulty Estimation)
        # 논문에서는 개별 평가 점수를 합산한 후, 다시 N개의 독립적인 선형레이어를 통과시켜 시그모이드를 적용합니다.
        # 여기서 Threshold (0.5) 이상 중, 가장 높은 점수를 선택합니다.
        self.difficulty_estimator = nn.ModuleList([nn.Linear(1, 1) for _ in range(num_features)])
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
        # (batch_size, 18) 크기의 입력을 받았다고 가정
        
        # 각 특징에 대한 최종 점수를 저장할 리스트
        final_scores = []

        for i in range(self.num_features):
            feature_input = x[:, i].unsqueeze(1)
            # 1단계: 개별 특징 평가 (선형 레이어 + Tanh)
            assessed_feature = self.tanh(self.feature_assessors[i](feature_input))
            # 2단계: 난이도 추정 (선형 레이어 + Sigmoid)
            difficulty_score = self.sigmoid(self.difficulty_estimator[i](assessed_feature))
            final_scores.append(difficulty_score)
            
        # final_scores 리스트를 하나의 텐서로 결합합니다.
        output = torch.cat(final_scores, dim=1)
        
        return output
    
    def run_on_batch(self, batch):
        """
        배치 단위로 모델을 실행합니다.
        Args:
            batch (torch.Tensor): 모델에 입력될 배치 데이터. 
                                  크기는 (batch_size, num_features)여야 합니다.
        Returns:
            torch.Tensor: 각 특징에 대한 최종 난이도 기여도 점수. 
                          크기는 (batch_size, num_features)입니다.
        """
        return self.forward(batch) # TODO 데이터셋 만들고 채워넣기