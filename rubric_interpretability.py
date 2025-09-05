import torch, json
from model import RubricNet
import numpy as np
import matplotlib.pyplot as plt

def get_features_data(runs_name = 'runs/p-est-250822-160040',
                      cipi_cached_path = '/Users/simhyeongju/AVAPT/data/CIPI/features/features_v1.json',
                      cipi_label_path = '/Users/simhyeongju/AVAPT/data/CIPI/index.json'):
    model_snapshot_path = f'/Users/simhyeongju/AVAPT/EDA/{runs_name}/model_snapshots/model_bestvalidation.pt'

    rubricnet = RubricNet()
    rubricnet.load_state_dict(torch.load(model_snapshot_path, weights_only=False))
    mean, std = rubricnet.get_scaler_infos()
    mean, std = mean.numpy(), std.numpy()


    cipi_features = json.load(open(cipi_cached_path, 'r'))
    features_mem = cipi_features['features_mem']
    features_name = cipi_features['features_names']
    cipi_labels = json.load(open(cipi_label_path, 'r'))

    # for i in range(len(features_name)):
    #     print(f"{features_name[i]} scaler: {mean[i]}, {std[i]}")

    label_features = [None for _ in range(9)] # [난이도][feature index]
    label_scores = [None for _ in range(9)] # [난이도][feature index]
    for key, features in features_mem.items():
        features = np.array(features)

        #label = int(cipi_labels[key]["henle"]) - 1
        label = rubricnet.predict(features).tolist()[0] - 1
        scaled_features = (features - mean) / std
        feature_scores = np.array(rubricnet.get_descriptive_scores(torch.tensor(features, dtype=torch.float32))).reshape(-1)
        if label_features[label] is None:
            label_features[label] = np.expand_dims(scaled_features, axis=0)
            label_scores[label] = np.expand_dims(feature_scores, axis=0)
        else:
            label_features[label] = np.concatenate((label_features[label], np.expand_dims(scaled_features, axis=0)), axis=0)
            label_scores[label] = np.concatenate((label_scores[label], np.expand_dims(feature_scores, axis=0)), axis=0)

        # print(key)
        # for i in range(len(features)):
        #     print(f"{features_name[i]}: {features[i]}, scaled: {scaled_features[i]}, score: {feature_scores[i] / 2 + 0.5}")

    # scaled_label_features_count = [features.shape[0] for features in label_features]
    # scaled_label_features_mean = [np.mean(features, axis=0) for features in label_features]

    return features_name, label_features, label_scores

# get_features_data()

def create_segmented_bar(widths, colormap='Greens', font_size=24):
    """
    너비가 다른 세그먼트로 구성된 수평 막대 그래프를 생성합니다.

    Args:
        widths (list of float): 각 세그먼트의 너비 리스트.
        colormap (str): 세그먼트 색상에 사용할 Matplotlib 컬러맵 이름.
        font_size (int): 세그먼트 안에 표시될 숫자 폰트 크기.
    """
    num_segments = len(widths)

    # 1. 색상 설정
    # -----------------
    # 지정된 컬러맵 가져오기
    cmap = plt.get_cmap(colormap)
    # 컬러맵의 특정 범위(밝은 색 ~ 어두운 색)에서 세그먼트 수만큼 색상 추출
    colors = cmap(np.linspace(0.2, 1.0, num_segments))

    # 2. 텍스트 색상 결정 함수
    # ---------------------------
    def get_text_color(background_color):
        """배경색의 밝기를 계산하여 적절한 텍스트 색상(검은색 또는 흰색)을 반환합니다."""
        # RGBA 색상에서 RGB 값만 사용하여 밝기(luminance) 계산
        r, g, b, _ = background_color
        luminance = 0.299 * r + 0.587 * g + 0.114 * b
        # 밝기가 0.5 이상이면 어두운 색 텍스트를, 그렇지 않으면 밝은 색 텍스트를 반환
        return 'black' if luminance > 0.5 else 'white'

    # 3. 그래프 생성
    # -----------------
    # 그래프 크기 설정 (가로로 길게)
    fig, ax = plt.subplots(figsize=(14, 2))
    fig.patch.set_facecolor('white') # 그림 전체 배경을 흰색으로

    # 각 세그먼트를 그리기 위한 시작 위치 변수
    current_left = 0

    for i, width in enumerate(widths):
        color = colors[i]
        
        # 수평 막대(세그먼트) 그리기
        ax.barh(
            y=0,                 # y 위치는 0으로 고정
            width=width,         # 현재 세그먼트의 너비
            height=1,            # 막대의 높이
            left=current_left,   # 왼쪽 시작 위치
            color=color,         # 배경색
            edgecolor='black',   # 테두리 색상
            linewidth=1.5        # 테두리 두께
        )

        # 세그먼트 중앙에 숫자 텍스트 추가
        text_x_position = current_left + width / 2
        text_color = get_text_color(color)
        
        ax.text(
            text_x_position,
            0,
            str(i + 1),
            ha='center',          # 수평 정렬: 가운데
            va='center',          # 수직 정렬: 가운데
            fontsize=font_size,
            color=text_color,
            fontweight='bold'     # 글자를 약간 굵게
        )

        # 다음 세그먼트의 시작 위치 업데이트
        current_left += width

    # 4. 그래프 스타일링
    # -------------------
    # 축, 눈금, 테두리 등 불필요한 요소 모두 제거
    # ax.axis('off')
    
    # 그래프 여백 최소화
    plt.tight_layout(pad=0)
    
    # 그래프 출력
    plt.show()

def create_descriptor_plot(y_labels, data_matrix):
    """
    Y축 레이블 리스트와 2차원 데이터를 기반으로 막대 그래프를 생성합니다.

    Args:
        y_labels (list of str): Y축에 표시될 레이블의 리스트.
        data_matrix (list of list of float or np.ndarray): 
            그래프에 표시할 2차원 데이터. 
            형태: (len(y_labels), num_data_points)
    """
    # 1. 데이터 및 스타일 설정
    # -----------------------------
    
    # 입력 데이터를 Numpy 배열로 변환하여 처리 용이성 확보
    data = np.array(data_matrix)
    data = data.transpose()
    
    # 데이터 형태를 기반으로 설명자 및 등급 수 결정
    if data.ndim != 2:
        raise ValueError("data_matrix는 반드시 2차원이어야 합니다.")
    num_descriptors, num_grades = data.shape
    print(f"설명자 수: {num_descriptors}, 등급 수: {num_grades}")

    if len(y_labels) != num_descriptors:
        raise ValueError("y_labels의 길이와 data_matrix의 행 수가 일치해야 합니다.")

    # X축 레이블 생성
    grades = np.arange(1, num_grades + 1)
    
    # 일반화된 스타일 생성 (색상과 해칭 패턴을 순환하여 사용)
    base_colors = ['#1a6429', '#97D094', '#C5E5C4']
    base_hatches = ['', '////', '----']
    styles_list = []
    for i in range(num_descriptors):
        color = base_colors[i % len(base_colors)]
        hatch = base_hatches[(i // len(base_colors)) % len(base_hatches)]
        styles_list.append({'color': color, 'hatch': hatch})

    # 2. 그래프 생성
    # -----------------

    # 그래프 크기 및 배경색 설정
    fig, ax = plt.subplots(figsize=(15, max(6, num_descriptors * 0.6)))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    # 각 설명자(descriptor)에 대해 막대 그래프 그리기
    for i in range(num_descriptors):
        # y축 위치 계산 (위쪽 레이블이 높은 y값을 갖도록)
        y_baseline = num_descriptors - 1 - i
        
        ax.bar(
            x=grades,
            height=data[i, :] * 0.8,  # 데이터 정규화 및 높이 조정
            bottom=y_baseline,
            width=0.7,
            color=styles_list[i]['color'],
            hatch=styles_list[i]['hatch'],
            edgecolor='black',
            linewidth=1.0,
            align='center'
        )

    # 3. 그래프 스타일링
    # -------------------

    # 축 설정
    ax.set_xticks(grades)
    ax.set_xlabel("Grades (X' axis)", fontsize=14)
    ax.set_xlim(0.5, num_grades + 0.5)
    ax.set_ylim(0, num_descriptors + 1)

    # Y축 레이블 설정
    tick_positions = np.arange(num_descriptors + 1)
    ax.set_yticks(tick_positions)
    ax.set_yticklabels(reversed([" "] + y_labels), fontsize=12)
    ax.tick_params(axis='y', length=0)

    # 수평 그리드 라인 추가
    ax.grid(axis='y', which='major', linestyle='-', color='gray')
    ax.set_axisbelow(True)

    # 테두리(spines) 제거
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)

    # 레이아웃 조정
    plt.tight_layout()
    plt.show()

def plot_descriptor_scores_vs_values(scores, values, columns):
    """
    Plots line graphs for multiple descriptors with each descriptor's values on the x-axis
    and the corresponding scores on the y-axis. Each line represents one descriptor and will
    have a unique color.

    Parameters:
    - scores: List of lists, where each inner list contains scores for a descriptor (12 x number of samples).
    - values: List of lists, where each inner list contains values for a descriptor (12 x number of samples).
    - columns: List of strings, names of the descriptors.
    """
    values = values.T.tolist()
    # Check if the number of descriptors in 'scores' and 'values' matches the number of names in 'columns'
    if len(scores) != len(values) or len(scores) != len(columns):
        print("Error: The number of descriptors, scores, and column names must match.")
        return

    # Generate a list of colors for the plots
    colors = [
        "#FF0000", "#FF7F00", "#FFFF00", "#7FFF00",
        "#00FF00", "#00FF7F", "#00FFFF", "#007FFF",
        "#0000FF", "#7F00FF", "#FF00FF", "#FF007F",
        "#000000", "#7F7F7F"
    ]

    plt.figure(figsize=(10, 6))

    # Plot each descriptor's scores against its values
    for i in range(len(scores)):
        plt.plot(values[i], scores[i], label=columns[i], color=colors[i], marker='o', linestyle='None')

    plt.title('Descriptor Scores vs. Descriptor Values')
    plt.xlabel('Descriptor Values')
    plt.ylabel('Descriptor Scores')
    plt.legend(loc='best')
    plt.grid(True)
    plt.show()