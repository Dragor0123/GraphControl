# GraphControl Modification Summary

## 목적
ControlNet의 입력으로 들어가는 condition A'에서 heterophilic link를 필터링하도록 수정

## 수정 내용

### 1. `utils/transforms.py` 수정

**파일 위치**: `/home/h2tae/wkspace/GraphControl/utils/transforms.py`

**함수**: `obtain_attributes` (Line 9)

**변경사항**:
- `labels` 파라미터 추가
- Heterophilic link 필터링 로직 추가 (Line 23-26)

```python
def obtain_attributes(data, use_adj=False, threshold=0.1, num_dim=32, labels=None):
    save_node_border = 30000

    if use_adj:
        # to undirected and remove self-loop
        edges = to_undirected(data.edge_index)
        edges, _ = remove_self_loops(edges)
        tmp = to_dense_adj(edges)[0]
    else:
        tmp = similarity(data.x, data.x)

        # discretize the similarity matrix by threshold
        tmp = torch.where(tmp>threshold, 1.0, 0.0)

        # Filter heterophilic links: set to 0 if nodes have different labels
        if labels is not None:
            label_matrix = labels.unsqueeze(1) == labels.unsqueeze(0)  # True if same class
            tmp = tmp * label_matrix.float()  # Keep only homophilic links
```

### 2. `graphcontrol.py` 수정

**파일 위치**: `/home/h2tae/wkspace/GraphControl/graphcontrol.py`

**함수**: `main` (Line 100)

**변경사항**:
- `obtain_attributes` 호출 시 ground truth labels 전달

```python
# Pass ground truth labels to filter heterophilic links in condition A'
x_sim = obtain_attributes(dataset_obj.data, use_adj=False, threshold=config.threshold, labels=dataset_obj.data.y).to(device)
```

## 동작 원리

### Condition A' 생성 과정
1. **Similarity Matrix 계산**: 노드 feature 간 similarity를 계산
2. **Threshold 적용**: Similarity가 threshold보다 큰 경우만 1.0으로 설정
3. **Heterophilic Link 필터링** (NEW):
   - Ground truth label을 사용하여 label matrix 생성
   - 같은 클래스 노드 쌍: True (유지)
   - 다른 클래스 노드 쌍: False (0으로 설정)
4. **Laplacian 계산 및 Eigen decomposition**: 필터링된 matrix로부터 positional encoding 생성

### 필터링 효과
- **이전**: 모든 similarity > threshold인 링크 사용
- **이후**: similarity > threshold 이면서 **같은 클래스**를 가진 노드 간 링크만 사용
- **결과**: Condition A'가 homophilic link만 포함하여 heterophilic graph에서 더 정확한 조건 제공

## 수정된 파일 목록
1. `utils/transforms.py` - Line 9, 23-26
2. `graphcontrol.py` - Line 100

## 주석 위치
- `utils/transforms.py:23-26` - Heterophilic link 필터링 로직 설명
- `graphcontrol.py:99` - Ground truth labels 전달 목적 설명

-----
 혹시 이 세팅에서 graph classification으로 변형하는 것은 쉽나? (Y/N with reasons) 
● N (어렵습니다)

  주요 이유:

  1. 서브그래프 샘플링 구조: 현재 코드는 random walk로 각 노드 주변의 서브그래프를 샘플링하여 그 노드의 클래스를 예측합니다 (graphcontrol.py:24-25). Graph classification은 전체 그래프를
   입력으로 받아야 합니다.
  2. 데이터 로더 구조: collect_subgraphs 함수가 노드별로 서브그래프를 생성하는 구조 (graphcontrol.py:24-25). Graph classification은 각 그래프가 하나의 샘플이 되어야 합니다.
  3. Condition A' 생성 방식: obtain_attributes가 전체 그래프의 노드 간 similarity matrix를 계산 (graphcontrol.py:100). Graph classification에서는 각 그래프마다 독립적으로 계산해야
  합니다.
  4. ControlNet 입력: x_sim[data.original_idx]로 전체 그래프에서 서브그래프의 노드들만 인덱싱 (graphcontrol.py:68, 142). Graph classification은 이런 인덱싱이 불필요합니다.
  5. 데이터셋 변경: NodeDataset을 GraphDataset으로 교체하고, train/test mask 대신 그래프 리스트를 사용해야 합니다.

  필요한 대규모 수정:
  - 데이터 로딩 파이프라인 전체 재구성
  - Condition A' 생성 방식 변경
  - 모델 forward 함수 수정
  - 평가 메트릭 변경