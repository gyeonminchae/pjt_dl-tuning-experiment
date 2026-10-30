# 0. 기본 설정
## 0-1. 기본 패키지 불러오기
from pathlib import Path
import pandas as pd
import torch
from torch import nn

## CPU 스레드 확인
torch.set_num_threads(1)

## 기본 Path 설정
base = Path(__file__).resolve().parent

## 0-2. 시각화 패키지 불러오기
import matplotlib
matplotlib.use("Agg")  # 창을 띄우지 않고 PNG에 저장합니다.
import matplotlib.pyplot as plt

# 1. 데이터 준비 및 train_test_split
df = pd.read_csv(base / "points.csv")
train = df[df["split"] == "train"]
valid = df[df["split"] == "valid"]
X_train = torch.tensor(train[["x1", "x2"]].values, dtype=torch.float32)
y_train = torch.tensor(train["label"].values, dtype=torch.long)
X_valid = torch.tensor(valid[["x1", "x2"]].values, dtype=torch.float32)
y_valid = torch.tensor(valid["label"].values, dtype=torch.long)
# test 4건은 읽은 표에만 있으며 학습/평가 Tensor로 만들지 않습니다.

# 2. 모델링 준비
## 2-1. 손실함수 준비
# 실습 07~08: 기준 lr과 비교 lr만 다릅니다. 더 나쁘게 나와도 그대로 보고합니다.
loss_fn = nn.CrossEntropyLoss()
summary = []

## 2-2. 파라미터 셋팅
for name, lr in [("baseline", 0.001), ("candidate", 0.01)]:
    ### 재현성 위한 시드 설정
    torch.manual_seed(42)
    ### 선형 변환(input layer:(2,8), 활성화 함수: ReLU(), output layer:(8,2))
    model = nn.Sequential(nn.Linear(2, 8), nn.ReLU(), nn.Linear(8, 2))
    ### 최적화(AdamW -> 학습률 후보: lr->0.001, lr -> 0.01 / weight_decay는 고정)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    history = []
    ### 모델 학습(epoch:20)
    for epoch in range(1, 21):
        model.train()
        #### 기울기 초기화
        optimizer.zero_grad()
        #### 손실함수 대입
        loss = loss_fn(model(X_train), y_train)
        # 이미 zero_grad와 loss 계산이 끝났습니다. loss.backward() 다음 optimizer.step()을 호출하세요.
        # 학습 데이터만 사용합니다. 비교 후보마다 모델을 처음부터 새로 만듭니다.
        # >>> B08
        # TODO B08: 위 요구사항에 맞게 이 블록만 완성합니다.
        #### 역전파 수행
        loss.backward()
        #### 최적화 수행
        optimizer.step()
        # <<< B08
        ### 모델 평가
        model.eval()
        with torch.no_grad():
            train_loss = loss_fn(model(X_train), y_train).item()
            valid_loss = loss_fn(model(X_valid), y_valid).item()
        # 한 epoch의 번호와 갱신 후 train_loss, valid_loss를 기록하세요.
        # history.append([epoch, train_loss, valid_loss]) 형태로 추가합니다.
        # >>> B09
        # TODO B09: 위 요구사항에 맞게 이 블록만 완성합니다.
        history.append([epoch, train_loss, 0.0])
        # <<< B09
    # 3. 결과 산출
    table = pd.DataFrame(history, columns=["epoch", "train_loss_eval", "valid_loss"])
    table.to_csv(base / f"history_{name}.csv", index=False)
    best = table.loc[table["valid_loss"].idxmin()]
    summary.append([name, lr, int(best["epoch"]), best["valid_loss"]])
    plt.figure(figsize=(8, 5))
    plt.plot(table["epoch"], table["train_loss_eval"], label="train (eval mode)")
    plt.plot(table["epoch"], table["valid_loss"], label="valid")
    plt.axvline(best["epoch"], linestyle="--", label="best epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(base / f"curves_{name}.png", dpi=150)
    plt.close()
result = pd.DataFrame(summary, columns=["setting", "lr", "best_epoch", "best_valid_loss"])
result.to_csv(base / "experiment_table.csv", index=False)
print(result.round(6).to_string(index=False))
print("test 미사용 / best epoch는 관찰한 최소 지점, 기본필수에서는 모델 저장·복원하지 않음")

## 결과 
## baseline(lr=0.001, weight_decay=0)과 비교군 lr_candidate(lr=0.01)의 best_epoch 모두 1이 나왔고 best_valid_loss 0으로 확인되었습니다.
## 그래서, 실제로 loss 그래프를 확인한 결과 두 집단에서 valid에서 모두 0으로 계속 이어진 것으로 보아,
## 기본적으로 Overfitting인 것으로 보여지며, 데이터의 양을 비롯하여 전반적인 부분을 다시 확인할 필요가 있어보입니다.