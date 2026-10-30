# 0. 기본 설정
## 0-1. 기본 패키지 불러오기
from pathlib import Path
import pandas as pd
import torch
from torch import nn

## CPU 스레드 설정
torch.set_num_threads(1)

## 기본 Path 설정
base = Path(__file__).resolve().parent

## 0-2. 데이터 로더 및 시각화 관련 패키지 불러오기
from torch.utils.data import TensorDataset, DataLoader
import copy
import matplotlib
matplotlib.use("Agg")  # 창을 띄우지 않고 PNG에 저장합니다.
import matplotlib.pyplot as plt

# 1. 데이터 준비 및 train_test_split
out = base / "outputs"
out.mkdir(exist_ok=True)
# 재현용 기본값 CPU. GPU 실습은 강사 확인 뒤 device 문자열만 cuda:0으로 바꿉니다.

## 연산 장치 설정: CPU
device = torch.device("cuda:0") if torch.cuda.is_available() else "cpu"

df = pd.read_csv(base / "data" / "points.csv")
train = df[df["split"] == "train"]
valid = df[df["split"] == "valid"]
X_train = torch.tensor(train[["x1", "x2"]].values, dtype=torch.float32)
y_train = torch.tensor(train["label"].values, dtype=torch.long)
X_valid = torch.tensor(valid[["x1", "x2"]].values, dtype=torch.float32).to(device)
y_valid = torch.tensor(valid["label"].values, dtype=torch.long).to(device)

# 2. 모델링 준비
## 2-1. 손실함수 설정
loss_fn = nn.CrossEntropyLoss()

## 2-2. 파라미터 조정
### 최적 파라미터 후보 설정
configs = [{'name': 'sgd', 'optimizer': 'SGD', 'lr': 0.01, 'dropout': 0.0}, 
           {'name': 'adamw', 'optimizer': 'AdamW', 'lr': 0.01, 'dropout': 0.0}, 
           {'name': 'adamw_dropout', 'optimizer': 'AdamW', 'lr': 0.01, 'dropout': 0.3}]
summary = []
for config in configs:
    ### 재현성 위한 시드 설정
    torch.manual_seed(42)
    ### width 설정: 32
    width = config.get("width", 32)
    ### 모델링 - 선형 변환(Input (2, 32), 활성화 함수 ReLU(), Dropout(0.0 따로, 0.0 따로, 0.3 따로)) 
    layers = [nn.Linear(2, width)]
    if config.get("bn", False):
        layers.append(nn.BatchNorm1d(width))
    layers.extend([nn.ReLU(), nn.Dropout(config.get("dropout", 0.0)), nn.Linear(width, width)])
    if config.get("bn", False):
        layers.append(nn.BatchNorm1d(width))
    layers.extend([nn.ReLU(), nn.Dropout(config.get("dropout", 0.0)), nn.Linear(width, 2)])
    model = nn.Sequential(*layers).to(device)
    # shuffle용 Generator를 후보마다 같은 seed로 다시 만듭니다.
    ### 데이터 로더 변환(batch_size:32, shuffle=True, seed:43)
    loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True,
                        generator=torch.Generator().manual_seed(43), num_workers=0)
    ### 최적화 설정(SGD or AdamW or AdamW)
    if config.get("optimizer", "AdamW") == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=config["lr"],
                                    momentum=config.get("momentum", 0.0), weight_decay=0.0)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=0.0)
    history = []
    ### best parameter(optimizer, dropout, epoch 등)와 그에 따른 loss 저장 준비
    best_loss = float("inf")
    best_state = None
    best_epoch = 0
    # 3. 모델 학습
    for epoch in range(1, 40 + 1):
        ## train 모드
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            ### 기울기 초기화
            optimizer.zero_grad()
            ### 손실함수 대입
            loss = loss_fn(model(xb), yb)
            # loss.backward() 다음 optimizer.step()을 호출하세요.
            # 같은 초기화·동일 shuffle 순서를 사용하며 세 후보의 변경 조건을 읽어 구분합니다.
            # >>> A07
            # TODO A07: 위 요구사항에 맞게 이 블록만 완성합니다.
            ### 역전파
            loss.backward()
            ### 최적화
            optimizer.step()
            # <<< A07
        # 같은 epoch 말 모델을 eval 모드로 비교합니다. 학습 중 Dropout loss와 구분합니다.
        # epoch 종료 후 model.eval()을 호출하세요.
        # 그 아래 no_grad 블록은 gradient 추적을 끄지만, Dropout 모드를 바꾸지는 않습니다.
        # >>> A08
        # TODO A08: 위 요구사항에 맞게 이 블록만 완성합니다.
        # 4. 모델 평가
        model.eval()
        # <<< A08
        with torch.no_grad():
            train_loss = loss_fn(model(X_train.to(device)), y_train.to(device)).item()
            logits = model(X_valid)
            valid_loss = loss_fn(logits, y_valid).item()
            valid_acc = (logits.argmax(1) == y_valid).float().mean().item()
            # 최저 valid loss를 관측하면 값·epoch·가중치 복사본을 함께 보관합니다.
            # A09 요구: 동률은 먼저 발견한 epoch 유지. copy.deepcopy(model.state_dict()) 사용.
            # model.state_dict()만 대입하면 이후 학습이 같은 Tensor를 바꾸므로 안 됩니다.
            # >>> A09
            # TODO A09: 최적 loss·epoch와 독립된 state_dict 복사본을 보관하세요.
            if valid_loss < best_loss:
                best_loss = valid_loss
                best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())
            # <<< A09
        history.append([epoch, train_loss, valid_loss, valid_acc])
    # 5. 결과 저장
    table = pd.DataFrame(history, columns=["epoch", "train_loss_eval", "valid_loss", "valid_acc"])
    best = table.loc[table["valid_loss"].idxmin()]
    name = config["name"]
    table.to_csv(out / f"07_{name}_history.csv", index=False)
    summary.append({**config, "best_epoch": int(best["epoch"]),
                    "best_valid_loss": float(best["valid_loss"]),
                    "valid_acc_at_best": float(best["valid_acc"]),
                    "last_train_loss_eval": train_loss, "last_valid_loss": valid_loss})
    # 제공 코드: 학습이 끝난 뒤 최적 상태로 복원하고 그 모델로 다시 평가합니다.
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        best_pred = model(X_valid).argmax(1).cpu()
        restored_loss = loss_fn(model(X_valid), y_valid).item()
    labels = y_valid.cpu()
    class_rows = []
    for label in (0, 1):
        # A10 요구: 실제 정답이 label인 행을 먼저 고른 뒤 support·correct·recall 계산.
        # 전체 accuracy나 '예측 label의 수'로 나누지 않습니다. 0건이면 recall=None.
        # >>> A10
        # TODO A10: 클래스별 실제 표본 수·정답 수·Recall을 기록하세요.
        mask = (labels == label)
        support = mask.sum().item()

        if support == 0:
            recall = None
            correct = 0
        else:
            correct = (best_pred[mask] == labels[mask]).sum().item()
            recall = correct / support

        class_rows.append({
            "name": name,
            "class": label,
            "support": support,
            "correct": correct,
            "recall": recall
        })
        # <<< A10
    pd.DataFrame(class_rows).to_csv(out / f"07_{name}_class_metrics.csv", index=False)
    pd.DataFrame({"id":valid["id"].tolist(),"actual":labels.tolist(),
                  "predicted":best_pred.tolist()}).to_csv(out / f"07_{name}_predictions.csv", index=False)
    summary[-1]["restored_valid_loss"] = restored_loss
    summary[-1]["restored_matches_best"] = abs(restored_loss-best_loss) < 1e-6
    plt.figure(figsize=(8, 5))
    plt.plot(table["epoch"], table["train_loss_eval"], label="train (eval)")
    plt.plot(table["epoch"], table["valid_loss"], label="valid")
    plt.axvline(best["epoch"], linestyle="--", label="best epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / f"07_{name}_curves.png", dpi=150)
    plt.close()
    print(name, "best epoch:", int(best["epoch"]), "best valid loss:", round(best["valid_loss"], 6))
result = pd.DataFrame(summary)
result.to_csv(out / "07_comparison.csv", index=False)
print("생성: outputs/07_*.csv, *.png / test 미사용")

# 결과
## 세 개의 파라미터 후보 중 수치 상으로는 adamW('lr': 0.01, 'dropout': 0.0, best epoch: 38)일 때의 valid loss(0.01656)가 가장 낮은 것으로 확인되었습니다.
## 또한, 그래프로도 확인하였을 때, adamW('lr': 0.01, 'dropout': 0.0, best epoch: 38)의 그래프가 loss간의 변동 폭의 안정성 등을 고려하였을 때도 가장 최적의 파라미터
## 인 것을 확인할 수 있었습니다.