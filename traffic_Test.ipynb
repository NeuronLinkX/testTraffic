import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


CSV_PATH = "ML-MATT-CompetitionQT2021_test.csv"

FEATURE_COLS = [
    "PRBUsageUL",
    "PRBUsageDL",
    "meanThr_DL",
    "meanThr_UL",
    "maxThr_DL",
    "maxThr_UL",
    "meanUE_DL",
    "meanUE_UL",
    "maxUE_DL",
    "maxUE_UL",
    "maxUE_UL+DL",
]

TARGET_COLS = ["PRBUsageDL"]

# ============================================================
# ①-보조 Sliding Window Dataset
# - 과거 window=12개의 KPI sequence를 입력 X로 사용한다.
# - 다음 시점의 PRBUsageDL을 정답 y로 사용한다.
# - 즉:
#     X = [t-11, t-10, ..., t]
#     y = [t+1의 PRBUsageDL]
# - 논문 실험 설정의 "previous 12 time steps → next time step prediction"에 해당한다.
# ============================================================
class SequenceDataset(Dataset):
    def __init__(self, x, y, window=12):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.window = window

    def __len__(self):
        return len(self.x) - self.window

    def __getitem__(self, idx):
        return (
            self.x[idx:idx + self.window],
            self.y[idx + self.window]
        )

# ============================================================
# ① 전처리 Preprocessing
# - CSV 파일을 읽는다.
# - Time, CellName 기준으로 정렬한다.
# - feature vector를 만든다.
# - train/valid/test를 시간 순서대로 70/20/10 분할한다.
# - train set 기준 평균(mean), 표준편차(std)로 정규화한다.
# - 논문 식 (6): (x - μ) / sqrt(σ² + ε)에 해당한다.
# - 목적:
#   서로 다른 KPI 수치 범위를 평균 0, 분산 1 수준으로 맞춰
#   특정 feature가 학습을 지배하지 않도록 한다.
# ============================================================
def load_data(path):
    df = pd.read_csv(path, sep=";")

    df["Time"] = pd.to_datetime(df["Time"], format="%H:%M").dt.hour * 60 + \
                 pd.to_datetime(df["Time"], format="%H:%M").dt.minute

    df = df.sort_values(["CellName", "Time"]).reset_index(drop=True)

    x = df[FEATURE_COLS].values.astype(np.float32)
    y = df[TARGET_COLS].values.astype(np.float32)

    n = len(df)
    train_end = int(n * 0.7)
    valid_end = int(n * 0.9)

    x_train, x_valid, x_test = x[:train_end], x[train_end:valid_end], x[valid_end:]
    y_train, y_valid, y_test = y[:train_end], y[train_end:valid_end], y[valid_end:]

    mean = x_train.mean(axis=0, keepdims=True)
    std = x_train.std(axis=0, keepdims=True) + 1e-8

    x_train = (x_train - mean) / std
    x_valid = (x_valid - mean) / std
    x_test = (x_test - mean) / std

    y_mean = y_train.mean(axis=0, keepdims=True)
    y_std = y_train.std(axis=0, keepdims=True) + 1e-8

    y_train = (y_train - y_mean) / y_std
    y_valid = (y_valid - y_mean) / y_std
    y_test = (y_test - y_mean) / y_std

    return x_train, y_train, x_valid, y_valid, x_test, y_test

# ============================================================
# ② MFE: Multi-head Self-Attention
# - 논문 식 (2)~(4)에 해당한다.
# - 입력 sequence 전체에서 어느 시점이 현재 예측에 중요한지 attention weight를 계산한다.
# - RNN처럼 순차적으로 처리하지 않고 전체 time step을 병렬로 본다.
# - 목적:
#   과거 12개 KPI 시점 중 현재 PRBUsageDL 예측에 중요한 시점을 전역적으로 포착한다.
#
# ③ MFE: FFN + LayerNorm + Residual
# - 논문 식 (5)~(8)에 해당한다.
# - Attention 출력에 FFN을 적용해 비선형 feature를 만든다.
# - Residual connection은 x를 더해서 gradient 소실을 줄인다.
# - LayerNorm은 layer 입력 분포를 안정화한다.
# ============================================================
class MFEBlock(nn.Module):
    def __init__(self, d_model=64, n_heads=8, dropout=0.2):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        # ② Multi-head Self-Attention
        # x를 Query, Key, Value로 동시에 사용한다.
        # 즉, 자기 자신 sequence 내부의 시간 의존성을 계산한다.
        a, _ = self.attn(x, x, x)

        # ③ Residual + LayerNorm
        # attention 결과 a를 원래 입력 x에 더한다.
        # 깊은 네트워크에서 정보 손실과 gradient vanishing을 줄인다.
        x = self.ln1(x + self.drop(a))

        # ③ FFN
        # Attention으로 얻은 표현을 비선형 변환한다.
        # ReLU를 통해 단순 선형 조합보다 풍부한 표현을 만든다.
        f = self.ffn(x)

        # ③ Residual + LayerNorm
        x = self.ln2(x + self.drop(f))

        return x

# ============================================================
# ④ TCF: Causal Convolution
# - 논문 식 (9)에 해당한다.
# - 현재 시점 t를 예측할 때 현재와 과거 정보만 사용한다.
# - 미래 정보를 convolution이 참조하지 않도록 padding 이후 뒤쪽을 잘라낸다.
# - 목적:
#   실제 예측 상황에서는 미래 데이터가 없으므로 information leakage를 차단한다.
#
# ⑤ TCF: Dilated Convolution
# - 논문 식 (10)에 해당한다.
# - dilation d = 1, 2, 4, 8로 확장한다.
# - 파라미터 수를 크게 늘리지 않고 receptive field를 넓힌다.
# - 목적:
#   짧은 fluctuation과 긴 시간 패턴을 동시에 포착한다.
#
# ⑥ TCF: LN + ReLU + Residual
# - 논문 식 (12)에 해당한다.
# - convolution 출력에 LayerNorm과 ReLU를 적용한다.
# - 마지막에 residual을 더해 gradient 소실을 줄인다.
# ============================================================

class TCFBlock(nn.Module):
    def __init__(self, d_model=64, kernel_size=3, dilation=1, dropout=0.2):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            d_model,
            d_model,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=self.padding
        )
        self.ln = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        # x: [batch, time, hidden_dim]
        r = x

        # Conv1D는 [batch, channel, time] 형식을 요구하므로 변환한다.
        y = x.transpose(1, 2)

        # ④ Causal Convolution + ⑤ Dilated Convolution
        # dilation 값에 따라 과거 참조 간격이 넓어진다.
        y = self.conv(y)

        # ④ 미래 정보 차단
        # padding으로 인해 생긴 미래 방향 출력을 제거한다.
        y = y[:, :, :-self.padding]

        # 다시 [batch, time, hidden_dim] 형식으로 복원한다.
        y = y.transpose(1, 2)

        # ⑥ LayerNorm + ReLU
        y = F.relu(self.ln(y))

        # ⑥ Residual
        # convolution 결과를 원래 입력 r과 더한다.
        return r + self.drop(y)

# ============================================================
# Transformer Only / MFE Only
# - Ablation Study의 only MFE 모델이다.
# - ① 전처리된 KPI sequence를 입력받는다.
# - ②~③ MFE만 사용한다.
# - ⑦ FC Layer로 다음 PRBUsageDL을 예측한다.
# ============================================================

class TransformerOnly(nn.Module):
    def __init__(self, input_dim, output_dim=1):
        super().__init__()
        self.proj = nn.Linear(input_dim, 64)
        self.mfe = nn.Sequential(MFEBlock(), MFEBlock())

        # ⑦ 출력 FC Layer
        # 마지막 time step의 hidden vector를 실제 예측값으로 압축한다.
        self.fc = nn.Linear(64, output_dim)

    def forward(self, x):
        # 입력 x: [batch, 12, feature_dim]
        x = self.proj(x)

        # ②~③ Transformer MFE
        x = self.mfe(x)

        # ⑦ 마지막 시점 representation으로 다음 시점 PRBUsageDL 예측
        return self.fc(x[:, -1, :])

# ============================================================
# TCN Only / TCF Only
# - Ablation Study의 only TCF 모델이다.
# - ① 전처리된 KPI sequence를 입력받는다.
# - ④ Causal Conv, ⑤ Dilated Conv, ⑥ LN+ReLU+Residual만 사용한다.
# - dilation=[1,2,4,8] 세트를 4번 반복한다.
# - ⑦ FC Layer로 다음 PRBUsageDL을 예측한다.
# ============================================================
class TCNOnly(nn.Module):
    def __init__(self, input_dim, output_dim=1):
        super().__init__()
        self.proj = nn.Linear(input_dim, 64)
        blocks = []
        for _ in range(4):
            for d in [1, 2, 4, 8]:
                blocks.append(TCFBlock(dilation=d))
        self.tcf = nn.Sequential(*blocks)

        # ⑦ 출력 FC Layer
        self.fc = nn.Linear(64, output_dim)

    def forward(self, x):
        x = self.proj(x)

        # ④~⑥ TCF only
        x = self.tcf(x)

        # ⑦ 다음 시점 PRBUsageDL 예측
        return self.fc(x[:, -1, :])

# ============================================================
# Transformer + TCN Hybrid / MFE + TCF
# - 논문에서 최종 채택한 Hybrid 구조다.
# - ① 전처리된 KPI sequence를 입력받는다.
# - ②~③ MFE로 전체 sequence의 전역 의존성을 먼저 추출한다.
# - ④~⑥ TCF로 causal/dilated temporal pattern을 추가 추출한다.
# - ⑦ FC Layer로 다음 PRBUsageDL을 예측한다.
# - 실증연구에서는 이 모델이 Transformer only, TCN only보다
#   MAE/RMSE가 낮은지 비교한다.
# ============================================================
class HybridMFE_TCF(nn.Module):
    def __init__(self, input_dim, output_dim=1):
        super().__init__()
        self.proj = nn.Linear(input_dim, 64)

        # ②~③ MFE
        self.mfe = nn.Sequential(MFEBlock(), MFEBlock())

        # ④~⑥ TCF
        blocks = []
        for _ in range(4):
            for d in [1, 2, 4, 8]:
                blocks.append(TCFBlock(dilation=d))
        self.tcf = nn.Sequential(*blocks)

        # ⑦ 출력 FC Layer
        self.fc = nn.Linear(64, output_dim)

    def forward(self, x):
        x = self.proj(x)

        # ②~③ MFE: 전역 temporal dependency 추출
        z = self.mfe(x)

        # ④~⑥ TCF: causal + dilated temporal pattern 추출
        y = self.tcf(z)

        # ⑦ 다음 시점 PRBUsageDL 예측
        return self.fc(y[:, -1, :])

# ============================================================
# ⑧ 학습 Training
# - 논문 식 (17): MSE + λΣθ²
# - MSE는 예측값과 실제값의 차이를 줄인다.
# - L2 regularization은 파라미터가 과도하게 커지는 것을 막아 과적합을 줄인다.
# - optimizer는 Adam을 사용한다.
# ============================================================
def loss_fn(pred, target, model, lam=1e-4):
    # MSE: 예측 PRBUsageDL과 실제 PRBUsageDL의 오차
    mse = F.mse_loss(pred, target)

    # L2: 모든 파라미터 제곱합
    l2 = sum((p ** 2).sum() for p in model.parameters())

    # 식 (17)
    return mse + lam * l2

def train(model, train_loader, valid_loader, device, epochs=30):
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    best_rmse = float("inf")
    best_state = None

    for ep in range(1, epochs + 1):
        model.train()

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            pred = model(x)

            # ⑧ Loss 계산
            loss = loss_fn(pred, y, model)

            # ⑧ 역전파 및 파라미터 업데이트
            opt.zero_grad()
            loss.backward()
            opt.step()

        # validation set에서 성능 확인
        val_mae, val_rmse = evaluate(model, valid_loader, device)

        # RMSE가 가장 낮은 모델 저장
        if val_rmse < best_rmse:
            best_rmse = val_rmse
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        print(f"Epoch {ep:03d} | Valid MAE={val_mae:.4f} | Valid RMSE={val_rmse:.4f}")

    model.load_state_dict(best_state)
    model.to(device)
    return model

# ============================================================
# ⑨ 검증 Validation / Inference
# - 논문 평가 지표 MAE / RMSE를 계산한다.
# - MAE:
#   예측값과 실제값의 절대 오차 평균.
#   평균적으로 얼마나 틀렸는지 직관적으로 보여준다.
# - RMSE:
#   제곱 오차 평균의 제곱근.
#   큰 오차에 더 민감하므로 burst 구간이나 급격한 변화 예측 실패를 더 강하게 반영한다.
# ============================================================

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    preds, trues = [], []

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        p = model(x)
        preds.append(p.cpu())
        trues.append(y.cpu())

    pred = torch.cat(preds)
    true = torch.cat(trues)

    # ⑨ MAE
    mae = torch.mean(torch.abs(pred - true)).item()

    # ⑨ RMSE
    rmse = torch.sqrt(torch.mean((pred - true) ** 2)).item()

    return mae, rmse

# ============================================================
# 전체 실증연구 실행부
# - Dataset 로드
# - ① 전처리
# - Sliding window 생성
# - 3개 모델 비교:
#   1) Transformer only / MFE only
#   2) TCN only / TCF only
#   3) Transformer + TCN / Hybrid
# - ⑧ 학습
# - ⑨ test set MAE/RMSE 비교
# ============================================================

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ① 데이터 로드 및 정규화
    x_train, y_train, x_valid, y_valid, x_test, y_test = load_data(CSV_PATH)

    # ① Sliding window dataset 생성
    train_ds = SequenceDataset(x_train, y_train)
    valid_ds = SequenceDataset(x_valid, y_valid)
    test_ds = SequenceDataset(x_test, y_test)

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    valid_loader = DataLoader(valid_ds, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)

    input_dim = len(FEATURE_COLS)

    # Ablation Study 대상 모델 3개
    models = {
        "Transformer only / MFE only": TransformerOnly(input_dim),
        "TCN only / TCF only": TCNOnly(input_dim),
        "Transformer + TCN / Hybrid": HybridMFE_TCF(input_dim),
    }

    results = {}

    for name, model in models.items():
        print("\n" + "=" * 80)
        print(name)
        print("=" * 80)

        # ⑧ 학습
        model = train(model, train_loader, valid_loader, device)

        # ⑨ 검증
        test_mae, test_rmse = evaluate(model, test_loader, device)

        results[name] = {
            "MAE": test_mae,
            "RMSE": test_rmse,
        }

    print("\nFinal Result")
    print("=" * 80)

    for name, r in results.items():
        print(f"{name:35s} | MAE={r['MAE']:.4f} | RMSE={r['RMSE']:.4f}")


if __name__ == "__main__":
    main()

# 본 구현은 논문 절차 ①~⑨를 그대로 코드 구조에 대응시켰다.
# ① 전처리에서 KPI를 정규화하고 sliding window를 생성하였다.
# ②~③ MFE는 Transformer self-attention과 FFN으로 전역 의존성을 추출하였다.
# ④~⑥ TCF는 causal/dilated convolution으로 시간 패턴을 추출하였다.
# ⑦ FC layer는 hidden representation을 다음 시점 PRBUsageDL 예측값으로 변환하였다.
# ⑧ MSE+L2 loss로 학습하였다.
# ⑨ MAE/RMSE로 Transformer only, TCN only, Hybrid를 비교하였다.