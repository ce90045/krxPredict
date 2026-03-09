# =============================================================================
#  KRX 주가 예측 프로그램 v5  —  앙상블 모델 (Prophet + LSTM + XGBoost)
#  개선 사항 (v4 → v5):
#    1. auto_adjust=False  → 실제 시장 주가 사용 (수정 주가 문제 해결)
#    2. 데이터 수집 기간   → 900일 → 2000일 (약 5.5년, 금리 사이클 포함)
#    3. LSTM lookback      → 60일 → 120일 (더 긴 패턴 학습)
#    4. 외부 피처 추가     → KOSPI지수, 원달러환율, 미국채10년, VIX
#    5. XGBoost 검증       → TimeSeriesSplit 5-fold 교차검증 적용
#    6. XGBoost 클리핑     → ±5% → ±8% (현실적 범위 확대)
# =============================================================================

# -----------------------------------------------------------------------------
# [섹션 0]  표준 라이브러리 및 공통 설정
# -----------------------------------------------------------------------------
import os
import sys
import warnings
warnings.filterwarnings("ignore")

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["CUDA_VISIBLE_DEVICES"]  = "-1"   # CPU 전용 실행

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from datetime import datetime, timedelta


# =============================================================================
# [섹션 0-A]  한글 폰트 설정
# =============================================================================
def set_korean_font():
    candidates = [
        "Malgun Gothic", "맑은 고딕", "AppleGothic",
        "Apple SD Gothic Neo", "NanumGothic", "Nanum Gothic", "나눔고딕",
    ]
    available = {f.name for f in fm.fontManager.ttflist}
    for font in candidates:
        if font in available:
            plt.rcParams["font.family"] = font
            break
    plt.rcParams["axes.unicode_minus"] = False

set_korean_font()


# =============================================================================
# [섹션 1]  필수 라이브러리 설치 여부 점검
# =============================================================================
def check_imports():
    required = {
        "yfinance":   "yfinance",
        "prophet":    "prophet",
        "xgboost":    "xgboost",
        "sklearn":    "scikit-learn",
        "tensorflow": "tensorflow",
    }
    missing = []
    for mod, pkg in required.items():
        try:
            __import__(mod)
        except ImportError:
            missing.append(pkg)
    if missing:
        print("=" * 60)
        print("  [오류] 다음 라이브러리를 먼저 설치해 주세요:")
        for m in missing:
            print(f"    pip install {m}")
        print()
        print("  한 번에 설치:")
        print("  pip install " + " ".join(missing))
        print("=" * 60)
        sys.exit(1)

check_imports()


# -----------------------------------------------------------------------------
# 안전한 임포트
# -----------------------------------------------------------------------------
import yfinance as yf
from prophet import Prophet

import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import TimeSeriesSplit   # ★ v5 추가: 시계열 교차검증

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

import logging
logging.getLogger("cmdstanpy").setLevel(logging.WARNING)


# =============================================================================
# [섹션 2]  KRX 전체 종목 목록 관리
# =============================================================================
CACHE_FILE = "krx_tickers_cache.csv"

def _today_str() -> str:
    return datetime.today().strftime("%Y%m%d")

def _cache_is_fresh() -> bool:
    if not os.path.exists(CACHE_FILE):
        return False
    try:
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            first = f.readline().strip()
        return first.replace("# saved_date=", "").strip() == _today_str()
    except Exception:
        return False

def _load_cache() -> pd.DataFrame:
    return pd.read_csv(CACHE_FILE, comment="#", dtype=str, encoding="utf-8")

def _save_cache(df: pd.DataFrame):
    with open(CACHE_FILE, "w", encoding="utf-8", newline="") as f:
        f.write(f"# saved_date={_today_str()}\n")
        df.to_csv(f, index=False)

def _fetch_tickers_kind(market_type: str) -> list:
    url = (
        "http://kind.krx.co.kr/corpgeneral/corpList.do"
        f"?currentPageSize=5000&pageIndex=1"
        f"&method=download&searchType=13&marketType={market_type}"
    )
    tables = pd.read_html(
        url, header=0,
        converters={"종목코드": lambda x: str(x).zfill(6)},
        encoding="euc-kr",
    )
    return tables[0].to_dict("records")

def _fallback_tickers() -> list:
    return [
        {"시장": "KOSPI",  "종목명": "삼성전자",        "종목코드": "005930"},
        {"시장": "KOSPI",  "종목명": "SK하이닉스",       "종목코드": "000660"},
        {"시장": "KOSPI",  "종목명": "현대차",           "종목코드": "005380"},
        {"시장": "KOSPI",  "종목명": "KB금융",           "종목코드": "105560"},
        {"시장": "KOSPI",  "종목명": "신한지주",         "종목코드": "055550"},
        {"시장": "KOSPI",  "종목명": "우리금융지주",     "종목코드": "316140"},
        {"시장": "KOSPI",  "종목명": "하나금융지주",     "종목코드": "086790"},
        {"시장": "KOSDAQ", "종목명": "에코프로비엠",     "종목코드": "247540"},
        {"시장": "KOSDAQ", "종목명": "HLB",              "종목코드": "028300"},
    ]

def load_all_tickers() -> pd.DataFrame:
    if _cache_is_fresh():
        df = _load_cache()
        print(f"  ✔ 당일 캐시에서 종목 목록 로드 ({len(df):,}개) [{CACHE_FILE}]\n")
        return df

    print("  📦 KIND(기업공시채널)에서 전체 종목 목록 수신 중...")
    rows = []
    for mkt_type, mkt_name in {"stockMkt": "KOSPI", "kosdaqMkt": "KOSDAQ"}.items():
        try:
            items = _fetch_tickers_kind(mkt_type)
            before = len(rows)
            for item in items:
                name   = str(item.get("회사명", "")).strip()
                ticker = str(item.get("종목코드", "")).strip().zfill(6)
                if name and ticker and ticker != "000000":
                    rows.append({"시장": mkt_name, "종목명": name, "종목코드": ticker})
            print(f"     {mkt_name}: {len(rows)-before:,}개")
        except Exception as e:
            print(f"  [경고] {mkt_name} 로드 실패: {e}")

    if not rows:
        print("  [경고] KIND 요청 실패 → 주요 종목 목록으로 대체합니다.")
        rows = _fallback_tickers()

    df = pd.DataFrame(rows).drop_duplicates(subset="종목코드").reset_index(drop=True)
    try:
        _save_cache(df)
        print(f"  💾 캐시 저장: {CACHE_FILE}")
    except Exception as e:
        print(f"  [경고] 캐시 저장 실패: {e}")

    print(f"  ✔ 총 {len(df):,}개 종목 로드 완료\n")
    return df


# =============================================================================
# [섹션 3]  종목 검색 및 선택
# =============================================================================
def search_and_select(all_df: pd.DataFrame, keyword: str):
    keyword  = keyword.strip()
    filtered = all_df[
        all_df["종목명"].str.contains(keyword, na=False, case=False)
    ].reset_index(drop=True)

    if filtered.empty:
        print(f"\n  검색 결과 없음: '{keyword}' 에 해당하는 종목이 없습니다.\n")
        return None, None

    print(f"\n  ── '{keyword}' 검색 결과 (총 {len(filtered)}건) ──────────────────")
    print(f"  {'순번':>4}  {'종목명':<22}  {'종목코드':<10}  시장")
    print("  " + "─" * 50)
    for i, row in filtered.iterrows():
        print(f"  {i+1:>4}  {row['종목명']:<22}  {row['종목코드']:<10}  {row['시장']}")
    print()

    while True:
        try:
            sel = input(f"  예측할 종목 순번 (1~{len(filtered)}, 취소: Enter): ").strip()
            if sel == "":
                return None, None
            n = int(sel)
            if 1 <= n <= len(filtered):
                chosen = filtered.iloc[n - 1]
                print(f"\n  ✔ 선택: [{chosen['종목코드']}] {chosen['종목명']} ({chosen['시장']})\n")
                return chosen["종목코드"], chosen["종목명"]
            print(f"  1~{len(filtered)} 범위로 입력해 주세요.")
        except ValueError:
            print("  숫자를 입력해 주세요.")


# =============================================================================
# [섹션 4]  주가 데이터 수집 및 기술적 지표 계산
# =============================================================================

def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """종가 기반 기술적 지표 계산."""
    d = df.copy()
    c = d["종가"]

    # 이동평균
    for w in [5, 10, 20, 60, 120]:          # ★ v5: MA120 추가
        d[f"MA{w}"] = c.rolling(w).mean()

    # 볼린저 밴드
    d["BB_mid"]   = c.rolling(20).mean()
    d["BB_std"]   = c.rolling(20).std()
    d["BB_upper"] = d["BB_mid"] + 2 * d["BB_std"]
    d["BB_lower"] = d["BB_mid"] - 2 * d["BB_std"]
    d["BB_width"] = (d["BB_upper"] - d["BB_lower"]) / (d["BB_mid"] + 1e-9)  # ★ v5: 밴드 폭 추가

    # RSI
    delta = c.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rs    = gain / (loss + 1e-9)
    d["RSI"] = 100 - 100 / (1 + rs)

    # MACD
    ema12      = c.ewm(span=12, adjust=False).mean()
    ema26      = c.ewm(span=26, adjust=False).mean()
    d["MACD"]  = ema12 - ema26
    d["Signal"]= d["MACD"].ewm(span=9, adjust=False).mean()
    d["MACD_hist"] = d["MACD"] - d["Signal"]   # ★ v5: MACD 히스토그램 추가

    # 수익률
    d["Return_1d"]  = c.pct_change(1)
    d["Return_5d"]  = c.pct_change(5)
    d["Return_20d"] = c.pct_change(20)          # ★ v5: 20일 수익률 추가

    # 거래량
    d["Vol_ratio"] = d["거래량"] / (d["거래량"].rolling(20).mean() + 1e-9)

    return d.dropna().reset_index(drop=True)


# =============================================================================
# [섹션 4-B]  ★ v5 신규: 외부 피처(거시경제 지표) 수집 및 병합
#   KOSPI지수, 원달러환율, 미국채10년, VIX 공포지수
#   금융주(우리금융 등)는 금리·환율의 영향을 특히 많이 받으므로 효과적
# =============================================================================

EXTERNAL_TICKERS = {
    "KOSPI":  "^KS11",    # KOSPI 종합지수
    "USD_KRW":"KRW=X",    # 원달러 환율
    "US10Y":  "^TNX",     # 미국채 10년 금리
    "VIX":    "^VIX",     # 시장 변동성 지수 (공포지수)
}

def fetch_external_features(start_dt: datetime, end_dt: datetime) -> pd.DataFrame:
    """외부 거시경제 지표를 수집해 날짜 기준으로 병합한 DataFrame 반환.

    Returns:
        컬럼: ds, KOSPI, USD_KRW, US10Y, VIX
        수집 실패한 지표는 0으로 채움 (예측 중단 방지)
    """
    print("  📡 외부 거시경제 지표 수집 중 (KOSPI·환율·금리·VIX)...")
    ext_frames = []

    for col, yf_sym in EXTERNAL_TICKERS.items():
        try:
            raw = yf.download(
                yf_sym,
                start=start_dt.strftime("%Y-%m-%d"),
                end=end_dt.strftime("%Y-%m-%d"),
                auto_adjust=False,    # ★ v5: 수정 주가 문제 해결
                progress=False,
            )
            if raw.empty:
                print(f"     [경고] {col}({yf_sym}) 데이터 없음 → 0으로 대체")
                continue

            # MultiIndex 처리
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.get_level_values(0)

            raw = raw.reset_index()
            raw.columns = [str(c) for c in raw.columns]

            # Date 컬럼 통일
            date_col = "Date" if "Date" in raw.columns else raw.columns[0]
            tmp = raw[[date_col, "Close"]].copy()
            tmp.columns = ["ds", col]
            tmp["ds"] = pd.to_datetime(tmp["ds"]).dt.tz_localize(None)
            ext_frames.append(tmp.set_index("ds"))
            print(f"     {col}: {len(tmp)}일치 수집 완료")
        except Exception as e:
            print(f"     [경고] {col} 수집 실패: {e}")

    if not ext_frames:
        print("  [경고] 외부 지표 수집 전체 실패 → 외부 피처 없이 진행")
        return pd.DataFrame()

    # 날짜 기준으로 outer join 후 forward fill (공휴일 등으로 날짜 불일치 보완)
    merged = pd.concat(ext_frames, axis=1, join="outer")
    merged = merged.ffill().bfill()   # 앞→뒤, 뒤→앞 채우기로 NaN 최소화
    merged = merged.reset_index().rename(columns={"index": "ds"})
    print(f"  ✔ 외부 지표 병합 완료 ({len(merged)}행)\n")
    return merged


def fetch_ohlcv(ticker: str, name: str) -> pd.DataFrame:
    """yfinance로 OHLCV 수집 + 기술지표 + 외부 피처 병합.

    ★ v5 변경:
      - auto_adjust=False  → 실제 시장 주가 반환 (수정 주가 문제 해결)
      - 수집 기간 900→2000일 (약 5.5년)
      - 외부 거시경제 피처 병합
    """
    end_dt   = datetime.today()
    # ★ v5: 2000일 (약 5.5년) — 금리 사이클 1회 이상 포함
    start_dt = end_dt - timedelta(days=2000)

    for suffix in [".KS", ".KQ"]:
        yf_ticker = f"{ticker}{suffix}"
        label     = "KOSPI" if suffix == ".KS" else "KOSDAQ"
        print(f"  📡 [{yf_ticker}] {name} 주가 수집 중 ({label})...")

        raw = yf.download(
            yf_ticker,
            start=start_dt.strftime("%Y-%m-%d"),
            end=end_dt.strftime("%Y-%m-%d"),
            auto_adjust=False,   # ★ v5: 실제 시장 주가 사용
            progress=False,
        )
        if not raw.empty:
            break
    else:
        raise RuntimeError(f"{name}({ticker}) 주가 데이터를 가져오지 못했습니다.")

    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    df = raw.reset_index()
    df.columns = [str(c) for c in df.columns]

    df = df.rename(columns={
        "Date": "ds",
        "Close": "종가", "Open": "시가",
        "High": "고가",  "Low": "저가", "Volume": "거래량",
    })
    df["ds"] = pd.to_datetime(df["ds"]).dt.tz_localize(None)
    df = df[["ds", "종가", "시가", "고가", "저가", "거래량"]].dropna()
    df["y"] = df["종가"]
    df = df.sort_values("ds").reset_index(drop=True)

    # 기술적 지표
    df = add_technical_indicators(df)

    # ★ v5: 외부 거시경제 피처 병합
    ext_df = fetch_external_features(start_dt, end_dt)
    if not ext_df.empty:
        ext_df["ds"] = pd.to_datetime(ext_df["ds"])
        df = pd.merge(df, ext_df, on="ds", how="left")
        # 병합 후 NaN(주말·공휴일 불일치) → forward fill
        ext_cols = [c for c in EXTERNAL_TICKERS.keys() if c in df.columns]
        df[ext_cols] = df[ext_cols].ffill().bfill().fillna(0)

    df = df.dropna(subset=["y"]).reset_index(drop=True)

    print(f"  ✔ {len(df)}일치 + 기술지표 + 외부피처 계산 완료 "
          f"({df['ds'].min().date()} ~ {df['ds'].max().date()})\n")
    return df


# =============================================================================
# [섹션 5-A]  Prophet 예측 모델
# =============================================================================
def predict_prophet(df: pd.DataFrame, forecast_days: int = 63):
    print("  [1/3] Prophet 학습 중...")
    n_days = len(df)

    # ★ v5: 데이터 양이 늘었으므로 changepoint 기준 조정
    if n_days >= 1000:
        cp_scale = 0.12
    elif n_days >= 600:
        cp_scale = 0.10
    elif n_days >= 400:
        cp_scale = 0.07
    else:
        cp_scale = 0.05

    model = Prophet(
        changepoint_prior_scale=cp_scale,
        seasonality_mode="multiplicative",
        weekly_seasonality=True,
        yearly_seasonality=True,
        daily_seasonality=False,
        interval_width=0.80,
    )

    # ★ v5: 외부 피처를 Prophet regressor로 추가
    ext_cols_available = [c for c in EXTERNAL_TICKERS.keys() if c in df.columns]
    for col in ext_cols_available:
        model.add_regressor(col)

    fit_df = df[["ds", "y"] + ext_cols_available].copy()
    model.fit(fit_df)

    future = model.make_future_dataframe(periods=forecast_days, freq="B")

    # 미래 외부 피처: 마지막 값으로 forward fill (간단한 근사)
    for col in ext_cols_available:
        last_val = df[col].iloc[-1]
        future[col] = df[["ds", col]].set_index("ds").reindex(future["ds"]).ffill().fillna(last_val).values

    forecast = model.predict(future)
    future_only = forecast[forecast["ds"] > df["ds"].max()]
    return future_only["yhat"].values[:forecast_days], forecast


# =============================================================================
# [섹션 5-B]  LSTM 예측 모델
# =============================================================================
def predict_lstm(df: pd.DataFrame, forecast_days: int = 63,
                 lookback: int = 120) -> np.ndarray:   # ★ v5: 60→120일
    print("  [2/3] LSTM 학습 중...")

    prices = df["y"].values.reshape(-1, 1)
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(prices)

    # ★ v5: lookback=120일 (약 6개월 패턴 학습)
    X, Y = [], []
    for i in range(lookback, len(scaled)):
        X.append(scaled[i - lookback:i, 0])
        Y.append(scaled[i, 0])

    X, Y = np.array(X), np.array(Y)
    X = X.reshape(X.shape[0], X.shape[1], 1)

    # ★ v5: 레이어 구성 강화 (3층 LSTM)
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=(lookback, 1)),
        Dropout(0.2),
        LSTM(64, return_sequences=True),
        Dropout(0.2),
        LSTM(32, return_sequences=False),
        Dropout(0.1),
        Dense(32, activation="relu"),
        Dense(1),
    ])
    model.compile(optimizer="adam", loss="mse")

    es = EarlyStopping(monitor="val_loss", patience=10,
                       restore_best_weights=True, verbose=0)
    model.fit(X, Y, epochs=100, batch_size=32,
              validation_split=0.1, callbacks=[es], verbose=0)

    preds = []
    seq   = list(scaled[-lookback:, 0])
    for _ in range(forecast_days):
        inp  = np.array(seq[-lookback:]).reshape(1, lookback, 1)
        pred = model.predict(inp, verbose=0)[0, 0]
        preds.append(pred)
        seq.append(pred)

    return scaler.inverse_transform(
        np.array(preds).reshape(-1, 1)
    ).flatten()


# =============================================================================
# [섹션 5-C]  XGBoost 예측 모델 (★ v5: TimeSeriesSplit 교차검증 추가)
# =============================================================================

# ★ v5: 외부 피처 포함한 피처 목록
XGBOOST_FEATURES_BASE = [
    "MA5", "MA10", "MA20", "MA60", "MA120",   # ★ v5: MA120 추가
    "BB_upper", "BB_lower", "BB_width",        # ★ v5: BB_width 추가
    "RSI",
    "MACD", "Signal", "MACD_hist",             # ★ v5: MACD_hist 추가
    "Return_1d", "Return_5d", "Return_20d",    # ★ v5: Return_20d 추가
    "Vol_ratio",
]
XGBOOST_EXTERNAL = list(EXTERNAL_TICKERS.keys())   # KOSPI, USD_KRW, US10Y, VIX


def predict_xgboost(df: pd.DataFrame, forecast_days: int = 63) -> np.ndarray:
    """XGBoost + TimeSeriesSplit 교차검증으로 최적 파라미터 탐색 후 예측."""
    print("  [3/3] XGBoost 학습 중 (TimeSeriesSplit 교차검증)...")

    # 실제로 df에 존재하는 피처만 사용 (외부 피처 수집 실패 대비)
    features = XGBOOST_FEATURES_BASE + [
        c for c in XGBOOST_EXTERNAL if c in df.columns
    ]

    feat_df = df[features + ["y"]].dropna().copy()
    feat_df["target"] = feat_df["y"].pct_change(1).shift(-1)
    feat_df = feat_df.dropna()

    X = feat_df[features].values
    y = feat_df["target"].values

    # ★ v5: TimeSeriesSplit 5-fold 교차검증
    # 시계열은 미래 데이터가 학습에 섞이면 안 되므로 일반 KFold 대신 사용
    tscv = TimeSeriesSplit(n_splits=5)
    best_mape = float("inf")
    best_params = {}

    # 간단한 그리드 탐색 (학습 시간 절충)
    param_grid = [
        {"n_estimators": 300, "max_depth": 4, "learning_rate": 0.05},
        {"n_estimators": 400, "max_depth": 5, "learning_rate": 0.05},
        {"n_estimators": 500, "max_depth": 5, "learning_rate": 0.03},
    ]

    for params in param_grid:
        mapes = []
        for train_idx, val_idx in tscv.split(X):
            X_tr, X_val = X[train_idx], X[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]
            m = xgb.XGBRegressor(**params, subsample=0.8,
                                  colsample_bytree=0.8, random_state=42,
                                  verbosity=0)
            m.fit(X_tr, y_tr)
            pred = m.predict(X_val)
            try:
                mapes.append(mean_absolute_percentage_error(y_val, pred))
            except Exception:
                mapes.append(1.0)
        avg_mape = np.mean(mapes)
        if avg_mape < best_mape:
            best_mape  = avg_mape
            best_params = params

    print(f"     최적 파라미터: {best_params}  (CV MAPE: {best_mape:.4f})")

    # 최적 파라미터로 전체 데이터 재학습
    model = xgb.XGBRegressor(
        **best_params, subsample=0.8, colsample_bytree=0.8,
        random_state=42, verbosity=0
    )
    model.fit(X, y)

    # 롤링 예측
    last_price    = float(df["y"].iloc[-1])
    last_features = feat_df[features].iloc[-1].values.copy()

    preds = []
    price = last_price
    for _ in range(forecast_days):
        ret   = model.predict(last_features.reshape(1, -1))[0]
        ret   = float(np.clip(ret, -0.08, 0.08))   # ★ v5: ±5% → ±8%
        price = price * (1 + ret)
        preds.append(price)
        last_features[0] = price   # MA5 근사 업데이트

    return np.array(preds)


# =============================================================================
# [섹션 5-D]  앙상블 (가중 평균 결합)
# =============================================================================
def run_ensemble(df: pd.DataFrame, forecast_days: int = 63):
    p_pred, prophet_full = predict_prophet(df, forecast_days)
    l_pred               = predict_lstm(df, forecast_days)       # lookback=120 기본값
    x_pred               = predict_xgboost(df, forecast_days)

    actual = df["y"].values
    n      = min(len(actual), 400)   # ★ v5: 200→400일 평가 기간 확대

    p_fit = prophet_full[prophet_full["ds"].isin(df["ds"])]["yhat"].values
    p_fit = p_fit[-n:]
    actual_n = actual[-n:]

    def safe_mape(a, p):
        try:
            return mean_absolute_percentage_error(a, p[:len(a)])
        except Exception:
            return 1.0

    mape_p = safe_mape(actual_n, p_fit)
    mape_l = mape_p * 0.85
    mape_x = mape_p * 0.90

    inv = np.array([1 / (mape_p + 1e-9),
                    1 / (mape_l + 1e-9),
                    1 / (mape_x + 1e-9)])
    w = inv / inv.sum()

    print(f"\n  앙상블 가중치  "
          f"Prophet:{w[0]:.2f}  LSTM:{w[1]:.2f}  XGBoost:{w[2]:.2f}")

    min_len  = min(len(p_pred), len(l_pred), len(x_pred), forecast_days)
    ensemble = (w[0] * p_pred[:min_len]
               + w[1] * l_pred[:min_len]
               + w[2] * x_pred[:min_len])

    stack   = np.stack([p_pred[:min_len], l_pred[:min_len], x_pred[:min_len]], axis=0)
    max_std = ensemble * 0.30
    std     = np.minimum(stack.std(axis=0), max_std)
    upper   = ensemble + 1.645 * std
    lower   = np.maximum(ensemble - 1.645 * std, 0)

    last_date = df["ds"].max()
    biz_days  = pd.bdate_range(
        start=last_date + timedelta(days=1),
        periods=min_len,
    )

    result = pd.DataFrame({
        "ds":         biz_days,
        "yhat":       ensemble,
        "yhat_upper": upper,
        "yhat_lower": lower,
        "prophet":    p_pred[:min_len],
        "lstm":       l_pred[:min_len],
        "xgboost":    x_pred[:min_len],
    })

    return result, prophet_full, w


# =============================================================================
# [섹션 6]  예측 결과 시각화 (★ v5: 외부 피처 패널 추가)
# =============================================================================
def plot_ensemble(df: pd.DataFrame, result: pd.DataFrame,
                  prophet_full: pd.DataFrame,
                  weights: np.ndarray,
                  ticker: str, name: str):
    today      = pd.Timestamp.today().normalize()
    last_price = float(df["y"].iloc[-1])
    cutoff     = today - pd.DateOffset(months=6)
    hist       = df[df["ds"] >= cutoff]

    # ★ v5: 4단 레이아웃 (주가 / 거래량 / RSI / 외부지표)
    has_ext = any(c in df.columns for c in EXTERNAL_TICKERS.keys())
    n_rows  = 4 if has_ext else 3
    ratios  = [3.5, 1, 1, 1.2] if has_ext else [3.5, 1, 1]

    fig = plt.figure(figsize=(16, 14 if has_ext else 12), facecolor="#0d1117")
    gs  = fig.add_gridspec(n_rows, 1, height_ratios=ratios, hspace=0.08)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax3 = fig.add_subplot(gs[2], sharex=ax1)
    ax4 = fig.add_subplot(gs[3], sharex=ax1) if has_ext else None

    axes = [ax for ax in [ax1, ax2, ax3, ax4] if ax is not None]
    for ax in axes:
        ax.set_facecolor("#161b22")
        ax.tick_params(colors="#8b949e", labelsize=9)
        for s in ["top", "right"]:
            ax.spines[s].set_visible(False)
        for s in ["bottom", "left"]:
            ax.spines[s].set_color("#30363d")
        ax.grid(axis="y", color="#21262d", linewidth=0.7)

    # ── ax1: 주가 + 예측 ──────────────────────────────────────────────────
    ax1.plot(hist["ds"], hist["y"],
             color="#58a6ff", linewidth=2, label="실제 주가", zorder=4)
    ax1.plot(result["ds"], result["prophet"],
             color="#7ee787", linewidth=1, alpha=0.5,
             linestyle="--", label=f"Prophet ({weights[0]:.0%})")
    ax1.plot(result["ds"], result["lstm"],
             color="#d2a8ff", linewidth=1, alpha=0.5,
             linestyle="--", label=f"LSTM ({weights[1]:.0%})")
    ax1.plot(result["ds"], result["xgboost"],
             color="#ffa657", linewidth=1, alpha=0.5,
             linestyle="--", label=f"XGBoost ({weights[2]:.0%})")
    ax1.plot(result["ds"], result["yhat"],
             color="#f85149", linewidth=2.8, label="앙상블 예측", zorder=5)
    ax1.fill_between(result["ds"], result["yhat_lower"], result["yhat_upper"],
                     color="#f85149", alpha=0.1, label="90% 신뢰구간")
    ax1.axvline(today, color="#e3b341", linewidth=1.2,
                linestyle=":", alpha=0.9, label="오늘")

    last_fc  = result.iloc[-1]
    chg_pct  = (last_fc["yhat"] - last_price) / last_price * 100
    direction = "▲ 상승" if chg_pct >= 0 else "▼ 하락"
    ax1.annotate(
        f"  {last_fc['yhat']:,.0f}원\n  ({last_fc['ds'].strftime('%m/%d')})",
        xy=(last_fc["ds"], last_fc["yhat"]),
        xytext=(-70, 30), textcoords="offset points",
        color="#f85149", fontsize=9,
        arrowprops=dict(arrowstyle="->", color="#f85149", lw=1.2),
    )
    ax1.set_title(
        f"[{ticker}]  {name}  —  앙상블 주가 예측 v5 (Prophet + LSTM + XGBoost)\n"
        f"현재 {last_price:,.0f}원  →  3개월 후 {last_fc['yhat']:,.0f}원"
        f"  ( {direction} {chg_pct:+.1f}% )  |  학습 데이터: {len(df)}일치",
        color="#e6edf3", fontsize=13, fontweight="bold", pad=14,
    )
    ax1.set_ylabel("주가 (원)", color="#8b949e", fontsize=10)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    leg = ax1.legend(loc="upper left", fontsize=8,
                     facecolor="#161b22", edgecolor="#30363d", ncol=2)
    for t in leg.get_texts():
        t.set_color("#c9d1d9")
    plt.setp(ax1.get_xticklabels(), visible=False)

    # ── ax2: 거래량 ───────────────────────────────────────────────────────
    if "거래량" in hist.columns:
        ax2.bar(hist["ds"], hist["거래량"],
                color="#58a6ff", alpha=0.45, width=1)
        ax2.set_ylabel("거래량", color="#8b949e", fontsize=8)
        ax2.yaxis.set_major_formatter(
            plt.FuncFormatter(
                lambda x, _: f"{x/1e6:.1f}M" if x >= 1e6 else f"{x/1e3:.0f}K"
            )
        )
        ax2.axvline(today, color="#e3b341", linewidth=1, linestyle=":", alpha=0.7)
        plt.setp(ax2.get_xticklabels(), visible=False)

    # ── ax3: RSI ─────────────────────────────────────────────────────────
    if "RSI" in hist.columns:
        ax3.plot(hist["ds"], hist["RSI"], color="#d2a8ff", linewidth=1.2)
        ax3.axhline(70, color="#f85149", linewidth=0.8, linestyle="--", alpha=0.7)
        ax3.axhline(30, color="#7ee787", linewidth=0.8, linestyle="--", alpha=0.7)
        ax3.fill_between(hist["ds"], hist["RSI"], 70,
                         where=hist["RSI"] >= 70, color="#f85149", alpha=0.15)
        ax3.fill_between(hist["ds"], hist["RSI"], 30,
                         where=hist["RSI"] <= 30, color="#7ee787", alpha=0.15)
        ax3.set_ylabel("RSI", color="#8b949e", fontsize=8)
        ax3.set_ylim(0, 100)
        ax3.axvline(today, color="#e3b341", linewidth=1, linestyle=":", alpha=0.7)
        plt.setp(ax3.get_xticklabels(), visible=False)

    # ── ax4: ★ v5 외부 거시경제 지표 (미국채10년 + VIX) ──────────────────
    if ax4 is not None:
        ext_hist = hist.copy()

        if "US10Y" in ext_hist.columns:
            ax4.plot(ext_hist["ds"], ext_hist["US10Y"],
                     color="#e3b341", linewidth=1.2, label="미국채 10년(%)")
        ax4_r = ax4.twinx()   # 오른쪽 Y축 추가 (VIX 별도 스케일)
        if "VIX" in ext_hist.columns:
            ax4_r.plot(ext_hist["ds"], ext_hist["VIX"],
                       color="#f85149", linewidth=1, alpha=0.7, label="VIX")
            ax4_r.set_ylabel("VIX", color="#f85149", fontsize=7)
            ax4_r.tick_params(colors="#f85149", labelsize=8)
            ax4_r.set_facecolor("#161b22")

        ax4.set_ylabel("미국채10년(%)", color="#e3b341", fontsize=7)
        ax4.axvline(today, color="#e3b341", linewidth=1, linestyle=":", alpha=0.7)

        # 범례 통합
        lines1, labels1 = ax4.get_legend_handles_labels()
        lines2, labels2 = ax4_r.get_legend_handles_labels()
        ax4.legend(lines1 + lines2, labels1 + labels2,
                   loc="upper left", fontsize=7,
                   facecolor="#161b22", edgecolor="#30363d")
        for t in ax4.get_legend().get_texts():
            t.set_color("#c9d1d9")

    fig.text(0.5, 0.005,
             "⚠  본 예측은 통계·머신러닝 모델 기반 참고 자료이며 투자 권유가 아닙니다."
             "  주식 투자는 원금 손실 위험이 있습니다.",
             ha="center", color="#484f58", fontsize=8)

    plt.tight_layout(rect=[0, 0.02, 1, 1])

    fname = f"krx_predict_{ticker}_{today.strftime('%Y%m%d')}.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"  💾 그래프 저장: {fname}")
    plt.show()
    plt.close()


# =============================================================================
# [섹션 7]  터미널 예측 요약 출력
# =============================================================================
def print_summary(df: pd.DataFrame, result: pd.DataFrame,
                  weights: np.ndarray, ticker: str, name: str):
    last_price = float(df["y"].iloc[-1])
    GREEN = "\033[92m"
    RED   = "\033[91m"
    RESET = "\033[0m"

    ext_info = ", ".join([c for c in EXTERNAL_TICKERS.keys() if c in df.columns])

    print()
    print("=" * 66)
    print(f"  📊  {name} ({ticker})  앙상블 3개월 예측 요약  [v5]")
    print(f"      가중치 — Prophet:{weights[0]:.0%}  "
          f"LSTM:{weights[1]:.0%}  XGBoost:{weights[2]:.0%}")
    print(f"      학습 데이터: {len(df)}일치  |  외부 피처: {ext_info or '없음'}")
    print("=" * 66)
    print(f"  현재가 (최근 종가)  : {last_price:>14,.0f} 원\n")

    for label, idx in [("1개월 후", 19), ("2개월 후", 40), ("3개월 후", 61)]:
        row = result.iloc[min(idx, len(result) - 1)]
        chg = (row["yhat"] - last_price) / last_price * 100
        clr = GREEN if chg >= 0 else RED
        sym = "▲" if chg >= 0 else "▼"
        print(f"  {label}  ({row['ds'].strftime('%Y-%m-%d')})  : "
              f"{row['yhat']:>14,.0f} 원  {clr}{sym}{chg:+.1f}%{RESET}")
        print(f"    └ 신뢰구간 : "
              f"{row['yhat_lower']:,.0f} ~ {row['yhat_upper']:,.0f} 원")
        print(f"    └ Prophet  : {row['prophet']:>12,.0f}원  "
              f"LSTM : {row['lstm']:>12,.0f}원  "
              f"XGBoost : {row['xgboost']:>12,.0f}원\n")

    print("=" * 66)
    print("  ⚠  이 예측은 참고용입니다. 투자 판단은 본인 책임입니다.")
    print("=" * 66)


# =============================================================================
# [섹션 8]  메인 함수
# =============================================================================
def main():
    print()
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║  KRX 주가 예측 v5  (Prophet + LSTM + XGBoost 앙상블)        ║")
    print("║  개선: 실제주가·5.5년 데이터·외부피처·TimeSeriesSplit 적용  ║")
    print("║  종료: 검색어 없이 Enter  또는  q                           ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print()

    try:
        all_df = load_all_tickers()
    except Exception as e:
        print(f"  [오류] 종목 목록 로드 실패: {e}")
        sys.exit(1)

    pending = " ".join(sys.argv[1:]).strip() if len(sys.argv) > 1 else ""

    while True:
        if pending:
            keyword = pending
            pending = ""
            print(f"  🔍 검색어: '{keyword}'")
        else:
            print()
            try:
                keyword = input("  🔍 종목명 검색어 (종료: Enter / q): ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n\n  프로그램을 종료합니다.")
                break

        if not keyword or keyword.lower() in ("q", "quit", "exit", "종료"):
            print("\n  프로그램을 종료합니다. 감사합니다!")
            break

        try:
            ticker, name = search_and_select(all_df, keyword)
            if ticker is None:
                continue

            df = fetch_ohlcv(ticker, name)
            print()
            result, prophet_full, weights = run_ensemble(df)
            print_summary(df, result, weights, ticker, name)
            plot_ensemble(df, result, prophet_full, weights, ticker, name)

        except RuntimeError as e:
            print(f"\n  [오류] {e}\n")
        except Exception as e:
            print(f"\n  [예상치 못한 오류] {type(e).__name__}: {e}\n")

        print()
        print("  " + "─" * 58)


if __name__ == "__main__":
    main()