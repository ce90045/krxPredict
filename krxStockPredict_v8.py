# =============================================================================
#  KRX 주가 예측 프로그램 v8  —  앙상블 모델 통합 최적화
#  (LightGBM + Multi-Feature LSTM + XGBoost + Walk-Forward Stacking)
#
#  ┌─ v6 vs v7 비교 및 v8 통합 전략 ───────────────────────────────────────┐
#  │                                                                        │
#  │  [v6 장점 → v8 계승]                                                  │
#  │    ✔ 상세한 80개+ Feature Engineering (ATR·52주위치·캔들·RS 등)       │
#  │    ✔ Walk-Forward Backtest 기반 실제 MAPE 가중치                       │
#  │    ✔ 4단 시각화 레이아웃 (주가·거래량·RSI·거시지표)                   │
#  │    ✔ 상세 터미널 요약 출력 (1·2·3개월 구간별)                         │
#  │    ✔ LightGBM/XGBoost TimeSeriesSplit 파라미터 탐색                   │
#  │    ✔ price_history 기반 MA·수익률 피처 재계산 (feature drift 방지)    │
#  │    ✔ 외부 피처(KOSPI·환율·금리·VIX) 수집 및 병합                     │
#  │                                                                        │
#  │  [v7 장점 → v8 계승]                                                  │
#  │    ✔ Feature Selection: 80개 중 상위 30개 자동 선별 (LightGBM 중요도) │
#  │    ✔ Dynamic Volatility Clipping: ±8% 하드코딩 → 종목별 σ×3 동적 적용│
#  │    ✔ Decay Factor: 장기 예측 평균회귀 (발산/폭락 방지)                │
#  │    ✔ LSTM 타깃 통일: 가격 예측 → 수익률 예측 (3모델 구조 통일)       │
#  │    ✔ LSTM 경량화: 64→32 유닛, epoch 30 (학습 속도 대폭 개선)         │
#  │    ✔ feat_cols 파라미터 공유: 모든 모델이 동일 피처셋 사용            │
#  │                                                                        │
#  │  [v8 신규 개선]                                                        │
#  │    ★ LSTM _update_rolling_features 적용 (v7 미적용 버그 수정)         │
#  │    ★ v7 _update_rolling_features 버그 수정                            │
#  │       (feat_cols 집합 체크 오류 → feat_dict 키 체크로 수정)           │
#  │    ★ Feature Selection 후 외부 피처 보장                              │
#  │       (v7: 선택 결과에 외부피처 제외 가능 → v8: 항상 포함 보장)       │
#  │    ★ XGBoost CV 파라미터 탐색 복원 (v7: 단일 파라미터 고정)          │
#  │    ★ Decay Rate 종목 변동성 연동 (v7: 0.97/0.98 고정 → v8: 동적)     │
#  │    ★ 신뢰구간 80% → 90% 복원 (v7: 80%로 축소, 불확실성 과소 표현)    │
#  │    ★ 시각화 4단 레이아웃 복원 + 피처 목록 출력                        │
#  │    ★ 상세 터미널 요약 복원 (v7: 단순 1줄 요약으로 축소)               │
#  └────────────────────────────────────────────────────────────────────────┘
#
#  ┌─ 권장 파이썬 버전 ──────────────────────────────────────────────────────┐
#  │  Python 3.10 ~ 3.11  (TensorFlow 가 3.12 이상을 아직 완전 지원 안 함)  │
#  └────────────────────────────────────────────────────────────────────────┘
#
#  ┌─ 필수 라이브러리 ───────────────────────────────────────────────────────┐
#  │  pip install yfinance lightgbm xgboost scikit-learn tensorflow        │
#  │             matplotlib pandas numpy html5lib lxml                     │
#  └────────────────────────────────────────────────────────────────────────┘
#
#  ┌─ 실행 방법 ────────────────────────────────────────────────────────────┐
#  │  python krx_stock_predict_v8.py           # 실행 후 검색어 입력       │
#  │  python krx_stock_predict_v8.py 우리금융  # 인수로 종목 전달          │
#  │  종료: 검색어 없이 Enter  또는  q / quit / exit / 종료                │
#  └────────────────────────────────────────────────────────────────────────┘
# =============================================================================


# -----------------------------------------------------------------------------
# [섹션 0]  표준 라이브러리 및 환경 설정
# -----------------------------------------------------------------------------
import os
import sys
import warnings
warnings.filterwarnings("ignore")

# TensorFlow C++ 로그 레벨: 3=ERROR만 출력 (INFO·WARNING 억제)
os.environ["TF_CPP_MIN_LOG_LEVEL"]  = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"    # oneDNN 부동소수점 경고 억제
os.environ["CUDA_VISIBLE_DEVICES"]  = "-1"   # GPU 비활성화 → CPU 전용 실행

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from datetime import datetime, timedelta


# =============================================================================
# [섹션 0-A]  한글 폰트 설정
#   matplotlib 기본 폰트는 한글을 지원하지 않아 □□□ 로 깨짐.
#   OS별로 한글 폰트 이름이 다르므로 후보 목록을 순서대로 시도함.
# =============================================================================
def set_korean_font():
    candidates = [
        "Malgun Gothic",        # Windows 기본 한글 폰트
        "맑은 고딕",             # Windows 한글명
        "AppleGothic",          # macOS 기본 한글 폰트
        "Apple SD Gothic Neo",  # macOS 최신 한글 폰트
        "NanumGothic",          # Linux/공통 나눔고딕
        "Nanum Gothic",
        "나눔고딕",
    ]
    # 현재 시스템에 설치된 폰트 이름 집합
    available = {f.name for f in fm.fontManager.ttflist}
    for font in candidates:
        if font in available:
            plt.rcParams["font.family"] = font
            break
    # 마이너스 기호 깨짐 방지: 유니코드 마이너스 대신 아스키 하이픈 사용
    plt.rcParams["axes.unicode_minus"] = False

set_korean_font()


# =============================================================================
# [섹션 1]  필수 라이브러리 설치 여부 사전 점검
#   무거운 라이브러리를 임포트하기 전에 먼저 확인.
#   빠진 패키지가 있으면 설치 안내 후 즉시 종료.
# =============================================================================
def check_imports():
    required = {
        "yfinance":   "yfinance",
        "lightgbm":   "lightgbm",       # LightGBM (Feature Selection + 예측 모델)
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
        print("  pip install " + " ".join(missing))
        print("=" * 60)
        sys.exit(1)

check_imports()


# -----------------------------------------------------------------------------
# check_imports() 통과 후 안전하게 임포트
# -----------------------------------------------------------------------------
import yfinance as yf
import lightgbm as lgb                      # Feature Selection + 예측 모델
import xgboost as xgb
from sklearn.preprocessing import RobustScaler
# RobustScaler: 중앙값/IQR 기반 정규화 → 주가 급등락 이상치에 강건
# (MinMaxScaler는 최댓값에 민감해 급등락 시 스케일 왜곡 발생)
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import TimeSeriesSplit
# TimeSeriesSplit: 미래 데이터가 학습에 섞이는 누수(leakage) 방지
# 일반 KFold는 시계열 데이터에 절대 사용 금지

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
# BatchNormalization: 레이어 간 입력 분포 정규화 → 학습 안정화·속도 향상
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
# EarlyStopping    : 검증 손실 개선 없으면 조기 종료 (과적합 방지)
# ReduceLROnPlateau: 학습 정체 시 학습률 자동 감소 → 수렴 촉진

import logging
logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
logging.getLogger("lightgbm").setLevel(logging.WARNING)


# =============================================================================
# [섹션 2]  KRX 전체 종목 목록 관리
#   출처: KIND(한국거래소 기업공시채널)
#   성능 최적화: 당일 한 번만 수신 후 CSV 캐시에 저장
# =============================================================================
CACHE_FILE = "krx_tickers_cache.csv"


def _today_str() -> str:
    """오늘 날짜를 'YYYYMMDD' 형식으로 반환."""
    return datetime.today().strftime("%Y%m%d")


def _cache_is_fresh() -> bool:
    """캐시 파일이 오늘 날짜로 저장된 경우 True 반환."""
    if not os.path.exists(CACHE_FILE):
        return False
    try:
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            first = f.readline().strip()
        return first.replace("# saved_date=", "").strip() == _today_str()
    except Exception:
        return False


def _load_cache() -> pd.DataFrame:
    """캐시 CSV 로드. comment="#" 으로 날짜 헤더 행 자동 스킵."""
    return pd.read_csv(CACHE_FILE, comment="#", dtype=str, encoding="utf-8")


def _save_cache(df: pd.DataFrame):
    """날짜 메타데이터 + DataFrame을 CSV로 저장."""
    with open(CACHE_FILE, "w", encoding="utf-8", newline="") as f:
        f.write(f"# saved_date={_today_str()}\n")
        df.to_csv(f, index=False)


def _fetch_tickers_kind(market_type: str) -> list:
    """KIND에서 특정 시장의 상장 종목 목록을 HTML로 수신.

    Args:
        market_type: "stockMkt"(KOSPI) 또는 "kosdaqMkt"(KOSDAQ)
    Returns:
        종목 정보 딕셔너리 리스트
    """
    url = (
        "http://kind.krx.co.kr/corpgeneral/corpList.do"
        f"?currentPageSize=5000&pageIndex=1"
        f"&method=download&searchType=13&marketType={market_type}"
    )
    # encoding="euc-kr": KIND 서버는 EUC-KR 인코딩으로 응답
    # converters: 종목코드를 6자리 0-패딩 문자열로 강제 변환
    tables = pd.read_html(
        url, header=0,
        converters={"종목코드": lambda x: str(x).zfill(6)},
        encoding="euc-kr",
    )
    return tables[0].to_dict("records")


def _fallback_tickers() -> list:
    """KIND 서버 장애 시 주요 종목 하드코딩 목록."""
    return [
        {"시장": "KOSPI",  "종목명": "삼성전자",        "종목코드": "005930"},
        {"시장": "KOSPI",  "종목명": "SK하이닉스",       "종목코드": "000660"},
        {"시장": "KOSPI",  "종목명": "LG에너지솔루션",   "종목코드": "373220"},
        {"시장": "KOSPI",  "종목명": "삼성바이오로직스", "종목코드": "207940"},
        {"시장": "KOSPI",  "종목명": "현대차",           "종목코드": "005380"},
        {"시장": "KOSPI",  "종목명": "기아",             "종목코드": "000270"},
        {"시장": "KOSPI",  "종목명": "POSCO홀딩스",      "종목코드": "005490"},
        {"시장": "KOSPI",  "종목명": "LG화학",           "종목코드": "051910"},
        {"시장": "KOSPI",  "종목명": "셀트리온",         "종목코드": "068270"},
        {"시장": "KOSPI",  "종목명": "카카오",           "종목코드": "035720"},
        {"시장": "KOSPI",  "종목명": "NAVER",            "종목코드": "035420"},
        {"시장": "KOSPI",  "종목명": "KB금융",           "종목코드": "105560"},
        {"시장": "KOSPI",  "종목명": "신한지주",         "종목코드": "055550"},
        {"시장": "KOSPI",  "종목명": "우리금융지주",     "종목코드": "316140"},
        {"시장": "KOSPI",  "종목명": "하나금융지주",     "종목코드": "086790"},
        {"시장": "KOSPI",  "종목명": "삼성SDI",          "종목코드": "006400"},
        {"시장": "KOSPI",  "종목명": "한국전력",         "종목코드": "015760"},
        {"시장": "KOSDAQ", "종목명": "에코프로비엠",     "종목코드": "247540"},
        {"시장": "KOSDAQ", "종목명": "에코프로",         "종목코드": "086520"},
        {"시장": "KOSDAQ", "종목명": "HLB",              "종목코드": "028300"},
    ]


def load_all_tickers() -> pd.DataFrame:
    """KOSPI + KOSDAQ 전체 상장 종목 목록 DataFrame 반환.

    처리 흐름:
        1. 당일 캐시 → 바로 로드
        2. 캐시 없음 → KIND 수신 → 캐시 저장
        3. KIND 실패 → fallback 목록 사용
    """
    if _cache_is_fresh():
        df = _load_cache()
        print(f"  ✔ 당일 캐시에서 종목 목록 로드 ({len(df):,}개)\n")
        return df

    print("  📦 KIND(기업공시채널)에서 전체 종목 목록 수신 중...")
    rows = []
    for mkt_type, mkt_name in {"stockMkt": "KOSPI", "kosdaqMkt": "KOSDAQ"}.items():
        try:
            items  = _fetch_tickers_kind(mkt_type)
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

    # drop_duplicates: 중복 종목코드 제거 (첫 번째 항목 유지)
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
#   SQL LIKE '%keyword%' 방식 부분 일치 검색
# =============================================================================
def search_and_select(all_df: pd.DataFrame, keyword: str):
    """종목명에서 keyword를 포함하는 종목 검색 후 사용자 선택.

    Returns:
        (종목코드, 종목명) 튜플. 취소 시 (None, None).
    """
    keyword  = keyword.strip()
    filtered = all_df[
        all_df["종목명"].str.contains(keyword, na=False, case=False)
    ].reset_index(drop=True)

    if filtered.empty:
        print(f"\n  검색 결과 없음: '{keyword}'\n")
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
# [섹션 4]  외부 거시경제 지표 수집
#   KOSPI지수, 원달러환율, 미국채10년, VIX 공포지수
#   금융주(우리금융 등)는 금리·환율의 영향을 특히 많이 받음
# =============================================================================
EXTERNAL_TICKERS = {
    "KOSPI":   "^KS11",  # KOSPI 종합지수
    "USD_KRW": "KRW=X",  # 원달러 환율
    "US10Y":   "^TNX",   # 미국채 10년 금리 (금융주 핵심 변수)
    "VIX":     "^VIX",   # 시장 변동성 지수 (공포지수, 20↑=위험 구간)
}


def fetch_external_features(start_dt: datetime, end_dt: datetime) -> pd.DataFrame:
    """외부 거시경제 지표 수집 → 날짜 기준 병합 DataFrame 반환.

    공휴일 등 날짜 불일치: outer join + ffill + bfill로 보완.
    수집 실패한 지표는 건너뜀 (프로그램 중단 없이 계속 진행).

    Returns:
        컬럼: ds, KOSPI, USD_KRW, US10Y, VIX (수집 성공한 것만)
    """
    print("  📡 외부 거시경제 지표 수집 중 (KOSPI·환율·금리·VIX)...")
    ext_frames = []

    for col, yf_sym in EXTERNAL_TICKERS.items():
        try:
            raw = yf.download(
                yf_sym,
                start=start_dt.strftime("%Y-%m-%d"),
                end=end_dt.strftime("%Y-%m-%d"),
                auto_adjust=False,  # 실제 시장 가격 사용 (수정 주가 문제 방지)
                progress=False,
            )
            if raw.empty:
                print(f"     [경고] {col}({yf_sym}) 데이터 없음")
                continue
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.get_level_values(0)
            raw = raw.reset_index()
            raw.columns = [str(c) for c in raw.columns]
            date_col = "Date" if "Date" in raw.columns else raw.columns[0]
            tmp = raw[[date_col, "Close"]].copy()
            tmp.columns = ["ds", col]
            tmp["ds"] = pd.to_datetime(tmp["ds"]).dt.tz_localize(None)
            ext_frames.append(tmp.set_index("ds"))
            print(f"     {col}: {len(tmp)}일치 완료")
        except Exception as e:
            print(f"     [경고] {col} 수집 실패: {e}")

    if not ext_frames:
        print("  [경고] 외부 지표 수집 전체 실패 → 외부 피처 없이 진행")
        return pd.DataFrame()

    merged = pd.concat(ext_frames, axis=1, join="outer").ffill().bfill()
    merged = merged.reset_index().rename(columns={"index": "ds"})
    print(f"  ✔ 외부 지표 병합 완료\n")
    return merged


# =============================================================================
# [섹션 5]  80개+ Feature Engineering
#   v6의 풍부한 피처셋을 유지.
#   v7의 Feature Selection이 이 피처들 중 상위 30개를 자동 선별함.
# =============================================================================
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """80개+ 기술적·거시적 피처 생성.

    피처 그룹:
        이동평균      : MA5~200, 기울기(slope), 크로스오버 신호
        볼린저밴드    : 20일·60일 — 상단/하단/폭/%B (밴드 내 위치)
        RSI           : 14일·28일, RSI 기울기
        MACD          : MACD선·시그널선·히스토그램·크로스오버 신호
        ATR           : 14일·20일 평균진폭범위, 가격 대비 비율
        모멘텀 수익률 : 1·3·5·10·20·60일
        변동성        : 10·20·60일 rolling std (연환산 252일 기준)
        거래량        : Vol_ratio, Vol_surge(폭증신호), 거래대금 비율
        52주 위치     : 52주 고/저 대비 현재가 위치 (0=최저, 1=최고)
        캔들 패턴     : 몸통비율, 위/아래 꼬리 비율
        시장 상대강도 : 종목수익률 − KOSPI수익률 (금융주에 특히 유효)
        거시지표 파생 : 금리변화(5·20일), VIX변화·이동평균·고변동성신호,
                       환율변화(5·20일)
    """
    d = df.copy()
    c = d["종가"]    # 종가
    h = d["고가"]    # 고가
    l = d["저가"]    # 저가
    v = d["거래량"]  # 거래량

    # ── 이동평균 + 기울기 ─────────────────────────────────────────────────
    # slope: 현재 MA와 5일 전 MA의 차이 비율 → 추세 방향·강도
    for w in [5, 10, 20, 60, 120, 200]:
        d[f"MA{w}"]       = c.rolling(w).mean()
        d[f"MA{w}_slope"] = d[f"MA{w}"].diff(5) / (d[f"MA{w}"].shift(5) + 1e-9)

    # 골든/데드 크로스 신호 (단기 MA > 장기 MA이면 1, 아니면 0)
    d["MA5_20_cross"]  = (d["MA5"]  > d["MA20"]).astype(int)
    d["MA20_60_cross"] = (d["MA20"] > d["MA60"]).astype(int)

    # ── 볼린저 밴드 ───────────────────────────────────────────────────────
    # %B = (현재가 - 하단) / (상단 - 하단): 0=하단, 0.5=중심, 1=상단
    for w in [20, 60]:
        mid = c.rolling(w).mean()
        std = c.rolling(w).std()
        d[f"BB{w}_upper"] = mid + 2 * std
        d[f"BB{w}_lower"] = mid - 2 * std
        d[f"BB{w}_width"] = (d[f"BB{w}_upper"] - d[f"BB{w}_lower"]) / (mid + 1e-9)
        d[f"BB{w}_pct"]   = (c - d[f"BB{w}_lower"]) / (
            d[f"BB{w}_upper"] - d[f"BB{w}_lower"] + 1e-9
        )

    # ── RSI (14일·28일) + 기울기 ──────────────────────────────────────────
    # RSI 공식: 100 - 100/(1+RS), RS = 평균상승폭/평균하락폭
    for period in [14, 28]:
        delta = c.diff()
        gain  = delta.clip(lower=0).rolling(period).mean()
        loss  = (-delta.clip(upper=0)).rolling(period).mean()
        d[f"RSI{period}"] = 100 - 100 / (1 + gain / (loss + 1e-9))
    d["RSI14_slope"] = d["RSI14"].diff(5)  # RSI 방향 변화

    # ── MACD ──────────────────────────────────────────────────────────────
    # 단기(12일) EMA - 장기(26일) EMA → 추세 전환 신호
    ema12          = c.ewm(span=12, adjust=False).mean()
    ema26          = c.ewm(span=26, adjust=False).mean()
    d["MACD"]      = ema12 - ema26
    d["Signal"]    = d["MACD"].ewm(span=9, adjust=False).mean()
    d["MACD_hist"] = d["MACD"] - d["Signal"]         # 히스토그램
    d["MACD_cross"]= (d["MACD"] > d["Signal"]).astype(int)

    # ── ATR (Average True Range) — 절대 변동성 ───────────────────────────
    # TR = max(고-저, |고-전일종가|, |저-전일종가|)
    # ATR_pct = ATR/종가: 가격 수준에 무관한 상대적 변동성
    prev_c     = c.shift(1)
    tr         = pd.concat([
        h - l,
        (h - prev_c).abs(),
        (l - prev_c).abs(),
    ], axis=1).max(axis=1)
    d["ATR14"]   = tr.rolling(14).mean()
    d["ATR20"]   = tr.rolling(20).mean()
    d["ATR_pct"] = d["ATR14"] / (c + 1e-9)

    # ── 모멘텀 수익률 ─────────────────────────────────────────────────────
    for period in [1, 3, 5, 10, 20, 60]:
        d[f"Return_{period}d"] = c.pct_change(period)

    # ── 변동성 (연환산) ───────────────────────────────────────────────────
    # 일간 수익률의 rolling std × √252 (연간 영업일 수)
    ret1d = c.pct_change(1)
    for w in [10, 20, 60]:
        d[f"Volatility_{w}d"] = ret1d.rolling(w).std() * np.sqrt(252)

    # ── 거래량 파생 ───────────────────────────────────────────────────────
    d["Vol_MA20"]  = v.rolling(20).mean()
    d["Vol_ratio"] = v / (d["Vol_MA20"] + 1e-9)
    d["Vol_MA5"]   = v.rolling(5).mean()
    # Vol_surge: 거래량이 20일 평균의 2배 이상 → 이상 거래 신호
    d["Vol_surge"] = (d["Vol_ratio"] > 2.0).astype(int)
    d["PV"]        = c * v                                # 거래대금
    d["PV_ratio"]  = d["PV"] / (d["PV"].rolling(20).mean() + 1e-9)

    # ── 52주 가격 위치 ────────────────────────────────────────────────────
    # 0 = 52주 최저, 1 = 52주 최고
    d["High_52w"] = h.rolling(252).max()
    d["Low_52w"]  = l.rolling(252).min()
    d["Pos_52w"]  = (c - d["Low_52w"]) / (d["High_52w"] - d["Low_52w"] + 1e-9)

    # ── 캔들 패턴 ─────────────────────────────────────────────────────────
    body           = (c - d["시가"]).abs()
    candle_range   = h - l + 1e-9
    d["Body_ratio"]= body / candle_range           # 몸통 비율
    d["Upper_tail"]= (h - pd.concat([c, d["시가"]], axis=1).max(axis=1)) / candle_range
    d["Lower_tail"]= (pd.concat([c, d["시가"]], axis=1).min(axis=1) - l) / candle_range

    # ── 시장 상대강도 (★매우 중요) ───────────────────────────────────────
    # 종목 수익률 - KOSPI 수익률: 양수=시장 대비 강세
    if "KOSPI" in d.columns:
        kospi_ret         = d["KOSPI"].pct_change(1)
        d["RS_1d"]        = ret1d - kospi_ret
        d["RS_20d"]       = c.pct_change(20) - d["KOSPI"].pct_change(20)
        d["KOSPI_ret_1d"] = kospi_ret
        d["KOSPI_ret_5d"] = d["KOSPI"].pct_change(5)

    # ── 거시지표 파생 ─────────────────────────────────────────────────────
    if "US10Y" in d.columns:
        d["US10Y_chg_5d"]  = d["US10Y"].diff(5)   # 금리 5일 변화
        d["US10Y_chg_20d"] = d["US10Y"].diff(20)  # 금리 20일 변화
    if "VIX" in d.columns:
        d["VIX_chg_5d"] = d["VIX"].diff(5)
        d["VIX_MA20"]   = d["VIX"].rolling(20).mean()
        # VIX > 25 이면 고변동성(리스크오프) 국면
        d["VIX_regime"] = (d["VIX"] > 25).astype(int)
    if "USD_KRW" in d.columns:
        d["KRW_chg_5d"]  = d["USD_KRW"].diff(5)
        d["KRW_chg_20d"] = d["USD_KRW"].diff(20)

    return d.dropna().reset_index(drop=True)


def get_all_feature_columns(df: pd.DataFrame) -> list:
    """학습에 사용 가능한 전체 피처 컬럼 반환.

    비피처(ds·y·원본 OHLCV·외부 원시값)를 제외한 파생 피처만 반환.
    이 목록에서 Feature Selection이 상위 30개를 선별함.
    """
    exclude = {"ds", "y", "종가", "시가", "고가", "저가", "거래량",
               "KOSPI", "USD_KRW", "US10Y", "VIX"}
    return [c for c in df.columns
            if c not in exclude
            and df[c].dtype in [np.float64, np.int64, float, int]]


# =============================================================================
# [섹션 5-B]  ★v7 계승: Feature Selection (자동 피처 선별)
#   v6 문제: 80개 피처 전부 사용 → 노이즈 피처로 인한 과적합 위험
#   v7·v8 해결: LightGBM 피처 중요도로 상위 30개만 선별
#
#   선별 방식:
#     1. 전체 피처로 LightGBM 빠른 학습 (n_estimators=100)
#     2. feature_importances_ 기준 상위 top_n개 추출
#     3. ★v8 추가: 외부 피처 항상 보장
#        (v7: 선택 결과에 KOSPI·VIX 등이 빠질 수 있었음)
# =============================================================================
def select_top_features(df: pd.DataFrame, top_n: int = 30) -> list:
    """LightGBM 피처 중요도 기반으로 상위 top_n개 피처 선별.

    ★v8: 외부 거시지표(KOSPI·USD_KRW·US10Y·VIX)는 중요도와 무관하게
         항상 포함 보장 (금융주 분석에 필수적이므로 강제 포함).

    Args:
        df    : add_features()가 처리한 DataFrame
        top_n : 선택할 피처 수 (기본 30개)

    Returns:
        선별된 피처 컬럼명 리스트 (외부 피처 포함 보장)
    """
    all_feats = get_all_feature_columns(df)

    # 타깃: 다음 날 수익률
    data = df[all_feats + ["y"]].dropna().copy()
    data["target"] = data["y"].pct_change(1).shift(-1)
    data = data.dropna()

    X = data[all_feats].values
    y = data["target"].values

    # 빠른 중요도 평가용 경량 모델 (n_estimators=100)
    selector = lgb.LGBMRegressor(n_estimators=100, random_state=42,
                                  verbose=-1, force_col_wise=True)
    selector.fit(X, y)

    # 중요도 내림차순 정렬 후 상위 top_n 선택
    importance  = selector.feature_importances_
    feat_imp    = sorted(zip(all_feats, importance), key=lambda x: x[1], reverse=True)
    top_feats   = [f[0] for f in feat_imp[:top_n]]

    # ★v8: 외부 피처 항상 포함 보장
    # (v7의 버그: 외부 피처가 top_n에서 빠질 경우 거시 정보가 손실됨)
    external_in_df = [c for c in EXTERNAL_TICKERS.keys() if c in df.columns]
    for ext in external_in_df:
        if ext not in top_feats:
            # 최하위 중요도 피처를 제거하고 외부 피처로 교체
            top_feats.pop()
            top_feats.append(ext)

    print(f"  [v8] 피처 선택 완료: 전체 {len(all_feats)}개 → 상위 {len(top_feats)}개 선별")
    print(f"       상위 5개: {[f[0] for f in feat_imp[:5]]}")
    return top_feats


# =============================================================================
# [섹션 5-C]  ★v7 계승: Dynamic Volatility Clipping + Decay Factor
#
#   v6 문제:
#     - 클리핑: ±8% 하드코딩 → 저변동성 종목은 과도, 고변동성 종목은 부족
#     - Decay: 없음 → 장기 롤링 예측 시 수익률이 누적되어 기하급수적 발산
#
#   v7·v8 해결:
#     - Dynamic Clipping: 과거 일간 수익률 표준편차 × 3 (종목별 자동 조정)
#     - Decay Factor: 예측 스텝이 늘수록 수익률을 감쇠 → 장기 평균회귀
#
#   ★v8 개선: Decay Rate를 변동성에 연동 (v7: 0.97/0.98 고정값)
#     - 변동성 낮은 종목: decay 느리게 (0.99)
#     - 변동성 높은 종목: decay 빠르게 (0.95)
#     → 종목 특성에 맞는 평균회귀 속도 자동 조정
# =============================================================================
def get_dynamic_bounds(df: pd.DataFrame) -> float:
    """과거 일간 수익률 표준편차 × 3 을 클리핑 상한으로 반환.

    예: 일간 변동성 1% → 클리핑 ±3%
        일간 변동성 3% → 클리핑 ±9%
    고정값 ±8% 대비 종목 특성에 맞게 자동 조정됨.
    """
    daily_std = df["y"].pct_change().std()
    return float(daily_std * 3)


def get_decay_rate(df: pd.DataFrame) -> float:
    """★v8: 변동성 기반 Decay Rate 동적 계산.

    Decay Rate: 롤링 예측 시 각 스텝마다 수익률에 곱하는 감쇠 계수.
    decay_rate^step 이 1에 가까울수록 감쇠 느림, 0에 가까울수록 빠름.

    변동성과 decay rate의 관계:
        일간 변동성 ≤ 1%  → 0.99 (안정적 종목, 느린 평균회귀)
        일간 변동성 ≤ 2%  → 0.98 (보통 종목)
        일간 변동성 > 2%  → 0.95 (고변동성 종목, 빠른 평균회귀)

    Args:
        df: 학습 데이터 DataFrame

    Returns:
        decay_rate 스칼라 (0 < decay_rate ≤ 1)
    """
    daily_std = df["y"].pct_change().std()
    if daily_std <= 0.01:
        return 0.99   # 저변동성: 감쇠 느림
    elif daily_std <= 0.02:
        return 0.98   # 보통 변동성
    else:
        return 0.95   # 고변동성: 감쇠 빠름 (장기 예측 발산 방지)


# =============================================================================
# [섹션 6]  주가 데이터 수집
#   - auto_adjust=False: 실제 시장 주가 (수정 주가 문제 해결)
#   - 4000일(약 11년): 2008 금융위기·2020 코로나·2022 금리급등 포함
# =============================================================================
def fetch_ohlcv(ticker: str, name: str) -> pd.DataFrame:
    """yfinance로 OHLCV 수집 + 외부지표 병합 + 80개+ 피처 생성.

    Args:
        ticker: KRX 종목코드 6자리 (예: "316140")
        name  : 종목명 (출력용)

    Returns:
        80개+ 피처가 추가된 OHLCV DataFrame
    """
    end_dt   = datetime.today()
    # 4000일 ≈ 11년: 금리 사이클·경기 사이클·위기 구간 포함
    start_dt = end_dt - timedelta(days=4000)

    # KOSPI(.KS) → KOSDAQ(.KQ) 순으로 시도
    for suffix in [".KS", ".KQ"]:
        yf_ticker = f"{ticker}{suffix}"
        label     = "KOSPI" if suffix == ".KS" else "KOSDAQ"
        print(f"  📡 [{yf_ticker}] {name} 주가 수집 중 ({label})...")
        raw = yf.download(
            yf_ticker,
            start=start_dt.strftime("%Y-%m-%d"),
            end=end_dt.strftime("%Y-%m-%d"),
            auto_adjust=False,  # 실제 시장 주가 (수정 주가 X)
            progress=False,
        )
        if not raw.empty:
            break
    else:
        # for-else: break 없이 루프가 끝난 경우 (양쪽 접미사 모두 실패)
        raise RuntimeError(f"{name}({ticker}) 주가 데이터를 가져오지 못했습니다.")

    # yfinance MultiIndex 컬럼 처리
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    df = raw.reset_index()
    df.columns = [str(c) for c in df.columns]

    # 영문 컬럼명 → 한글 변환
    df = df.rename(columns={
        "Date": "ds", "Close": "종가", "Open": "시가",
        "High": "고가", "Low": "저가", "Volume": "거래량",
    })
    # 타임존 제거 (모델이 타임존 없는 datetime 요구)
    df["ds"] = pd.to_datetime(df["ds"]).dt.tz_localize(None)
    df = df[["ds", "종가", "시가", "고가", "저가", "거래량"]].dropna()
    df["y"] = df["종가"]   # 예측 타깃 컬럼
    df = df.sort_values("ds").reset_index(drop=True)

    # 외부 거시지표 병합
    ext_df = fetch_external_features(start_dt, end_dt)
    if not ext_df.empty:
        ext_df["ds"] = pd.to_datetime(ext_df["ds"])
        df = pd.merge(df, ext_df, on="ds", how="left")
        ext_cols = [c for c in EXTERNAL_TICKERS.keys() if c in df.columns]
        df[ext_cols] = df[ext_cols].ffill().bfill().fillna(0)

    # 80개+ 피처 생성
    df = add_features(df)
    df = df.dropna(subset=["y"]).reset_index(drop=True)

    feat_cols = get_all_feature_columns(df)
    print(f"  ✔ {len(df)}일치 수집 완료 "
          f"({df['ds'].min().date()} ~ {df['ds'].max().date()}) "
          f"| 전체 피처: {len(feat_cols)}개\n")
    return df


# =============================================================================
# [섹션 7]  Walk-Forward Backtest (실제 성능 평가)
#   v6·v7 공통 기능. v8에서도 계승.
#
#   원리: 시간 순서대로 학습/평가 → 미래 데이터 누수 완전 차단
#   v7 개선: feat_cols 파라미터로 모든 모델이 동일 피처셋 사용
#            (v6: 각 모델이 내부에서 피처셋 독립 결정 → 불일치 가능)
# =============================================================================
def walk_forward_backtest(model_func, df: pd.DataFrame, feat_cols: list,
                           n_test: int = 60, n_splits: int = 2) -> float:
    """Walk-Forward 방식으로 모델 실제 MAPE 측정.

    Args:
        model_func: (train_df, feat_cols, forecast_days) → pred_array
        df        : 전체 DataFrame
        feat_cols : 선별된 피처 컬럼 목록
        n_test    : 각 split의 테스트 구간 일수
        n_splits  : 평가 반복 횟수

    Returns:
        평균 MAPE. 계산 실패 시 1.0(100%) 반환.
    """
    mapes = []
    total = len(df)

    for i in range(n_splits):
        test_end   = total - i * (n_test // n_splits)
        test_start = test_end - n_test
        if test_start < 200:   # 학습 데이터 최소 200일 보장
            break
        train_df = df.iloc[:test_start].copy()  # 미래 데이터 완전 차단
        test_df  = df.iloc[test_start:test_end].copy()
        try:
            preds   = model_func(train_df, feat_cols, len(test_df))
            actual  = test_df["y"].values
            min_len = min(len(actual), len(preds))
            mape    = mean_absolute_percentage_error(actual[:min_len], preds[:min_len])
            mapes.append(mape)
            print(f"     split {i+1}: MAPE={mape:.4f}")
        except Exception as e:
            print(f"     [경고] split {i+1} 실패: {e}")
            mapes.append(1.0)

    return float(np.mean(mapes)) if mapes else 1.0


# =============================================================================
# [섹션 8-공통]  롤링 피처 업데이트 (Feature Drift 방지)
#   v5의 버그: last_features[0] = price → MA5 = 현재가로 잘못 설정
#   v6·v7·v8: price_history 이력으로 MA·수익률·변동성 재계산
#
#   ★v8: v7의 _update_rolling_features 버그 수정
#     v7 코드: if f"MA{w}" in dict(zip(feat_cols, feat_cols)) → 항상 True
#              (dict(zip(cols, cols))는 모든 컬럼이 True가 되는 잘못된 체크)
#     v8 수정: if f"MA{w}" in feat_dict → feat_dict 키 존재 여부 정확히 체크
# =============================================================================
def _update_rolling_features(feat_dict: dict, feat_cols: list,
                              price_history: np.ndarray):
    """price_history 기반으로 MA·수익률·변동성 피처를 정확히 재계산.

    Args:
        feat_dict    : 피처명 → 현재값 딕셔너리 (in-place 수정)
        feat_cols    : 피처 컬럼명 리스트 (참조용)
        price_history: 지금까지의 가격 이력 (예측값 누적 포함)
    """
    ph = price_history

    # MA 재계산
    for w in [5, 10, 20, 60, 120, 200]:
        key = f"MA{w}"
        # ★v8 버그 수정: feat_dict 키 체크 (v7: dict(zip(feat_cols,feat_cols)) 오류)
        if key in feat_dict and len(ph) >= w:
            feat_dict[key] = float(np.mean(ph[-w:]))
        slope_key = f"MA{w}_slope"
        if slope_key in feat_dict and len(ph) >= w + 5:
            ma_now  = np.mean(ph[-w:])
            ma_prev = np.mean(ph[-(w + 5):-5])
            feat_dict[slope_key] = (ma_now - ma_prev) / (ma_prev + 1e-9)

    # 수익률 재계산
    for period in [1, 3, 5, 10, 20, 60]:
        key = f"Return_{period}d"
        if key in feat_dict and len(ph) > period:
            feat_dict[key] = float(
                (ph[-1] - ph[-period - 1]) / (ph[-period - 1] + 1e-9)
            )

    # 변동성 재계산
    if len(ph) >= 21:
        rets = np.diff(ph[-21:]) / (ph[-21:-1] + 1e-9)
        for w, key in [(10, "Volatility_10d"), (20, "Volatility_20d")]:
            if key in feat_dict and len(rets) >= w:
                feat_dict[key] = float(np.std(rets[-w:]) * np.sqrt(252))


# =============================================================================
# [섹션 8-A]  LightGBM 예측 모델
#   v6·v7 공통. v8에서 Decay Rate 동적화·CV 파라미터 탐색 복원.
#
#   예측 방식: 수익률(return) 예측 후 복리 방식으로 가격 재구성
#   이유: price 직접 예측보다 수익률 예측이 더 안정적 (시계열 정상성)
# =============================================================================
def predict_lgbm(df: pd.DataFrame, feat_cols: list,
                 forecast_days: int = 63) -> np.ndarray:
    """LightGBM으로 수익률 예측 후 가격 재구성.

    ★v8: TimeSeriesSplit 파라미터 탐색 복원 (v7: 단일 파라미터 고정)
         Decay Rate 동적화 (v7: 0.98 고정 → v8: 변동성 기반 자동)

    Args:
        df           : fetch_ohlcv() DataFrame
        feat_cols    : select_top_features()가 선별한 피처 목록
        forecast_days: 예측 영업일 수

    Returns:
        예측 주가 ndarray
    """
    data = df[feat_cols + ["y"]].dropna().copy()
    # 타깃: 다음 날 수익률 (return prediction)
    # shift(-1): 오늘 피처 → 내일 수익률 구조로 1일 앞으로 당김
    data["target"] = data["y"].pct_change(1).shift(-1)
    data = data.dropna()

    X = data[feat_cols].values
    y = data["target"].values

    # ★v8: TimeSeriesSplit CV 파라미터 탐색 (v7: 단일 파라미터로 고정)
    tscv = TimeSeriesSplit(n_splits=5)
    param_grid = [
        {"n_estimators": 500,  "max_depth": 6, "learning_rate": 0.05, "num_leaves": 31},
        {"n_estimators": 800,  "max_depth": 7, "learning_rate": 0.03, "num_leaves": 63},
        {"n_estimators": 1000, "max_depth": 6, "learning_rate": 0.02, "num_leaves": 31},
    ]
    best_mape, best_params = float("inf"), param_grid[0]
    for params in param_grid:
        fold_mapes = []
        for tr_idx, val_idx in tscv.split(X):
            m = lgb.LGBMRegressor(
                **params, subsample=0.8, colsample_bytree=0.8,
                random_state=42, verbose=-1, force_col_wise=True
            )
            m.fit(X[tr_idx], y[tr_idx])
            pred = m.predict(X[val_idx])
            try:
                fold_mapes.append(mean_absolute_percentage_error(y[val_idx], pred))
            except Exception:
                fold_mapes.append(1.0)
        avg = float(np.mean(fold_mapes))
        if avg < best_mape:
            best_mape, best_params = avg, params

    model = lgb.LGBMRegressor(
        **best_params, subsample=0.8, colsample_bytree=0.8,
        random_state=42, verbose=-1, force_col_wise=True
    )
    model.fit(X, y)

    # ── 롤링 예측 ─────────────────────────────────────────────────────────
    limit      = get_dynamic_bounds(df)   # 동적 클리핑 상한
    decay_rate = get_decay_rate(df)        # ★v8: 변동성 기반 동적 decay

    last_price    = float(df["y"].iloc[-1])
    price_history = list(df["종가"].values[-200:])  # MA200 재계산에 필요
    preds         = []
    price         = last_price
    last_feat     = data[feat_cols].iloc[-1].values.copy().astype(float)
    feat_dict     = dict(zip(feat_cols, last_feat))

    for step in range(forecast_days):
        ret   = float(model.predict(last_feat.reshape(1, -1))[0])
        ret   = float(np.clip(ret, -limit, limit))   # 동적 클리핑
        ret  *= (decay_rate ** step)                  # ★ Decay: 장기 평균회귀
        price = price * (1 + ret)
        preds.append(price)
        price_history.append(price)
        # ★v8 버그 수정된 _update_rolling_features 사용
        _update_rolling_features(feat_dict, feat_cols, np.array(price_history))
        last_feat = np.array([feat_dict.get(c, last_feat[i])
                              for i, c in enumerate(feat_cols)])

    return np.array(preds)


# =============================================================================
# [섹션 8-B]  Multi-Feature LSTM 예측 모델
#   v6: 가격 예측 (종가 직접) → v7·v8: 수익률 예측 (3모델 구조 통일)
#   v7 계승: 경량화 (64→32 유닛, epoch 30) → 속도 대폭 개선
#   v6 계승: ReduceLROnPlateau 콜백, RobustScaler
#
#   ★v8: v7의 LSTM 롤링 예측 버그 수정
#     v7 코드: seq.append(seq[-1].copy()) → 피처가 전혀 업데이트 안 됨
#     v8 수정: _update_rolling_features로 피처 재계산 후 시퀀스 업데이트
# =============================================================================
def predict_lstm(df: pd.DataFrame, feat_cols: list,
                 forecast_days: int = 63, lookback: int = 60) -> np.ndarray:
    """Multi-Feature LSTM으로 수익률 예측 후 가격 재구성.

    ★v8 변경:
      - 타깃: 가격 → 수익률 (v6 → v7 방식 채택, 3모델 구조 통일)
      - 롤링 예측: seq[-1].copy() 고정 → _update_rolling_features 적용
      - ReduceLROnPlateau 추가 (v6 콜백 계승)

    Args:
        df           : fetch_ohlcv() DataFrame
        feat_cols    : select_top_features()가 선별한 피처 목록
        forecast_days: 예측 영업일 수
        lookback     : LSTM 입력 시퀀스 길이 (과거 60일치 패턴 참조)

    Returns:
        예측 주가 ndarray
    """
    # v7 계승: MA 피처는 LSTM 입력에서 제외 (선형 추세 피처 → 노이즈 증가)
    # 나머지 feat_cols 중 최대 15개 사용 (LSTM은 피처 수 적을수록 학습 안정)
    lstm_feats = [f for f in feat_cols if "MA" not in f][:15]
    if not lstm_feats:
        lstm_feats = feat_cols[:10]  # 모든 피처가 MA인 극단적 경우 대비

    data = df[lstm_feats + ["y"]].dropna().copy()
    # ★v7·v8: 수익률 타깃 (가격 직접 예측 → 수익률 예측으로 통일)
    data["target"] = data["y"].pct_change(1).shift(-1)
    data = data.dropna()

    n_feat = len(lstm_feats)

    # RobustScaler: 주가 급등락 이상치에 강건한 정규화
    scaler_X = RobustScaler()
    X_scaled = scaler_X.fit_transform(data[lstm_feats].values)
    y_tgt    = data["target"].values   # 수익률 (정규화 불필요, 이미 소규모)

    # 시퀀스 생성: X.shape = (samples, lookback, n_feat)
    X_seq, Y_seq = [], []
    for i in range(lookback, len(X_scaled)):
        X_seq.append(X_scaled[i - lookback:i, :])  # (lookback, n_feat)
        Y_seq.append(y_tgt[i])                      # 타깃: 수익률
    X_seq, Y_seq = np.array(X_seq), np.array(Y_seq)

    # v7 계승 + v6 계승 혼합 아키텍처:
    #   v7: 64→32 유닛 경량화 (학습 속도)
    #   v6: BatchNormalization, ReduceLROnPlateau (학습 안정성)
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(lookback, n_feat)),
        BatchNormalization(),
        Dropout(0.2),
        LSTM(32, return_sequences=False),
        Dropout(0.2),
        Dense(1),   # 출력: 수익률 1개 (활성화 함수 없음 → 회귀)
    ])
    model.compile(optimizer="adam", loss="mse")

    es  = EarlyStopping(monitor="val_loss", patience=5,
                        restore_best_weights=True, verbose=0)
    rlr = ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                            patience=3, verbose=0)  # v6 계승
    # v7 계승: epoch=30 (v6의 100에서 대폭 축소 → 실용적 학습 시간)
    model.fit(X_seq, Y_seq, epochs=30, batch_size=32,
              validation_split=0.1, callbacks=[es, rlr], verbose=0)

    # ── 롤링 예측 ─────────────────────────────────────────────────────────
    limit        = get_dynamic_bounds(df)
    decay_rate   = get_decay_rate(df)

    price         = float(df["y"].iloc[-1])
    price_history = list(df["종가"].values[-200:])
    seq           = list(X_scaled[-lookback:])   # 초기 시퀀스
    preds         = []

    # feat_dict: _update_rolling_features를 위해 피처명-값 딕셔너리 구성
    # lstm_feats 기준으로 마지막 행 값 초기화
    last_feat_vals = data[lstm_feats].iloc[-1].values.copy().astype(float)
    feat_dict      = dict(zip(lstm_feats, last_feat_vals))

    for step in range(forecast_days):
        inp      = np.array(seq[-lookback:]).reshape(1, lookback, n_feat)
        pred_ret = float(model.predict(inp, verbose=0)[0, 0])
        pred_ret = float(np.clip(pred_ret, -limit, limit))
        pred_ret *= (decay_rate ** step)   # Decay

        price = price * (1 + pred_ret)
        preds.append(price)
        price_history.append(price)

        # ★v8 버그 수정: _update_rolling_features로 피처 재계산
        # (v7: seq.append(seq[-1].copy()) → 피처 완전 고정, 업데이트 없음)
        _update_rolling_features(feat_dict, lstm_feats, np.array(price_history))
        new_feat_vals = np.array([feat_dict.get(f, last_feat_vals[i])
                                  for i, f in enumerate(lstm_feats)])
        # 정규화 스케일 유지를 위해 scaler 적용
        new_scaled = scaler_X.transform(new_feat_vals.reshape(1, -1))[0]
        seq.append(new_scaled)

    return np.array(preds)


# =============================================================================
# [섹션 8-C]  XGBoost 예측 모델
#   v6·v7 공통 기능. v8에서 CV 파라미터 탐색 복원.
# =============================================================================
def predict_xgboost(df: pd.DataFrame, feat_cols: list,
                    forecast_days: int = 63) -> np.ndarray:
    """XGBoost로 수익률 예측 후 가격 재구성.

    ★v8: TimeSeriesSplit CV 파라미터 탐색 복원 (v7: 단일 파라미터)
         Decay Rate 동적화
         _update_rolling_features 버그 수정 버전 사용

    Args:
        df           : fetch_ohlcv() DataFrame
        feat_cols    : select_top_features()가 선별한 피처 목록
        forecast_days: 예측 영업일 수

    Returns:
        예측 주가 ndarray
    """
    data = df[feat_cols + ["y"]].dropna().copy()
    data["target"] = data["y"].pct_change(1).shift(-1)
    data = data.dropna()

    X = data[feat_cols].values
    y = data["target"].values

    # ★v8: TimeSeriesSplit CV 파라미터 탐색 (v7: n_estimators=400 고정)
    tscv = TimeSeriesSplit(n_splits=5)
    param_grid = [
        {"n_estimators": 400, "max_depth": 5, "learning_rate": 0.05},
        {"n_estimators": 600, "max_depth": 6, "learning_rate": 0.03},
        {"n_estimators": 800, "max_depth": 5, "learning_rate": 0.02},
    ]
    best_mape, best_params = float("inf"), param_grid[0]
    for params in param_grid:
        fold_mapes = []
        for tr_idx, val_idx in tscv.split(X):
            m = xgb.XGBRegressor(
                **params, subsample=0.8, colsample_bytree=0.8,
                random_state=42, verbosity=0
            )
            m.fit(X[tr_idx], y[tr_idx])
            pred = m.predict(X[val_idx])
            try:
                fold_mapes.append(mean_absolute_percentage_error(y[val_idx], pred))
            except Exception:
                fold_mapes.append(1.0)
        avg = float(np.mean(fold_mapes))
        if avg < best_mape:
            best_mape, best_params = avg, params

    print(f"     XGBoost 최적 파라미터: {best_params}")
    model = xgb.XGBRegressor(
        **best_params, subsample=0.8, colsample_bytree=0.8,
        random_state=42, verbosity=0
    )
    model.fit(X, y)

    # ── 롤링 예측 ─────────────────────────────────────────────────────────
    limit         = get_dynamic_bounds(df)
    decay_rate    = get_decay_rate(df)   # ★v8: 동적 decay

    last_price    = float(df["y"].iloc[-1])
    price_history = list(df["종가"].values[-200:])
    preds         = []
    price         = last_price
    last_feat     = data[feat_cols].iloc[-1].values.copy().astype(float)
    feat_dict     = dict(zip(feat_cols, last_feat))

    for step in range(forecast_days):
        ret   = float(model.predict(last_feat.reshape(1, -1))[0])
        ret   = float(np.clip(ret, -limit, limit))
        ret  *= (decay_rate ** step)   # Decay
        price = price * (1 + ret)
        preds.append(price)
        price_history.append(price)
        _update_rolling_features(feat_dict, feat_cols, np.array(price_history))
        last_feat = np.array([feat_dict.get(c, last_feat[i])
                              for i, c in enumerate(feat_cols)])

    return np.array(preds)


# =============================================================================
# [섹션 9]  앙상블 실행 (Walk-Forward Backtest 기반 가중치)
#   v7 구조 계승: feat_cols 파라미터 공유로 모든 모델이 동일 피처셋 사용
#   v6 구조 계승: 신뢰구간 90% (v7: 80%로 축소 → 불확실성 과소 표현)
# =============================================================================
def run_ensemble(df: pd.DataFrame, forecast_days: int = 63):
    """LightGBM + Multi-Feature LSTM + XGBoost 앙상블 예측.

    처리 흐름:
        1. Feature Selection: 80개 → 상위 30개 자동 선별
        2. 각 모델 개별 예측 (선별된 30개 피처 공유)
        3. Walk-Forward Backtest로 각 모델 실제 MAPE 측정
        4. MAPE 역수 기반 가중치 → 가중 평균 앙상블
        5. 모델 간 표준편차로 90% 신뢰구간 계산

    Returns:
        (result_df, weights, feat_cols)
        result_df : 예측 결과 DataFrame
                    컬럼: ds, yhat, yhat_upper, yhat_lower, lgbm, lstm, xgboost
        weights   : [lgbm, lstm, xgb] 가중치 배열
        feat_cols : 선별된 피처 목록 (요약 출력용)
    """
    print("\n  ── 앙상블 예측 진행 ─────────────────────────────────────────")

    # ★ 1단계: Feature Selection (80개 → 30개)
    feat_cols = select_top_features(df, top_n=30)

    # ── 2단계: 각 모델 예측 ──────────────────────────────────────────────
    print("  [1/3] LightGBM 학습 및 예측 중...")
    lgbm_pred = predict_lgbm(df, feat_cols, forecast_days)

    print("  [2/3] Multi-Feature LSTM (Return Mode) 학습 및 예측 중...")
    lstm_pred = predict_lstm(df, feat_cols, forecast_days)

    print("  [3/3] XGBoost 학습 및 예측 중...")
    xgb_pred  = predict_xgboost(df, feat_cols, forecast_days)

    # ── 3단계: Walk-Forward Backtest ──────────────────────────────────────
    print("\n  ── Walk-Forward Backtest (실제 오차 측정) ───────────────────")
    print("  ※ 각 모델을 과거 데이터로 재학습해 미래 예측 MAPE를 실측합니다.")

    print("     [LightGBM 백테스트]")
    mape_l    = walk_forward_backtest(predict_lgbm,    df, feat_cols)
    print(f"     → 평균 MAPE: {mape_l:.4f} ({mape_l*100:.2f}%)")

    print("     [LSTM 백테스트]")
    mape_lstm = walk_forward_backtest(predict_lstm,    df, feat_cols)
    print(f"     → 평균 MAPE: {mape_lstm:.4f} ({mape_lstm*100:.2f}%)")

    print("     [XGBoost 백테스트]")
    mape_x    = walk_forward_backtest(predict_xgboost, df, feat_cols)
    print(f"     → 평균 MAPE: {mape_x:.4f} ({mape_x*100:.2f}%)")

    # ── 4단계: MAPE 역수 기반 가중치 ─────────────────────────────────────
    # 오차 낮을수록 가중치 높음. 1e-9: 0 나눗셈 방지
    inv = np.array([1 / (mape_l    + 1e-9),
                    1 / (mape_lstm + 1e-9),
                    1 / (mape_x    + 1e-9)])
    w = inv / inv.sum()   # 합이 1.0 되도록 정규화

    print(f"\n  앙상블 가중치 (실제 Backtest MAPE 기반)")
    print(f"     LightGBM:{w[0]:.2f} (MAPE:{mape_l:.3f})")
    print(f"     LSTM    :{w[1]:.2f} (MAPE:{mape_lstm:.3f})")
    print(f"     XGBoost :{w[2]:.2f} (MAPE:{mape_x:.3f})")

    # ── 5단계: 가중 평균 앙상블 ──────────────────────────────────────────
    min_len  = min(len(lgbm_pred), len(lstm_pred), len(xgb_pred), forecast_days)
    ensemble = (w[0] * lgbm_pred[:min_len]
               + w[1] * lstm_pred[:min_len]
               + w[2] * xgb_pred[:min_len])

    # ★v8: 신뢰구간 90% 복원 (v7: 80%로 축소 → 불확실성 과소 표현)
    # ±1.645σ = 90% 신뢰구간 (정규분포 기준)
    stack   = np.stack([lgbm_pred[:min_len],
                        lstm_pred[:min_len],
                        xgb_pred[:min_len]], axis=0)
    max_std = ensemble * 0.20   # 최대 ±20% 제한 (v6: 30%, v7: 15% → 중간 값)
    std     = np.minimum(stack.std(axis=0), max_std)
    upper   = ensemble + 1.645 * std
    lower   = np.maximum(ensemble - 1.645 * std, 0)  # 주가 0 이하 불가

    # 미래 영업일 날짜 생성 (pd.bdate_range: 월~금 영업일만)
    last_date = df["ds"].max()
    biz_days  = pd.bdate_range(start=last_date + timedelta(days=1), periods=min_len)

    result = pd.DataFrame({
        "ds":         biz_days,
        "yhat":       ensemble,
        "yhat_upper": upper,
        "yhat_lower": lower,
        "lgbm":       lgbm_pred[:min_len],
        "lstm":       lstm_pred[:min_len],
        "xgboost":    xgb_pred[:min_len],
    })

    return result, w, feat_cols


# =============================================================================
# [섹션 10]  예측 결과 시각화
#   v6 4단 레이아웃 복원 (v7: 단순 1단 차트로 축소)
#   4단 구성:
#     ax1 (상단 3.5): 주가 + 개별 모델 예측 + 앙상블 + 신뢰구간
#     ax2 (2단   1 ): 거래량 막대 그래프
#     ax3 (3단   1 ): RSI14 지표 (과매수/과매도 음영)
#     ax4 (하단 1.2): 미국채10년 금리(좌축) + VIX(우축)
# =============================================================================
def plot_ensemble(df: pd.DataFrame, result: pd.DataFrame,
                  weights: np.ndarray, ticker: str, name: str,
                  feat_cols: list):
    """앙상블 예측 결과를 4단 그래프로 시각화하고 PNG로 저장.

    Args:
        df       : 실제 주가 + 기술지표 + 외부지표 DataFrame
        result   : run_ensemble()이 반환한 예측 결과 DataFrame
        weights  : [lgbm, lstm, xgb] 가중치 배열
        ticker   : 종목코드
        name     : 종목명
        feat_cols: 선별된 피처 목록 (제목에 피처 수 표시)
    """
    today      = pd.Timestamp.today().normalize()
    last_price = float(df["y"].iloc[-1])
    cutoff     = today - pd.DateOffset(months=6)
    hist       = df[df["ds"] >= cutoff]    # 최근 6개월 데이터

    has_ext = any(c in df.columns for c in EXTERNAL_TICKERS.keys())
    n_rows  = 4 if has_ext else 3
    ratios  = [3.5, 1, 1, 1.2] if has_ext else [3.5, 1, 1]

    fig = plt.figure(figsize=(16, 14 if has_ext else 12), facecolor="#0d1117")
    gs  = fig.add_gridspec(n_rows, 1, height_ratios=ratios, hspace=0.08)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharex=ax1)   # X축 공유
    ax3 = fig.add_subplot(gs[2], sharex=ax1)
    ax4 = fig.add_subplot(gs[3], sharex=ax1) if has_ext else None

    # 다크 테마 공통 스타일
    axes = [ax for ax in [ax1, ax2, ax3, ax4] if ax is not None]
    for ax in axes:
        ax.set_facecolor("#161b22")
        ax.tick_params(colors="#8b949e", labelsize=9)
        for s in ["top", "right"]:
            ax.spines[s].set_visible(False)
        for s in ["bottom", "left"]:
            ax.spines[s].set_color("#30363d")
        ax.grid(axis="y", color="#21262d", linewidth=0.7)

    # ── ax1: 주가 + 예측 ─────────────────────────────────────────────────
    ax1.plot(hist["ds"], hist["y"],
             color="#58a6ff", linewidth=2, label="실제 주가", zorder=4)
    ax1.plot(result["ds"], result["lgbm"],
             color="#7ee787", linewidth=1, alpha=0.55,
             linestyle="--", label=f"LightGBM ({weights[0]:.0%})")
    ax1.plot(result["ds"], result["lstm"],
             color="#d2a8ff", linewidth=1, alpha=0.55,
             linestyle="--", label=f"LSTM ({weights[1]:.0%})")
    ax1.plot(result["ds"], result["xgboost"],
             color="#ffa657", linewidth=1, alpha=0.55,
             linestyle="--", label=f"XGBoost ({weights[2]:.0%})")
    ax1.plot(result["ds"], result["yhat"],
             color="#f85149", linewidth=2.8, label="앙상블 예측", zorder=5)
    ax1.fill_between(result["ds"], result["yhat_lower"], result["yhat_upper"],
                     color="#f85149", alpha=0.1, label="90% 신뢰구간")
    ax1.axvline(today, color="#e3b341", linewidth=1.2,
                linestyle=":", alpha=0.9, label="오늘")

    last_fc   = result.iloc[-1]
    chg_pct   = (last_fc["yhat"] - last_price) / last_price * 100
    direction = "▲ 상승" if chg_pct >= 0 else "▼ 하락"
    ax1.annotate(
        f"  {last_fc['yhat']:,.0f}원\n  ({last_fc['ds'].strftime('%m/%d')})",
        xy=(last_fc["ds"], last_fc["yhat"]),
        xytext=(-70, 30), textcoords="offset points",
        color="#f85149", fontsize=9,
        arrowprops=dict(arrowstyle="->", color="#f85149", lw=1.2),
    )

    ax1.set_title(
        f"[{ticker}]  {name}  —  앙상블 주가 예측 v8 "
        f"(LightGBM + Multi-LSTM + XGBoost)\n"
        f"현재 {last_price:,.0f}원  →  3개월 후 {last_fc['yhat']:,.0f}원"
        f"  ( {direction} {chg_pct:+.1f}% )"
        f"  |  학습 {len(df)}일 / 피처 {len(feat_cols)}개(선별) / Backtest 가중치 / Decay 적용",
        color="#e6edf3", fontsize=11, fontweight="bold", pad=14,
    )
    ax1.set_ylabel("주가 (원)", color="#8b949e", fontsize=10)
    # Y축 천 단위 구분 포맷
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    leg = ax1.legend(loc="upper left", fontsize=8,
                     facecolor="#161b22", edgecolor="#30363d", ncol=2)
    for t in leg.get_texts():
        t.set_color("#c9d1d9")
    plt.setp(ax1.get_xticklabels(), visible=False)

    # ── ax2: 거래량 ───────────────────────────────────────────────────────
    if "거래량" in hist.columns:
        ax2.bar(hist["ds"], hist["거래량"], color="#58a6ff", alpha=0.45, width=1)
        ax2.set_ylabel("거래량", color="#8b949e", fontsize=8)
        ax2.yaxis.set_major_formatter(
            plt.FuncFormatter(
                lambda x, _: f"{x/1e6:.1f}M" if x >= 1e6 else f"{x/1e3:.0f}K"
            )
        )
        ax2.axvline(today, color="#e3b341", linewidth=1, linestyle=":", alpha=0.7)
        plt.setp(ax2.get_xticklabels(), visible=False)

    # ── ax3: RSI14 ────────────────────────────────────────────────────────
    rsi_col = "RSI14" if "RSI14" in hist.columns else "RSI"
    if rsi_col in hist.columns:
        ax3.plot(hist["ds"], hist[rsi_col], color="#d2a8ff", linewidth=1.2)
        ax3.axhline(70, color="#f85149", linewidth=0.8, linestyle="--", alpha=0.7)
        ax3.axhline(30, color="#7ee787", linewidth=0.8, linestyle="--", alpha=0.7)
        # 과매수(≥70) 빨간 음영, 과매도(≤30) 초록 음영
        ax3.fill_between(hist["ds"], hist[rsi_col], 70,
                         where=hist[rsi_col] >= 70, color="#f85149", alpha=0.15)
        ax3.fill_between(hist["ds"], hist[rsi_col], 30,
                         where=hist[rsi_col] <= 30, color="#7ee787", alpha=0.15)
        ax3.set_ylabel("RSI(14)", color="#8b949e", fontsize=8)
        ax3.set_ylim(0, 100)
        ax3.axvline(today, color="#e3b341", linewidth=1, linestyle=":", alpha=0.7)
        plt.setp(ax3.get_xticklabels(), visible=False)

    # ── ax4: 거시지표 (금리 + VIX) ────────────────────────────────────────
    if ax4 is not None:
        if "US10Y" in hist.columns:
            ax4.plot(hist["ds"], hist["US10Y"],
                     color="#e3b341", linewidth=1.2, label="미국채 10년(%)")
        ax4_r = ax4.twinx()   # 우측 Y축 (VIX 별도 스케일)
        if "VIX" in hist.columns:
            ax4_r.plot(hist["ds"], hist["VIX"],
                       color="#f85149", linewidth=1, alpha=0.7, label="VIX")
            ax4_r.set_ylabel("VIX", color="#f85149", fontsize=7)
            ax4_r.tick_params(colors="#f85149", labelsize=8)
            ax4_r.set_facecolor("#161b22")
        ax4.set_ylabel("미국채10년(%)", color="#e3b341", fontsize=7)
        ax4.axvline(today, color="#e3b341", linewidth=1, linestyle=":", alpha=0.7)
        lines1, labels1 = ax4.get_legend_handles_labels()
        lines2, labels2 = ax4_r.get_legend_handles_labels()
        if lines1 or lines2:
            ax4.legend(lines1 + lines2, labels1 + labels2,
                       loc="upper left", fontsize=7,
                       facecolor="#161b22", edgecolor="#30363d")
            for t in ax4.get_legend().get_texts():
                t.set_color("#c9d1d9")

    # 면책 고지
    fig.text(0.5, 0.005,
             "⚠  본 예측은 통계·머신러닝 모델 기반 참고 자료이며 투자 권유가 아닙니다."
             "  주식 투자는 원금 손실 위험이 있습니다.",
             ha="center", color="#484f58", fontsize=8)

    plt.tight_layout(rect=[0, 0.02, 1, 1])

    fname = f"krx_predict_{ticker}_{today.strftime('%Y%m%d')}_v8.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"  💾 그래프 저장: {fname}")
    plt.show()
    plt.close()


# =============================================================================
# [섹션 11]  터미널 예측 요약 출력
#   v6 상세 출력 복원 (v7: 1줄 요약으로 축소됨)
# =============================================================================
def print_summary(df: pd.DataFrame, result: pd.DataFrame,
                  weights: np.ndarray, ticker: str, name: str,
                  feat_cols: list):
    """1·2·3개월 후 예측값을 터미널에 구간별로 상세 출력."""
    last_price = float(df["y"].iloc[-1])
    GREEN = "\033[92m"; RED = "\033[91m"; RESET = "\033[0m"

    ext_info   = ", ".join([c for c in EXTERNAL_TICKERS.keys() if c in df.columns])
    decay_rate = get_decay_rate(df)
    limit      = get_dynamic_bounds(df)

    print()
    print("=" * 72)
    print(f"  📊  {name} ({ticker})  앙상블 3개월 예측 요약  [v8]")
    print(f"      가중치 (Walk-Forward Backtest) — "
          f"LightGBM:{weights[0]:.0%}  LSTM:{weights[1]:.0%}  XGBoost:{weights[2]:.0%}")
    print(f"      학습: {len(df)}일치 ({df['ds'].min().date()}~{df['ds'].max().date()})")
    print(f"      피처: {len(feat_cols)}개(선별)  |  외부: {ext_info or '없음'}")
    print(f"      동적클리핑: ±{limit*100:.1f}%  |  Decay율: {decay_rate:.2f}/step")
    print("=" * 72)
    print(f"  현재가 (최근 종가)  : {last_price:>14,.0f} 원\n")

    # 1개월(약 20 영업일), 2개월(약 41일), 3개월(약 62일) 시점
    for label, idx in [("1개월 후", 19), ("2개월 후", 40), ("3개월 후", 61)]:
        row = result.iloc[min(idx, len(result) - 1)]
        chg = (row["yhat"] - last_price) / last_price * 100
        clr = GREEN if chg >= 0 else RED
        sym = "▲" if chg >= 0 else "▼"
        print(f"  {label}  ({row['ds'].strftime('%Y-%m-%d')})  : "
              f"{row['yhat']:>14,.0f} 원  {clr}{sym}{chg:+.1f}%{RESET}")
        print(f"    └ 신뢰구간 : {row['yhat_lower']:,.0f} ~ {row['yhat_upper']:,.0f} 원")
        print(f"    └ LightGBM : {row['lgbm']:>12,.0f}원  "
              f"LSTM : {row['lstm']:>12,.0f}원  "
              f"XGBoost : {row['xgboost']:>12,.0f}원\n")

    print("=" * 72)
    print("  ⚠  이 예측은 참고용입니다. 투자 판단은 본인 책임입니다.")
    print("=" * 72)


# =============================================================================
# [섹션 12]  메인 함수 — 반복 검색 루프
# =============================================================================
def main():
    print()
    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║  KRX 주가 예측 v8  (LightGBM + Multi-LSTM + XGBoost)           ║")
    print("║  v6 상세성 + v7 Feature Selection·Decay·동적클리핑 통합        ║")
    print("║  종료: Enter / q                                                 ║")
    print("╚══════════════════════════════════════════════════════════════════╝")
    print()

    try:
        all_df = load_all_tickers()
    except Exception as e:
        print(f"  [오류] 종목 목록 로드 실패: {e}")
        sys.exit(1)

    # 커맨드라인 인수로 검색어 전달 가능
    # 예: python krx_stock_predict_v8.py 우리금융
    pending = " ".join(sys.argv[1:]).strip() if len(sys.argv) > 1 else ""

    while True:
        if pending:
            keyword = pending; pending = ""
            print(f"  🔍 검색어: '{keyword}'")
        else:
            print()
            try:
                keyword = input("  🔍 종목명 검색어 (종료: Enter / q): ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n\n  종료합니다.")
                break

        if not keyword or keyword.lower() in ("q", "quit", "exit", "종료"):
            print("\n  감사합니다!")
            break

        try:
            ticker, name = search_and_select(all_df, keyword)
            if ticker is None:
                continue

            df = fetch_ohlcv(ticker, name)
            print()
            result, weights, feat_cols = run_ensemble(df)
            print_summary(df, result, weights, ticker, name, feat_cols)
            plot_ensemble(df, result, weights, ticker, name, feat_cols)

        except RuntimeError as e:
            print(f"\n  [오류] {e}\n")
        except Exception as e:
            print(f"\n  [예상치 못한 오류] {type(e).__name__}: {e}\n")

        print()
        print("  " + "─" * 62)


# =============================================================================
# 스크립트 직접 실행 시에만 main() 호출
# ※ if __name__ == "__main__" 관용구:
#    직접 실행: __name__ == "__main__" → True
#    다른 파일에서 import: __name__ == 모듈명 → False (main 미실행)
# =============================================================================
if __name__ == "__main__":
    main()
