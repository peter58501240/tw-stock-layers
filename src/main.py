from __future__ import annotations

# =============================
# Config（初版：先讓你「看得到東西」）
# =============================
BACKFILL_DAYS = 30          # 往前補 30 天，保證 MA20 成形（遇到假日也夠）
E_DRAWDOWN = 0.08           # 強勢股回落 -8% 內仍視為強勢（初版用）
HISTORY_PATH = "outputs/history_prices.csv"
OUT_HTML = "docs/index.html"

TWSE_BASE = "https://openapi.twse.com.tw/v1"
TWSE_DAILY_ALL = f"{TWSE_BASE}/exchangeReport/STOCK_DAY_ALL"

# 初版：只做 MA5/10/20（不追求 MA60/MA240，否則要很長歷史）
MA_WINDOWS = (5, 10, 20)

# =============================
# Imports
# =============================
import os
import datetime as dt
from dataclasses import dataclass
from typing import Optional, Tuple, Any, List

import numpy as np
import pandas as pd
import requests


# =============================
# Data Structures
# =============================
@dataclass
class FetchResult:
    df: pd.DataFrame
    source: str
    ok: bool
    error: Optional[str] = None
    warn: Optional[str] = None


# =============================
# Helpers
# =============================
def _http_get(url: str, timeout: int = 30) -> Tuple[int, str, str]:
    r = requests.get(url, timeout=timeout, headers={"User-Agent": "tw-stock-layers/1.0"})
    ct = r.headers.get("content-type", "")
    return r.status_code, ct, r.text


def _try_parse_json(text: str) -> Tuple[bool, Optional[Any], Optional[str]]:
    try:
        return True, requests.models.complexjson.loads(text), None
    except Exception as e:
        return False, None, f"{type(e).__name__}: {e}"


def normalize_twse(df: pd.DataFrame) -> pd.DataFrame:
    """
    TWSE STOCK_DAY_ALL 常見欄位：
    Date, code, name, volume, TradeValue, OpeningPrice, HighestPrice, LowestPrice, ClosingPrice, Change, Transaction
    我們只需要 code/name/close/volume
    """
    if df is None or df.empty:
        return df

    # 欄位映射
    col_map = {
        "Date": "date",
        "日期": "date",
        "code": "code",
        "證券代號": "code",
        "name": "name",
        "證券名稱": "name",
        "volume": "volume",
        "成交股數": "volume",
        "ClosingPrice": "close",
        "收盤價": "close",
        "收盤": "close",
    }
    df = df.rename(columns={c: col_map.get(c, c) for c in df.columns})

    # 清理 code
    if "code" in df.columns:
        df["code"] = df["code"].astype(str).str.strip()

    # close/volume 轉數字
    if "close" in df.columns:
        df["close"] = pd.to_numeric(df["close"], errors="coerce")
    if "volume" in df.columns:
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce")

    df["market"] = "TWSE"
    return df


# =============================
# Fetch
# =============================
def fetch_twse_daily_all(for_date: Optional[pd.Timestamp] = None) -> FetchResult:
    """
    取得 TWSE 全市場日資料（MVP 寬鬆版）
    """
    try:
        url = TWSE_DAILY_ALL
        if for_date is not None:
            ymd = for_date.strftime("%Y%m%d")
            url = f"{TWSE_DAILY_ALL}?date={ymd}"

        status, ct, text = _http_get(url)
        if status != 200:
            return FetchResult(pd.DataFrame(), "TWSE", False, f"HTTP {status} from TWSE")

        ok, data, jerr = _try_parse_json(text)
        if not ok or data is None:
            return FetchResult(pd.DataFrame(), "TWSE", False, f"TWSE JSON parse failed: {jerr}")

        df = pd.DataFrame(data)
        df = normalize_twse(df)

 # MVP：只要 normalize 後有資料就放行
return FetchResult(df, "TWSE", True)
        return FetchResult(df, "TWSE", True)

    except Exception as e:
        return FetchResult(pd.DataFrame(), "TWSE", False, f"{type(e).__name__}: {e}")


# =============================
# History IO
# =============================
def load_history() -> pd.DataFrame:
    if os.path.exists(HISTORY_PATH):
        try:
            hist = pd.read_csv(HISTORY_PATH, dtype={"code": str})
            hist["date"] = pd.to_datetime(hist["date"])
            return hist
        except Exception:
            pass

    return pd.DataFrame(columns=["date", "code", "market", "name", "close", "volume"])


def append_day(history: pd.DataFrame, day_df: pd.DataFrame, day: pd.Timestamp) -> pd.DataFrame:
    keep = ["code", "market", "close"]
    if "name" in day_df.columns:
        keep.append("name")
    if "volume" in day_df.columns:
        keep.append("volume")

    df = day_df[keep].copy()
    df["date"] = day
    df["code"] = df["code"].astype(str).str.strip()

    out = pd.concat([history, df], ignore_index=True)
    out["date"] = pd.to_datetime(out["date"])
    out = out.drop_duplicates(subset=["date", "code", "market"], keep="last")
    return out


def backfill_twse_recent_days(history: pd.DataFrame, today: pd.Timestamp, target_days: int) -> Tuple[pd.DataFrame, int]:
    """
    往前補 target_days 個「可能的交易日」。
    這裡用日曆往前掃（含假日），抓不到就跳過；直到補到 target_days 次成功為止。
    """
    if target_days <= 0:
        return history, 0

    # 已有的日期集合（TWSE）
    have_dates = set()
    if not history.empty:
        h = history[history["market"] == "TWSE"].copy()
        if not h.empty:
            have_dates = set(pd.to_datetime(h["date"]).dt.date.astype(str).tolist())

    filled = 0
    # 往前掃 target_days * 2，假日多也夠
    for i in range(1, target_days * 3 + 1):
        d = today - pd.Timedelta(days=i)
        d_key = d.date().isoformat()
        if d_key in have_dates:
            continue

        r = fetch_twse_daily_all(d)
        if not r.ok or r.df.empty:
            continue

        history = append_day(history, r.df, d)
        have_dates.add(d_key)
        filled += 1
        if filled >= target_days:
            break

    return history, filled


# =============================
# Indicators（初版：MA5/10/20 + 回落）
# =============================
def compute_indicators(hist: pd.DataFrame) -> pd.DataFrame:
    hist = hist.sort_values(["market", "code", "date"]).copy()
    hist["close"] = pd.to_numeric(hist["close"], errors="coerce")
    hist["volume"] = pd.to_numeric(hist.get("volume", np.nan), errors="coerce")

    def _grp(g: pd.DataFrame) -> pd.DataFrame:
        g = g.copy()
        g["ma5"] = g["close"].rolling(5).mean()
        g["ma10"] = g["close"].rolling(10).mean()
        g["ma20"] = g["close"].rolling(20).mean()
        g["hi_close"] = g["close"].cummax()
        g["dd_from_hi"] = g["close"] / g["hi_close"] - 1.0
        return g

    return hist.groupby(["market", "code"], group_keys=False).apply(_grp)


# =============================
# Layer Logic（初版：保證分層有東西）
# =============================
def layer_logic(today_row: pd.Series) -> Tuple[str, str]:
    """
    初版分層（只用 MA5/10/20）：
    - E：close > MA5/10/20 且 dd > -8%
    - B：MA5 > MA10 > MA20（多頭排列）
    - C：close > MA20（整理）
    - D：close <= MA20（轉弱）
    - Z：close 或 MA20 缺
    """
    close = pd.to_numeric(today_row.get("close"), errors="coerce")
    if pd.isna(close):
        return "Z", "資料不足：close 缺"

    ma5 = pd.to_numeric(today_row.get("ma5"), errors="coerce")
    ma10 = pd.to_numeric(today_row.get("ma10"), errors="coerce")
    ma20 = pd.to_numeric(today_row.get("ma20"), errors="coerce")

    if pd.isna(ma20):
        return "Z", "資料不足：MA20 未成形（初版）"

    dd = pd.to_numeric(today_row.get("dd_from_hi"), errors="coerce")
    if pd.isna(dd):
        dd = 0.0

    if (not pd.isna(ma5)) and (not pd.isna(ma10)) and close > ma5 and close > ma10 and close > ma20 and dd > -E_DRAWDOWN:
        return "E", "強勢：收盤>MA5/10/20 且未回落-8%（初版）"

    if (not pd.isna(ma5)) and (not pd.isna(ma10)) and ma5 > ma10 > ma20:
        return "B", "趨勢：MA5>MA10>MA20（初版）"

    if close > ma20:
        return "C", "整理：收盤>MA20（初版）"

    return "D", "轉弱：收盤≤MA20（初版）"


# =============================
# Report HTML（摘要警示）
# =============================
def build_html_report(date_str: str, layers: pd.DataFrame, warnings: List[str], errors: List[str]) -> str:
    def _table(df: pd.DataFrame, title: str) -> str:
        if df.empty:
            return f"<h2>{title}</h2><p>(空)</p>"
        cols = ["market", "code", "name", "close", "layer", "reason"]
        df2 = df.copy()
        for c in cols:
            if c not in df2.columns:
                df2[c] = ""
        df2 = df2[cols]
        return f"<h2>{title}</h2>" + df2.to_html(index=False, escape=True)

    html = []
    html.append("<!doctype html><html><head><meta charset='utf-8'>")
    html.append("<meta name='viewport' content='width=device-width, initial-scale=1'>")
    html.append("<title>TW Stock Layers - Daily</title>")
    html.append(
        "<style>"
        "body{font-family:system-ui,-apple-system,Segoe UI,Roboto,Arial; margin:24px;}"
        "table{border-collapse:collapse; width:100%;}"
        "th,td{border:1px solid #ddd; padding:8px;}"
        "th{background:#f5f5f5;}"
        ".pill{display:inline-block;padding:2px 8px;border-radius:999px;background:#f2f2f2;margin-right:8px;}"
        "</style>"
    )
    html.append("</head><body>")
    html.append(f"<h1>每日分層報表</h1><p>日期：{date_str}（Asia/Taipei）</p>")

    if errors:
        html.append("<h2>抓取錯誤</h2><ul>")
        for e in errors:
            html.append(f"<li>{e}</li>")
        html.append("</ul>")

    # 摘要警示（不要滿版）
    if warnings:
        html.append("<h2>即時警示（摘要｜MVP）</h2><ul>")
        for w in warnings[:8]:
            html.append(f"<li>{w}</li>")
        if len(warnings) > 8:
            html.append(f"<li>…另有 {len(warnings)-8} 則提示省略</li>")
        html.append("</ul>")

    # 分層
    for layer in ["E", "B", "C", "D", "Z"]:
        df_layer = layers[layers["layer"] == layer].sort_values(["market", "code"])
        html.append(_table(df_layer, f"{layer} 層"))

    html.append("<hr>")
    html.append("<p style='color:#666'>註：初版先以 TWSE + MA5/10/20 讓分層「可用可看」。後續再逐步加入 TPEx、MA60/240、RS、營收與規則全文。</p>")
    html.append("</body></html>")
    return "\n".join(html)


# =============================
# Main
# =============================
def main() -> int:
    os.makedirs("docs", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    today = pd.Timestamp(dt.datetime.now().date())
    date_str = today.strftime("%Y-%m-%d")

    errors: List[str] = []
    warnings: List[str] = []

    # 1) 取 TWSE 今日資料
    twse = fetch_twse_daily_all(None)
    if not twse.ok or twse.df.empty:
        errors.append(f"TWSE 取得失敗：{twse.error}")
        # 仍輸出空報表避免 Pages 空白
        empty = pd.DataFrame(columns=["market", "code", "name", "close", "layer", "reason"])
        html = build_html_report(date_str, empty, warnings, errors)
        with open(OUT_HTML, "w", encoding="utf-8") as f:
            f.write(html)
        return 0

    warnings.append("🟢 MVP：目前只跑 TWSE（TPEx 暫停）")

    # 2) 載入歷史
    hist = load_history()

    # 3) 回補歷史（加速 MA20 成形）
    hist, filled = backfill_twse_recent_days(hist, today, BACKFILL_DAYS)
    warnings.append(f"🟢 TWSE 回補：成功補到 {filled} 天（目標 {BACKFILL_DAYS}）")

    # 4) 寫入今日
    hist = append_day(hist, twse.df, today)
    hist.to_csv(HISTORY_PATH, index=False, encoding="utf-8")

    # 5) 算指標
    hist_ind = compute_indicators(hist)

    # 6) 取今日切片
    today_ind = hist_ind[hist_ind["date"] == today].copy()
    if today_ind.empty:
        errors.append("今日指標資料為空（可能日期寫入失敗）")
        empty = pd.DataFrame(columns=["market", "code", "name", "close", "layer", "reason"])
        html = build_html_report(date_str, empty, warnings, errors)
        with open(OUT_HTML, "w", encoding="utf-8") as f:
            f.write(html)
        return 0

    # 7) 分層
    out_rows = []
    for _, row in today_ind.iterrows():
        layer, reason = layer_logic(row)
        out_rows.append({
            "market": row.get("market", ""),
            "code": str(row.get("code", "")).strip(),
            "name": row.get("name", ""),
            "close": row.get("close", ""),
            "layer": layer,
            "reason": reason,
        })
    layers = pd.DataFrame(out_rows)

    # 8) 摘要統計（讓你爽：一眼看到有沒有分層）
    cnt = layers["layer"].value_counts().to_dict()
    warnings.append("📊 分佈：" + " / ".join([f"{k}:{v}" for k, v in cnt.items()]))

    # 9) 輸出 HTML
    html = build_html_report(date_str, layers, warnings, errors)
    with open(OUT_HTML, "w", encoding="utf-8") as f:
        f.write(html)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
