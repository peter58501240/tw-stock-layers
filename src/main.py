from __future__ import annotations

import os
import sys
import json
import math
import datetime as dt
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import requests


# -----------------------------
# Config (可調整/替換資料源)
# -----------------------------
TWSE_BASE = "https://openapi.twse.com.tw/v1"
TPEX_BASE = "https://www.tpex.org.tw/openapi/v1"  # 若實際 base 不同，之後以回傳錯誤調整

# 嘗試抓「當日全市場行情」的端點（若端點不同，先讓程式輸出錯誤，便於我快速修正）
TWSE_DAILY_ALL = f"{TWSE_BASE}/exchangeReport/STOCK_DAY_ALL"
TPEX_DAILY_ALL_CANDIDATES = [
    f"{TPEX_BASE}/tpex_mainboard_daily",   # 候選1（可能需調整）
    f"{TPEX_BASE}/stock_aftertrading_daily_trading_info",  # 候選2（可能需調整）
]

HISTORY_PATH = "outputs/history_prices.csv"
OUT_HTML = "docs/index.html"


# -----------------------------
# v7.9.9.1 核心參數（MVP 子集）
# -----------------------------
SDR_DAYS = 30
SDR_MIN_RETURN = 0.05  # +5%
E_STOP_MA = 20
E_DRAWDOWN = 0.08      # -8%
CYCLE_DRAWDOWN = 0.12  # -12%
GLOBAL_STOP = 0.12     # -12%
CYCLE_MA = 240

# 當資料不足時依 §11：不可放寬 → 進 Z
MIN_HISTORY_FOR_MA240 = 240
MIN_HISTORY_FOR_MA60 = 60
MIN_HISTORY_FOR_MA20 = 20


@dataclass
class FetchResult:
    df: pd.DataFrame
    source: str
    ok: bool
    error: Optional[str] = None


def _http_get_json(url: str, timeout: int = 30) -> Tuple[bool, Optional[object], Optional[str]]:
    try:
        r = requests.get(url, timeout=timeout, headers={"User-Agent": "tw-stock-layers/1.0"})
        if r.status_code != 200:
            return False, None, f"HTTP {r.status_code}: {url}"
        # 有些端點回 JSON，有些回 text/json；統一嘗試 json()
        return True, r.json(), None
    except Exception as e:
        return False, None, f"{type(e).__name__}: {e}"


def fetch_twse_daily_all() -> FetchResult:
    ok, data, err = _http_get_json(TWSE_DAILY_ALL)
    if not ok:
        return FetchResult(pd.DataFrame(), "TWSE", False, err)

    # 嘗試自動辨識欄位（不同端點可能欄名不同）
    df = pd.DataFrame(data)
    if df.empty:
        return FetchResult(df, "TWSE", False, "TWSE returned empty dataset")

    # 常見欄位猜測：Code/StockNo/證券代號, Name/證券名稱, Close/收盤價, Volume/成交股數
    colmap = {}
    for c in df.columns:
        lc = str(c).lower()
        if "code" in lc or "stockno" in lc or "證券代號" in c:
            colmap[c] = "code"
        elif "name" in lc or "證券名稱" in c:
            colmap[c] = "name"
        elif "close" in lc or "收盤" in c:
            colmap[c] = "close"
        elif "volume" in lc or "成交股數" in c or "成交量" in c:
            colmap[c] = "volume"

    df = df.rename(columns=colmap)
    need = {"code", "close"}
    if not need.issubset(df.columns):
        return FetchResult(df, "TWSE", False, f"TWSE schema unexpected: columns={list(df.columns)[:30]}")

    df["market"] = "TWSE"
    return FetchResult(df, "TWSE", True)


def fetch_tpex_daily_all() -> FetchResult:
    last_err = None
    for url in TPEX_DAILY_ALL_CANDIDATES:
        ok, data, err = _http_get_json(url)
        if not ok:
            last_err = err
            continue
        df = pd.DataFrame(data)
        if df.empty:
            last_err = f"TPEX empty dataset: {url}"
            continue

        colmap = {}
        for c in df.columns:
            lc = str(c).lower()
            if "code" in lc or "stock" in lc or "代號" in c:
                colmap[c] = "code"
            elif "name" in lc or "名稱" in c:
                colmap[c] = "name"
            elif "close" in lc or "收盤" in c:
                colmap[c] = "close"
            elif "volume" in lc or "成交" in c:
                colmap[c] = "volume"

        df = df.rename(columns=colmap)
        if {"code", "close"}.issubset(df.columns):
            df["market"] = "TPEx"
            return FetchResult(df, "TPEx", True)

        last_err = f"TPEX schema unexpected: {url}, columns={list(df.columns)[:30]}"

    return FetchResult(pd.DataFrame(), "TPEx", False, last_err or "TPEX fetch failed")


def load_history() -> pd.DataFrame:
    if os.path.exists(HISTORY_PATH):
        try:
            hist = pd.read_csv(HISTORY_PATH, dtype={"code": str})
            hist["date"] = pd.to_datetime(hist["date"])
            return hist
        except Exception:
            pass
    return pd.DataFrame(columns=["date", "code", "market", "close", "volume"])


def append_today(history: pd.DataFrame, today_df: pd.DataFrame, today: pd.Timestamp) -> pd.DataFrame:
    # 保留必要欄位
    keep = ["code", "market", "close"]
    if "volume" in today_df.columns:
        keep.append("volume")
    df = today_df[keep].copy()
    df["date"] = today
    df["code"] = df["code"].astype(str).str.strip()

    # 去重（同日同股只留最後）
    out = pd.concat([history, df], ignore_index=True)
    out["date"] = pd.to_datetime(out["date"])
    out = out.drop_duplicates(subset=["date", "code", "market"], keep="last")
    return out


def compute_indicators(hist: pd.DataFrame) -> pd.DataFrame:
    # 以 (code, market) 分組計算 MA 與回落
    hist = hist.sort_values(["market", "code", "date"]).copy()
    hist["close"] = pd.to_numeric(hist["close"], errors="coerce")
    hist["volume"] = pd.to_numeric(hist.get("volume", np.nan), errors="coerce")

    def _grp(g: pd.DataFrame) -> pd.DataFrame:
        g = g.copy()
        g["ma20"] = g["close"].rolling(MIN_HISTORY_FOR_MA20).mean()
        g["ma60"] = g["close"].rolling(MIN_HISTORY_FOR_MA60).mean()
        g["ma240"] = g["close"].rolling(MIN_HISTORY_FOR_MA240).mean()
        g["hi_close"] = g["close"].cummax()
        g["dd_from_hi"] = g["close"] / g["hi_close"] - 1.0
        return g

    return hist.groupby(["market", "code"], group_keys=False).apply(_grp)


def layer_logic(today_row: pd.Series, hist_tail: pd.DataFrame) -> Tuple[str, str]:
    """
    MVP 分層：
    - 資料不足：Z（依 §11）
    - E：今日收盤 > ma60 且 dd_from_hi > -8% 且 close > ma20（近似強勢短驗證）
    - A/B/C/D：先用 ma240/ma60 作粗分（後續再把營收YoY、RS、量能等補齊）
    - 循環股/金融股判定：MVP 暫不自動辨識（先留待下一步加產業分類）
    """
    # 資料不足 → Z
    if pd.isna(today_row.get("ma60")) or pd.isna(today_row.get("ma240")) or pd.isna(today_row.get("ma20")):
        return "Z", "§11 資料不足：MA 不足，列觀察"

    close = float(today_row["close"])
    ma20 = float(today_row["ma20"])
    ma60 = float(today_row["ma60"])
    ma240 = float(today_row["ma240"])
    dd = float(today_row.get("dd_from_hi", 0.0))

    # 近似 E（強動能短驗證）：站上 ma60、站上 ma20、未回落 -8%
    if close > ma60 and close > ma20 and dd > -E_DRAWDOWN:
        return "E", "近似E：收盤>MA60且>MA20且未回落-8%"

    # 趨勢粗分
    if close > ma240 and close > ma60:
        return "B", "趨勢：收盤>MA240且>MA60（MVP）"
    if close > ma240 and close <= ma60:
        return "C", "回檔：收盤>MA240但≤MA60（MVP）"
    if close <= ma240 and close > ma60:
        return "D", "反彈：收盤≤MA240但>MA60（MVP）"

    return "Z", "弱勢：未達趨勢條件，列觀察（MVP）"


def build_html_report(date_str: str, layers: pd.DataFrame, warnings: list[str], errors: list[str]) -> str:
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
    html.append("<style>body{font-family:system-ui,-apple-system,Segoe UI,Roboto,Arial; margin:24px;} table{border-collapse:collapse; width:100%;} th,td{border:1px solid #ddd; padding:8px;} th{background:#f5f5f5;}</style>")
    html.append("</head><body>")
    html.append(f"<h1>每日分層報表</h1><p>日期：{date_str}（Asia/Taipei）</p>")

    if errors:
        html.append("<h2>抓取錯誤</h2><ul>")
        for e in errors:
            html.append(f"<li>{e}</li>")
        html.append("</ul>")

    if warnings:
        html.append("<h2>即時警示（MVP）</h2><ul>")
        for w in warnings:
            html.append(f"<li>{w}</li>")
        html.append("</ul>")

    # 分層輸出
    for layer in ["A", "B", "C", "D", "E", "Z"]:
        df_layer = layers[layers["layer"] == layer].sort_values(["market", "code"])
        html.append(_table(df_layer, f"{layer} 層"))

    html.append("<hr><p style='color:#666'>註：本版本為可上線 MVP。當歷史資料累積達 MA/量能/基本面需求後，分層將逐步符合 v7.9.9.1 全條文。</p>")
    html.append("</body></html>")
    return "\n".join(html)


def main() -> int:
    os.makedirs("docs", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    today = pd.Timestamp(dt.datetime.now().date())
    date_str = today.strftime("%Y-%m-%d")

    errors: list[str] = []
    warnings: list[str] = []

    twse = fetch_twse_daily_all()
    if not twse.ok:
        errors.append(f"TWSE 取得失敗：{twse.error}")
    tpex = fetch_tpex_daily_all()
    if not tpex.ok:
        errors.append(f"TPEx 取得失敗：{tpex.error}")

    daily = pd.concat([twse.df, tpex.df], ignore_index=True) if (twse.ok or tpex.ok) else pd.DataFrame()
    if daily.empty:
        html = build_html_report(date_str, pd.DataFrame(columns=["market","code","name","close","layer","reason"]), warnings, errors)
        with open(OUT_HTML, "w", encoding="utf-8") as f:
            f.write(html)
        return 0

    # 讀取/累積歷史
    hist = load_history()
    hist = append_today(hist, daily, today)
    hist.to_csv(HISTORY_PATH, index=False, encoding="utf-8")

    # 計算指標
    hist_ind = compute_indicators(hist)
    # 取今日資料（含指標）
    today_ind = hist_ind[hist_ind["date"] == today].copy()
    if today_ind.empty:
        errors.append("今日指標資料為空（可能是日期格式或寫入失敗）")

    # 分層
    out_rows = []
    for _, row in today_ind.iterrows():
        layer, reason = layer_logic(row, hist_ind)
        out_rows.append({
            "market": row.get("market", ""),
            "code": str(row.get("code", "")).strip(),
            "name": row.get("name", ""),
            "close": row.get("close", ""),
            "layer": layer,
            "reason": reason,
        })
    layers = pd.DataFrame(out_rows)

    # MVP 警示（先做工程級）
    if any("取得失敗" in e for e in errors):
        warnings.append("🟠 資料源部分失效：依 §11 降級，今日分層可能偏向 Z")
    z_ratio = (layers["layer"] == "Z").mean() if len(layers) else 1.0
    if z_ratio > 0.8:
        warnings.append("🟡 Z 層占比偏高：歷史資料尚在累積（屬正常MVP階段）")

    # 產出 HTML
    html = build_html_report(date_str, layers, warnings, errors)
    with open(OUT_HTML, "w", encoding="utf-8") as f:
        f.write(html)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
