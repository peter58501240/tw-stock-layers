from __future__ import annotations

BACKFILL_DAYS = 10  # 往前補 10 個交易日

import os
import sys
import datetime as dt
from dataclasses import dataclass
from typing import Optional, Tuple, Any, List

import numpy as np
import pandas as pd
import requests

# -----------------------------
# Config
# -----------------------------
TWSE_BASE = "https://openapi.twse.com.tw/v1"
TPEX_BASE = "https://www.tpex.org.tw/openapi/v1"  # 可能需要再調，但本版已容錯

# 端點
TWSE_DAILY_ALL = f"{TWSE_BASE}/exchangeReport/STOCK_DAY_ALL"

# TPEx 端點候選（任一成功即可）
TPEX_DAILY_ALL_CANDIDATES = [
    f"{TPEX_BASE}/stock_aftertrading_daily_trading_info",
    f"{TPEX_BASE}/tpex_mainboard_daily",
]

HISTORY_PATH = "outputs/history_prices.csv"
OUT_HTML = "docs/index.html"

# v7.9.9.1 MVP 參數（技術面子集）
E_DRAWDOWN = 0.08      # -8%
MIN_HISTORY_FOR_MA20 = 20
MIN_HISTORY_FOR_MA60 = 60
MIN_HISTORY_FOR_MA240 = 240

# 欄位別名對照：不同資料源命名不同，一律轉成統一欄位
COLUMN_ALIASES = {
    # 日期
    "Date": "date",
    "日期": "date",
    # 代號/名稱
    "code": "code",
    "Code": "code",
    "StockNo": "code",
    "證券代號": "code",
    "name": "name",
    "Name": "name",
    "證券名稱": "name",
    # 成交量
    "volume": "volume",
    "Volume": "volume",
    "成交量": "volume",
    "成交股數": "volume",
    # 收盤價
    "close": "close",
    "Close": "close",
    "ClosingPrice": "close",
    "收盤價": "close",
    "收盤": "close",
    # 開高低（備用）
    "OpeningPrice": "open",
    "HighestPrice": "high",
    "LowestPrice": "low",
    # 成交金額（備用）
    "TradeValue": "trade_value",
    # 漲跌（備用）
    "Change": "change",
    # 成交筆數（備用）
    "Transaction": "transactions",
}


@dataclass
class FetchResult:
    df: pd.DataFrame
    source: str
    ok: bool
    error: Optional[str] = None
    warn: Optional[str] = None


def _http_get(url: str, timeout: int = 30) -> Tuple[int, str, str]:
    """
    回傳 (status_code, content_type, text)
    """
    r = requests.get(url, timeout=timeout, headers={"User-Agent": "tw-stock-layers/1.1"})
    ct = r.headers.get("content-type", "")
    return r.status_code, ct, r.text


def _try_parse_json(text: str) -> Tuple[bool, Optional[Any], Optional[str]]:
    try:
        return True, requests.models.complexjson.loads(text), None
    except Exception as e:
        return False, None, f"{type(e).__name__}: {e}"


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    將不同來源欄位名稱統一成：date/code/name/close/volume/...
    """
    if df is None or df.empty:
        return df

    new_cols = {}
    for c in df.columns:
        if c in COLUMN_ALIASES:
            new_cols[c] = COLUMN_ALIASES[c]
        else:
            # 也嘗試以 lower 去對應
            lc = str(c).strip()
            if lc in COLUMN_ALIASES:
                new_cols[c] = COLUMN_ALIASES[lc]
            else:
                new_cols[c] = c  # 保留原欄位（不影響核心）
    df = df.rename(columns=new_cols)

    # code 一律字串
    if "code" in df.columns:
        df["code"] = df["code"].astype(str).str.strip()

    return df


def fetch_twse_daily_all(for_date: Optional[pd.Timestamp] = None) -> FetchResult:
    """
    取得 TWSE 全市場日資料。
    若 for_date 有值，嘗試用 querystring 帶入日期（YYYYMMDD）。
    取不到就回 ok=False 但不丟例外。
    """
    try:
        url = TWSE_DAILY_ALL
        if for_date is not None:
            ymd = for_date.strftime("%Y%m%d")
            # 嘗試常見參數名稱：date
            url = f"{TWSE_DAILY_ALL}?date={ymd}"

        status, ct, text = _http_get(url)
        if status != 200:
            return FetchResult(pd.DataFrame(), "TWSE", False, f"HTTP {status} from TWSE: {url}")

        ok, data, jerr = _try_parse_json(text)
        if not ok or data is None:
            return FetchResult(pd.DataFrame(), "TWSE", False, f"TWSE JSON parse failed: {jerr}")

        df = pd.DataFrame(data)
        df = normalize_columns(df)

        if not {"code", "close"}.issubset(df.columns):
            return FetchResult(df, "TWSE", False, f"TWSE missing required columns after normalize: {list(df.columns)[:30]}")

        df["market"] = "TWSE"
        return FetchResult(df, "TWSE", True)

    except Exception as e:
        return FetchResult(pd.DataFrame(), "TWSE", False, f"{type(e).__name__}: {e}")

    def backfill_twse_recent_days(history: pd.DataFrame, today: pd.Timestamp, days: int) -> Tuple[pd.DataFrame, list[str]]:
    """
    往前回補最近 days 個「交易日」的資料（以日為步進，抓不到就跳過）。
    只回補 TWSE（TPEx 先不強求）。
    """
    notes = []
    if days <= 0:
        return history, notes

    # 已經有資料的日期集合（TWSE）
    have_dates = set(
        pd.to_datetime(history.loc[history["market"] == "TWSE", "date"]).dt.date.astype(str).tolist()
    ) if not history.empty else set()

    filled = 0
    # 往前最多掃 2*days 天（避免遇到連假完全補不到）
    for i in range(1, days * 2 + 1):
        d = today - pd.Timedelta(days=i)
        d_key = d.date().isoformat()
        if d_key in have_dates:
            continue

        r = fetch_twse_daily_all(d)
        if not r.ok or r.df.empty:
            continue

        history = append_today(history, r.df, d)
        have_dates.add(d_key)
        filled += 1
        if filled >= days:
            break

    if filled > 0:
        notes.append(f"🟢 TWSE 已回補近 {filled} 個交易日（目標 {days}）")
    else:
        notes.append("🟠 TWSE 回補失敗：可能端點不支援 date 參數或被限制（仍可每日累積）")

    return history, notes


def fetch_tpex_daily_all() -> FetchResult:
    last_warn = None
    last_err = None

    for url in TPEX_DAILY_ALL_CANDIDATES:
        try:
            status, ct, text = _http_get(url)
            if status != 200:
                last_err = f"TPEX HTTP {status}: {url}"
                continue

            # 有些時候會回 HTML 或空字串
            if text is None or len(text.strip()) == 0:
                last_warn = f"TPEX empty response (likely blocked or no data): {url}"
                continue

            # 如果 content-type 不是 json，也先嘗試 parse；失敗就當容錯略過
            ok, data, jerr = _try_parse_json(text)
            if not ok or data is None:
                # 容錯：記 warn，不中斷
                snippet = text.strip()[:120].replace("\n", " ")
                last_warn = f"TPEX non-JSON response: {url} ({jerr}); head='{snippet}'"
                continue

            df = pd.DataFrame(data)
            df = normalize_columns(df)

            if {"code", "close"}.issubset(df.columns):
                df["market"] = "TPEx"
                return FetchResult(df, "TPEx", True, warn=last_warn)
            else:
                last_warn = f"TPEX schema unexpected after normalize: {url}, columns={list(df.columns)[:30]}"
                continue

        except Exception as e:
            last_err = f"TPEX exception: {type(e).__name__}: {e}"
            continue

    # 這裡改成「ok=False 但不致命」：主流程會用 warning 呈現並繼續跑 TWSE
    return FetchResult(pd.DataFrame(), "TPEx", False, error=last_err, warn=last_warn)


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
    keep = ["code", "market", "close"]
    if "volume" in today_df.columns:
        keep.append("volume")

    df = today_df[keep].copy()
    df["date"] = today
    df["code"] = df["code"].astype(str).str.strip()

    out = pd.concat([history, df], ignore_index=True)
    out["date"] = pd.to_datetime(out["date"])
    out = out.drop_duplicates(subset=["date", "code", "market"], keep="last")
    return out


def compute_indicators(hist: pd.DataFrame) -> pd.DataFrame:
    hist = hist.sort_values(["market", "code", "date"]).copy()
    hist["close"] = pd.to_numeric(hist["close"], errors="coerce")
    if "volume" in hist.columns:
        hist["volume"] = pd.to_numeric(hist["volume"], errors="coerce")
    else:
        hist["volume"] = np.nan

    def _grp(g: pd.DataFrame) -> pd.DataFrame:
        g = g.copy()
        g["ma20"] = g["close"].rolling(MIN_HISTORY_FOR_MA20).mean()
        g["ma60"] = g["close"].rolling(MIN_HISTORY_FOR_MA60).mean()
        g["ma240"] = g["close"].rolling(MIN_HISTORY_FOR_MA240).mean()
        g["hi_close"] = g["close"].cummax()
        g["dd_from_hi"] = g["close"] / g["hi_close"] - 1.0
        return g

    return hist.groupby(["market", "code"], group_keys=False).apply(_grp)


def layer_logic(today_row: pd.Series) -> Tuple[str, str]:
    """
    MVP 分層（技術面子集）：
    - MA 不足：Z（依 §11：缺值不放寬）
    - 近似 E：收盤 > MA60 且 > MA20 且未回落 -8%
    - B/C/D：以 MA240/MA60 粗分
    - A：此 MVP 先保留空（待補 RS/營收/量能條件後再開）
    """
    if pd.isna(today_row.get("ma60")) or pd.isna(today_row.get("ma240")) or pd.isna(today_row.get("ma20")):
        return "Z", "§11 資料不足：MA 不足，列觀察"

    close = float(today_row["close"])
    ma20 = float(today_row["ma20"])
    ma60 = float(today_row["ma60"])
    ma240 = float(today_row["ma240"])
    dd = float(today_row.get("dd_from_hi", 0.0))

    if close > ma60 and close > ma20 and dd > -E_DRAWDOWN:
        return "E", "近似E：收盤>MA60且>MA20且未回落-8%"

    if close > ma240 and close > ma60:
        return "B", "趨勢：收盤>MA240且>MA60（MVP）"
    if close > ma240 and close <= ma60:
        return "C", "回檔：收盤>MA240但≤MA60（MVP）"
    if close <= ma240 and close > ma60:
        return "D", "反彈：收盤≤MA240但>MA60（MVP）"

    return "Z", "弱勢：未達趨勢條件，列觀察（MVP）"


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

    for layer in ["A", "B", "C", "D", "E", "Z"]:
        df_layer = layers[layers["layer"] == layer].sort_values(["market", "code"])
        html.append(_table(df_layer, f"{layer} 層"))

    html.append("<hr><p style='color:#666'>註：本版本為可上線 MVP。MA/RS/營收/量能等資料補齊後，分層將逐步貼近 v7.9.9.1 全條文。</p>")
    html.append("</body></html>")
    return "\n".join(html)


def main() -> int:
    os.makedirs("docs", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    today = pd.Timestamp(dt.datetime.now().date())
    date_str = today.strftime("%Y-%m-%d")

    errors: List[str] = []
    warnings: List[str] = []

    twse = fetch_twse_daily_all()
    if not twse.ok:
        errors.append(f"TWSE 取得失敗：{twse.error}")

    tpex = fetch_tpex_daily_all()
    if not tpex.ok:
        # TPEx 不致命：改成 warning（不讓整個流程掛掉）
        if tpex.warn:
            warnings.append(f"🟠 TPEx 取得異常：{tpex.warn}")
        if tpex.error:
            warnings.append(f"🟠 TPEx 例外：{tpex.error}")
    else:
        if tpex.warn:
            warnings.append(f"🟡 TPEx 提示：{tpex.warn}")

    # 合併資料（只要其中一個有資料就繼續）
    frames = []
    if twse.ok and not twse.df.empty:
        frames.append(twse.df)
    if tpex.ok and not tpex.df.empty:
        frames.append(tpex.df)

    daily = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if daily.empty:
        if not errors:
            errors.append("今日資料為空（可能兩市場資料源皆暫時不可用）")
        html = build_html_report(date_str, pd.DataFrame(columns=["market", "code", "name", "close", "layer", "reason"]), warnings, errors)
        with open(OUT_HTML, "w", encoding="utf-8") as f:
            f.write(html)
        return 0

    # 讀取/累積歷史
    hist = load_history()

# 若歷史不足，先回補最近 N 個交易日（TWSE）
hist, bf_notes = backfill_twse_recent_days(hist, today, BACKFILL_DAYS)
warnings.extend(bf_notes)

# 再把今天資料寫入
hist = append_today(hist, daily, today)
hist.to_csv(HISTORY_PATH, index=False, encoding="utf-8")


    # 計算指標
    hist_ind = compute_indicators(hist)
    today_ind = hist_ind[hist_ind["date"] == today].copy()
    if today_ind.empty:
        errors.append("今日指標資料為空（可能是日期格式或寫入失敗）")

    # 分層
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

    # MVP 警示
    z_ratio = (layers["layer"] == "Z").mean() if len(layers) else 1.0
    if z_ratio > 0.8:
        warnings.append("🟡 Z 層占比偏高：歷史資料尚在累積（符合 §11 缺值降級）")

    if twse.ok and tpex.ok:
        warnings.append("🟢 TWSE/TPEx 皆已取得（若 TPEx 為空屬正常時段差異）")
    elif twse.ok and not tpex.ok:
        warnings.append("🟠 今日僅 TWSE 可用：TPEx 依 §11 降級處理")
    elif (not twse.ok) and tpex.ok:
        warnings.append("🟠 今日僅 TPEx 可用：TWSE 依 §11 降級處理")

    # 產出 HTML
    html = build_html_report(date_str, layers, warnings, errors)
    with open(OUT_HTML, "w", encoding="utf-8") as f:
        f.write(html)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
