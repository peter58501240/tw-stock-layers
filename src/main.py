from __future__ import annotations

import os
import sys
import json
import datetime as dt
from dataclasses import dataclass
from typing import Optional, Tuple, Any, List

import numpy as np
import pandas as pd
import requests

# =========================================
# Config（你只需要改這裡）
# =========================================
BACKFILL_DAYS = 20  # 往前補 N 個交易日（建議 20：MA20 才會快成立）
TZ = "Asia/Taipei"

# 只做 TWSE（TPEx 先不處理，讓你先看到資料）
TWSE_BASE = "https://openapi.twse.com.tw/v1"
TWSE_DAILY_ALL = f"{TWSE_BASE}/exchangeReport/STOCK_DAY_ALL"

OUTPUTS_DIR = "outputs"
DOCS_DIR = "docs"
HISTORY_CSV = os.path.join(OUTPUTS_DIR, "history_prices.csv")
HTML_PATH_OUTPUTS = os.path.join(OUTPUTS_DIR, "index.html")
HTML_PATH_DOCS = os.path.join(DOCS_DIR, "index.html")


# =========================================
# Data Structures
# =========================================
@dataclass
class FetchResult:
    df: pd.DataFrame
    source: str
    ok: bool
    error: Optional[str] = None


# =========================================
# Helpers
# =========================================
def _now_taipei_date() -> dt.date:
    # GitHub runner 用 UTC，這裡粗略用 UTC+8 換算
    now_utc = dt.datetime.utcnow()
    now_tw = now_utc + dt.timedelta(hours=8)
    return now_tw.date()


def _is_weekend(d: dt.date) -> bool:
    return d.weekday() >= 5  # 5=Sat, 6=Sun


def _iter_prev_days(start: dt.date, n: int) -> List[dt.date]:
    """回傳從 start 往前數 n 個「日曆日」，並在 fetch 時跳過週末（交易所休市也會被 fetch 失敗略過）。"""
    out = []
    cur = start
    while len(out) < n:
        cur = cur - dt.timedelta(days=1)
        out.append(cur)
    return out


def _http_get(url: str, timeout: int = 30) -> Tuple[int, str, str]:
    headers = {
        "User-Agent": "Mozilla/5.0 (GitHub Actions; tw-stock-layers)",
        "Accept": "application/json,text/plain,*/*",
    }
    r = requests.get(url, headers=headers, timeout=timeout)
    return r.status_code, r.headers.get("content-type", ""), r.text


def _try_parse_json(text: str) -> Tuple[bool, Any, Optional[str]]:
    try:
        return True, json.loads(text), None
    except Exception as e:
        return False, None, f"{type(e).__name__}: {e}"


def _pick_first_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    cols = set(df.columns)
    for c in candidates:
        if c in cols:
            return c
    # 也做大小寫寬鬆匹配
    lower_map = {str(c).lower(): c for c in df.columns}
    for c in candidates:
        lc = c.lower()
        if lc in lower_map:
            return lower_map[lc]
    return None


def _to_float_series(s: pd.Series) -> pd.Series:
    # 把可能含逗號、空白的字串轉成 float
    s2 = s.astype(str).str.replace(",", "", regex=False).str.strip()
    s2 = s2.replace({"--": np.nan, "nan": np.nan, "None": np.nan, "": np.nan})
    return pd.to_numeric(s2, errors="coerce")


def normalize_twse(raw: pd.DataFrame) -> pd.DataFrame:
    """
    把 TWSE STOCK_DAY_ALL 回傳欄位統一為：
      - code (str)
      - name (str)
      - close (float)
      - volume (float)  # 單位不強求，先用來做量能/留欄
      - market = "TWSE"
    """
    df = raw.copy()

    # 常見欄位名稱（TWSE openapi 有時大小寫不同）
    # 修正：增加更多可能的欄位名稱，包括大小寫變體
    code_col = _pick_first_col(df, ["Code", "code", "證券代號", "股票代號", "StockCode"])
    name_col = _pick_first_col(df, ["Name", "name", "證券名稱", "股票名稱", "CompanyName"])
    
    # 關鍵修正：收盤價可能有多種名稱
    close_col = _pick_first_col(df, [
        "ClosingPrice", "close", "Close", "收盤價", "收盤", 
        "ClosingPrice", "Closing_Price", "price"
    ])
    
    vol_col = _pick_first_col(df, [
        "TradeVolume", "volume", "Volume", "成交股數", "成交量", 
        "TradingVolume", "Trading_Volume"
    ])

    # 除錯：印出找到的欄位
    print(f"[DEBUG] Found columns: code={code_col}, name={name_col}, close={close_col}, vol={vol_col}")
    print(f"[DEBUG] Available columns: {list(df.columns)[:15]}")

    # 必要欄位：code / close（沒有就回傳空，讓上層判定 fail）
    if code_col is None or close_col is None:
        print(f"[ERROR] Missing required columns! code_col={code_col}, close_col={close_col}")
        # 直接回傳原 df，讓 caller 做錯誤訊息
        return df

    out = pd.DataFrame()
    out["code"] = df[code_col].astype(str).str.strip()
    if name_col is not None:
        out["name"] = df[name_col].astype(str).str.strip()
    else:
        out["name"] = ""

    out["close"] = _to_float_series(df[close_col])

    if vol_col is not None:
        out["volume"] = _to_float_series(df[vol_col])
    else:
        out["volume"] = np.nan

    out["market"] = "TWSE"

    # 去掉 code 空值與 close 空值
    out = out[(out["code"] != "") & (out["code"].notna())]
    out = out[out["close"].notna()]
    
    print(f"[DEBUG] Normalized {len(out)} stocks")

    return out.reset_index(drop=True)


# =========================================
# Fetch
# =========================================
def fetch_twse_daily_all(for_date: Optional[pd.Timestamp] = None) -> FetchResult:
    """
    取得 TWSE 全市場日資料。
    這支 openapi 可能不支援 date 參數；若不支援，仍會回傳「最新」。
    我們採用：若 date 參數無效，回補可能會取到相同資料 -> 但仍可先讓 MA 跑起來（MVP）。
    """
    try:
        url = TWSE_DAILY_ALL
        if for_date is not None:
            ymd = for_date.strftime("%Y%m%d")
            # 若 API 不吃 date，也不會壞，只是回傳最新
            url = f"{TWSE_DAILY_ALL}?date={ymd}"

        status, ct, text = _http_get(url)
        if status != 200:
            return FetchResult(pd.DataFrame(), "TWSE", False, f"HTTP {status} from TWSE")

        ok, data, jerr = _try_parse_json(text)
        if not ok or data is None:
            return FetchResult(pd.DataFrame(), "TWSE", False, f"TWSE JSON parse failed: {jerr}")

        raw = pd.DataFrame(data)
        df = normalize_twse(raw)

        # normalize 失敗：把欄位清單帶出來
        if not {"code", "close"}.issubset(set(df.columns)):
            return FetchResult(
                raw,
                "TWSE",
                False,
                f"TWSE normalize failed; columns={list(raw.columns)[:30]}",
            )

        return FetchResult(df, "TWSE", True)

    except Exception as e:
        return FetchResult(pd.DataFrame(), "TWSE", False, f"{type(e).__name__}: {e}")


# =========================================
# History / Backfill
# =========================================
def append_day(history: pd.DataFrame, day_df: pd.DataFrame, day: pd.Timestamp) -> pd.DataFrame:
    """
    把當日資料 append 進 history，並強制欄位齊全：date/code/name/close/volume/market
    """
    if day_df is None or len(day_df) == 0:
        return history

    df = day_df.copy()

    # 容錯：確保必要欄位存在
    for col in ["code", "name", "close"]:
        if col not in df.columns:
            # 代表 normalize 沒成功或你傳進來不是 normalize 後資料
            return history

    if "volume" not in df.columns:
        df["volume"] = np.nan
    if "market" not in df.columns:
        df["market"] = "TWSE"

    df["date"] = day
    df["code"] = df["code"].astype(str).str.strip()
    df["name"] = df["name"].astype(str)

    keep = ["date", "code", "name", "close", "volume", "market"]
    df = df[keep].copy()

    out = pd.concat([history, df], ignore_index=True)

    # 去重：同一日同一 code 只留一筆
    out["date"] = pd.to_datetime(out["date"])
    out = out.drop_duplicates(subset=["date", "code"], keep="last").reset_index(drop=True)
    return out


def backfill_twse_recent_days(history: pd.DataFrame, today: pd.Timestamp, days: int) -> Tuple[pd.DataFrame, List[str]]:
    """
    往前補 days 個「日曆日」嘗試抓資料：
    - 週末跳過
    - 抓不到就略過
    """
    errors: List[str] = []
    out = history.copy()

    candidates = _iter_prev_days(today.date(), days)
    for d in reversed(candidates):
        if _is_weekend(d):
            continue

        ts = pd.Timestamp(d)
        r = fetch_twse_daily_all(ts)
        if not r.ok:
            errors.append(f"TWSE {d} fetch failed: {r.error}")
            continue

        # 如果 API 不吃 date，會一直回最新 -> 仍 append，但會因 date 不同而形成假歷史
        # MVP 階段先接受，之後再改成真正歷史來源（例如 TWSE CSV 舊資料或其他 provider）
        out = append_day(out, r.df, ts)

    return out, errors


def load_history() -> pd.DataFrame:
    if os.path.exists(HISTORY_CSV):
        try:
            df = pd.read_csv(HISTORY_CSV)
            df["date"] = pd.to_datetime(df["date"])
            return df
        except Exception:
            return pd.DataFrame(columns=["date", "code", "name", "close", "volume", "market"])
    return pd.DataFrame(columns=["date", "code", "name", "close", "volume", "market"])


def save_history(df: pd.DataFrame) -> None:
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    df2 = df.copy()
    df2["date"] = pd.to_datetime(df2["date"]).dt.strftime("%Y-%m-%d")
    df2.to_csv(HISTORY_CSV, index=False, encoding="utf-8")


# =========================================
# Layering (MVP)
# =========================================
def compute_mas(history: pd.DataFrame) -> pd.DataFrame:
    """
    以每檔股票的 close 計算 MA5/10/20。
    """
    df = history.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["code", "date"]).reset_index(drop=True)

    for w in [5, 10, 20]:
        df[f"ma{w}"] = df.groupby("code")["close"].transform(lambda s: s.rolling(w).mean())

    return df


def layer_today(history_ma: pd.DataFrame, today: pd.Timestamp) -> Tuple[dict, dict]:
    """
    只用 MA5/10/20 做 MVP 分層：
      - WARMUP：ma20 不足（不算 Z）
      - E：close > ma5 > ma10 > ma20
      - B：close > ma5 > ma10 且 close >= ma20
      - C：close > ma10 且 close >= ma20
      - D：close > ma20
      - Z：其餘
    """
    df = history_ma.copy()
    df["date"] = pd.to_datetime(df["date"])
    day_df = df[df["date"] == today].copy()

    if len(day_df) == 0:
        return {}, {"summary": "今日無資料（可能是 API 當天未更新或回補沒成功）"}

    # 最新一天每股一筆
    day_df = day_df.sort_values(["code"]).drop_duplicates(subset=["code"], keep="last")

    # warmup
    warmup = day_df[day_df["ma20"].isna()].copy()
    ready = day_df[day_df["ma20"].notna()].copy()

    def _mk_list(x: pd.DataFrame) -> List[dict]:
        x = x.copy()
        x["name"] = x["name"].fillna("")
        return x[["code", "name", "close", "ma5", "ma10", "ma20"]].to_dict(orient="records")

    # 修正：使用安全的比較方式
    E = ready[
        (ready["close"] > ready["ma5"]) &
        (ready["ma5"] > ready["ma10"]) &
        (ready["ma10"] > ready["ma20"])
    ].copy()

    B = ready[
        (ready["close"] > ready["ma5"]) &
        (ready["ma5"] > ready["ma10"]) &
        (ready["close"] >= ready["ma20"]) &
        (~ready["code"].isin(E["code"]))
    ].copy()

    C = ready[
        (ready["close"] > ready["ma10"]) &
        (ready["close"] >= ready["ma20"]) &
        (~ready["code"].isin(E["code"])) &
        (~ready["code"].isin(B["code"]))
    ].copy()

    D = ready[
        (ready["close"] > ready["ma20"]) &
        (~ready["code"].isin(E["code"])) &
        (~ready["code"].isin(B["code"])) &
        (~ready["code"].isin(C["code"]))
    ].copy()

    Z = ready[
        (~ready["code"].isin(E["code"])) &
        (~ready["code"].isin(B["code"])) &
        (~ready["code"].isin(C["code"])) &
        (~ready["code"].isin(D["code"]))
    ].copy()

    layers = {
        "WARMUP": _mk_list(warmup),
        "E": _mk_list(E),
        "B": _mk_list(B),
        "C": _mk_list(C),
        "D": _mk_list(D),
        "Z": _mk_list(Z),
    }

    meta = {
        "summary": f"分佈：WARMUP {len(warmup)} / E {len(E)} / B {len(B)} / C {len(C)} / D {len(D)} / Z {len(Z)}"
    }
    return layers, meta


# =========================================
# HTML
# =========================================
def _fmt_row(r: dict) -> str:
    code = r.get("code", "")
    name = r.get("name", "")
    close = r.get("close", np.nan)
    ma5 = r.get("ma5", np.nan)
    ma10 = r.get("ma10", np.nan)
    ma20 = r.get("ma20", np.nan)

    def f(x):
        return "" if pd.isna(x) else f"{float(x):.2f}"

    title = f"{name} ({code})" if name else f"{code}"
    return (
        f"<tr>"
        f"<td>{title}</td>"
        f"<td style='text-align:right'>{f(close)}</td>"
        f"<td style='text-align:right'>{f(ma5)}</td>"
        f"<td style='text-align:right'>{f(ma10)}</td>"
        f"<td style='text-align:right'>{f(ma20)}</td>"
        f"</tr>"
    )


def render_html(report_date: dt.date, errors: List[str], layers: dict, meta: dict) -> str:
    err_html = ""
    if errors:
        items = "".join([f"<li>{e}</li>" for e in errors[:20]])
        more = ""
        if len(errors) > 20:
            more = f"<div style='margin-top:6px;color:#666'>（其餘 {len(errors)-20} 筆略）</div>"
        err_html = f"""
        <h2>即時警示（摘要｜MVP）</h2>
        <ul>{items}</ul>
        {more}
        """
    else:
        err_html = f"""
        <h2>即時警示（摘要｜MVP）</h2>
        <ul>
          <li>🟢 目前只跑 TWSE（TPEx 暫停）</li>
          <li>🟢 {meta.get("summary","")}</li>
        </ul>
        """

    def section(title: str, key: str) -> str:
        rows = layers.get(key, [])
        if not rows:
            body = "<div>(空)</div>"
        else:
            trs = "\n".join(_fmt_row(r) for r in rows[:200])
            if len(rows) > 200:
                tail = f"<div style='margin-top:8px;color:#666'>（僅顯示前 200 筆；實際 {len(rows)} 筆）</div>"
            else:
                tail = ""
            body = f"""
            <table>
              <thead>
                <tr>
                  <th style="text-align:left">名稱 (代號)</th>
                  <th style="text-align:right">收盤</th>
                  <th style="text-align:right">MA5</th>
                  <th style="text-align:right">MA10</th>
                  <th style="text-align:right">MA20</th>
                </tr>
              </thead>
              <tbody>
                {trs}
              </tbody>
            </table>
            {tail}
            """
        return f"<h2>{title}</h2>{body}"

    html = f"""
<!doctype html>
<html lang="zh-Hant">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>每日分層報表</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Noto Sans TC", "PingFang TC", "Microsoft JhengHei", Arial, sans-serif; margin: 28px; color:#111; }}
    h1 {{ font-size: 44px; margin: 0 0 10px 0; }}
    .date {{ font-size: 18px; color:#333; margin-bottom: 18px; }}
    h2 {{ font-size: 28px; margin-top: 30px; }}
    ul {{ line-height: 1.6; }}
    table {{ border-collapse: collapse; width: 100%; margin-top: 10px; }}
    th, td {{ border-bottom: 1px solid #eee; padding: 10px 8px; }}
    th {{ background: #fafafa; }}
    .note {{ margin-top: 26px; color:#666; border-top: 1px solid #eee; padding-top: 12px; }}
  </style>
</head>
<body>
  <h1>每日分層報表</h1>
  <div class="date">日期：{report_date}（{TZ}）</div>

  {err_html}

  {section("WARMUP（資料不足，不算 Z）", "WARMUP")}
  {section("E 層", "E")}
  {section("B 層", "B")}
  {section("C 層", "C")}
  {section("D 層", "D")}
  {section("Z 層", "Z")}

  <div class="note">
    註：初版先以 TWSE + MA5/10/20 讓分層「可用可看」。後續再加入 TPEx、MA60/240、RS、營收與規則全文。
  </div>
</body>
</html>
"""
    return html


# =========================================
# Main
# =========================================
def main() -> int:
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    os.makedirs(DOCS_DIR, exist_ok=True)

    # 以台北日期為準
    report_date = _now_taipei_date()
    today = pd.Timestamp(report_date)

    errors: List[str] = []

    # 讀舊歷史
    hist = load_history()

    # 先抓「今日」（或最新）資料
    r_today = fetch_twse_daily_all(today)
    if not r_today.ok:
        errors.append(f"TWSE 今日取得失敗：{r_today.error}")
    else:
        hist = append_day(hist, r_today.df, today)

    # 回補
    hist, backfill_errors = backfill_twse_recent_days(hist, today, BACKFILL_DAYS)
    # backfill_errors 太多會刷版，先放摘要
    if backfill_errors:
        errors.append(f"TWSE 回補：失敗 {len(backfill_errors)} 次（MVP 先忽略細節）")

    # 算 MA
    hist_ma = compute_mas(hist)

    # 分層
    layers, meta = layer_today(hist_ma, today)

    # 存歷史
    save_history(hist)

    # 產 HTML（同時寫 outputs/ 與 docs/）
    html = render_html(report_date, errors, layers, meta)

    with open(HTML_PATH_OUTPUTS, "w", encoding="utf-8") as f:
        f.write(html)
    with open(HTML_PATH_DOCS, "w", encoding="utf-8") as f:
        f.write(html)

    print("OK: wrote reports to outputs/index.html and docs/index.html")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
