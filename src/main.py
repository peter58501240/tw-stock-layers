from __future__ import annotations

import os
import json
import datetime as dt
from dataclasses import dataclass
from typing import Optional, Tuple, Any, List, Dict

import numpy as np
import pandas as pd
import requests

# =========================================
# Config（你只需要改這裡）
# =========================================
TZ = "Asia/Taipei"

# 往前補「交易日」(不是日曆日)
# 建議至少 30：確保 MA20 成立、還有些緩衝
BACKFILL_TRADING_DAYS = 35

# MA 視窗（先做 MVP）
MA_WINDOWS = [5, 10, 20]

# 只做 TWSE（TPEx 先不處理）
# 這個是「舊站 JSON」，date 真的有效，能拿歷史
TWSE_MI_INDEX = "https://www.twse.com.tw/exchangeReport/MI_INDEX"

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
# Time / Helpers
# =========================================
def _now_taipei_date() -> dt.date:
    # GitHub runner 多半是 UTC
    now_utc = dt.datetime.utcnow()
    now_tw = now_utc + dt.timedelta(hours=8)
    return now_tw.date()


def _is_weekend(d: dt.date) -> bool:
    return d.weekday() >= 5  # Sat/Sun


def _http_get_json(url: str, params: Dict[str, str], timeout: int = 30) -> Tuple[bool, Any, str]:
    headers = {
        "User-Agent": "Mozilla/5.0 (GitHub Actions; tw-stock-layers)",
        "Accept": "application/json,text/plain,*/*",
        "Referer": "https://www.twse.com.tw/",
    }
    r = requests.get(url, headers=headers, params=params, timeout=timeout)
    if r.status_code != 200:
        return False, None, f"HTTP {r.status_code}"
    try:
        return True, r.json(), ""
    except Exception as e:
        return False, None, f"JSONDecodeError: {e}"


def _to_float(x: Any) -> float:
    if x is None:
        return np.nan
    s = str(x).strip().replace(",", "")
    if s in ("", "--", "nan", "None"):
        return np.nan
    try:
        return float(s)
    except Exception:
        return np.nan


def _pick_index(fields: List[str], candidates: List[str]) -> Optional[int]:
    # 先完全匹配
    for c in candidates:
        if c in fields:
            return fields.index(c)
    # 再做大小寫/空白寬鬆
    norm = {str(f).strip().lower(): i for i, f in enumerate(fields)}
    for c in candidates:
        k = str(c).strip().lower()
        if k in norm:
            return norm[k]
    return None


# =========================================
# Fetch (TWSE MI_INDEX by date)
# =========================================
def fetch_twse_day_all(day: dt.date) -> FetchResult:
    """
    用 TWSE MI_INDEX 抓「指定日期」的整市場資料（真正歷史）。
    """
    ymd = day.strftime("%Y%m%d")
    params = {
        "response": "json",
        "date": ymd,
        "type": "ALL",  # ALL / ALLBUT0999 等；先用 ALL
    }

    ok, data, err = _http_get_json(TWSE_MI_INDEX, params=params)
    if not ok:
        return FetchResult(pd.DataFrame(), "TWSE", False, f"TWSE MI_INDEX {ymd} fetch failed: {err}")

    # 常見：休市會回傳 stat != OK
    stat = str(data.get("stat", "")).upper()
    if stat != "OK":
        return FetchResult(pd.DataFrame(), "TWSE", False, f"TWSE MI_INDEX {ymd} stat={data.get('stat')}")

    fields = data.get("fields", [])
    rows = data.get("data", [])
    if not fields or not rows:
        return FetchResult(pd.DataFrame(), "TWSE", False, f"TWSE MI_INDEX {ymd} empty fields/data")

    # 必要欄位（會因版本略不同，所以多放幾個候選）
    i_code = _pick_index(fields, ["證券代號", "股票代號", "Code"])
    i_name = _pick_index(fields, ["證券名稱", "股票名稱", "Name"])
    i_close = _pick_index(fields, ["收盤價", "收盤", "ClosingPrice", "close"])
    i_vol = _pick_index(fields, ["成交股數", "成交量", "TradeVolume", "volume"])

    if i_code is None or i_close is None:
        return FetchResult(
            pd.DataFrame(),
            "TWSE",
            False,
            f"TWSE MI_INDEX {ymd} missing required fields; fields_sample={fields[:20]}",
        )

    out = []
    for r in rows:
        # rows 通常是 list[str]
        code = str(r[i_code]).strip() if i_code < len(r) else ""
        if not code:
            continue
        name = str(r[i_name]).strip() if (i_name is not None and i_name < len(r)) else ""
        close = _to_float(r[i_close]) if i_close < len(r) else np.nan
        if pd.isna(close):
            continue
        vol = _to_float(r[i_vol]) if (i_vol is not None and i_vol < len(r)) else np.nan
        out.append((code, name, close, vol))

    df = pd.DataFrame(out, columns=["code", "name", "close", "volume"])
    df["market"] = "TWSE"
    df["date"] = pd.Timestamp(day)
    df = df.drop_duplicates(subset=["date", "code"], keep="last").reset_index(drop=True)

    return FetchResult(df, "TWSE", True)


# =========================================
# History I/O
# =========================================
def load_history() -> pd.DataFrame:
    if os.path.exists(HISTORY_CSV):
        try:
            df = pd.read_csv(HISTORY_CSV)
            df["date"] = pd.to_datetime(df["date"])
            return df
        except Exception:
            pass
    return pd.DataFrame(columns=["date", "code", "name", "close", "volume", "market"])


def save_history(df: pd.DataFrame) -> None:
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    df2 = df.copy()
    df2["date"] = pd.to_datetime(df2["date"]).dt.strftime("%Y-%m-%d")
    df2.to_csv(HISTORY_CSV, index=False, encoding="utf-8")


def merge_day(history: pd.DataFrame, day_df: pd.DataFrame) -> pd.DataFrame:
    if day_df is None or day_df.empty:
        return history
    keep = ["date", "code", "name", "close", "volume", "market"]
    df = day_df.copy()
    for c in keep:
        if c not in df.columns:
            return history
    out = pd.concat([history, df[keep]], ignore_index=True)
    out["date"] = pd.to_datetime(out["date"])
    out["code"] = out["code"].astype(str).str.strip()
    out = out.drop_duplicates(subset=["date", "code"], keep="last").reset_index(drop=True)
    return out


def backfill_trading_days(history: pd.DataFrame, end_day: dt.date, n_trading_days: int) -> Tuple[pd.DataFrame, List[str]]:
    """
    從 end_day 往前回補「成功抓到的交易日」達 n_trading_days。
    - 週末跳過
    - 休市/抓不到：略過但記錄
    """
    errors: List[str] = []
    out = history.copy()

    got = 0
    cur = end_day
    max_lookback = n_trading_days * 3  # 緩衝：避免連假/錯誤導致找太久
    tried = 0

    while got < n_trading_days and tried < max_lookback:
        tried += 1
        cur = cur - dt.timedelta(days=1)
        if _is_weekend(cur):
            continue

        r = fetch_twse_day_all(cur)
        if not r.ok:
            errors.append(r.error or f"TWSE {cur} failed")
            continue

        out = merge_day(out, r.df)
        got += 1

    return out, errors


# =========================================
# Indicators / Layering (MVP)
# =========================================
def compute_mas(history: pd.DataFrame) -> pd.DataFrame:
    df = history.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["code", "date"]).reset_index(drop=True)

    for w in MA_WINDOWS:
        df[f"ma{w}"] = df.groupby("code")["close"].transform(lambda s: s.rolling(w).mean())

    return df


def layer_today(history_ma: pd.DataFrame, today: pd.Timestamp) -> Tuple[dict, dict]:
    """
    只用 MA5/10/20 做 MVP 分層（你先要「看到真的資料」）
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

    if day_df.empty:
        return {}, {"summary": "今日無資料（可能是交易所尚未更新或今天非交易日）"}

    day_df = day_df.sort_values(["code"]).drop_duplicates(subset=["code"], keep="last")

    # 以 ma20 作為「是否可分層」的門檻
    warmup = day_df[day_df["ma20"].isna()].copy()
    ready = day_df[day_df["ma20"].notna()].copy()

    def _mk_list(x: pd.DataFrame) -> List[dict]:
        x = x.copy()
        x["name"] = x["name"].fillna("")
        cols = ["code", "name", "close", "ma5", "ma10", "ma20"]
        for c in cols:
            if c not in x.columns:
                x[c] = np.nan
        return x[cols].to_dict(orient="records")

    E = ready[(ready["close"] > ready["ma5"]) & (ready["ma5"] > ready["ma10"]) & (ready["ma10"] > ready["ma20"])].copy()
    B = ready[(ready["close"] > ready["ma5"]) & (ready["ma5"] > ready["ma10"]) & (ready["close"] >= ready["ma20"]) & (~ready["code"].isin(E["code"]))].copy()
    C = ready[(ready["close"] > ready["ma10"]) & (ready["close"] >= ready["ma20"]) & (~ready["code"].isin(E["code"])) & (~ready["code"].isin(B["code"]))].copy()
    D = ready[(ready["close"] > ready["ma20"]) & (~ready["code"].isin(E["code"])) & (~ready["code"].isin(B["code"])) & (~ready["code"].isin(C["code"]))].copy()
    Z = ready[(~ready["code"].isin(E["code"])) & (~ready["code"].isin(B["code"])) & (~ready["code"].isin(C["code"])) & (~ready["code"].isin(D["code"]))].copy()

    layers = {
        "WARMUP": _mk_list(warmup),
        "E": _mk_list(E),
        "B": _mk_list(B),
        "C": _mk_list(C),
        "D": _mk_list(D),
        "Z": _mk_list(Z),
    }

    meta = {"summary": f"分佈：WARMUP {len(warmup)} / E {len(E)} / B {len(B)} / C {len(C)} / D {len(D)} / Z {len(Z)}"}
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
    if errors:
        items = "".join([f"<li>{e}</li>" for e in errors[:12]])
        more = f"<div style='margin-top:6px;color:#666'>（其餘 {max(0,len(errors)-12)} 筆略）</div>" if len(errors) > 12 else ""
        alert = f"<h2>即時警示（摘要｜MVP）</h2><ul>{items}</ul>{more}"
    else:
        alert = f"<h2>即時警示（摘要｜MVP）</h2><ul><li>🟢 目前只跑 TWSE（TPEx 暫停）</li><li>🟢 {meta.get('summary','')}</li></ul>"

    def section(title: str, key: str) -> str:
        rows = layers.get(key, [])
        if not rows:
            return f"<h2>{title}</h2><div>(空)</div>"
        trs = "\n".join(_fmt_row(r) for r in rows[:200])
        tail = f"<div style='margin-top:8px;color:#666'>（僅顯示前 200 筆；實際 {len(rows)} 筆）</div>" if len(rows) > 200 else ""
        return f"""
        <h2>{title}</h2>
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

    return f"""<!doctype html>
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
  {alert}
  {section("WARMUP（資料不足，不算 Z）", "WARMUP")}
  {section("E 層", "E")}
  {section("B 層", "B")}
  {section("C 層", "C")}
  {section("D 層", "D")}
  {section("Z 層", "Z")}
  <div class="note">
    註：本版改用 TWSE MI_INDEX（可抓指定日期）建立「真歷史」，MA5/10/20 才有意義。下一步再加 TPEx 與你的 v7.9.9.x 規則全文。
  </div>
</body>
</html>
"""


# =========================================
# Main
# =========================================
def main() -> int:
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    os.makedirs(DOCS_DIR, exist_ok=True)

    report_date = _now_taipei_date()
    today = pd.Timestamp(report_date)

    errors: List[str] = []

    hist = load_history()

    # 先抓「今天」（若非交易日可能失敗，仍可用回補的最近交易日）
    r_today = fetch_twse_day_all(report_date)
    if r_today.ok:
        hist = merge_day(hist, r_today.df)
    else:
        errors.append(f"TWSE 今日取得失敗：{r_today.error}")

    # 回補：補足交易日數
    hist, backfill_errors = backfill_trading_days(hist, report_date, BACKFILL_TRADING_DAYS)
    if backfill_errors:
:
        # 不要刷版：只摘要
        errors.append(f"TWSE 回補失敗 {len(backfill_errors)} 次（常見原因：休市/連假/交易所暫時擋）")

    # 算 MA
    hist_ma = compute_mas(hist)

    # 分層：如果今天沒資料，會顯示「今日無資料」
    layers, meta = layer_today(hist_ma, today)

    save_history(hist)

    html = render_html(report_date, errors, layers, meta)
    with open(HTML_PATH_OUTPUTS, "w", encoding="utf-8") as f:
        f.write(html)
    with open(HTML_PATH_DOCS, "w", encoding="utf-8") as f:
        f.write(html)

    print("OK: wrote reports to outputs/index.html and docs/index.html")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
