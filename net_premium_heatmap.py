#!/usr/bin/env python3
"""
Net Premium Heat Map
====================
Visualizes directional net premium per strike × expiry using
Unusual Whales' option contract screener endpoint.

Directional logic (same as top_bullish_bearish_flows.py):
  CALL at ask  → bullish  (customers aggressively buying calls)
  CALL at bid  → bearish  (customers selling / writing calls)
  PUT  at ask  → bearish  (customers aggressively buying puts)
  PUT  at bid  → bullish  (customers selling / writing puts)

Net premium = sum(bullish premium − bearish premium) grouped by (strike, expiry).
  Positive (green)  → net buying pressure at that strike
  Negative (red)    → net selling / put-buying pressure at that strike

Uses:
  GET /api/screener/option-contracts   — contract-level premium + ask/bid side
  GET /api/stock/{ticker}/stock-state  — spot price
  GET /api/stock/{ticker}/expiry-breakdown — available expirations

NOTE: The screener only returns contracts with volume ≥ 200, so very quiet
strikes will be absent. This is a flow-based heatmap, not a full-chain view.

Author: William Stewart
Date: February 2026
API: Unusual Whales (https://api.unusualwhales.com/docs)
"""

import os
import sys
import re
import math
import time
import logging
import argparse
import tkinter as tk
from tkinter import ttk, font as tkfont, messagebox
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Tuple

import requests
import pandas as pd
from dotenv import load_dotenv

# ── stdout for Windows ──────────────────────────────────────────────────────
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)
    except Exception:
        pass
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', buffering=1)

# ── Ticker Search ────────────────────────────────────────────────────────────
_TICKER_SEARCH_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'Ticker Search'
)
if _TICKER_SEARCH_DIR not in sys.path:
    sys.path.insert(0, _TICKER_SEARCH_DIR)
from ticker_search import stock_search

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger(__name__)

# ── Load .env ────────────────────────────────────────────────────────────────
_script_dir = os.path.dirname(os.path.abspath(__file__))
for _p in [os.path.join(_script_dir, '.env'),
           os.path.join(os.path.dirname(_script_dir), '.env')]:
    if os.path.isfile(_p):
        load_dotenv(_p)
        break

UW_TOKEN = os.getenv("UNUSUAL_WHALES_API_TOKEN") or os.getenv("UNUSUAL_WHALES_API_KEY")
if not UW_TOKEN:
    print("ERROR: Unusual Whales API key not found.")
    print("  Set UNUSUAL_WHALES_API_TOKEN or UNUSUAL_WHALES_API_KEY in your .env file.")
    sys.exit(1)

# ============================================================================
# CONFIGURATION
# ============================================================================

BASE_URL         = "https://api.unusualwhales.com/api"
REQUEST_TIMEOUT  = 30
RATE_LIMIT_SLEEP = 0.3

DEFAULT_EXPIRIES = 6    # today (or last market open) + next 5 market dates
DEFAULT_STRIKES  = 20   # strikes above/below ATM in each direction

# Weighting: ask-side (aggressive buy) is a stronger signal than bid-side (passive sell)
ASK_WEIGHT = 1.0
BID_WEIGHT = 0.5

# ============================================================================
# MARKET DATE HELPERS
# ============================================================================

def get_last_market_open(dt: Optional[date] = None) -> date:
    """Return today if it's a weekday; otherwise the most recent Friday."""
    d = dt or datetime.now().date()
    while d.weekday() >= 5:
        d -= timedelta(days=1)
    return d


# ============================================================================
# OCC OPTION SYMBOL PARSER
# ============================================================================

_OPT_RE = re.compile(
    r'^(?P<sym>[A-Z.]+)(?P<yy>\d{2})(?P<mm>\d{2})(?P<dd>\d{2})(?P<cp>[PC])(?P<strike>\d{8})$'
)


def parse_option_symbol(symbol: str) -> Optional[Dict]:
    """Parse OCC symbol (e.g. SPY260228C00690000) → {expiry_date, type, strike}."""
    m = _OPT_RE.match((symbol or '').upper().strip())
    if not m:
        return None
    try:
        exp = date(2000 + int(m.group('yy')), int(m.group('mm')), int(m.group('dd')))
    except ValueError:
        return None
    return {
        'expiry_date': exp.isoformat(),
        'type':        'call' if m.group('cp') == 'C' else 'put',
        'strike':      round(int(m.group('strike')) / 1000.0, 2),
    }


# ============================================================================
# API CLIENT
# ============================================================================

class UWClient:
    def __init__(self, api_key: str):
        self.session = requests.Session()
        self.session.headers.update({
            'Authorization': f'Bearer {api_key}',
            'Accept': 'application/json',
        })

    def _get(self, endpoint: str, params: Optional[Dict] = None) -> Dict:
        url = f"{BASE_URL}{endpoint}"
        for attempt in range(3):
            try:
                r = self.session.get(url, params=params, timeout=REQUEST_TIMEOUT)
                if r.status_code == 429:
                    wait = int(r.headers.get('Retry-After', 60))
                    logger.warning("Rate limited; sleeping %ss", wait)
                    time.sleep(wait)
                    continue
                r.raise_for_status()
                time.sleep(RATE_LIMIT_SLEEP)
                return r.json()
            except requests.HTTPError as e:
                status = e.response.status_code if e.response else None
                if status in (401, 403, 404, 422) or attempt == 2:
                    raise
                time.sleep(2 ** attempt)
        raise RuntimeError(f"Max retries for {endpoint}")

    # ── Spot price ────────────────────────────────────────────────────────────

    def get_spot_price(self, ticker: str) -> float:
        data = self._get(f"/stock/{ticker.upper()}/stock-state")
        inner = data.get('data', data)
        price = float(inner.get('close', inner.get('last', 0)))
        if price == 0:
            raise ValueError(f"Could not parse price from stock-state: {inner}")
        return price

    # ── Expirations ──────────────────────────────────────────────────────────

    def get_expirations(
        self,
        ticker: str,
        limit: int = 10,
        as_of_date: Optional[date] = None,
    ) -> List[str]:
        params: Dict = {}
        if as_of_date:
            params['date'] = as_of_date.isoformat()
        data = self._get(f"/stock/{ticker.upper()}/expiry-breakdown",
                         params=params or None)
        rows = data.get('data', data) if isinstance(data, dict) else data
        if not isinstance(rows, list):
            rows = []
        ref = as_of_date or get_last_market_open()
        valid = []
        for row in rows:
            exp_str = str(
                (row.get('expires') or row.get('expiry') or '') if isinstance(row, dict) else row
            )[:10]
            try:
                if date.fromisoformat(exp_str) >= ref:
                    valid.append(exp_str)
            except ValueError:
                pass
        return sorted(set(valid))[:limit]

    # ── Option contracts (screener) ──────────────────────────────────────────

    def get_contracts(
        self,
        ticker: str,
        expiry_dates: List[str],
        min_strike: Optional[float] = None,
        max_strike: Optional[float] = None,
        as_of_date: Optional[date] = None,
    ) -> pd.DataFrame:
        """
        GET /api/screener/option-contracts
        Returns all contracts for ticker at given expiries within the strike range.
        Paginates through up to max 250/page.
        NOTE: Contracts with volume < 200 are not returned by the API.
        """
        params: Dict = {
            'ticker_symbol':   ticker.upper(),
            'expiry_dates[]':  expiry_dates,
            'order':           'volume',
            'order_direction': 'desc',
            'limit':           250,
        }
        if min_strike is not None:
            params['min_strike'] = str(min_strike)
        if max_strike is not None:
            params['max_strike'] = str(max_strike)
        if as_of_date:
            params['date'] = as_of_date.isoformat()

        all_rows: List[Dict] = []
        page = 1
        while True:
            p = dict(params)
            if page > 1:
                p['page'] = page
            data = self._get("/screener/option-contracts", params=p)
            rows = data.get('data', data) if isinstance(data, dict) else data
            if not isinstance(rows, list) or not rows:
                break
            all_rows.extend(rows)
            if len(rows) < 250:
                break
            page += 1
            time.sleep(RATE_LIMIT_SLEEP)

        if not all_rows:
            logger.warning("No contracts returned for %s", ticker)
            return pd.DataFrame()

        records = []
        for row in all_rows:
            sym    = row.get('option_symbol', '')
            parsed = parse_option_symbol(sym) if sym else None
            if parsed:
                strike = parsed['strike']
                expiry = parsed['expiry_date']
                cp     = parsed['type']
            else:
                strike = round(float(row.get('strike', 0) or 0), 2)
                expiry = str(row.get('expiry_date') or row.get('expiry') or '')[:10]
                cp     = (row.get('type') or '').lower()

            if not strike or not expiry or cp not in ('call', 'put'):
                continue

            premium = float(row.get('premium') or 0)
            vol     = float(row.get('volume') or 0)

            # ask_perc / bid_perc: named fields are 0–100 percentages;
            # fall back to ask_side_volume / bid_side_volume ratios.
            ask_p = row.get('ask_perc')
            if ask_p is not None:
                ask_p = float(ask_p) / 100.0
                bid_p = float(row.get('bid_perc') or 0) / 100.0
            else:
                ask_vol = float(row.get('ask_side_volume') or 0)
                bid_vol = float(row.get('bid_side_volume') or 0)
                ask_p   = ask_vol / vol if vol > 0 else 0.0
                bid_p   = bid_vol / vol if vol > 0 else 0.0

            records.append({
                'strike':        strike,
                'expiry_date':   expiry,
                'type':          cp,
                'premium':       premium,
                'ask_perc':      ask_p,
                'bid_perc':      bid_p,
                'volume':        int(vol),
                'open_interest': int(row.get('open_interest') or 0),
            })

        return pd.DataFrame(records)


# ============================================================================
# NET PREMIUM COMPUTATION
# ============================================================================

def compute_net_premium(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute directional net premium per (strike, expiry_date).

    Per contract:
      CALL ask-side → bullish  (premium * ask_perc * ASK_WEIGHT)
      CALL bid-side → bearish  (premium * bid_perc * BID_WEIGHT)
      PUT  ask-side → bearish  (premium * ask_perc * ASK_WEIGHT)
      PUT  bid-side → bullish  (premium * bid_perc * BID_WEIGHT)

    net_premium = bullish − bearish
      Positive → net buying / bullish pressure
      Negative → net selling / bearish pressure
    """
    if df.empty:
        return pd.DataFrame(columns=['strike', 'expiry_date', 'net_premium',
                                     'bullish_premium', 'bearish_premium', 'volume'])

    rows = []
    for _, row in df.iterrows():
        p     = row['premium']
        ask_p = row['ask_perc']
        bid_p = row['bid_perc']

        ask_prem = p * ask_p * ASK_WEIGHT
        bid_prem = p * bid_p * BID_WEIGHT

        if row['type'] == 'call':
            bullish  = ask_prem   # calls bought at ask → bullish
            bearish  = bid_prem   # calls sold at bid   → bearish
        else:
            bullish  = bid_prem   # puts sold at bid    → bullish (put selling)
            bearish  = ask_prem   # puts bought at ask  → bearish (put buying)

        rows.append({
            'strike':           row['strike'],
            'expiry_date':      row['expiry_date'],
            'net_premium':      bullish - bearish,
            'bullish_premium':  bullish,
            'bearish_premium':  bearish,
            'volume':           row['volume'],
        })

    out = pd.DataFrame(rows)
    out = out.groupby(['strike', 'expiry_date'], as_index=False).agg({
        'net_premium':     'sum',
        'bullish_premium': 'sum',
        'bearish_premium': 'sum',
        'volume':          'sum',
    })
    return out


# ============================================================================
# DATA FETCH
# ============================================================================

def fetch_premium_data(
    ticker: str,
    n_expiries: int,
    n_strikes: int,
    as_of_date: Optional[date] = None,
) -> Tuple[float, pd.DataFrame, List[str], date]:
    """
    Returns (spot_price, net_premium_df, expiry_dates, data_date).
    net_premium_df has columns: strike, expiry_date, net_premium,
                                bullish_premium, bearish_premium, volume.
    """
    client    = UWClient(UW_TOKEN)
    data_date = as_of_date or get_last_market_open()

    logger.info("Fetching spot price for %s...", ticker)
    spot = client.get_spot_price(ticker)
    logger.info("  Underlying: $%.2f", spot)

    logger.info("Fetching expirations (as of %s)...", data_date.isoformat())
    expiry_dates = client.get_expirations(ticker, limit=n_expiries, as_of_date=data_date)
    if not expiry_dates:
        raise ValueError(f"No valid expirations found for {ticker}")
    expiry_dates = expiry_dates[:n_expiries]
    logger.info("  Expiries: %s", ', '.join(expiry_dates))

    step       = 1.0 if spot < 100 else (5.0 if spot < 500 else 10.0)
    min_strike = spot - (n_strikes + 3) * step
    max_strike = spot + (n_strikes + 3) * step

    logger.info("Fetching contracts (%.0f – %.0f)...", min_strike, max_strike)
    contracts = client.get_contracts(
        ticker, expiry_dates,
        min_strike=min_strike, max_strike=max_strike,
        as_of_date=data_date,
    )

    if contracts.empty:
        raise ValueError(
            "No contract data returned.\n"
            "The market may be closed or this ticker has no options flow today.\n"
            "NOTE: Contracts with volume < 200 are excluded by the screener API."
        )

    logger.info("  Got %d contract rows across all expiries", len(contracts))

    premium_df = compute_net_premium(contracts)
    premium_df['strike'] = premium_df['strike'].round(2)
    premium_df = premium_df.groupby(['strike', 'expiry_date'], as_index=False).agg({
        'net_premium':     'sum',
        'bullish_premium': 'sum',
        'bearish_premium': 'sum',
        'volume':          'sum',
    })

    # Keep only n_strikes above and below ATM
    all_strikes = sorted(premium_df['strike'].unique())
    if all_strikes:
        atm = min(all_strikes, key=lambda s: abs(s - spot))
        above = sorted([s for s in all_strikes if s > atm])[:n_strikes]
        below = sorted([s for s in all_strikes if s < atm], reverse=True)[:n_strikes]
        keep  = set(above + below + [atm])
        premium_df = premium_df[premium_df['strike'].isin(keep)].copy()

    return (spot, premium_df, expiry_dates, data_date)


# ============================================================================
# FORMATTING HELPERS
# ============================================================================

def _fmt_prem(value: float) -> str:
    abs_val = abs(value)
    if abs_val >= 1e9:  return f"{value / 1e9:+.2f} B"
    if abs_val >= 1e6:  return f"{value / 1e6:+.2f} M"
    if abs_val >= 1e3:  return f"{value / 1e3:+.2f} K"
    if abs_val > 0:     return f"{value:+.0f}"
    return "—"


def _cell_bg(value: float, max_abs: float) -> str:
    """Green = net bullish premium; Red = net bearish premium (dark theme)."""
    if max_abs <= 0:
        return "#2d2d2d"
    intensity = min(abs(value) / max_abs, 1.0)
    if value > 0:
        r = int(26  + (13  - 26)  * intensity)
        g = int(46  + (107 - 46)  * intensity)
        b = int(26  + (13  - 26)  * intensity)
        return f"#{r:02x}{g:02x}{b:02x}"
    elif value < 0:
        r = int(46  + (165 - 46)  * intensity)
        g = int(26  + (42  - 26)  * intensity)
        b = int(26  + (42  - 26)  * intensity)
        return f"#{r:02x}{g:02x}{b:02x}"
    return "#2d2d2d"


def _strike_label(strike: float) -> str:
    return f"${int(round(strike))}" if abs(strike - round(strike)) < 1e-6 else f"${strike:.2f}"


# ============================================================================
# GUI
# ============================================================================

class NetPremiumHeatMapGUI:
    """Desktop GUI — ticker button, metrics bar, scrollable color grid."""

    BG_DARK   = "#1e1e1e"
    FG        = "#e0e0e0"
    HEADER_BG = "#252526"
    TICKER_BG = "#0e639c"
    TICKER_FG = "#ffffff"

    def __init__(
        self,
        n_expiries: int = DEFAULT_EXPIRIES,
        n_strikes: int  = DEFAULT_STRIKES,
        initial_ticker: Optional[str] = None,
    ):
        self.n_expiries   = n_expiries
        self.n_strikes    = n_strikes
        self.ticker       = (initial_ticker or "SPY").strip().upper()
        self.spot_price   = 0.0
        self.premium_df   = pd.DataFrame()
        self.expiry_dates: List[str] = []
        self.as_of_date: Optional[date] = None

        self.root = tk.Tk()
        self.root.title("Net Premium Heat Map")
        self.root.configure(bg=self.BG_DARK)
        self.root.minsize(900, 600)
        self.root.geometry("1150x720")

        self._build_ui()
        self._load_data()

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self):
        # Top bar
        top = tk.Frame(self.root, bg=self.HEADER_BG, height=48)
        top.pack(fill=tk.X, side=tk.TOP)
        top.pack_propagate(False)

        self.ticker_btn = tk.Button(
            top, text=self.ticker, font=("Segoe UI", 11, "bold"),
            bg=self.TICKER_BG, fg=self.TICKER_FG, activebackground="#1177bb",
            relief=tk.FLAT, padx=16, pady=6, cursor="hand2",
            command=self._on_change_ticker,
        )
        self.ticker_btn.pack(side=tk.LEFT, padx=12, pady=8)

        self._date_label = tk.Label(
            top, text="—", fg=self.FG, bg=self.HEADER_BG, font=("Segoe UI", 10),
        )
        self._date_label.pack(side=tk.RIGHT, padx=16, pady=12)

        tk.Button(
            top, text="Net Premium", fg=self.FG, bg=self.HEADER_BG,
            relief=tk.FLAT, font=("Segoe UI", 10), state=tk.DISABLED,
        ).pack(side=tk.RIGHT, padx=4, pady=8)

        tk.Button(
            top, text="↻ Refresh", fg=self.FG, bg=self.HEADER_BG,
            relief=tk.FLAT, font=("Segoe UI", 10), cursor="hand2",
            command=self._load_data,
        ).pack(side=tk.RIGHT, padx=4, pady=8)

        # Content area
        content = tk.Frame(self.root, bg=self.BG_DARK, padx=16, pady=12)
        content.pack(fill=tk.BOTH, expand=True)

        self._title_label = tk.Label(
            content, text=f"Net Premium Heat Map - {self.ticker}",
            fg=self.FG, bg=self.BG_DARK, font=("Segoe UI", 16, "bold"),
        )
        self._title_label.pack(anchor=tk.W)

        # Metrics row
        metrics = tk.Frame(content, bg=self.BG_DARK)
        metrics.pack(anchor=tk.W, pady=(8, 12))

        self._net_label = tk.Label(
            metrics, text="Total Net Premium (—)",
            fg="#c586c0", bg=self.BG_DARK, font=("Segoe UI", 10),
        )
        self._net_label.pack(side=tk.LEFT)

        self._underlying_label = tk.Label(
            metrics, text="Underlying (—)",
            fg="#569cd6", bg=self.BG_DARK, font=("Segoe UI", 10),
        )
        self._underlying_label.pack(side=tk.LEFT, padx=(20, 0))

        self._note_label = tk.Label(
            metrics,
            text="Note: only contracts with volume ≥ 200 shown",
            fg="#6e6e6e", bg=self.BG_DARK, font=("Segoe UI", 9, "italic"),
        )
        self._note_label.pack(side=tk.LEFT, padx=(20, 0))

        # Scrollable grid
        grid_frame = tk.Frame(content, bg=self.BG_DARK)
        grid_frame.pack(fill=tk.BOTH, expand=True)

        canvas = tk.Canvas(grid_frame, bg=self.BG_DARK, highlightthickness=0)
        vbar   = ttk.Scrollbar(grid_frame, orient=tk.VERTICAL,   command=canvas.yview)
        hbar   = ttk.Scrollbar(grid_frame, orient=tk.HORIZONTAL, command=canvas.xview)

        self._table_frame = tk.Frame(canvas, bg=self.BG_DARK)
        self._table_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all")),
        )
        canvas.create_window((0, 0), window=self._table_frame, anchor=tk.NW)
        canvas.configure(yscrollcommand=vbar.set, xscrollcommand=hbar.set)

        vbar.pack(side=tk.RIGHT, fill=tk.Y)
        hbar.pack(side=tk.BOTTOM, fill=tk.X)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.root.bind("<MouseWheel>",
            lambda e: canvas.yview_scroll(int(-1 * e.delta / 120), "units"))
        self.root.bind("<Shift-MouseWheel>",
            lambda e: canvas.xview_scroll(int(-1 * e.delta / 120), "units"))

    # ── Events ────────────────────────────────────────────────────────────────

    def _on_change_ticker(self):
        new = stock_search()
        if new and str(new).strip():
            self.ticker = str(new).strip().upper()
            self.ticker_btn.configure(text=self.ticker)
            self._load_data()

    def _load_data(self):
        self._title_label.configure(text=f"Net Premium Heat Map - {self.ticker}")
        self.root.config(cursor="wait")
        self.root.update()
        try:
            spot, premium_df, expiry_dates, data_date = fetch_premium_data(
                self.ticker, self.n_expiries, self.n_strikes,
            )
            self.spot_price   = spot
            self.premium_df   = premium_df
            self.expiry_dates = expiry_dates
            self.as_of_date   = data_date

            self._date_label.configure(text=data_date.strftime("%b %d, %Y"))
            self._underlying_label.configure(text=f"Underlying (${spot:.2f})")
            self._update_metrics()
            self._render_grid()
        except Exception as e:
            messagebox.showerror("Net Premium Error", str(e))
            logger.exception("GUI load failed")
        finally:
            self.root.config(cursor="")

    # ── Metrics ───────────────────────────────────────────────────────────────

    def _update_metrics(self):
        if self.premium_df.empty:
            self._net_label.configure(text="Total Net Premium (—)", fg="#c586c0")
            return
        total = self.premium_df['net_premium'].sum()
        color = "#6ab04c" if total >= 0 else "#e55039"
        self._net_label.configure(
            text=f"Total Net Premium ({_fmt_prem(total)})", fg=color,
        )

    # ── Grid ─────────────────────────────────────────────────────────────────

    def _render_grid(self):
        for w in self._table_frame.winfo_children():
            w.destroy()

        if self.premium_df.empty:
            tk.Label(
                self._table_frame, text="No data",
                fg=self.FG, bg=self.BG_DARK, font=("Segoe UI", 12),
            ).pack(pady=20)
            return

        pivot = self.premium_df.pivot_table(
            index='strike', columns='expiry_date', values='net_premium',
            fill_value=0, aggfunc='sum',
        )
        valid_expiries = [e for e in self.expiry_dates if e in pivot.columns]
        if not valid_expiries:
            valid_expiries = sorted(pivot.columns.tolist())
        pivot = pivot[valid_expiries].sort_index(ascending=False)
        if pivot.index.duplicated().any():
            pivot = pivot.groupby(level=0).sum()

        max_abs = pivot.abs().values.max() if pivot.size else 1.0

        # Find the cell with the largest absolute value per column → one purple per expiry
        purple_cells: set = set()
        for exp in valid_expiries:
            col = pivot[exp]
            if col.abs().max() > 0:
                top_strike = col.abs().idxmax()
                purple_cells.add((top_strike, exp))

        cell_font   = tkfont.Font(family="Segoe UI", size=9)
        header_font = tkfont.Font(family="Segoe UI", size=9, weight="bold")

        # ── Header row ──────────────────────────────────────────────────────
        tk.Label(
            self._table_frame, text="Strike Price", fg=self.FG, bg=self.HEADER_BG,
            font=header_font, width=12, anchor=tk.W, padx=6, pady=4,
        ).grid(row=0, column=0, sticky=tk.NSEW, padx=1, pady=1)

        for c, exp in enumerate(valid_expiries, start=1):
            try:
                label = datetime.strptime(exp, '%Y-%m-%d').strftime("%b %d, %Y")
            except Exception:
                label = exp
            tk.Label(
                self._table_frame, text=label, fg=self.FG, bg=self.HEADER_BG,
                font=header_font, width=14, anchor=tk.CENTER, padx=6, pady=4,
            ).grid(row=0, column=c, sticky=tk.NSEW, padx=1, pady=1)

        # ── Data rows ───────────────────────────────────────────────────────
        atm = min(pivot.index, key=lambda s: abs(s - self.spot_price))

        for r, strike in enumerate(pivot.index, start=1):
            is_atm = strike == atm
            lbl    = _strike_label(strike)
            tk.Label(
                self._table_frame, text=lbl,
                fg="#569cd6" if is_atm else self.FG,
                bg=self.HEADER_BG if is_atm else self.BG_DARK,
                font=header_font if is_atm else cell_font,
                width=12, anchor=tk.E, padx=6, pady=3,
            ).grid(row=r, column=0, sticky=tk.NSEW, padx=1, pady=1)

            for c, exp in enumerate(valid_expiries, start=1):
                val = float(pivot.loc[strike, exp]) if exp in pivot.columns else 0.0
                txt = _fmt_prem(val)
                is_purple = (strike, exp) in purple_cells
                bg  = "#6a0dad" if is_purple else _cell_bg(val, max_abs)
                fg  = "#ffffff"
                tk.Label(
                    self._table_frame, text=txt, fg=fg, bg=bg,
                    font=cell_font, width=14, anchor=tk.CENTER, padx=6, pady=3,
                ).grid(row=r, column=c, sticky=tk.NSEW, padx=1, pady=1)

        for i in range(len(valid_expiries) + 1):
            self._table_frame.columnconfigure(i, weight=0 if i == 0 else 1)
        for i in range(len(pivot.index) + 1):
            self._table_frame.rowconfigure(i, weight=0)

    def run(self):
        self.root.mainloop()


# ============================================================================
# CLI ENTRY POINT
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Net Premium Heat Map — Directional flow by strike × expiry',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
By default opens the desktop GUI (ticker search then heatmap).

  --ticker SYMBOL   Ticker to analyze (e.g. SPY)
  --expiries N      Number of expirations — today + next N-1 (default: 6)
  --strikes N       Strikes above/below ATM (default: 20)
  --no-gui          Print summary to terminal instead of opening GUI
  --debug           Enable debug logging
        """
    )
    parser.add_argument('--ticker',   type=str, default=None)
    parser.add_argument('--expiries', type=int, default=DEFAULT_EXPIRIES)
    parser.add_argument('--strikes',  type=int, default=DEFAULT_STRIKES)
    parser.add_argument('--no-gui',   action='store_true')
    parser.add_argument('--debug',    action='store_true')
    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    if args.ticker:
        ticker = args.ticker.strip().upper()
    else:
        ticker = stock_search()
        if not ticker:
            print("No ticker selected. Exiting.")
            sys.exit(0)
        ticker = ticker.strip().upper()

    if not args.no_gui:
        app = NetPremiumHeatMapGUI(
            n_expiries=args.expiries,
            n_strikes=args.strikes,
            initial_ticker=ticker,
        )
        app.run()
        return

    # ── Terminal output ───────────────────────────────────────────────────────
    print(f"Net Premium Heat Map — {ticker}")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    try:
        spot, premium_df, expiry_dates, data_date = fetch_premium_data(
            ticker, args.expiries, args.strikes,
        )
        pivot = premium_df.pivot_table(
            index='strike', columns='expiry_date', values='net_premium',
            fill_value=0, aggfunc='sum',
        ).sort_index(ascending=False)
        valid = [e for e in expiry_dates if e in pivot.columns]
        pivot = pivot[valid]

        print(f"Underlying: ${spot:.2f}  |  Data date: {data_date.isoformat()}")
        print(f"Expiries  : {', '.join(valid)}")
        print(f"Total net : {_fmt_prem(premium_df['net_premium'].sum())}")
        print()
        print(pivot.to_string(
            float_format=lambda x: f"{x/1e6:+.2f}M",
        ))
    except Exception as e:
        logger.error("Fatal: %s", e, exc_info=args.debug)
        sys.exit(1)


if __name__ == '__main__':
    main()
