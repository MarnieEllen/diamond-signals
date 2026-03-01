#!/usr/bin/env python3
"""
DIAMOND Daily Signal Generator  (Claude-driven version)
=========================================================
Usage:
  python3 generate_signal.py \
      --spx-price    6900.39 \
      --spx-prev     6946.13 \
      --spx-52w-high 7002.28 \
      --spx-high     6910.00 \
      --spx-low      6880.00 \
      --vix-level    15.06 \
      --vix-prev     14.90

Claude fetches these values each morning via WebSearch, then calls this script.
The script maintains a rolling local history (market_history.json) for EMA and
trend analysis, and generates a fully self-contained index.html.

Updated with intelligence from TEN PORTAL daily updates and Dec 2025 trade log.
"""

import argparse, json, os
from datetime import datetime, date
import numpy as np
import pytz

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
HISTORY_FILE = os.path.join(SCRIPT_DIR, 'market_history.json')
OUTPUT_PATH  = os.path.join(SCRIPT_DIR, 'index.html')

# ── Site password (change this to update the access password) ───────────────
SITE_PASSWORD = os.environ.get('SITE_PASSWORD', 'Diamonds')

# ════════════════════════════════════════════════════════════════════════════
# HISTORY  (rolling local datastore, updated each run)
# ════════════════════════════════════════════════════════════════════════════

def load_history():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE) as f:
            return json.load(f)
    return {"records": []}

def save_history(hist):
    with open(HISTORY_FILE, 'w') as f:
        json.dump(hist, f, indent=2)

def update_history(hist, spx_price, spx_high, spx_low, vix_level, today_str):
    existing = {r['date'] for r in hist['records']}
    if today_str not in existing:
        hist['records'].append({
            'date':  today_str,
            'spx':   round(spx_price, 2),
            'spx_h': round(spx_high, 2),
            'spx_l': round(spx_low, 2),
            'vix':   round(vix_level, 2),
        })
    hist['records'] = sorted(hist['records'], key=lambda r: r['date'])[-252:]
    return hist

# ════════════════════════════════════════════════════════════════════════════
# EMA CALCULATION
# ════════════════════════════════════════════════════════════════════════════

def compute_ema(arr, period):
    """Compute EMA for a given period. Returns None if insufficient data."""
    if len(arr) < period:
        return None
    ema = float(np.mean(arr[:period]))  # seed with SMA
    k = 2.0 / (period + 1)
    for p in arr[period:]:
        ema = float(p) * k + ema * (1 - k)
    return round(ema, 2)

def compute_emas(records):
    """Compute EMA 9, 21, and 50 from history records.
    Returns dict — any value may be None if insufficient history.
    Accuracy improves as local history accumulates (needs 50+ days for full stack).
    """
    spx_arr = [r['spx'] for r in records]
    return {
        'ema9':  compute_ema(spx_arr, 9),
        'ema21': compute_ema(spx_arr, 21),
        'ema50': compute_ema(spx_arr, 50),
    }

# ════════════════════════════════════════════════════════════════════════════
# KEY LEVELS
# ════════════════════════════════════════════════════════════════════════════

def compute_key_levels(records, spx_now, spx_52w_high, emas):
    """Compute key support/resistance levels to display on the site."""
    n = len(records)
    levels = {}

    # EMAs — the 9/21/50 stack is the primary framework
    if emas.get('ema9'):
        levels['EMA 9'] = (emas['ema9'], 'short-term momentum')
    if emas.get('ema21'):
        levels['EMA 21'] = (emas['ema21'], 'directional anchor')
    if emas.get('ema50'):
        levels['EMA 50'] = (emas['ema50'], 'trend baseline')

    # 10-day price range
    if n >= 10:
        hi_10 = round(max(r['spx_h'] for r in records[-10:]), 0)
        lo_10 = round(min(r['spx_l'] for r in records[-10:]), 0)
        levels['10d High'] = (hi_10, 'recent resistance')
        levels['10d Low']  = (lo_10, 'recent support')

    # 52-week high
    levels['52w High'] = (round(spx_52w_high, 0), 'all-time high zone')

    # Nearest round-number levels (every 50 pts)
    base = round(spx_now / 50) * 50
    for offset in [-100, -50, 0, 50, 100]:
        lvl = base + offset
        if abs(lvl - spx_now) > 20 and abs(lvl - spx_now) <= 150:
            direction = 'support' if lvl < spx_now else 'resistance'
            levels[f'{int(lvl)}'] = (float(lvl), f'round {direction}')

    return levels

# ════════════════════════════════════════════════════════════════════════════
# ENVIRONMENT CLASSIFICATION
# ════════════════════════════════════════════════════════════════════════════

def classify_environment(records, spx_now, vix_now, emas):
    n = len(records)
    spx_arr = np.array([r['spx'] for r in records])
    vix_arr = np.array([r['vix'] for r in records])

    # ── VIX stress detection ─────────────────────────────────────────────
    vix_10d    = float(np.mean(vix_arr[-10:])) if n >= 10 else float(np.mean(vix_arr))
    vix_5d     = float(np.mean(vix_arr[-5:]))  if n >= 5  else float(np.mean(vix_arr))
    vix_stress = vix_now > 25
    vix_spike  = vix_now > vix_10d * 1.30
    is_stress  = vix_stress or vix_spike

    # ── Post-event detection ─────────────────────────────────────────────
    vix_20d_max   = float(np.max(vix_arr[-20:])) if n >= 20 else float(np.max(vix_arr))
    was_elevated  = vix_20d_max > 22
    vix_declining = vix_now < vix_5d and vix_now < vix_10d * 0.95
    is_post_event = was_elevated and vix_declining and vix_now < 22 and not is_stress

    # ── EMA stack analysis (primary directional signal) ──────────────────
    e9  = emas.get('ema9')
    e21 = emas.get('ema21')
    e50 = emas.get('ema50')

    ema_uptrend   = False
    ema_downtrend = False
    ema_bunching  = False

    if e9 and e21 and e50:
        ema_uptrend   = (e9 > e21 > e50) and (spx_now > e9)
        ema_downtrend = (e9 < e21 < e50) and (spx_now < e9)
        ema_spread    = (max(e9, e21, e50) - min(e9, e21, e50)) / spx_now * 100
        ema_bunching  = ema_spread < 0.75
    elif e9 and e21:
        ema_uptrend   = (e9 > e21) and (spx_now > e9)
        ema_downtrend = (e9 < e21) and (spx_now < e9)

    # ── Simple trend fallback (when EMA history is limited) ──────────────
    simple_uptrend = simple_downtrend = False
    if n >= 10:
        s5, s10 = float(spx_arr[-5]), float(spx_arr[-10])
        simple_uptrend   = spx_now > s5 > s10
        simple_downtrend = spx_now < s5 < s10

    use_ema   = bool(e9 and e21)
    uptrend   = ema_uptrend   if use_ema else simple_uptrend
    downtrend = ema_downtrend if use_ema else simple_downtrend

    # ── Compression detection ────────────────────────────────────────────
    spx_10_slice  = spx_arr[-10:] if n >= 10 else spx_arr
    spx_10d_high  = float(np.max(spx_10_slice))
    spx_10d_low   = float(np.min(spx_10_slice))
    spx_10d_range = (spx_10d_high - spx_10d_low) / spx_now * 100
    vix_flat      = vix_now <= vix_10d * 1.05
    is_compression = (spx_10d_range < 3.5 and vix_flat) or ema_bunching

    # ── EMA description fragment ─────────────────────────────────────────
    ema_desc = ""
    if e9 and e21 and e50:
        ema_desc = f" EMA stack: 9={e9:,.0f} / 21={e21:,.0f} / 50={e50:,.0f}."
    elif e9 and e21:
        ema_desc = f" EMA 9={e9:,.0f} / 21={e21:,.0f} (50d EMA still building)."
    else:
        ema_desc = " (EMA stack building — more history needed for full signal.)"

    # ── Classification ───────────────────────────────────────────────────
    if is_stress:
        return ("Stress / Expansion", "stress",
                f"VIX at {vix_now:.2f} — elevated stress regime. "
                "Survival mode: no new income trades. Simplify and reduce urgency.")

    elif is_post_event:
        return ("Post-Event Decay", "post_event",
                f"VIX rolling over from recent high of {vix_20d_max:.2f}. "
                f"Currently {vix_now:.2f} and declining. DIAMONDS' home environment.{ema_desc}")

    elif is_compression:
        detail = (f"SPX in a {spx_10d_range:.1f}% range over 10 sessions "
                  f"({spx_10d_low:,.0f}–{spx_10d_high:,.0f}). VIX at {vix_now:.2f}.")
        if ema_bunching:
            detail += " EMAs converging — textbook compression setup."
        else:
            detail += " Patience is the edge."
        detail += ema_desc
        return ("Compression / Range", "compression", detail)

    elif uptrend:
        if ema_uptrend and e9 and e21:
            detail = f"EMA stack in bullish alignment.{ema_desc} "
        else:
            detail = "SPX making higher highs and higher lows. "
        detail += f"VIX stable at {vix_now:.2f}. Watch for delta creep to the upside."
        return ("Directional Drift — Up", "drift_up", detail)

    elif downtrend:
        if ema_downtrend and e9 and e21:
            detail = f"EMA stack inverted — bearish alignment.{ema_desc} "
        else:
            detail = "SPX making lower lows. "
        detail += f"VIX at {vix_now:.2f}. Wait for sellers to lose momentum before adding PCS."
        return ("Directional Drift — Down", "drift_down", detail)

    else:
        return ("Compression / Range", "compression",
                f"No clear directional conviction. EMAs mixed or history still building.{ema_desc} "
                "If you cannot name the environment, do not trade.")

# ════════════════════════════════════════════════════════════════════════════
# GI TRIGGERS  (any one is sufficient — GI goes on BEFORE fear arrives)
# ════════════════════════════════════════════════════════════════════════════

def gi_triggers(records, spx_now, spx_52w_high, vix_now):
    triggers = []
    n = len(records)
    spx_arr = np.array([r['spx'] for r in records])
    vix_arr = np.array([r['vix'] for r in records])

    # T1: Near All-Time Highs (within 2%)
    pct_from_ath = (spx_now / spx_52w_high - 1) * 100
    if pct_from_ath >= -2.0:
        triggers.append(("Near All-Time Highs",
                         f"SPX is {pct_from_ath:+.1f}% from 52-week high ({spx_52w_high:,.0f}). "
                         "GI structure: 150pt wide put spread (e.g., buy 6800/sell 6650 put), "
                         "4–5 months out, target cost ~$29–$30/spread."))

    # T2: Price Extension (ATR-based)
    if n >= 15:
        hi_arr = np.array([r['spx_h'] for r in records])
        lo_arr = np.array([r['spx_l'] for r in records])
        pc_arr = np.roll(spx_arr, 1); pc_arr[0] = spx_arr[0]
        tr = np.maximum(hi_arr - lo_arr,
             np.maximum(abs(hi_arr - pc_arr), abs(lo_arr - pc_arr)))
        atr14    = float(np.mean(tr[-14:]))
        mean20   = float(np.mean(spx_arr[-20:])) if n >= 20 else float(np.mean(spx_arr))
        ext      = abs(spx_now - mean20)
        atr_mult = ext / atr14 if atr14 > 0 else 0

        if atr_mult >= 3.0:
            triggers.append(("Price Extension ≥ 3 ATR",
                             f"SPX is {ext:.0f} pts ({atr_mult:.1f}× ATR) from 20d mean. "
                             f"Daily ATR ≈ {atr14:.0f}. Market is extended — GI before the gap, not after."))
        elif atr_mult >= 2.0:
            triggers.append(("Price Extension ≥ 2 ATR",
                             f"SPX is {ext:.0f} pts ({atr_mult:.1f}× ATR) from 20d mean. "
                             f"Daily ATR ≈ {atr14:.0f}. Approaching 3 ATR threshold — consider GI now."))

    # T3: Volatility Compression (calm rally)
    if n >= 10:
        spx_10d_ago = float(spx_arr[-10])
        vix_10d     = float(np.mean(vix_arr[-10:]))
        spx_up_pct  = (spx_now / spx_10d_ago - 1) * 100
        if spx_up_pct > 0.5 and vix_now <= vix_10d * 0.98:
            triggers.append(("Volatility Compression",
                             f"SPX up {spx_up_pct:+.1f}% over 10 days while VIX ({vix_now:.2f}) "
                             f"falling below 10d mean ({vix_10d:.2f}). "
                             "Calm rallies can end abruptly — GI is cheapest right now."))

    # T4: Extended Compression / Inventory Risk
    if n >= 20:
        spx_20    = spx_arr[-20:]
        range_20d = (float(np.max(spx_20)) - float(np.min(spx_20))) / spx_now * 100
        if range_20d < 3.0:
            triggers.append(("Extended Compression / Inventory Risk",
                             f"SPX has moved only {range_20d:.1f}% over 20 sessions. "
                             "Long-running compression = large open inventory. "
                             "GI manages the tail risk that builds during quiet periods."))

    return triggers

# ════════════════════════════════════════════════════════════════════════════
# PERMISSIONS  (environment × component permission matrix)
# ════════════════════════════════════════════════════════════════════════════

def permissions(env_key):
    matrix = {
        "stress": {
            "money_press": ("no",   "Forbidden — no new short puts in a stress regime"),
            "ccs":         ("no",   "Suspended — do not add directional risk"),
            "pcs":         ("no",   "Suspended — wait for VIX to stabilize before adding"),
            "ic":          ("no",   "Suspended — no new income structures in stress"),
            "gi":          ("req",  "Mandatory — if not already on, add immediately"),
            "primary":     "Simplify. Reduce urgency. Survive the stress regime.",
            "note":        "Stress is about survival, not optimization. Every action should reduce exposure, not add to it.",
        },
        "post_event": {
            "money_press": ("yes",  "Preferred — rebuild inventory here. Short put 50–100 pts above visible support/EMA."),
            "ccs":         ("cond", "Selective — one at a time, only if delta needs balancing toward neutral"),
            "pcs":         ("cond", "Selective — VIX must be rolling over, not still elevated or spiking"),
            "ic":          ("no",   "Not yet — wait for full compression to establish (20+ days of range)"),
            "gi":          ("opt",  "Assess inventory size. Add if rebuilding positions quickly."),
            "primary":     "Add / normalize Money Press. Rebuild without urgency.",
            "note":        "Post-Event Decay is DIAMONDS' home environment. Inventory rebuilds naturally here — let it.",
        },
        "compression": {
            "money_press": ("yes",  "Allowed — structure only. Roll on Tuesday at 3–5 DTE. Move short put UP 50 pts if SPX moved +50 pts."),
            "ccs":         ("cond", "Rarely — only if delta is clearly unbalanced. Strike must be outside 1-day ATR from current price."),
            "pcs":         ("cond", "Rarely — only if delta is clearly unbalanced. Strike must be outside 1-day ATR from current price."),
            "ic":          ("cond", "Rare — mature compression only (20+ sessions). Min credit: $2.25–$2.50. Exit at 50–60% profit (GTC on fill). Width: 15 pts each side."),
            "gi":          ("opt",  "Preferred if near ATH or VIX compressed. Structure: 150pt wide put spread, 4–5 months out, ~$29–$30/spread."),
            "primary":     "Let shorts decay. Roll MPs on Tuesday at 3–5 DTE. Do not over-manage.",
            "note":        "Compression punishes activity. The most correct action is often nothing. No trade is a trade.",
        },
        "drift_up": {
            "money_press": ("yes",  "Allowed — normalize positive delta. Roll short put up to follow market. Tuesday roll preferred."),
            "ccs":         ("yes",  "Allowed — prevents delta creep upward. Strike must be outside 1-day ATR from current price."),
            "pcs":         ("no",   "Not in upward drift — only adds downside risk while market moves away"),
            "ic":          ("no",   "Not in drift — one-directional move kills the IC structure"),
            "gi":          ("opt",  "Increasingly important as drift extends. Add after 3+ ATR extension."),
            "primary":     "Watch delta daily. One CCS only if delta creeping positive. Keep it mechanical.",
            "note":        "Drift kills through delta creep. Trim early and mechanically — don't wait for conviction.",
        },
        "drift_down": {
            "money_press": ("yes",  "Allowed — normalize negative delta. Keep short put well below current price."),
            "ccs":         ("no",   "Not in downward drift — adds upside risk while market moves away"),
            "pcs":         ("cond", "After fear starts to fade — VIX must be stalling or clearly rolling over. Wait for the turn first."),
            "ic":          ("no",   "Not in drift — one-directional move kills the IC structure"),
            "gi":          ("opt",  "Increasingly important as drift extends. Protects existing inventory."),
            "primary":     "Watch delta. PCS only after sellers clearly lose momentum (VIX stalling/rolling over).",
            "note":        "PCSs pay for patience, not bravery. Wait for the turn to show first — then act.",
        },
    }
    return matrix.get(env_key, matrix["compression"])

# ════════════════════════════════════════════════════════════════════════════
# HTML GENERATION
# ════════════════════════════════════════════════════════════════════════════

STOP_QUESTIONS = [
    "What environment is today?",
    "What problem does this trade solve?",
    "What does it do to delta?",
    "Does it increase urgency later?",
]

ENV_COLORS = {
    "stress":      "#ef4444",
    "post_event":  "#10b981",
    "compression": "#60a5fa",
    "drift_up":    "#a78bfa",
    "drift_down":  "#f59e0b",
}

def badge(status, note):
    colors = {
        "yes":  ("#10b981", "✓ Allowed"),
        "no":   ("#ef4444", "✗ Forbidden"),
        "cond": ("#f59e0b", "⚡ Conditional"),
        "req":  ("#c9a84c", "⚠ Required"),
        "opt":  ("#60a5fa", "◎ Optional"),
    }
    color, label = colors.get(status, ("#6b7280", "—"))
    return (f'<span class="badge" style="color:{color};border-color:{color}40">{label}</span>'
            f'<span class="perm-note">{note}</span>')


def generate_html(spx_price, spx_prev, spx_52w, vix_level, vix_prev,
                  records, emas, key_levels,
                  env_name, env_key, env_detail, triggers, perms):

    et     = pytz.timezone('US/Eastern')
    now_et = datetime.now(et)

    spx_chg     = spx_price - spx_prev
    spx_pct     = spx_chg / spx_prev * 100
    vix_chg     = vix_level - vix_prev
    vix_pct     = vix_chg / vix_prev * 100
    spx_ath_pct = (spx_price / spx_52w - 1) * 100
    env_color   = ENV_COLORS.get(env_key, "#60a5fa")

    def cc(v): return "pos" if v >= 0 else "neg"
    def fc(v, p): return f"{'+'if v>=0 else''}{v:.2f} ({'+'if p>=0 else''}{p:.2f}%)"

    # ── Permission rows ──────────────────────────────────────────────────
    perm_rows = ""
    for label, key in [
        ("Money Press (MP)", "money_press"),
        ("Call Credit Spread (CCS)", "ccs"),
        ("Put Credit Spread (PCS)", "pcs"),
        ("Iron Condor (IC)", "ic"),
        ("Gap Insurance (GI)", "gi"),
    ]:
        s, n_txt = perms[key]
        perm_rows += (f'<div class="perm-row">'
                      f'<span class="perm-label">{label}</span>'
                      f'{badge(s, n_txt)}</div>\n')

    # ── GI section ───────────────────────────────────────────────────────
    if triggers:
        tcards = "".join(
            f'<div class="trigger-card">'
            f'<div class="trigger-name">⚡ {t[0]}</div>'
            f'<div class="trigger-detail">{t[1]}</div></div>'
            for t in triggers
        )
        gi_html = (
            f'<section class="section">'
            f'<h2 class="section-title">Gap Insurance Triggers — Active</h2>'
            f'<div class="trigger-grid">{tcards}</div>'
            f'<p class="gi-note">One trigger is sufficient. '
            f'Structure: 150pt wide put spread (e.g. buy 6800/sell 6650 put), '
            f'4–5 months to expiry, ~$29–$30/spread. '
            f'Place GI before fear arrives — it is expensive after.</p>'
            f'</section>'
        )
    else:
        gi_html = (
            '<section class="section">'
            '<h2 class="section-title">Gap Insurance Triggers</h2>'
            '<div class="no-triggers">No GI triggers active — continue monitoring. '
            'GI goes on before you feel you need it.</div>'
            '</section>'
        )

    # ── VIX label ────────────────────────────────────────────────────────
    vix_label = (
        "⚠ Extreme — stress protocol active" if vix_level > 30 else
        "⚠ Elevated — monitor closely"        if vix_level > 25 else
        "Moderate — heightened awareness"     if vix_level > 20 else
        "Normal — standard operations"        if vix_level > 15 else
        "Low — watch for compression triggers"
    )

    # ── Key Levels section ───────────────────────────────────────────────
    if key_levels:
        ema_k   = [k for k in key_levels if k.startswith('EMA')]
        rng_k   = [k for k in key_levels if k.startswith('10d')]
        round_k = [k for k in key_levels if k[0].isdigit()]
        other_k = [k for k in key_levels if k not in ema_k + rng_k + round_k]
        ordered = ema_k + rng_k + round_k + other_k

        level_items = ""
        for k in ordered:
            val, desc = key_levels[k]
            dist = val - spx_price
            dist_str  = f"{dist:+.0f} pts" if abs(dist) > 1 else "← current"
            lvl_color = "var(--green)" if val < spx_price else "var(--blue)" if val > spx_price else "var(--gold)"
            level_items += (
                f'<div class="level-item">'
                f'<div class="level-label">{k}</div>'
                f'<div class="level-val" style="color:{lvl_color}">{val:,.0f}</div>'
                f'<div class="level-dist">{dist_str} &middot; {desc}</div>'
                f'</div>'
            )
        kl_html = (
            f'<section class="section">'
            f'<h2 class="section-title">Key Levels</h2>'
            f'<div class="level-grid">{level_items}</div>'
            f'</section>'
        )
    else:
        kl_html = ""

    # ── Misc ─────────────────────────────────────────────────────────────
    hist_note  = (f"Signal quality improving — {len(records)} day(s) of history on file."
                  if len(records) < 15 else f"Based on {len(records)} days of history.")
    stop_items = "".join(f"<li>{q}</li>" for q in STOP_QUESTIONS)
    date_str   = now_et.strftime('%A, %B %d, %Y')
    time_str   = now_et.strftime('%I:%M %p ET')

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>DIAMOND — Daily Signal</title>
<style>
:root{{
  --bg:#050a15;--card:#0d1526;--border:rgba(201,168,76,.18);
  --gold:#c9a84c;--gold-l:#e8c97c;--text:#e2e8f0;--muted:#94a3b8;
  --green:#10b981;--red:#ef4444;--blue:#60a5fa;--env:{env_color};
}}
*{{box-sizing:border-box;margin:0;padding:0;}}
body{{font-family:'Segoe UI',system-ui,-apple-system,sans-serif;background:var(--bg);color:var(--text);min-height:100vh;padding-bottom:60px;}}
header{{background:linear-gradient(180deg,#080f22 0%,#050a15 100%);border-bottom:1px solid var(--border);padding:32px 24px 24px;text-align:center;}}
.logo{{font-size:2.6rem;font-weight:700;letter-spacing:.22em;color:var(--gold);text-shadow:0 0 40px rgba(201,168,76,.35);}}
.logo span{{color:#fff;opacity:.92;}}
.tagline{{font-size:.72rem;letter-spacing:.2em;text-transform:uppercase;color:var(--muted);margin-top:5px;}}
.ts{{font-size:.7rem;color:var(--muted);margin-top:12px;}}
.container{{max-width:980px;margin:0 auto;padding:0 20px;}}
.section{{margin-top:34px;}}
.section-title{{font-size:.66rem;letter-spacing:.2em;text-transform:uppercase;color:var(--muted);margin-bottom:14px;border-bottom:1px solid rgba(255,255,255,.05);padding-bottom:8px;}}
.market-grid{{display:grid;grid-template-columns:1fr 1fr;gap:14px;}}
@media(max-width:540px){{.market-grid{{grid-template-columns:1fr;}}}}
.market-card{{background:var(--card);border:1px solid var(--border);border-radius:12px;padding:22px 24px;}}
.mlabel{{font-size:.65rem;letter-spacing:.15em;text-transform:uppercase;color:var(--muted);margin-bottom:6px;}}
.mprice{{font-size:2.2rem;font-weight:700;font-variant-numeric:tabular-nums;line-height:1;}}
.mchange{{font-size:.86rem;margin-top:6px;font-variant-numeric:tabular-nums;}}
.mmeta{{font-size:.7rem;color:var(--muted);margin-top:10px;}}
.pos{{color:var(--green);}} .neg{{color:var(--red);}}
.level-grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(140px,1fr));gap:10px;}}
.level-item{{background:var(--card);border:1px solid var(--border);border-radius:10px;padding:14px 16px;}}
.level-label{{font-size:.6rem;letter-spacing:.1em;text-transform:uppercase;color:var(--muted);margin-bottom:4px;}}
.level-val{{font-size:1.15rem;font-weight:700;font-variant-numeric:tabular-nums;}}
.level-dist{{font-size:.65rem;color:var(--muted);margin-top:4px;}}
.env-card{{background:var(--card);border:1px solid var(--env);border-radius:12px;padding:28px 30px;position:relative;overflow:hidden;}}
.env-card::before{{content:'';position:absolute;inset:0;background:radial-gradient(ellipse at top right,rgba(201,168,76,.04) 0%,transparent 65%);pointer-events:none;}}
.env-chip{{display:inline-block;font-size:.63rem;letter-spacing:.14em;text-transform:uppercase;padding:4px 11px;border-radius:99px;background:rgba(255,255,255,.05);border:1px solid rgba(255,255,255,.1);color:var(--env);margin-bottom:14px;}}
.env-name{{font-size:1.7rem;font-weight:700;color:#fff;line-height:1.2;margin-bottom:12px;}}
.env-detail{{font-size:.88rem;color:var(--muted);line-height:1.65;}}
.action-card{{background:linear-gradient(135deg,rgba(201,168,76,.07) 0%,rgba(201,168,76,.02) 100%);border:1px solid rgba(201,168,76,.28);border-radius:12px;padding:22px 26px;}}
.alabel{{font-size:.63rem;letter-spacing:.16em;text-transform:uppercase;color:var(--gold);margin-bottom:8px;}}
.atext{{font-size:1.05rem;font-weight:600;color:#fff;}}
.anote{{font-size:.83rem;color:var(--muted);margin-top:9px;line-height:1.55;}}
.perm-grid{{background:var(--card);border:1px solid var(--border);border-radius:12px;overflow:hidden;}}
.perm-row{{display:flex;align-items:flex-start;gap:14px;flex-wrap:wrap;padding:14px 20px;border-bottom:1px solid rgba(255,255,255,.04);}}
.perm-row:last-child{{border-bottom:none;}}
.perm-label{{font-size:.84rem;font-weight:600;min-width:200px;color:var(--text);}}
.badge{{font-size:.68rem;font-weight:700;letter-spacing:.08em;padding:3px 10px;border-radius:99px;border:1px solid;white-space:nowrap;}}
.perm-note{{font-size:.76rem;color:var(--muted);line-height:1.45;flex:1;}}
.trigger-grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(260px,1fr));gap:12px;}}
.trigger-card{{background:rgba(201,168,76,.07);border:1px solid rgba(201,168,76,.22);border-radius:10px;padding:16px 18px;}}
.trigger-name{{font-size:.8rem;font-weight:700;color:var(--gold-l);margin-bottom:6px;}}
.trigger-detail{{font-size:.74rem;color:var(--muted);line-height:1.5;}}
.gi-note{{font-size:.74rem;color:var(--muted);margin-top:12px;font-style:italic;}}
.no-triggers{{font-size:.84rem;color:var(--muted);background:var(--card);border:1px solid var(--border);border-radius:10px;padding:16px 20px;}}
.roll-card{{background:var(--card);border:1px solid var(--border);border-radius:12px;padding:20px 24px;}}
.roll-intro{{font-size:.82rem;color:var(--muted);margin-bottom:14px;}}
.roll-grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(220px,1fr));gap:12px;}}
.roll-item{{border-radius:8px;padding:14px 16px;background:rgba(255,255,255,.03);border:1px solid rgba(255,255,255,.07);}}
.roll-label{{font-size:.63rem;letter-spacing:.1em;text-transform:uppercase;color:var(--gold);margin-bottom:6px;}}
.roll-val{{font-size:.82rem;color:var(--text);line-height:1.55;}}
.delta-card{{background:var(--card);border:1px solid var(--border);border-radius:12px;padding:20px 24px;}}
.delta-intro{{font-size:.82rem;color:var(--muted);margin-bottom:14px;}}
.delta-bands{{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;}}
@media(max-width:600px){{.delta-bands{{grid-template-columns:1fr;}}}}
.delta-band{{border-radius:8px;padding:14px 16px;}}
.band-inside{{background:rgba(16,185,129,.08);border:1px solid rgba(16,185,129,.22);}}
.band-edge{{background:rgba(245,158,11,.08);border:1px solid rgba(245,158,11,.22);}}
.band-out{{background:rgba(239,68,68,.08);border:1px solid rgba(239,68,68,.22);}}
.blabel{{font-size:.8rem;font-weight:700;margin-bottom:6px;}}
.band-inside .blabel{{color:var(--green);}}
.band-edge .blabel{{color:#f59e0b;}}
.band-out .blabel{{color:var(--red);}}
.bdesc{{font-size:.74rem;color:var(--muted);line-height:1.5;}}
.stop-card{{background:rgba(239,68,68,.04);border:1px solid rgba(239,68,68,.18);border-radius:12px;padding:20px 24px;}}
.stop-title{{font-size:.66rem;letter-spacing:.16em;text-transform:uppercase;color:#ef4444;margin-bottom:12px;}}
.stop-card ul{{list-style:none;display:grid;gap:9px;}}
.stop-card li{{font-size:.84rem;color:var(--muted);padding-left:22px;position:relative;line-height:1.4;}}
.stop-card li::before{{content:'→';position:absolute;left:0;color:#ef4444;}}
.hist-note{{font-size:.7rem;color:rgba(148,163,184,.5);margin-top:8px;font-style:italic;}}
footer{{margin-top:52px;padding:24px 20px;border-top:1px solid rgba(255,255,255,.05);text-align:center;font-size:.66rem;color:rgba(148,163,184,.55);line-height:1.7;}}
</style>
</head>
<body>

<!-- ── Password Gate ─────────────────────────────────────────────────── -->
<div id="gate" style="position:fixed;inset:0;background:#050a15;display:flex;align-items:center;justify-content:center;z-index:9999;font-family:'Segoe UI',system-ui,-apple-system,sans-serif;">
  <div style="text-align:center;max-width:340px;width:90%;padding:40px 32px;background:#0d1526;border:1px solid rgba(201,168,76,.22);border-radius:16px;">
    <div style="font-size:2rem;font-weight:700;letter-spacing:.22em;color:#c9a84c;text-shadow:0 0 30px rgba(201,168,76,.35);margin-bottom:6px;">◆ <span style="color:#fff;opacity:.92">DIAMOND</span></div>
    <div style="font-size:.62rem;letter-spacing:.2em;text-transform:uppercase;color:#94a3b8;margin-bottom:36px;">Private Access</div>
    <input id="pw" type="password" placeholder="Password"
      style="width:100%;background:#050a15;border:1px solid rgba(201,168,76,.3);border-radius:8px;padding:14px 18px;color:#e2e8f0;font-size:.95rem;outline:none;text-align:center;letter-spacing:.15em;margin-bottom:12px;box-sizing:border-box;"
      onkeydown="if(event.key==='Enter')checkPw()">
    <button onclick="checkPw()"
      style="width:100%;background:linear-gradient(135deg,rgba(201,168,76,.15),rgba(201,168,76,.05));border:1px solid rgba(201,168,.4);border-radius:8px;padding:14px;color:#c9a84c;font-size:.85rem;font-weight:600;letter-spacing:.12em;cursor:pointer;text-transform:uppercase;">
      Enter
    </button>
    <div id="pw-err" style="color:#ef4444;font-size:.74rem;margin-top:10px;min-height:18px;"></div>
  </div>
</div>
<script>
(function(){{
  if(sessionStorage.getItem('diamond_auth')==='1'){{
    document.getElementById('gate').style.display='none';
  }}
}})();
function checkPw(){{
  var pw=document.getElementById('pw');
  if(pw.value==='{SITE_PASSWORD}'){{
    sessionStorage.setItem('diamond_auth','1');
    document.getElementById('gate').style.display='none';
  }}else{{
    document.getElementById('pw-err').textContent='Incorrect password — try again.';
    pw.value=''; pw.focus();
  }}
}}
</script>
<!-- ─────────────────────────────────────────────────────────────────── -->

<header>
  <div class="logo">◆ <span>DIAMOND</span></div>
  <div class="tagline">Daily DIAMONDS Signal &mdash; Options Inventory Intelligence</div>
  <div class="ts">Generated {time_str} &middot; {date_str}</div>
</header>

<div class="container">

  <section class="section">
    <h2 class="section-title">Market Data</h2>
    <div class="market-grid">
      <div class="market-card">
        <div class="mlabel">S&amp;P 500 Index (SPX)</div>
        <div class="mprice">{spx_price:,.2f}</div>
        <div class="mchange {cc(spx_chg)}">{fc(spx_chg, spx_pct)}</div>
        <div class="mmeta">52-week high: {spx_52w:,.0f} &middot; {spx_ath_pct:+.1f}% from ATH</div>
      </div>
      <div class="market-card">
        <div class="mlabel">CBOE Volatility Index (VIX)</div>
        <div class="mprice">{vix_level:.2f}</div>
        <div class="mchange {cc(vix_chg)}">{fc(vix_chg, vix_pct)}</div>
        <div class="mmeta">{vix_label}</div>
      </div>
    </div>
  </section>

  {kl_html}

  <section class="section">
    <h2 class="section-title">Today's Environment</h2>
    <div class="env-card">
      <div class="env-chip">DIAMONDS Environment ID</div>
      <div class="env-name">{env_name}</div>
      <div class="env-detail">{env_detail}</div>
    </div>
    <div class="hist-note">{hist_note}</div>
  </section>

  <section class="section">
    <div class="action-card">
      <div class="alabel">Primary Action</div>
      <div class="atext">{perms['primary']}</div>
      <div class="anote">{perms['note']}</div>
    </div>
  </section>

  <section class="section">
    <h2 class="section-title">Today's Permissions</h2>
    <div class="perm-grid">{perm_rows}</div>
  </section>

  {gi_html}

  <section class="section">
    <h2 class="section-title">Money Press — Roll Mechanics</h2>
    <div class="roll-card">
      <div class="roll-intro">Mechanical rules for managing the Money Press. Apply consistently — no discretion needed.</div>
      <div class="roll-grid">
        <div class="roll-item">
          <div class="roll-label">Roll Day</div>
          <div class="roll-val">Tuesday preferred. Trigger at 3–5 DTE on the short put side.</div>
        </div>
        <div class="roll-item">
          <div class="roll-label">Strike Adjustment</div>
          <div class="roll-val">If SPX moved +50 pts since last roll, move short put UP 50 pts on next roll.</div>
        </div>
        <div class="roll-item">
          <div class="roll-label">Short Put Target</div>
          <div class="roll-val">50–100 pts above visible EMA / support level. Gives cushion plus premium collected.</div>
        </div>
        <div class="roll-item">
          <div class="roll-label">Event (FOMC) Protocol</div>
          <div class="roll-val">Roll INTO the event on Tuesday before. Roll OUT after Wednesday announcement. Never hold short unrolled over the Fed decision.</div>
        </div>
        <div class="roll-item">
          <div class="roll-label">Long Protection</div>
          <div class="roll-val">~90 DTE long put. Cost recovered in ~3 weekly rolls. Remaining rolls = free cash flow.</div>
        </div>
        <div class="roll-item">
          <div class="roll-label">Patience Rule</div>
          <div class="roll-val">We sell time — time must pass to get paid. Focus on weekly premium collected, not Net Liq.</div>
        </div>
      </div>
    </div>
  </section>

  <section class="section">
    <h2 class="section-title">Portfolio Delta Guardrails</h2>
    <div class="delta-card">
      <div class="delta-intro">Delta measures total directional exposure across all open positions. Monitor daily. These thresholds trigger specific responses regardless of environment:</div>
      <div class="delta-bands">
        <div class="delta-band band-inside">
          <div class="blabel">Inside ±150Δ</div>
          <div class="bdesc">Smooth operation. MP additions allowed. One DD if environment permits. GI by trigger only.</div>
        </div>
        <div class="delta-band band-edge">
          <div class="blabel">Edge ±150 to ±500Δ</div>
          <div class="bdesc">No new DDs. MP roll only to reduce risk. Consider GI if not on. Daily monitoring required.</div>
        </div>
        <div class="delta-band band-out">
          <div class="blabel">Outside &gt;±500Δ</div>
          <div class="bdesc">Must adjust immediately. Defense only — no new MP or DD. GI first priority if missing. Stop and re-assess.</div>
        </div>
      </div>
    </div>
  </section>

  <section class="section">
    <h2 class="section-title">The Stop Rule</h2>
    <div class="stop-card">
      <div class="stop-title">Before any trade — answer each in one sentence. If you can't, stop.</div>
      <ul>{stop_items}</ul>
    </div>
  </section>

</div>

<footer>
  ◆ DIAMOND Investment Group &mdash; Signal generated at {time_str} on {date_str}<br><br>
  <strong>Disclaimer:</strong> This page applies the DIAMONDS operating system rules mechanically to publicly available market data.
  It is for informational and educational purposes only and does not constitute financial advice, investment recommendations,
  or a solicitation to buy or sell any security or options contract. Options trading involves substantial risk of loss.
  Always consult a qualified financial professional before making investment decisions.
</footer>
</body>
</html>"""

# ════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='DIAMOND Daily Signal Generator')
    parser.add_argument('--spx-price',    type=float, required=True, help='SPX current price')
    parser.add_argument('--spx-prev',     type=float, required=True, help='SPX previous close')
    parser.add_argument('--spx-52w-high', type=float, required=True, help='SPX 52-week high')
    parser.add_argument('--spx-high',     type=float, default=None,  help="Today's SPX session high (optional)")
    parser.add_argument('--spx-low',      type=float, default=None,  help="Today's SPX session low (optional)")
    parser.add_argument('--vix-level',    type=float, required=True, help='VIX current level')
    parser.add_argument('--vix-prev',     type=float, required=True, help='VIX previous close')
    args = parser.parse_args()

    spx_high  = args.spx_high or args.spx_price
    spx_low   = args.spx_low  or args.spx_price
    today_str = date.today().isoformat()

    print("◆ DIAMOND Signal Generator (v2 — enhanced intelligence)")
    print(f"  Date:  {today_str}")
    print(f"  SPX:   {args.spx_price:,.2f}  (prev {args.spx_prev:,.2f})")
    print(f"  VIX:   {args.vix_level:.2f}  (prev {args.vix_prev:.2f})")

    hist = load_history()
    hist = update_history(hist, args.spx_price, spx_high, spx_low, args.vix_level, today_str)
    save_history(hist)
    records = hist['records']
    print(f"  History: {len(records)} day(s) on file")

    emas       = compute_emas(records)
    key_levels = compute_key_levels(records, args.spx_price, args.spx_52w_high, emas)

    e9, e21, e50 = emas.get('ema9'), emas.get('ema21'), emas.get('ema50')
    if e9 and e21:
        e50_str = f"{e50:,.0f}" if e50 else "building"
        print(f"  EMA 9/21/50: {e9:,.0f} / {e21:,.0f} / {e50_str}")

    env_name, env_key, env_detail = classify_environment(records, args.spx_price, args.vix_level, emas)
    print(f"  Environment: {env_name}")

    triggers = gi_triggers(records, args.spx_price, args.spx_52w_high, args.vix_level)
    print(f"  GI triggers active: {len(triggers)}")

    perms = permissions(env_key)

    html = generate_html(
        args.spx_price, args.spx_prev, args.spx_52w_high,
        args.vix_level, args.vix_prev,
        records, emas, key_levels,
        env_name, env_key, env_detail, triggers, perms
    )

    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        f.write(html)

    print(f"  ✓ index.html written to: {OUTPUT_PATH}")
    print("  Done.")


if __name__ == '__main__':
    main()
