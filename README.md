# FX Predictor — H1 · GitHub Actions + Pages

## What this does
- Fetches **real H1 candles** from Yahoo Finance (free, no API key)
- Runs **Markov Chain + RSI + MACD + Z-Score + Kelly Criterion** analysis
- **Embeds data directly into `dashboard.html`** — no server needed
- **GitHub Actions** runs this every hour automatically
- **GitHub Pages** hosts the HTML — accessible from any device, including your phone

---

## One-Time Setup (5 minutes)

### Step 1 — Create a GitHub repository
1. Go to https://github.com/new
2. Name it e.g. `fx-predictor`
3. Set it to **Public** (required for free GitHub Pages)
4. Click **Create repository**

### Step 2 — Upload these files
Upload all files keeping the folder structure:
```
fx-predictor/
├── predictor.py
├── dashboard.html
├── forex_data.json          ← will be auto-generated, upload an empty {} for now
└── .github/
    └── workflows/
        └── hourly_update.yml
```
You can drag & drop in the GitHub web UI, or use git:
```bash
git init
git remote add origin https://github.com/YOUR_USERNAME/fx-predictor.git
git add .
git commit -m "initial commit"
git push -u origin main
```

### Step 3 — Enable GitHub Pages
1. In your repo → **Settings** → **Pages** (left sidebar)
2. Under **Source** → select **GitHub Actions**
3. Click **Save**

### Step 4 — Run the workflow once manually
1. Go to **Actions** tab in your repo
2. Click **FX Predictor — Hourly Update**
3. Click **Run workflow** → **Run workflow**
4. Wait ~2 minutes for it to complete

### Step 5 — Get your phone URL
Your dashboard will be live at:
```
https://YOUR_USERNAME.github.io/fx-predictor/dashboard.html
```
Bookmark this on your phone. It updates every hour automatically.

---

## How it works end-to-end

```
Every hour (GitHub Actions cron)
        ↓
predictor.py runs
        ↓
yfinance fetches H1 candles for all 7 pairs
        ↓
Markov Chain + Indicators computed
        ↓
Results embedded directly into dashboard.html
        ↓
dashboard.html committed back to repo
        ↓
GitHub Pages re-deploys (< 1 min)
        ↓
You open the URL on your phone → live data ✓
```

---

## Local usage (no GitHub needed)
```bash
pip install yfinance pandas numpy
python predictor.py
# open dashboard.html in browser
```

---

## Files
```
predictor.py              ← Python engine
dashboard.html            ← Self-contained HTML dashboard
forex_data.json           ← Raw JSON output (for debugging)
.github/workflows/
  hourly_update.yml       ← GitHub Actions schedule
README.md
```

---

## Pairs covered
EUR/USD · GBP/USD · USD/JPY · USD/CHF · AUD/USD · USD/CAD · NZD/USD

---

## ⚠ Disclaimer
Educational and research use only. Not financial advice.
Forex trading involves substantial risk of loss.
Statistical patterns do not guarantee future results.
