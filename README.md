# Hybrid Anomaly Detection System for Stock Market Manipulation

A Data Science project for detecting suspicious stock-trading sessions and screening for potential pump-and-dump patterns from historical OHLCV data.

The system combines financial rule-based signals with an unsupervised machine learning model, **Isolation Forest**, to identify unusual price and volume behavior.

> **Academic-use disclaimer:** The system generates risk signals for research and exploratory analysis. A flagged event is not proof of market manipulation and must not be interpreted as investment advice.

---

## System Architecture

![Hybrid Anomaly Detection Pipeline](assets/system_architecture.png)

The workflow covers:

1. Data collection  
2. Data preprocessing and quality checks  
3. Financial feature engineering  
4. Rule-based anomaly detection  
5. Isolation Forest anomaly detection  
6. Hybrid scoring and risk labeling  
7. Anomaly clustering and pump-and-dump screening  
8. Visualization and dashboard analysis  

---

## Objectives

- Analyze historical daily OHLCV stock-market data.
- Detect unusual combinations of price movements and trading volume.
- Combine domain-based technical signals with unsupervised machine learning.
- Group suspicious sessions into clusters for pump-and-dump screening.
- Produce visual and dashboard-ready outputs for exploratory analysis.

---

## Covered Stocks

| Ticker | Company | Market | Role in Project |
|---|---|---|---|
| LKNCY | Luckin Coffee | NASDAQ | Main reproducible anomaly-detection experiment |
| FLC | FLC Group | HOSE | Comparative high-volatility case |
| VNM | Vinamilk | HOSE | Comparative stable-stock case |

Each dataset contains daily:

- Date
- Open
- High
- Low
- Close
- Volume

### Data Sources

- **LKNCY:** Yahoo Finance data retrieved with `yfinance`.
- **FLC** and **VNM:** Vietnamese stock data collected with `vnstock3`.

---

## Pipeline

### 1. Data Preprocessing

The preprocessing stage:

- Converts price and volume columns to numeric values.
- Parses and sorts the `Date` column chronologically.
- Removes missing and duplicate observations.
- Checks OHLC consistency:

```text
Low ≤ Open / Close ≤ High
```

- Logs invalid or zero-value observations.
- Detects trading-date gaps.

Typical outputs:

```text
df_clean.csv
data_issue_log.csv
```

---

### 2. Feature Engineering

The system generates financial and statistical indicators that describe price movement, volatility, trend, and abnormal volume behavior.

| Feature Group | Features |
|---|---|
| Price movement | Daily Return, Log Return, Close-to-Open Return |
| Intraday volatility | Intraday Range, Range Percentage |
| Rolling volatility | 120-day return volatility |
| Volume behavior | 30-day Volume Ratio, 120-day Volume Z-score |
| Price anomaly | 120-day Return Z-score |
| Trend | MA-5, MA-20, MA-50 |
| Technical indicators | RSI-14, Bollinger Bands, MACD, ATR-14 |

Key calculations:

```text
Daily Return = (Close_t - Close_(t-1)) / Close_(t-1)

Volume Ratio = Volume_t / Mean(Volume_(t-30:t))

Volume Z-score =
(Volume_t - RollingMean_120(Volume)) / RollingStd_120(Volume)
```

Typical output:

```text
df_feat.csv
```

---

### 3. Rule-Based Anomaly Detection

The rule-based component captures financial patterns associated with unusual trading activity.

```text
Volume-Return Signal:
Volume Ratio > 1.8 AND |Daily Return| > 2%

Bollinger Breakout:
Close > Upper Bollinger Band × 1.002
AND Volume > 1.2 × 30-day average volume

RSI-Volume Signal:
RSI-14 > 70 AND Volume Ratio > 1.5
```

The rule score combines Z-score intensity, binary rule signals, and a bonus for stronger signals:

```text
Rule Score =
0.35 × Z-score Intensity
+ 0.65 × Binary Rule Signals
+ Severity Bonus
```

The score is normalized to the range `[0, 1]`.

Typical output:

```text
df_rule.csv
```

---

### 4. Isolation Forest

Isolation Forest is used to identify unusual observations without requiring manually labeled manipulation events.

#### Input Features

```text
log_return
vol_ratio
range_pct
RSI_14
ret_z
vol_z
volatility_120
MACD_hist
BB_pos
ATR_14
```

#### Configuration

```python
IsolationForest(
    n_estimators=500,
    contamination=0.01,
    random_state=42
)
```

Input features are standardized using `StandardScaler`. The Isolation Forest decision score is inverted and normalized into an `ml_score` between `0` and `1`, where higher values indicate more unusual behavior.

Typical output:

```text
df_ml.csv
```

---

### 5. Hybrid Ensemble and Risk Labeling

The final anomaly score combines the machine-learning and financial-rule components:

```text
Final Anomaly Score =
0.4 × ML Score + 0.6 × Rule-Based Score
```

A trading session is flagged as suspicious when:

```text
Final Anomaly Score ≥ 0.40
```

Severity labels are assigned from the anomaly score, return Z-score, and volume Z-score:

| Severity | Criteria |
|---|---|
| High | `|ret_z| ≥ 3`, `vol_z ≥ 3`, or score `≥ 0.75` |
| Medium | score `0.55–0.75`, or Z-score between `2–3` |
| Low | score `0.40–0.55`, or Z-score between `1.5–2` |
| Normal | No suspicious signal |

Typical output:

```text
df_final_output.csv
```

---

### 6. Anomaly Clustering and Pump-and-Dump Screening

Suspicious sessions are grouped into clusters when adjacent signals are no more than two calendar days apart.

A cluster is marked as a **potential pump-and-dump candidate** when:

1. At least one session in the cluster has:

```text
Volume Z-score ≥ 2
```

2. The price falls by at least 5% within the following three trading sessions after the cluster peak:

```text
Price Drop ≤ -5%
```

Each cluster records:

- Start and end date
- Cluster length
- Number of high-severity sessions
- Maximum volume Z-score
- Average return
- Post-event returns after 1, 3, and 5 trading days
- Pump-and-dump candidate flag

Typical output:

```text
clusters.csv
```

---

### 7. Sensitivity Analysis

The rule-based component is evaluated under several Z-score thresholds.

| Z-score Threshold | Detected Sessions | Detection Rate |
|---|---:|---:|
| 2.0 | 99 | 7.86% |
| 2.5 | 59 | 4.68% |
| 3.0 | 40 | 3.17% |

Higher thresholds generate fewer alerts and create a stricter anomaly-screening setting.

Typical output:

```text
validation_summary.csv
```

---

## Results

### LKNCY Detection Summary

The main notebook produced the following results for the LKNCY experiment:

| Metric | Result |
|---|---:|
| Feature-ready trading sessions | 1,260 |
| Suspicious sessions detected | 143 |
| Detection rate | 11.35% |
| High-severity sessions | 48 |
| Medium-severity sessions | 58 |
| Low-severity sessions | 37 |
| Anomaly sequences identified | 74 |
| Longest consecutive anomaly sequence | 8 sessions |
| Clusters with positive 3-day post-event movement | 50.0% |

The anomalies may reflect temporary volatility, corporate events, speculative activity, or potential market irregularities. They are intended to prioritize periods for manual review rather than prove misconduct.

### Computational Efficiency

| Metric | Result |
|---|---:|
| Isolation Forest training time | ~0.4568 seconds |
| Average inference time | ~32.7 µs/sample |
| Approximate RAM usage | ~5 MB |

### Comparative Findings

- **VNM:** Predominantly stable behavior with lower volatility.
- **FLC:** Strong abnormal volatility, particularly during 2021–2022.
- **LKNCY:** Caution-level anomaly signals, especially around high-volatility periods.

---

## Visual Outputs

The project includes visualizations for:

- Closing-price movement with anomaly markers.
- Top anomaly-score sessions.
- Distribution of anomaly severity.
- Return Z-score versus Volume Z-score behavior.
- Time-based anomaly heatmaps.
- Cluster-level post-event price behavior.

Common figure files include:

```text
fig_price_anomalies.png
bieudo_top15_ngay.png
bieudo_muc_do_bat_thuong.png
heatmap_retZ_volZ.png
heatmap_lich_bat_thuong.png
```

---

## Repository Structure

```text
.
├── assets/
│   └── system_architecture.png
│
├── data/
│   └── raw/
│       └── LKNCY.csv
│
├── docs/
│   └── Present_NCKH.pdf
│
├── notebooks/
│   ├── hybrid_stock_anomaly_detection.ipynb
│   └── archive/
│
├── outputs/
│   ├── figures/
│   └── tables/
│
├── plots/
│   └── additional exploratory charts
│
├── stocks/
│   ├── FLC/
│   │   └── stock-specific data, notebooks, and outputs
│   └── VNM/
│       └── stock-specific data, notebooks, and outputs
│
└── README.md
```

---

## Technologies

- Python
- Pandas
- NumPy
- Scikit-learn
- Isolation Forest
- StandardScaler
- MinMaxScaler
- Matplotlib
- Seaborn
- yfinance
- vnstock3
- Jupyter Notebook
- Power BI

---

## Run the Project

Open the main notebook:

```bash
jupyter notebook
```

Then run:

```text
notebooks/hybrid_stock_anomaly_detection.ipynb
```

The original notebook reads and writes CSV files using relative paths. Before rerunning it, update the data-loading and output paths to match the current folder structure.

For example:

```python
df = pd.read_csv("../data/raw/LKNCY.csv")
```

Run the cells in sequence to generate cleaned data, engineered features, rule-based scores, Isolation Forest scores, hybrid scores, clusters, validation summaries, and visualizations.

---

## Limitations

- The project combines unsupervised detection with heuristic financial rules.
- An anomaly signal does not confirm market manipulation.
- Isolation Forest uses a fixed `contamination=0.01`.
- The current analysis does not adjust anomalies using broad-market movements such as VN-Index or NASDAQ index changes.
- The repository does not include independently verified manipulation labels.
- The current permutation-style check is exploratory and should not be interpreted as formal statistical proof.
- Results must not be used as financial or investment advice.

---

## Future Work

- Add broad-market index normalization.
- Integrate financial-news sentiment analysis using NLP.
- Evaluate Autoencoder-based anomaly detection.
- Validate against independently confirmed manipulation cases.
- Extend analysis to more tickers and industry sectors.
- Build a real-time dashboard for continuous anomaly monitoring.

---

## Disclaimer

This repository is developed for educational and academic research purposes only.

It does not provide financial advice and does not confirm that a company, stock, or trading session was involved in market manipulation.
