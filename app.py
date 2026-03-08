import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import io

# =============================================================================
# CÁC HÀM XỬ LÝ (Lấy từ Notebook của bạn)
# =============================================================================
# Thêm @st.cache_data để lưu kết quả, giúp app chạy nhanh hơn
@st.cache_data
def load_and_clean_data(uploaded_file):
    """
    Tải và làm sạch dữ liệu từ file CSV được tải lên.
    (Tương đương các cell 9-18)
    """
    # Đọc dữ liệu từ file upload (thay vì đường dẫn D:\...)
    # Thêm encoding='utf-8-sig' để xử lý ký tự đặc biệt nếu có
    try:
        df = pd.read_csv(uploaded_file, encoding='utf-8-sig')
    except Exception as e:
        try:
            # Thử đọc lại với encoding khác nếu utf-8 lỗi
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file, encoding='latin1')
        except Exception as e2:
            st.error(f"Không thể đọc file. Lỗi: {e2}")
            return None, None

    # Chuyển dữ liệu số về numeric (cell 12)
    coin_labels = df.columns
    for label in coin_labels:
        if label != 'Date':
            df[label] = pd.to_numeric(df[label], errors="coerce")

    # Drop các dòng có NAN (không hợp lệ/ thiếu)
    df.dropna(inplace=True)

    # Chuyển cột Date về dạng datetime
    df['Date'] = pd.to_datetime(df['Date'])

    # Xóa dòng trùng lặp
    df.drop_duplicates(inplace=True)

    # Sắp xếp lại theo ngày để đảm bảo logic
    df = df.sort_values('Date').reset_index(drop=True)

    # Kiểm tra logic OHLC (cell 15)
    mask_invalid = (df['High'] < df[['Open', 'Close', 'Low']].max(axis=1)) | (df['Low'] > df[['Open', 'Close', 'Low']].min(axis=1))
    data_issue_log = df.loc[mask_invalid, ['Date', 'Open', 'High', 'Low', 'Close']]

    # Kiểm tra giá và khối lượng = 0 (cell 16)
    mask_zero = (df['Volume'] == 0) | (df[['Open', 'High', 'Low', 'Close']] == 0).any(axis=1)
    data_issue_log = pd.concat([data_issue_log, df.loc[mask_zero, ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]])
    
    # Loại bỏ các dòng có vấn đề
    df_clean = df.drop(index=data_issue_log.index).reset_index(drop=True)

    return df_clean, data_issue_log.drop_duplicates()

@st.cache_data
def create_features(df):
    """
    Tạo các đặc trưng kỹ thuật từ dữ liệu đã làm sạch.
    (Tương đương cell 19)
    """
    CLOSE, OPEN, HIGH, LOW, VOL = 'Close', 'Open', 'High', 'Low', 'Volume'
    
    df['intraday_range'] = df[HIGH] - df[LOW]
    df['return'] = df['Close'].pct_change()
    df['volatility_120'] = df['return'].rolling(120, min_periods=120).std()

    df['log_return'] = np.log(df[CLOSE] / df[CLOSE].shift(1))
    df['range_pct'] = (df['High'] - df['Low']) / df['Close']
    df['close_open_ret'] = (df['Close'] - df['Open']) / df['Open']

    win = 120
    df['vol_mean'] = df['Volume'].rolling(win).mean()
    df['vol_std'] = df['Volume'].rolling(win).std()
    df['ret_mean'] = df['return'].rolling(win).mean()
    df['ret_std'] = df['return'].rolling(win).std()

    df['vol_z'] = (df['Volume'] - df['vol_mean']) / df['vol_std']
    df['ret_z'] = (df['return'] - df['ret_mean']) / df['ret_std']

    df['vol_ratio'] = df['Volume'] / df['Volume'].rolling(30).mean()

    df['MA_5'] = df['Close'].rolling(5).mean()
    df['MA_20'] = df['Close'].rolling(20).mean()
    df['MA_50'] = df['Close'].rolling(50).mean()

    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df['RSI_14'] = 100 - (100 / (1 + rs))

    ma20 = df['Close'].rolling(20).mean()
    std20 = df['Close'].rolling(20).std()
    df['BB_upper'] = ma20 + 2 * std20
    df['BB_lower'] = ma20 - 2 * std20
    df['BB_pos'] = (df['Close'] - ma20) / (2 * std20)

    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD_line'] = ema12 - ema26
    df['MACD_signal'] = df['MACD_line'].ewm(span=9, adjust=False).mean()
    df['MACD_hist'] = df['MACD_line'] - df['MACD_signal']

    tr = pd.concat([
        df['High'] - df['Low'],
        (df['High'] - df['Close'].shift(1)).abs(),
        (df['Low'] - df['Close'].shift(1)).abs()
    ], axis=1).max(axis=1)
    df['ATR_14'] = tr.rolling(14).mean()

    for col in ['return', 'vol_ratio', 'range_pct', 'RSI_14']:
        df[col + '_z'] = (df[col] - df[col].rolling(30).mean()) / df[col].rolling(30).std()

    # Xử lý NaN sau khi tính toán (cell 20)
    df_feat = df.dropna().reset_index(drop=True)
    return df_feat

@st.cache_data
def run_rule_based(df_feat):
    """
    Chạy mô hình dựa trên luật.
    (Tương đương cell 21)
    """
    df = df_feat.copy()
    ma30_vol = df['Volume'].rolling(30, min_periods=30).mean()

    df['flag_vol_return'] = (df['vol_ratio'] > 1.8) & (df['return'].abs() > 0.02)
    df['flag_bollinger'] = (df['Close'] > df['BB_upper'] * 1.002) & (df['Volume'] > 1.2 * ma30_vol)

    abs_ret_z = df['ret_z'].abs()
    vol_z = df['vol_z']
    
    cond_high = ((abs_ret_z >= 1.8) | (vol_z >= 1.8)) & (df['close_open_ret'].abs() >= 0.015)
    cond_med = ((abs_ret_z.between(1.2, 1.8, inclusive='left')) |
                 (vol_z.between(1.2, 1.8, inclusive='left')) |
                 (df['intraday_range'] >= 0.015))
    cond_low = ((abs_ret_z.between(0.8, 1.2, inclusive='left')) |
                 (vol_z.between(0.8, 1.2, inclusive='left')))

    df['z_band'] = 'None'
    df.loc[cond_low, 'z_band'] = 'Low'
    df.loc[cond_med, 'z_band'] = 'Medium'
    df.loc[cond_high, 'z_band'] = 'High'

    z_star = np.maximum(abs_ret_z, vol_z)
    df['z_intensity'] = np.clip((z_star - 0.8) / (1.8 - 0.8), 0, 1)
    
    df['flag_rsi_pump'] = (df['RSI_14'] > 70) & (df['vol_ratio'] > 1.5)
    df['binary_flags'] = (
        df['flag_vol_return'].astype(int) + 
        df['flag_bollinger'].astype(int) + 
        df['flag_rsi_pump'].astype(int)
    ) / 3.0

    df['rule_score'] = 0.35 * df['z_intensity'] + 0.65 * df['binary_flags']
    bonus = (
        0.15 * ((df['binary_flags'] > 0).astype(float)) +
        0.10 * (df['z_band'] == 'High'))
    df['rule_score'] = np.clip(df['rule_score'] + bonus, 0, 1)

    return df[['Date', 'rule_score']]

@st.cache_data
def run_ml_model(df_feat):
    """
    Chạy mô hình ML (IsolationForest).
    (Tương đương cell 22)
    """
    df = df_feat.copy()
    features = [
        'log_return', 'vol_ratio', 'range_pct', 'RSI_14',
        'ret_z', 'vol_z', 'volatility_120', 'MACD_hist', 'BB_pos', 'ATR_14'
    ]
    
    # Đảm bảo không còn NaN trong các cột features
    df_ml_input = df[features].dropna()
    
    scaler_in = StandardScaler()
    X_scaled = scaler_in.fit_transform(df_ml_input)

    iso = IsolationForest(n_estimators=500, contamination=0.05, random_state=42)
    iso.fit(X_scaled)

    raw = -iso.decision_function(X_scaled).reshape(-1, 1)
    scaler_out = MinMaxScaler()
    
    # Gán điểm ml_score lại cho df_ml_input (đã dropna)
    df_ml_input['ml_score'] = scaler_out.fit_transform(raw)
    
    # Merge lại vào df gốc bằng index
    df = df.merge(df_ml_input[['ml_score']], left_index=True, right_index=True, how='left')
    
    return df[['Date', 'ml_score']]

@st.cache_data
def ensemble_and_label(df_ml, df_rule, df_feat):
    """
    Kết hợp điểm số và gán nhãn.
    (Tương đương cell 23 và logic từ cell 29)
    """
    # Gộp df_feat (chứa các cột gốc) với df_ml và df_rule
    df_final = pd.merge(df_feat, df_ml[['Date', 'ml_score']], on='Date', how='left')
    df_final = pd.merge(df_final, df_rule[['Date', 'rule_score']], on='Date', how='left')
    
    # Điền 0 cho các ngày đầu (NaN)
    df_final['ml_score'] = df_final['ml_score'].fillna(0)
    df_final['rule_score'] = df_final['rule_score'].fillna(0)

    df_final['anomaly_score'] = 0.4 * df_final['ml_score'] + 0.6 * df_final['rule_score']
    df_final['label'] = np.where(df_final['anomaly_score'] >= 0.4, 'Yes', 'No')

    def classify_severity(row):
        abs_ret_z = abs(row.get('ret_z', 0))
        vol_z = row.get('vol_z', 0)
        score = row['anomaly_score']
        if (abs_ret_z >= 3) or (vol_z >= 3) or (score >= 0.75):
            return 'High'
        elif (0.55 <= score < 0.75) or (2 <= abs_ret_z < 3) or (2 <= vol_z < 3):
            return 'Medium'
        elif (0.4 <= score < 0.55) or (1.5 <= abs_ret_z < 2) or (1.5 <= vol_z < 2):
            return 'Low'
        else:
            return 'Normal'
    
    df_final['severity'] = df_final.apply(classify_severity, axis=1)

    # Thêm logic từ cell 29
    def detect_type(row):
        cond_price = abs(row.get('ret_z', 0)) >= 1.8
        cond_vol = row.get('vol_z', 0) >= 1.8
        if cond_price and cond_vol: return "Giá + Khối lượng"
        elif cond_price: return "Giá"
        elif cond_vol: return "Khối lượng"
        else: return "Bình thường"
    
    df_final['loai_bat_thuong'] = df_final.apply(detect_type, axis=1)
    
    df_final['group'] = (df_final['label'] != df_final['label'].shift()).cumsum()
    df_final['so_ngay_lien_tiep'] = df_final.groupby('group')['label'].transform(lambda x: len(x) if x.iloc[0] == 'Yes' else 0)

    return df_final

@st.cache_data
def find_clusters(df_final):
    """
    Gom cụm các ngày bất thường.
    (Tương đương cell 24)
    """
    final = df_final.copy() # Đã có đủ cột Close, return, ret_z, vol_z
    
    for n in (1, 3, 5):
        final[f'Close_f{n}'] = final['Close'].shift(-n)
        final[f'post_{n}d'] = (final[f'Close_f{n}'] / final['Close'] - 1.0) * 100.0

    yes = final[final['label'] == 'Yes'].copy()
    if yes.empty:
        return pd.DataFrame() # Trả về DF rỗng nếu không có bất thường

    yes['diff_days'] = yes['Date'].diff().dt.days
    yes['new_cluster'] = (yes['diff_days'].isna()) | (yes['diff_days'] > 2)
    yes['cluster_id'] = yes['new_cluster'].cumsum()

    clusters = []
    for cid, block in yes.groupby('cluster_id'):
        start_date = block['Date'].iloc[0]
        end_date = block['Date'].iloc[-1]
        length_days = (end_date - start_date).days + 1
        n_high_days = (block['severity'] == 'High').sum()
        max_vol_z = block['vol_z'].max()
        avg_ret = block['return'].mean() * 100
        end_idx = block.index[-1]
        
        # Đảm bảo end_idx nằm trong final.loc
        if end_idx not in final.index:
            continue

        post_1d = final.loc[end_idx, 'post_1d']
        post_3d = final.loc[end_idx, 'post_3d']
        post_5d = final.loc[end_idx, 'post_5d']

        pump_dump = 0
        pump_pct = dump_pct = time_to_peak = time_to_dump = np.nan

        if (block['vol_z'] >= 2).any():
            peak_idx = block['Close'].idxmax()
            peak_close = block.loc[peak_idx, 'Close']
            start_close = block['Close'].iloc[0]
            pump_pct = (peak_close / start_close - 1) * 100
            time_to_peak = (block.loc[peak_idx, 'Date'] - start_date).days

            look = final.loc[peak_idx + 1 : peak_idx + 3].copy()
            if not look.empty:
                min_close = look['Close'].min()
                dump_pct = (min_close / peak_close - 1) * 100
                time_to_dump = (look.loc[look['Close'].idxmin(), 'Date'] - block.loc[peak_idx, 'Date']).days
                if dump_pct <= -5:
                    pump_dump = 1

        clusters.append({
            'cluster_id': cid, 'start_date': start_date.date(), 'end_date': end_date.date(),
            'length_days': length_days, 'n_high_days': n_high_days, 'max_vol_z': max_vol_z,
            'avg_ret_%': avg_ret, 'post_1d_%': post_1d, 'post_3d_%': post_3d, 'post_5d_%': post_5d,
            'pump_dump': pump_dump, 'pump_pct_%': pump_pct, 'dump_pct_%': dump_pct,
            'time_to_peak_days': time_to_peak, 'time_to_dump_days': time_to_dump
        })

    return pd.DataFrame(clusters)

@st.cache_data
def generate_summary_text(df_final):
    """
    Tạo kết luận văn bản.
    (Tương đương cell 29)
    """
    total_days = len(df_final)
    abnormal_df = df_final[df_final['label'] == 'Yes']
    n_abnormal = len(abnormal_df)
    if total_days == 0: return "Không có dữ liệu.", "Không có dữ liệu."
    
    pct_abnormal = n_abnormal / total_days * 100
    sev_counts = abnormal_df['severity'].value_counts().reindex(['High', 'Medium', 'Low'], fill_value=0)
    high_n, med_n, low_n = sev_counts['High'], sev_counts['Medium'], sev_counts['Low']
    
    avg_vol_z = df_final['vol_z'].mean()
    avg_ret_std = df_final['return'].std()

    if avg_vol_z < 1.2 and avg_ret_std < 0.015:
        conclusion = "Cổ phiếu ổn định – khó có khả năng thao túng."
    elif high_n > 20 or pct_abnormal > 25:
        conclusion = "Có dấu hiệu thao túng đáng kể."
    elif pct_abnormal > 12 or high_n > 10:
        conclusion = "Có thể có thao túng, cần kiểm tra thêm."
    else:
        conclusion = "Chưa phát hiện dấu hiệu thao túng rõ ràng."

    summary = f"""
    TỔNG HỢP BẤT THƯỜNG:
    - Tổng số ngày bất thường: {n_abnormal}
       • High  : {high_n} ngày
       • Medium: {med_n} ngày
       • Low   : {low_n} ngày
    - Tỷ lệ ngày bất thường: {pct_abnormal:.2f}% trên toàn bộ dữ liệu
    """
    return summary, conclusion

# =============================================================================
# CÁC HÀM TRỰC QUAN HÓA (Lấy từ Notebook)
# =============================================================================

def plot_price_anomalies(df, clu):
    """ Vẽ biểu đồ giá và đánh dấu bất thường (cell 26) """
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(df['Date'], df['Close'], lw=1.5, label='Giá Đóng Cửa', color='blue')

    if not clu.empty:
        clu['start_date'] = pd.to_datetime(clu['start_date'])
        clu['end_date'] = pd.to_datetime(clu['end_date'])
        for _, r in clu.iterrows():
            ax.axvspan(r['start_date'], r['end_date'], color='#90caf9', alpha=0.3)

    colors = {'High': '#e53935', 'Medium': '#fb8c00', 'Low': '#f6d32d'}
    for sev in ['High', 'Medium', 'Low']:
        m = (df['label'] == 'Yes') & (df['severity'] == sev)
        if m.any():
            ax.scatter(df.loc[m, 'Date'], df.loc[m, 'Close'], s=40, color=colors[sev], label=f'Bất thường - {sev}', zorder=3, edgecolors='black', linewidth=0.5)

    ax.set_title('Giá Đóng Cửa và Các Ngày Giao Dịch Bất Thường', fontsize=16)
    ax.set_ylabel('Giá Đóng Cửa', fontsize=12)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
    ax.grid(True, alpha=0.2)
    ax.legend()
    plt.tight_layout()
    return fig

def plot_top_anomalies(df):
    """ Vẽ biểu đồ cột ngang top 15 ngày (cell 27) """
    topn = (df.sort_values('anomaly_score', ascending=False)
              .head(15)
              .sort_values('anomaly_score'))
    labels = topn['Date'].dt.strftime('%d/%m/%Y')
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(labels, topn['anomaly_score'].values, color='#ef5350')
    ax.set_title('Top 15 Ngày Có Điểm Bất Thường Cao Nhất', fontsize=16)
    ax.set_xlabel('Điểm Bất Thường (0–1)', fontsize=12)
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    return fig

def plot_calendar_heatmap(df):
    """ Vẽ heatmap lịch (cell 27) """
    cal = df.copy()
    cal['Tháng'] = cal['Date'].dt.month
    cal['Ngày'] = cal['Date'].dt.day
    pivot = cal.pivot_table(index='Ngày', columns='Tháng', values='anomaly_score', aggfunc='mean')
    
    fig, ax = plt.subplots(figsize=(10, 6))
    c = ax.imshow(pivot.values, aspect='auto', origin='lower', cmap='YlOrRd')
    fig.colorbar(c, label='Điểm Bất Thường Trung Bình')
    ax.set_yticks(ticks=np.arange(len(pivot.index)), labels=pivot.index)
    ax.set_xticks(ticks=np.arange(len(pivot.columns)), labels=pivot.columns)
    ax.set_title('Heatmap: Điểm Bất Thường Trung Bình (Ngày x Tháng)', fontsize=16)
    ax.set_xlabel('Tháng', fontsize=12)
    ax.set_ylabel('Ngày trong tháng', fontsize=12)
    plt.tight_layout()
    return fig

def plot_z_score_heatmap(df):
    """ Vẽ heatmap 2D ret_z vs vol_z (cell 27) """
    x = df['ret_z'].clip(-5, 5).values # Clip để loại bỏ ngoại lệ cực lớn
    y = df['vol_z'].clip(-2, 10).values # Clip
    
    fig, ax = plt.subplots(figsize=(8, 6.5))
    h = ax.hist2d(x, y, bins=40, cmap='YlGnBu', cmin=1) # cmin=1 để bỏ ô trống
    fig.colorbar(h[3], label='Số Lượng Điểm Dữ Liệu')
    for thr in [1.8, 3.0]:
        ax.axvline(thr, ls='--', lw=1, color='red')
        ax.axvline(-thr, ls='--', lw=1, color='red')
        ax.axhline(thr, ls='--', lw=1, color='red')
    ax.set_xlabel('Biến Động Giá (ret_z)', fontsize=12)
    ax.set_ylabel('Biến Động Khối Lượng (vol_z)', fontsize=12)
    ax.set_title('Heatmap: Mối Quan Hệ ret_z và vol_z', fontsize=16)
    plt.tight_layout()
    return fig

def plot_severity_distribution(df):
    """ Vẽ biểu đồ cột phân bố mức độ (cell 28) """
    severity_counts = df[df['label']=='Yes']['severity'].value_counts().reindex(['High', 'Medium', 'Low'], fill_value=0)
    
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ['#e53935', '#fb8c00', '#f6d32d']
    bars = ax.bar(severity_counts.index, severity_counts.values, color=colors)
    
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height + 0.5, f'{int(height)}', 
                 ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax.set_title('Phân Bố Số Ngày Theo Mức Độ Bất Thường', fontsize=16)
    ax.set_ylabel('Số Ngày', fontsize=12)
    ax.set_yticks(np.arange(0, severity_counts.max() + 20, 10)) # Điều chỉnh trục y
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    return fig

# =============================================================================
# GIAO DIỆN STREAMLIT
# =============================================================================

st.set_page_config(layout="wide", page_title="Stock Anomaly Dashboard")
st.title(" Dashboard Phát Hiện Bất Thường Cổ Phiếu ")

# --- Sidebar ---
st.sidebar.title("Tải Dữ Liệu")
st.sidebar.info(
    """
    Tải lên file CSV của bạn. 
    File phải chứa các cột: `Date`, `Open`, `High`, `Low`, `Close`, `Volume`.
    """
)
uploaded_file = st.sidebar.file_uploader("Chọn file CSV", type="csv")

# --- Main App Body ---
if uploaded_file is not None:
    
    # 1. Tải và làm sạch dữ liệu
    with st.spinner("Đang tải và làm sạch dữ liệu..."):
        df_clean, issue_log = load_and_clean_data(uploaded_file)
    
    if df_clean is None:
        st.stop()
        
    if not issue_log.empty:
        with st.expander("Cảnh báo: Đã loại bỏ các dòng dữ liệu không hợp lệ"):
            st.warning(f"Đã tìm thấy và loại bỏ {len(issue_log)} dòng dữ liệu có vấn đề (OHLC không logic hoặc Volume=0).")
            st.dataframe(issue_log)

    # 2. Tạo đặc trưng
    with st.spinner(f"Đang tính toán {len(df_clean)} dòng dữ liệu (RSI, MACD, Z-Scores...). Vui lòng chờ..."):
        df_feat = create_features(df_clean.copy()) # .copy() để đảm bảo cache hoạt động

    # 3. Chạy mô hình
    with st.spinner("Đang chạy mô hình Rule-based và Isolation Forest..."):
        df_rule = run_rule_based(df_feat.copy())
        df_ml = run_ml_model(df_feat.copy())

    # 4. Ensemble và Gán nhãn
    with st.spinner("Đang tổng hợp điểm số và gán nhãn..."):
        df_final = ensemble_and_label(df_ml, df_rule, df_feat.copy())
    
    # 5. Gom cụm
    with st.spinner("Đang gom cụm các bất thường..."):
        clusters_df = find_clusters(df_final.copy())

    # 6. Tạo kết luận
    summary_text, conclusion = generate_summary_text(df_final)

    # --- Hiển thị Kết quả ---
    st.subheader("Kết Luận Tổng Quan")
    
    if "thao túng đáng kể" in conclusion:
        st.error(f"**KẾT LUẬN: {conclusion}**")
    elif "Có thể có" in conclusion:
        st.warning(f"**KẾT LUẬN: {conclusion}**")
    else:
        st.success(f"**KẾT LUẬN: {conclusion}**")
    
    st.text(summary_text)

    # --- Hiển thị Biểu đồ ---
    st.subheader("Trực Quan Hóa Dữ Liệu")

    # Sử dụng tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Biểu Đồ Giá", 
        "📊 Phân Bố Mức Độ", 
        "🏆 Top 15 Bất Thường", 
        "🗓️ Heatmap Lịch", 
        "📉 Heatmap Z-Score"
    ])

    with tab1:
        st.pyplot(plot_price_anomalies(df_final, clusters_df))
    
    with tab2:
        st.pyplot(plot_severity_distribution(df_final))
        
    with tab3:
        st.pyplot(plot_top_anomalies(df_final))

    with tab4:
        st.pyplot(plot_calendar_heatmap(df_final))

    with tab5:
        st.pyplot(plot_z_score_heatmap(df_final))

    # --- Hiển thị Bảng Dữ Liệu ---
    st.subheader("Bảng Dữ Liệu Chi Tiết")
    
    tab_df1, tab_df2 = st.tabs(["Các Ngày Bất Thường", "Các Cụm Bất Thường (Clusters)"])
    
    with tab_df1:
        st.info("Danh sách tất cả các ngày được gán nhãn 'Yes' (bất thường).")
        cols_to_show = ['Date', 'anomaly_score', 'label', 'severity', 'loai_bat_thuong', 'so_ngay_lien_tiep', 'Close', 'Volume', 'ret_z', 'vol_z']
        st.dataframe(df_final[df_final['label'] == 'Yes'][cols_to_show].sort_values(by='anomaly_score', ascending=False))

    with tab_df2:
        st.info("Tổng hợp các cụm (clusters) bất thường liên tiếp (cách nhau không quá 2 ngày).")
        st.dataframe(clusters_df)

else:
    st.info("Vui lòng tải lên file CSV (OHLCV) từ thanh bên để bắt đầu phân tích.")