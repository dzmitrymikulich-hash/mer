import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
import numpy as np

# --- VERSION 2.20 MASTER CONFIGURATION ---
FILE_NAME = 'Consumption_export_Mer_Jan_Maerz_2026_Public Site 1.csv'
INTERVAL_HOURS = 0.25 
BIN_WIDTH = 10.0
BASE_FIXED_START = 100.0 

try:
    # 0. MANUAL INPUTS
    in_limit = input("Enter Grid Power Limit (kW) [default 100]: ")
    USER_LIMIT = float(in_limit.strip()) if in_limit.strip() else 100.0
    
    in_bess = input(f"Enter BESS Capacity (kWh) to simulate [at {USER_LIMIT}kW]: ")
    USER_BESS = float(in_bess.strip()) if in_bess.strip() else 50.0

    # 1. LOAD DATA
    df = pd.read_csv(FILE_NAME, sep=';|,', engine='python', header=None, 
                     usecols=[0, 1], names=['timestamp', 'station_1'], decimal=',')
    df['timestamp'] = pd.to_datetime(df['timestamp'].astype(str).str.strip(), dayfirst=True, errors='coerce')
    df = df.dropna(subset=['timestamp']).sort_values('timestamp')
    df['station_1'] = pd.to_numeric(df['station_1'], errors='coerce').fillna(0)
    
    # 3-Month Filter
    start_date = df['timestamp'].min()
    end_date_3m = start_date + pd.DateOffset(months=3)
    df = df[df['timestamp'] < end_date_3m].copy()
    
    df['time_only'] = df['timestamp'].dt.time
    df['day_name'] = df['timestamp'].dt.day_name()
    dummy_date = pd.to_datetime('2026-01-01')

    # 2. CALCULATION FUNCTIONS
    def get_event_counts_by_time(series, limit):
        is_above = series > limit
        event_starts = is_above & (~is_above.shift(fill_value=False))
        temp_df = pd.DataFrame({'time': df['time_only'], 'is_start': event_starts})
        return temp_df.groupby('time')['is_start'].sum()

    def calculate_bess_for_exceedance(power_series, limit, target_pct):
        target_count = int(round((target_pct / 100.0) * len(power_series)))
        is_above = power_series > limit
        if not is_above.any(): return 0.0
        v_ids = (is_above != is_above.shift()).cumsum()
        groups = (power_series[is_above] - limit) * INTERVAL_HOURS
        all_prefix_sums = []
        for _, group in groups.groupby(v_ids):
            all_prefix_sums.extend(group.cumsum().tolist())
        if len(all_prefix_sums) <= target_count: return 0.0
        all_prefix_sums.sort(reverse=True)
        return all_prefix_sums[target_count]

    # 3. BESS SIMULATION
    current_bess = USER_BESS
    grid_with_bess = []
    for val in df['station_1']:
        if val > USER_LIMIT:
            energy_needed = (val - USER_LIMIT) * INTERVAL_HOURS
            if current_bess >= energy_needed:
                current_bess -= energy_needed
                grid_with_bess.append(USER_LIMIT)
            else:
                shortfall = energy_needed - current_bess
                current_bess = 0
                grid_with_bess.append(USER_LIMIT + (shortfall / INTERVAL_HOURS))
        else:
            current_bess = USER_BESS 
            grid_with_bess.append(val)
    df['grid_with_bess'] = np.array(grid_with_bess)

    # 4. DATA AUDITS & CSV EXPORTS
    def get_full_audit(series, limit):
        is_v = series > limit
        if not is_v.any(): return pd.DataFrame()
        v_id = (is_v != is_v.shift()).cumsum()
        v_rows = df[is_v].copy()
        v_rows['e'] = (series[is_v] - limit) * INTERVAL_HOURS
        res = v_rows.groupby(v_id).agg({'timestamp':['min','max','count'], series.name:['max','mean'], 'e':'sum'})
        res.columns = ['Start','End','Intervals','Peak_kW','Avg_Power_kW','Energy_kWh']
        res['Duration_Min'] = res['Intervals'] * 15
        res['Day'] = pd.to_datetime(res['Start']).dt.day_name()
        return res

    audit_no_bess = get_full_audit(df['station_1'], USER_LIMIT)
    audit_with_bess = get_full_audit(df['grid_with_bess'], USER_LIMIT)
    audit_no_bess.to_csv('V2_Audit_Baseline.csv', index=False)
    audit_with_bess.to_csv(f'V2_Audit_BESS_{int(USER_BESS)}kWh.csv', index=False)

    total_data_hours = len(df) * INTERVAL_HOURS
    hours_orig = (df['station_1'] > USER_LIMIT).sum() * INTERVAL_HOURS
    hours_bess = (df['grid_with_bess'] > USER_LIMIT).sum() * INTERVAL_HOURS
    pct_orig = (hours_orig / total_data_hours) * 100 if total_data_hours > 0 else 0
    pct_bess = (hours_bess / total_data_hours) * 100 if total_data_hours > 0 else 0

    # 5. VISUALIZATION SUITE
    plot_times = [pd.Timestamp.combine(dummy_date, t) for t in sorted(df['time_only'].unique())]
    event_freq_orig = get_event_counts_by_time(df['station_1'], USER_LIMIT)
    event_freq_bess = get_event_counts_by_time(df['grid_with_bess'], USER_LIMIT)
    MAX_EVENTS_Y = max(event_freq_orig.max(), event_freq_bess.max()) + 1
    MAX_POWER_Y = df['station_1'].max() * 1.15

    # V2_1: Baseline Frequency
    fig, ax1 = plt.subplots(figsize=(14, 8))
    ax1.plot(plot_times, df.groupby('time_only')['station_1'].mean(), color='#1f77b4', label='Avg Power (kW)', alpha=0.6)
    ax1.axhline(y=USER_LIMIT, color='black', linestyle='--', linewidth=1.5, label='Power Limit')
    ax1.set_ylabel('Average Power [kW]'); ax1.set_ylim(0, MAX_POWER_Y); ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax2 = ax1.twinx(); ax2.fill_between(plot_times, 0, event_freq_orig, color='#d62728', alpha=0.3, step='post', label='Exceedance Events')
    ax2.set_ylabel('Exceedance Events'); ax2.set_ylim(0, MAX_EVENTS_Y); ax2.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    plt.title(f'Baseline Frequency: {hours_orig:.1f}h Over Limit ({pct_orig:.2f}%)')
    fig.legend(loc="upper right", bbox_to_anchor=(0.9, 0.9)); plt.savefig('V2_1_Baseline_Events.png'); plt.close()

    # V2_2: Distribution Histogram
    plt.figure(figsize=(12, 7))
    max_e = audit_no_bess['Energy_kWh'].max() if not audit_no_bess.empty else 10
    bin_edges = np.arange(0, max_e + BIN_WIDTH + 1, BIN_WIDTH)
    plt.hist(audit_no_bess['Energy_kWh'] if not audit_no_bess.empty else [], bins=bin_edges, alpha=0.3, label='Baseline', color='red', edgecolor='black')
    if not audit_with_bess.empty:
        plt.hist(audit_with_bess['Energy_kWh'], bins=bin_edges, alpha=0.7, label=f'{USER_BESS}kWh BESS', color='green', edgecolor='black')
    plt.title('Violation Distribution Comparison'); plt.xlabel('Excess Energy [kWh]'); plt.legend(); plt.savefig('V2_2_Distribution_Comparison.png'); plt.close()

    # V2_3: Dependency Curve
    max_p = df['station_1'].max(); sweep_limits = np.arange(BASE_FIXED_START, max_p + 5, 2)
    cap_0 = [calculate_bess_for_exceedance(df['station_1'], l, 0.0) for l in sweep_limits]
    cap_2 = [calculate_bess_for_exceedance(df['station_1'], l, 2.0) for l in sweep_limits]
    plt.figure(figsize=(12, 8))
    plt.fill_between(sweep_limits, cap_0, max(cap_0) + 10, color='green', alpha=0.15, label='Safe Zone')
    plt.plot(sweep_limits, cap_0, color='darkred', linewidth=2, label='0% Exceedance')
    plt.plot(sweep_limits, cap_2, color='blue', linewidth=2, linestyle='--', label='2% Exceedance')
    marker_steps = np.arange(BASE_FIXED_START, max_p + 1, 10)
    for ms in marker_steps:
        m0 = calculate_bess_for_exceedance(df['station_1'], ms, 0.0); m2 = calculate_bess_for_exceedance(df['station_1'], ms, 2.0)
        plt.plot(ms, m0, 'ko', markersize=4); plt.annotate(f"{m0:.1f}", (ms, m0), textcoords="offset points", xytext=(0, 10), ha='center', fontsize=8, color='darkred', fontweight='bold')
        if m2 > 0: plt.plot(ms, m2, 'o', color='blue', markersize=3); plt.annotate(f"{m2:.1f}", (ms, m2), textcoords="offset points", xytext=(0, -15), ha='center', fontsize=8, color='blue')
    plt.title('V2_3 Dependency Curve (0% vs 2%)'); plt.grid(True, alpha=0.1); plt.savefig('V2_3_Dependency_Curve.png'); plt.close()

    # V2_4: BESS Frequency
    fig, ax1 = plt.subplots(figsize=(14, 8))
    ax1.plot(plot_times, df.groupby('time_only')['grid_with_bess'].mean(), color='#2ca02c', label='Avg Power (BESS)', alpha=0.6)
    ax1.axhline(y=USER_LIMIT, color='black', linestyle='--', linewidth=1.5, label='Power Limit')
    ax1.set_ylabel('Average Power [kW]'); ax1.set_ylim(0, MAX_POWER_Y); ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax2 = ax1.twinx(); ax2.fill_between(plot_times, 0, event_freq_bess, color='#1f77b4', alpha=0.3, step='post', label='Remaining Events')
    ax2.set_ylabel('Exceedance Events'); ax2.set_ylim(0, MAX_EVENTS_Y); ax2.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    plt.title(f'BESS Frequency: {hours_bess:.1f}h Over Limit ({pct_bess:.2f}%)')
    fig.legend(loc="upper right", bbox_to_anchor=(0.9, 0.9)); plt.savefig('V2_4_BESS_Events.png'); plt.close()

    # --- NEW V2_5: WEEKLY VIOLATION ANALYSIS ---
    days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    if not audit_no_bess.empty:
        weekly_orig = audit_no_bess['Day'].value_counts().reindex(days_order).fillna(0)
        weekly_bess = audit_with_bess['Day'].value_counts().reindex(days_order).fillna(0) if not audit_with_bess.empty else pd.Series(0, index=days_order)
        
        plt.figure(figsize=(12, 7))
        x_indices = np.arange(len(days_order))
        plt.bar(x_indices - 0.2, weekly_orig, width=0.4, label='Baseline Violations', color='#d62728', alpha=0.6)
        plt.bar(x_indices + 0.2, weekly_bess, width=0.4, label=f'With {USER_BESS}kWh BESS', color='#2ca02c', alpha=0.8)
        
        plt.xticks(x_indices, days_order)
        plt.ylabel('Number of Unique Violation Events')
        plt.title('Violation Frequency by Day of Week')
        plt.legend(); plt.grid(axis='y', alpha=0.3); plt.savefig('V2_5_Weekly_Analysis.png'); plt.close()

    print(f"\nALL FILES GENERATED: {hours_orig:.1f}h baseline vs {hours_bess:.1f}h with BESS.")

except Exception as e:
    print(f"Error: {e}")
    