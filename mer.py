import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
import numpy as np

# --- VERSION 2.26 MASTER CONFIGURATION ---
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

    # Economic Inputs (Updated Defaults)
    in_kwh_cost = input("One-time Cost per 1 kWh BESS (EUR) [default 50]: ")
    COST_KWH = float(in_kwh_cost.strip()) if in_kwh_cost.strip() else 50.0

    in_kw_monthly = input("Monthly Fee per 1 kW Grid Limit (EUR) [default 0.5]: ")
    MONTHLY_KW_FEE = float(in_kw_monthly.strip()) if in_kw_monthly.strip() else 0.5

    in_roi = input("Expected RoI Time (Years) [default 5]: ")
    ROI_YEARS = float(in_roi.strip()) if in_roi.strip() else 5.0

    # 1. LOAD & FILTER DATA (3-MONTH SNAPSHOT)
    df = pd.read_csv(FILE_NAME, sep=';|,', engine='python', header=None, 
                     usecols=[0, 1], names=['timestamp', 'station_1'], decimal=',')
    df['timestamp'] = pd.to_datetime(df['timestamp'].astype(str).str.strip(), dayfirst=True, errors='coerce')
    df = df.dropna(subset=['timestamp']).sort_values('timestamp')
    df['station_1'] = pd.to_numeric(df['station_1'], errors='coerce').fillna(0)
    
    start_date = df['timestamp'].min()
    end_date_3m = start_date + pd.DateOffset(months=3)
    df = df[df['timestamp'] < end_date_3m].copy()
    df['time_only'] = df['timestamp'].dt.time
    df['day_name'] = df['timestamp'].dt.day_name()
    dummy_date = pd.to_datetime('2026-01-01')

    # 2. CALCULATION LOGIC
    def get_event_counts(series, limit):
        is_above = series > limit
        event_starts = is_above & (~is_above.shift(fill_value=False))
        return pd.DataFrame({'time': df['time_only'], 'is_start': event_starts}).groupby('time')['is_start'].sum()

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

    # 4. AUDITS & ECONOMIC OPTIMIZATION
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
    pct_orig = (hours_orig / total_data_hours) * 100
    pct_bess = (hours_bess / total_data_hours) * 100

    # Optimization Sweep (3-Month Basis)
    max_peak_val = df['station_1'].max()
    sweep_limits = np.arange(BASE_FIXED_START, max_peak_val + 1, 1)
    roi_months = ROI_YEARS * 12
    
    total_3m_expenses = []
    for l in sweep_limits:
        needed_kwh = calculate_bess_for_exceedance(df['station_1'], l, 0.0)
        # BESS Investment depreciated for 3 months
        bess_expense_3m = ((needed_kwh * COST_KWH) / roi_months) * 3
        # Grid Fees for 3 months
        grid_expense_3m = l * MONTHLY_KW_FEE * 3
        total_3m_expenses.append(bess_expense_3m + grid_expense_3m)

    opt_idx = np.argmin(total_3m_expenses)
    OPT_LIMIT = sweep_limits[opt_idx]
    OPT_KWH = calculate_bess_for_exceedance(df['station_1'], OPT_LIMIT, 0.0)
    OPT_TOTAL_COST = total_3m_expenses[opt_idx]

    # 5. VISUALIZATION SUITE
    plot_times = [pd.Timestamp.combine(dummy_date, t) for t in sorted(df['time_only'].unique())]
    ev_freq_orig = get_event_counts(df['station_1'], USER_LIMIT)
    ev_freq_bess = get_event_counts(df['grid_with_bess'], USER_LIMIT)
    MAX_EV = max(ev_freq_orig.max(), ev_freq_bess.max()) + 1
    MAX_PW = max_peak_val * 1.15

    # V2_1 Baseline Profile
    fig, ax1 = plt.subplots(figsize=(14, 8))
    ax1.plot(plot_times, df.groupby('time_only')['station_1'].mean(), color='#1f77b4', alpha=0.6, label='Avg Power')
    ax1.axhline(USER_LIMIT, color='black', linestyle='--', label='Limit')
    ax1.set_ylabel('Power [kW]'); ax1.set_ylim(0, MAX_PW)
    ax2 = ax1.twinx(); ax2.fill_between(plot_times, 0, ev_freq_orig, color='#d62728', alpha=0.3, step='post', label='Events')
    ax2.set_ylabel('Event Count'); ax2.set_ylim(0, MAX_EV); ax2.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    plt.title(f'V2_1 Baseline: {hours_orig:.1f}h Over Limit ({pct_orig:.2f}%)')
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M')); fig.legend(loc="upper right", bbox_to_anchor=(0.9, 0.9)); plt.savefig('V2_1_Baseline_Events.png'); plt.close()

    # V2_2 Histogram
    plt.figure(figsize=(10, 6)); bin_edges = np.arange(0, audit_no_bess['Energy_kWh'].max() + BIN_WIDTH, BIN_WIDTH)
    plt.hist(audit_no_bess['Energy_kWh'], bins=bin_edges, alpha=0.3, label='Baseline', color='red', edgecolor='black')
    if not audit_with_bess.empty: plt.hist(audit_with_bess['Energy_kWh'], bins=bin_edges, alpha=0.7, label='BESS', color='green', edgecolor='black')
    plt.title('V2_2 Violation Energy Distribution'); plt.xlabel('kWh'); plt.legend(); plt.savefig('V2_2_Distribution.png'); plt.close()

    # V2_3 Dependency & ROI Optimum
    cap_0 = [calculate_bess_for_exceedance(df['station_1'], l, 0.0) for l in sweep_limits]
    cap_2 = [calculate_bess_for_exceedance(df['station_1'], l, 2.0) for l in sweep_limits]
    plt.figure(figsize=(12, 8)); plt.fill_between(sweep_limits, cap_0, max(cap_0)+5, color='green', alpha=0.1)
    plt.plot(sweep_limits, cap_0, color='darkred', label='0% Curve'); plt.plot(sweep_limits, cap_2, color='blue', linestyle='--', label='2% Curve')
    plt.scatter(OPT_LIMIT, OPT_KWH, color='gold', s=200, edgecolors='black', label='Economic Optimum', zorder=5)
    plt.title(f'V2_3 Optimization: {ROI_YEARS}yr ROI\nBESS: {COST_KWH}€/kWh | Grid: {MONTHLY_KW_FEE}€/kW/mo')
    plt.xlabel('Limit [kW]'); plt.ylabel('BESS [kWh]'); plt.legend(); plt.grid(True, alpha=0.1)
    plt.savefig('V2_3_Dependency_Curve.png'); plt.close()

    # V2_4 BESS Profile
    fig, ax1 = plt.subplots(figsize=(14, 8))
    ax1.plot(plot_times, df.groupby('time_only')['grid_with_bess'].mean(), color='#2ca02c', alpha=0.6)
    ax1.axhline(USER_LIMIT, color='black', linestyle='--'); ax1.set_ylabel('Power [kW]'); ax1.set_ylim(0, MAX_PW)
    ax2 = ax1.twinx(); ax2.fill_between(plot_times, 0, ev_freq_bess, color='#1f77b4', alpha=0.3, step='post')
    ax2.set_ylabel('Remaining Events'); ax2.set_ylim(0, MAX_EV); ax2.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    plt.title(f'V2_4 BESS Impact: {hours_bess:.1f}h Over Limit ({pct_bess:.2f}%)')
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M')); plt.savefig('V2_4_BESS_Events.png'); plt.close()

    # V2_5 Weekly Breakdown
    days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    if not audit_no_bess.empty:
        plt.figure(figsize=(10, 6))
        w_orig = audit_no_bess['Day'].value_counts().reindex(days_order).fillna(0)
        w_bess = audit_with_bess['Day'].value_counts().reindex(days_order).fillna(0) if not audit_with_bess.empty else pd.Series(0, index=days_order)
        plt.bar(np.arange(7)-0.2, w_orig, width=0.4, label='Baseline', color='red', alpha=0.5)
        plt.bar(np.arange(7)+0.2, w_bess, width=0.4, label='BESS', color='green')
        plt.xticks(np.arange(7), days_order); plt.title('V2_5 Weekly Violation Frequency'); plt.legend(); plt.savefig('V2_5_Weekly.png'); plt.close()

    print(f"\n--- ECONOMIC SUMMARY (3-MONTH ANALYSIS) ---")
    print(f"Optimal Grid Limit:      {OPT_LIMIT} kW")
    print(f"Optimal BESS Capacity:  {OPT_KWH:.2f} kWh")
    print(f"Min. 3m Expense (Sum):  {OPT_TOTAL_COST:,.2f} EUR")

except Exception as e:
    print(f"Error: {e}")
