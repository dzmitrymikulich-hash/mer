import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
import numpy as np

# --- VERSION 2.39 MASTER CONFIGURATION ---
FILE_NAME = 'Consumption_export_Mer_Jan_Maerz_2026_Public Site 1.csv'
INTERVAL_HOURS = 0.25 
BIN_WIDTH = 10.0
BASE_FIXED_START = 100.0 
TARGET_UTIL_THRESHOLD = 2500.0
EXCEEDANCE_TARGET = 2.0 

try:
    # 0. MANUAL INPUTS
    in_limit = input("Enter Grid Power Limit (kW) [default 100]: ")
    USER_LIMIT = float(in_limit.strip()) if in_limit.strip() else 100.0
    
    in_bess = input(f"Enter BESS Capacity (kWh) to simulate [at {USER_LIMIT}kW]: ")
    USER_BESS = float(in_bess.strip()) if in_bess.strip() else 50.0

    # 1. LOAD & FILTER DATA (Q1 SNAPSHOT)
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

    # --- CALCULATIONS: ENERGY & UTILIZATION ---
    ENERGY_Q1 = df['station_1'].sum() * INTERVAL_HOURS
    ANNUAL_ENERGY_EST = ENERGY_Q1 * 4
    MAX_PEAK_BASE = df['station_1'].max()
    TARGET_PEAK_2500 = ANNUAL_ENERGY_EST / TARGET_UTIL_THRESHOLD
    UTIL_HOURS_BASE = ANNUAL_ENERGY_EST / MAX_PEAK_BASE if MAX_PEAK_BASE > 0 else 0

    # 2. CORE LOGIC FUNCTIONS
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

    # 3. BESS SIMULATION & AUDITS
    grid_with_bess = []
    current_bess = USER_BESS
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
    
    MAX_PEAK_BESS = df['grid_with_bess'].max()
    UTIL_HOURS_BESS = ANNUAL_ENERGY_EST / MAX_PEAK_BESS if MAX_PEAK_BESS > 0 else 0

    def get_audit(series, limit):
        is_v = series > limit
        if not is_v.any(): return pd.DataFrame()
        v_id = (is_v != is_v.shift()).cumsum()
        v_rows = df[is_v].copy()
        v_rows['e'] = (series[is_v] - limit) * INTERVAL_HOURS
        res = v_rows.groupby(v_id).agg({'timestamp':['min','max','count'], series.name:['max','mean'], 'e':'sum'})
        res.columns = ['Start','End','Intervals','Peak_kW','Avg_Power_kW','Energy_kWh']
        res['Day'] = pd.to_datetime(res['Start']).dt.day_name()
        return res

    audit_no_bess = get_audit(df['station_1'], USER_LIMIT)
    audit_with_bess = get_audit(df['grid_with_bess'], USER_LIMIT)
    audit_no_bess.to_csv('V2_Audit_Baseline.csv', index=False)
    audit_with_bess.to_csv(f'V2_Audit_BESS_{int(USER_BESS)}kWh.csv', index=False)

    # 4. VISUALIZATION (7 CHARTS)
    plot_times = [pd.Timestamp.combine(dummy_date, t) for t in sorted(df['time_only'].unique())]
    ev_orig = get_event_counts(df['station_1'], USER_LIMIT)
    ev_bess = get_event_counts(df['grid_with_bess'], USER_LIMIT)
    MAX_EV = max(ev_orig.max(), ev_bess.max()) + 1
    MAX_PW = MAX_PEAK_BASE * 1.15

    # V2_1 & V2_4: PROFILES (SYNCED AXES)
    for suffix, data, events, color in [('Baseline', df['station_1'], ev_orig, '#d62728'), ('BESS', df['grid_with_bess'], ev_bess, '#1f77b4')]:
        fig, ax1 = plt.subplots(figsize=(12, 7))
        ax1.plot(plot_times, df.groupby('time_only')[data.name].mean(), color='gray', alpha=0.5, label='Avg Power')
        ax1.axhline(USER_LIMIT, color='black', linestyle='--', label=f'Limit: {USER_LIMIT}kW')
        ax1.set_ylim(0, MAX_PW); ax1.set_ylabel('Power [kW]'); ax1.set_xlabel('Time')
        ax2 = ax1.twinx(); ax2.fill_between(plot_times, 0, events, color=color, alpha=0.3, step='post', label='Exceedance Events')
        ax2.set_ylim(0, MAX_EV); ax2.set_ylabel('Events'); ax2.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
        plt.title(f'Profile: {suffix}'); ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        plt.savefig(f'V2_{"1" if suffix=="Baseline" else "4"}_{suffix}_Events.png'); plt.close()

    # V2_2: HISTOGRAM
    plt.figure(figsize=(10, 6))
    plt.hist(audit_no_bess['Energy_kWh'] if not audit_no_bess.empty else [], bins=20, alpha=0.3, color='red', label='Baseline')
    if not audit_with_bess.empty: plt.hist(audit_with_bess['Energy_kWh'], bins=20, alpha=0.7, color='green', label='BESS')
    plt.title('V2_2 Violation Distribution'); plt.xlabel('kWh'); plt.ylabel('Count'); plt.legend(); plt.savefig('V2_2_Distribution.png'); plt.close()

    # V2_3A, B, C: TRIPLE DEPENDENCY (0%, 2%, 2500h Zone)
    sweep = np.arange(BASE_FIXED_START, MAX_PEAK_BASE + 5, 2)
    c0 = np.array([calculate_bess_for_exceedance(df['station_1'], l, 0.0) for l in sweep])
    c2 = np.array([calculate_bess_for_exceedance(df['station_1'], l, EXCEEDANCE_TARGET) for l in sweep])

    # A - Baseline
    plt.figure(figsize=(10, 6)); plt.plot(sweep, c0, color='darkred', label='0%'); plt.xlabel('Limit [kW]'); plt.ylabel('BESS [kWh]')
    plt.title('V2_3A: 0% Curve'); plt.legend(); plt.savefig('V2_3A_Baseline.png'); plt.close()
    
    # B - Comparison
    plt.figure(figsize=(10, 6)); plt.plot(sweep, c0, color='darkred', label='0%'); plt.plot(sweep, c2, 'orange', linestyle='--', label='2%')
    plt.title('V2_3B: Reliability Sensitivity'); plt.xlabel('Limit [kW]'); plt.ylabel('BESS [kWh]'); plt.legend(); plt.savefig('V2_3B_Comparison.png'); plt.close()
    
    # C - FINAL STRATEGIC MAP
    plt.figure(figsize=(10, 6)); t_mask = sweep <= TARGET_PEAK_2500
    plt.fill_between(sweep[t_mask], c2[t_mask], c0[t_mask], color='cyan', alpha=0.15, label='Efficiency Zone (<=2%)')
    plt.fill_between(sweep[t_mask], c0[t_mask], max(c0)+5, color='dodgerblue', alpha=0.25, label='Zero-Risk Zone (0%)')
    plt.plot(sweep, c0, color='darkred', label='0% Curve'); plt.plot(sweep, c2, 'orange', linestyle='--', label='2% Curve')
    plt.axvline(TARGET_PEAK_2500, color='blue', linestyle='-.', label=f'2500h Line ({TARGET_PEAK_2500:.1f}kW)')
    plt.title('V2_3C: Strategic Success Map'); plt.xlabel('Limit [kW]'); plt.ylabel('BESS [kWh]'); plt.legend(); plt.savefig('V2_3C_Final.png'); plt.close()

    # V2_5: WEEKLY
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    plt.figure(figsize=(10, 6))
    if not audit_no_bess.empty:
        w_o = audit_no_bess['Day'].value_counts().reindex(days).fillna(0)
        plt.bar(np.arange(7), w_o, color='red', alpha=0.5)
    plt.xticks(np.arange(7), days); plt.title('V2_5 Weekly Frequency'); plt.xlabel('Day'); plt.ylabel('Violations'); plt.savefig('V2_5_Weekly.png'); plt.close()

    print(f"\n--- ALL FEATURES GENERATED ---")
    print(f"Annual Utilization (BESS): {UTIL_HOURS_BESS:,.0f} h")

except Exception as e:
    import traceback
    traceback.print_exc()