"""
AMYRA Synthetic Data Generator
================================
Tạo dữ liệu giả lập siêu thực tế cho công ty thời trang trung niên AMYRA.
Chỉ sử dụng Pandas và Numpy.
"""

import numpy as np
import pandas as pd

# ── Seed để tái tạo được ──────────────────────────────────────────────────────
RNG = np.random.default_rng(42)

# ═══════════════════════════════════════════════════════════════════════════════
# 1. BẢNG CUSTOMERS (3,000 dòng)
# ═══════════════════════════════════════════════════════════════════════════════

N_CUSTOMERS = 3_000

# --- customer_id: định dạng CUST-XXXXX ---
raw_cust_ids = RNG.choice(range(10_000, 99_999), size=N_CUSTOMERS, replace=False)
customer_ids = [f"CUST-{i}" for i in raw_cust_ids]

# --- age: 40–75, phân phối chuẩn lệch về nhóm 45–60 ---
ages = np.clip(
    RNG.normal(loc=52, scale=8, size=N_CUSTOMERS).astype(int),
    40, 75
)

# --- gender: Nữ 85%, Nam 15% ---
genders = RNG.choice(["Female", "Male"], size=N_CUSTOMERS, p=[0.85, 0.15])

# --- location: các tỉnh/thành phố Việt Nam ---
locations = [
    "Hà Nội", "TP. Hồ Chí Minh", "Đà Nẵng", "Hải Phòng",
    "Cần Thơ", "Nha Trang", "Huế", "Đà Lạt", "Biên Hòa", "Vũng Tàu"
]
loc_weights = [0.22, 0.30, 0.10, 0.08, 0.07, 0.06, 0.05, 0.04, 0.05, 0.03]
customer_locations = RNG.choice(locations, size=N_CUSTOMERS, p=loc_weights)

# --- customer_tier: New 50%, Silver 35%, Gold 15% ---
tiers = RNG.choice(["New", "Silver", "Gold"], size=N_CUSTOMERS, p=[0.50, 0.35, 0.15])

# --- EXTRA COL 1: preferred_style (phong cách ưa thích) ---
# Thực tế: khách hàng trung niên thường thích Elegant, Classic; ít Sporty hơn
styles = ["Elegant", "Classic", "Casual", "Formal", "Sporty"]
style_weights = [0.30, 0.28, 0.22, 0.14, 0.06]
preferred_styles = RNG.choice(styles, size=N_CUSTOMERS, p=style_weights)

# --- EXTRA COL 2: lifetime_purchases (tổng đơn hàng lịch sử) ---
# New → ít, Gold → nhiều; phân phối Poisson có điều chỉnh theo tier
tier_lambda = {"New": 1.5, "Silver": 5.0, "Gold": 14.0}
lifetime_purchases = np.array([
    int(RNG.poisson(lam=tier_lambda[t])) for t in tiers
])

df_customers = pd.DataFrame({
    "customer_id":         customer_ids,
    "age":                 ages,
    "gender":              genders,
    "location":            customer_locations,
    "customer_tier":       tiers,
    "preferred_style":     preferred_styles,
    "lifetime_purchases":  lifetime_purchases,
})

# ═══════════════════════════════════════════════════════════════════════════════
# 2. BẢNG TELESALES_CALLS (10,000 dòng)
# ═══════════════════════════════════════════════════════════════════════════════

N_CALLS = 10_000

# --- Anti-join: 20% khách hàng CHƯA TỪNG nhận cuộc gọi ---
n_no_call = int(N_CUSTOMERS * 0.20)          # 600 khách hàng không có call
n_callable = N_CUSTOMERS - n_no_call         # 2,400 khách có thể bị gọi

# Chọn ngẫu nhiên 2,400 customer_id để phân bổ cuộc gọi
callable_indices = RNG.choice(N_CUSTOMERS, size=n_callable, replace=False)
callable_cust_ids = [customer_ids[i] for i in callable_indices]

# Mỗi khách hàng có thể được gọi nhiều lần; phân phối số lần gọi theo NegBinom
# Trung bình ~4.2 cuộc gọi/khách; đảm bảo tổng ≈ 10,000
call_counts_raw = RNG.negative_binomial(n=2, p=0.32, size=n_callable)
call_counts_raw = np.maximum(call_counts_raw, 1)            # ít nhất 1 cuộc
# Scale để tổng = N_CALLS
scale_factor = N_CALLS / call_counts_raw.sum()
call_counts = np.round(call_counts_raw * scale_factor).astype(int)
# Điều chỉnh sai lệch làm tròn
diff = N_CALLS - call_counts.sum()
call_counts[: abs(diff)] += int(np.sign(diff))

# Mở rộng customer_id theo số cuộc gọi
call_customer_ids = np.repeat(callable_cust_ids, call_counts)

# --- call_id: định dạng CALL-XXXXX ---
raw_call_ids = RNG.choice(range(10_000, 99_999), size=N_CALLS, replace=False)
call_ids = [f"CALL-{i}" for i in raw_call_ids]

# --- call_duration_seconds: 5–600s ---
# Dùng phân phối Gamma để có đuôi dài thực tế
raw_durations = RNG.gamma(shape=1.8, scale=80, size=N_CALLS)
call_durations = np.clip(raw_durations, 5, 600).astype(int)

# --- call_outcome: logic phụ thuộc duration ---
outcomes = []
for dur in call_durations:
    if dur < 20:
        # 99% Not Interested hoặc No Answer
        outcomes.append(
            RNG.choice(
                ["Not Interested", "No Answer", "Success", "Call Back Later"],
                p=[0.55, 0.44, 0.005, 0.005]
            )
        )
    elif dur <= 60:
        # Trung gian
        outcomes.append(
            RNG.choice(
                ["Not Interested", "No Answer", "Success", "Call Back Later"],
                p=[0.35, 0.30, 0.15, 0.20]
            )
        )
    elif dur <= 120:
        # Bắt đầu cải thiện
        outcomes.append(
            RNG.choice(
                ["Not Interested", "No Answer", "Success", "Call Back Later"],
                p=[0.20, 0.15, 0.45, 0.20]
            )
        )
    else:
        # > 120s: Success vọt lên trên 60%
        outcomes.append(
            RNG.choice(
                ["Not Interested", "No Answer", "Success", "Call Back Later"],
                p=[0.10, 0.05, 0.72, 0.13]
            )
        )

outcomes = np.array(outcomes)

# --- call_time_of_day: khung giờ gọi điện thực tế ---
# Telesales thường gọi sáng và chiều, ít buổi tối
time_slots = ["Morning (8-11h)", "Noon (11-13h)", "Afternoon (13-17h)", "Evening (17-20h)"]
time_weights = [0.30, 0.15, 0.40, 0.15]
call_times = RNG.choice(time_slots, size=N_CALLS, p=time_weights)

# --- order_value_vnd: chỉ có giá trị nếu Success ---
# Phân phối log-normal để mô phỏng chi tiêu thực tế
log_mean = np.log(900_000)     # trung bình hình học ~900k
log_std  = 0.55
raw_values = np.exp(RNG.normal(loc=log_mean, scale=log_std, size=N_CALLS))
raw_values = np.clip(raw_values, 350_000, 2_500_000)
# Làm tròn đến hàng nghìn (thực tế giá bán lẻ)
raw_values = (np.round(raw_values / 1_000) * 1_000).astype(int)

order_values = np.where(outcomes == "Success", raw_values, 0)

# --- EXTRA COL 3: agent_id – nhân viên telesales ---
# 20 nhân viên, phân phối không đều (Pareto: vài người gọi nhiều hơn)
n_agents = 20
agent_ids = [f"AGT-{i:03d}" for i in range(1, n_agents + 1)]
# Trọng số Pareto: top 5 nhân viên xử lý ~50% cuộc gọi
agent_weights = np.array([1/(i**0.8) for i in range(1, n_agents + 1)])
agent_weights /= agent_weights.sum()
assigned_agents = RNG.choice(agent_ids, size=N_CALLS, p=agent_weights)

# --- EXTRA COL 4: follow_up_required (boolean) ---
# Cuộc gọi "Call Back Later" luôn cần follow-up; Success/Not Interested thì ít hơn
follow_up = np.where(
    outcomes == "Call Back Later", True,
    np.where(outcomes == "Success",
             RNG.random(N_CALLS) < 0.08,       # 8% Success cũng cần follow-up
             RNG.random(N_CALLS) < 0.03)        # các trạng thái khác 3%
)

df_calls = pd.DataFrame({
    "call_id":                call_ids,
    "customer_id":            call_customer_ids,
    "call_duration_seconds":  call_durations,
    "call_time_of_day":       call_times,
    "call_outcome":           outcomes,
    "order_value_vnd":        order_values,
    "agent_id":               assigned_agents,
    "follow_up_required":     follow_up,
})

# ═══════════════════════════════════════════════════════════════════════════════
# 3. VALIDATION – kiểm tra ràng buộc trước khi xuất file
# ═══════════════════════════════════════════════════════════════════════════════

assert len(df_customers) == N_CUSTOMERS, "Sai số dòng customers!"
assert len(df_calls) == N_CALLS, "Sai số dòng calls!"
assert df_customers["age"].between(40, 75).all(), "age ngoài khoảng 40-75!"
assert df_calls["call_duration_seconds"].between(5, 600).all(), "duration ngoài khoảng!"
assert (df_calls.loc[df_calls["call_outcome"] != "Success", "order_value_vnd"] == 0).all(), \
    "order_value_vnd phải = 0 nếu không Success!"
assert (df_calls.loc[df_calls["call_outcome"] == "Success", "order_value_vnd"]
        .between(350_000, 2_500_000).all()), "order_value ngoài khoảng 350k-2.5M!"

# Kiểm tra anti-join: số customer_id không có trong calls phải ≥ 20%
called_ids = set(df_calls["customer_id"].unique())
all_ids    = set(df_customers["customer_id"])
never_called = all_ids - called_ids
pct_never_called = len(never_called) / N_CUSTOMERS
assert pct_never_called >= 0.18, f"Anti-join thất bại: {pct_never_called:.1%} < 18%"

# ═══════════════════════════════════════════════════════════════════════════════
# 4. XUẤT FILE CSV
# ═══════════════════════════════════════════════════════════════════════════════

df_customers.to_csv("amyra_customers.csv", index=False, encoding="utf-8-sig")
df_calls.to_csv("amyra_telesales_calls.csv", index=False, encoding="utf-8-sig")

print("✅ Hoàn tất! Đã xuất 2 file:")
print(f"   • amyra_customers.csv        — {len(df_customers):,} dòng")
print(f"   • amyra_telesales_calls.csv  — {len(df_calls):,} dòng")
print()
print("─── Thống kê nhanh (customers) ──────────────────────────")
print(f"  Age range   : {df_customers['age'].min()} – {df_customers['age'].max()} tuổi")
print(f"  Gender (F%) : {(df_customers['gender']=='Female').mean():.1%}")
print(f"  Tier split  : {df_customers['customer_tier'].value_counts().to_dict()}")
print()
print("─── Thống kê nhanh (calls) ───────────────────────────────")
print(f"  Khách chưa từng bị gọi : {len(never_called):,} ({pct_never_called:.1%})")
print(f"  Outcome split           :")
for outcome, cnt in df_calls["call_outcome"].value_counts().items():
    print(f"    {outcome:<20} {cnt:>5,}  ({cnt/N_CALLS:.1%})")
print(f"  Avg order (Success)     : {df_calls.loc[df_calls['call_outcome']=='Success','order_value_vnd'].mean():,.0f} VND")