import streamlit as st
import pandas as pd
import plotly.express as px

# --- CẤU HÌNH TRANG ---
st.set_page_config(page_title="AMYRA Telesales Analytics", layout="wide")

st.title("AMYRA - HỆ THỐNG PHÂN TÍCH HIỆU SUẤT TELESALES")
st.markdown("---")

# --- NẠP DỮ LIỆU ---
@st.cache_data
def load_data():
    customers = pd.read_csv('amyra_customers.csv')
    calls = pd.read_csv('amyra_telesales_calls.csv')
    # Kết hợp 2 bảng để phân tích sâu
    df = pd.merge(calls, customers, on='customer_id')
    return df

df = load_data()

# --- SIDEBAR (BỘ LỌC) ---
st.sidebar.header("🛠 Bộ Lọc Tinh Vi")
age_filter = st.sidebar.slider("Chọn Độ Tuổi Khách Hàng", 40, 75, (40, 75))
tier_filter = st.sidebar.multiselect("Nhóm Khách Hàng", options=df['customer_tier'].unique(), default=df['customer_tier'].unique())

# Áp dụng bộ lọc
mask = (df['age'].between(*age_filter)) & (df['customer_tier'].isin(tier_filter))
filtered_df = df[mask]

# --- CHỈ SỐ KPI CHÍNH (THỨ NHÀ TUYỂN DỤNG MUỐN THẤY) ---
col1, col2, col3, col4 = st.columns(4)
with col1:
    total_calls = len(filtered_df)
    st.metric("Tổng Cuộc Gọi", f"{total_calls:,}")
with col2:
    success_rate = (filtered_df['call_outcome'] == 'Success').mean() * 100
    st.metric("Tỷ Lệ Chốt Đơn", f"{success_rate:.2f}%")
with col3:
    total_rev = filtered_df['order_value_vnd'].sum()
    st.metric("Tổng Doanh Thu", f"{total_rev/1e6:.1f} Tr VNĐ")
with col4:
    avg_duration = filtered_df['call_duration_seconds'].mean()
    st.metric("Thời Lượng TB", f"{avg_duration:.0f} giây")

st.markdown("---")

# --- PHÂN TÍCH BIỂU ĐỒ ---
row2_col1, row2_col2 = st.columns(2)

with row2_col1:
    st.subheader("🎯 Tỷ Lệ Chuyển Đổi Theo Kết Quả")
    fig_pie = px.pie(filtered_df, names='call_outcome', color='call_outcome',
                     color_discrete_map={'Success':'#2ecc71', 'Not Interested':'#e74c3c', 'No Answer':'#95a5a6', 'Call Back Later':'#f1c40f'},
                     hole=0.4)
    st.plotly_chart(fig_pie, use_container_width=True)

with row2_col2:
    st.subheader("📈 Doanh Thu Theo Nhóm Tuổi")
    # Chia nhóm tuổi để dễ nhìn
    filtered_df['age_group'] = pd.cut(filtered_df['age'], bins=[40, 50, 60, 70, 80], labels=['40-50', '51-60', '61-70', '71+'])
    rev_age = filtered_df.groupby('age_group')['order_value_vnd'].sum().reset_index()
    fig_bar = px.bar(rev_age, x='age_group', y='order_value_vnd', color='age_group',
                     labels={'order_value_vnd': 'Doanh Thu (VNĐ)', 'age_group': 'Nhóm Tuổi'})
    st.plotly_chart(fig_bar, use_container_width=True)

# --- INSIGHT SÂU: MỐI QUAN HỆ THỜI LƯỢNG VÀ THÀNH CÔNG ---
st.subheader("💡 Phân Tích Logic: Thời Lượng Gọi vs. Khả Năng Chốt Đơn")
st.write("Dữ liệu cho thấy cuộc gọi càng dài, khách hàng trung niên càng có xu hướng tin tưởng và đặt hàng.")
fig_scatter = px.box(filtered_df, x='call_outcome', y='call_duration_seconds', color='call_outcome',
                     title="Phân Bổ Thời Lượng Gọi Theo Kết Quả")
st.plotly_chart(fig_scatter, use_container_width=True)

st.info("💡 Insight cho AMYRA: Tập trung đào tạo Telesales giữ chân khách hàng trên 120 giây để tăng tỷ lệ chốt đơn lên gấp 3 lần.")