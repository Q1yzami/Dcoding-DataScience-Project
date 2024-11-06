import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import numpy as np
from babel.numbers import format_currency

sns.set(style='dark')

# Helper function yang dibutuhkan untuk menyiapkan berbagai dataframe

def create_category_sum_df(df):
    category_sum_df = df.groupby('product_category_name_english').agg(
        total_orders=('order_item_id', 'count'), total_revenue=('price', 'sum')
    ).reset_index()
    category_sum_df = category_sum_df.sort_values(by='total_revenue', ascending=False).head(5)

    return category_sum_df

def create_monthly_sales_df(df):
    df['order_purchase_timestamp'] = pd.to_datetime(df['order_purchase_timestamp'])

    top_categories = df.groupby('product_category_name_english').agg(
        total_revenue=('price', 'sum')
    ).sort_values(by='total_revenue', ascending=False).head(3).index

    filtered_df = df[df['product_category_name_english'].isin(top_categories)]
    filtered_df['year_month'] = filtered_df['order_purchase_timestamp'].dt.to_period('M')

    monthly_sales = filtered_df.groupby(['product_category_name_english', 'year_month']).agg(
        total_sales=('price', 'sum')
    ).reset_index()

    monthly_sales['year_month'] = monthly_sales['year_month'].astype(str)
    monthly_sales = monthly_sales.sort_values(by='year_month')

    return monthly_sales

def create_seller_segmentation_df(df):
    seller_segmentation_df = df.groupby('order_id').agg(total_order=('price', 'sum')).reset_index()
    seller_segmentation_df = pd.merge(seller_segmentation_df, df[['order_id', 'seller_id']], on='order_id')
    seller_segmentation_df = seller_segmentation_df.groupby('seller_id').agg(total_spent=('total_order', 'sum')).reset_index()

    high_spender_threshold = seller_segmentation_df['total_spent'].quantile(0.75)
    low_spender_threshold = seller_segmentation_df['total_spent'].quantile(0.25)

    seller_segmentation_df['spending_segment'] = np.where(
        seller_segmentation_df['total_spent'] >= high_spender_threshold, 'High Spender',
        np.where(seller_segmentation_df['total_spent'] <= low_spender_threshold, 'Low Spender', 'Medium Spender')
    )

    return seller_segmentation_df
    
def create_seller_segmentation_sum_df(seller_segmentation_df):
    segment_counts = seller_segmentation_df['spending_segment'].value_counts().reset_index()
    segment_counts.columns = ['spending_segment', 'seller_count']

    return segment_counts

def create_avg_cust_segment_spent_df(seller_segmentation_df):
    avg_cust_segment_spent_df = seller_segmentation_df.groupby('spending_segment').agg(avg_spending=('total_spent', 'mean')).sort_values(by='avg_spending', ascending=False).reset_index()

    return avg_cust_segment_spent_df

def create_rfm_df(df):
    rfm_df = df.groupby('seller_id', as_index=False).agg({
        'order_purchase_timestamp': 'max',
        'order_id': 'nunique',
        'price': 'sum'
    })

    rfm_df.columns = ['seller_id', 'last_order_date', 'frequency', 'monetary']
    rfm_df['seller_id'] = rfm_df['seller_id'].str[:5]

    rfm_df['last_order_date'] = rfm_df['last_order_date'].dt.date
    recent_date = df['order_purchase_timestamp'].dt.date.max()
    rfm_df['recency'] = rfm_df['last_order_date'].apply(lambda x: (recent_date - x).days)
    rfm_df.drop('last_order_date', axis=1, inplace=True)

    return rfm_df

# Load cleaned data
all_df = pd.read_csv("all_data.csv")

datetime_col = ['order_purchase_timestamp', 'order_delivered_customer_date']
all_df.sort_values(by='order_purchase_timestamp', inplace=True)
all_df.reset_index(drop=True, inplace=True)

for column in datetime_col:
    all_df[column] = pd.to_datetime(all_df[column])

min_date = all_df["order_purchase_timestamp"].min()
max_date = all_df["order_purchase_timestamp"].max()
 
with st.sidebar:

    st.text_input(
        "Dasboard by:",
        "Zakiy Qiros M/Quby_z, say hii to quby",
    )
    
    start_date, end_date = st.date_input(
        label='Rentang Waktu',min_value=min_date,
        max_value=max_date,
        value=[min_date, max_date]
    )

main_df = all_df[(all_df['order_purchase_timestamp'] >= str(start_date)) &
            (all_df['order_purchase_timestamp'] <= str(end_date))]

category_sum_df =  create_category_sum_df(main_df)
monthly_sales_df = create_monthly_sales_df(main_df)
seller_segmentation_df = create_seller_segmentation_df(main_df)
seller_segmentation_sum_df = create_seller_segmentation_sum_df(seller_segmentation_df)
avg_cust_segment_spent_df = create_avg_cust_segment_spent_df(seller_segmentation_df)
rfm_df = create_rfm_df(main_df)

st.header('Dasboard by Quby_Z🤖')
st.subheader('Top Kategori')

col1, col2, col3 = st.columns(3)

with col1:
    top_category = category_sum_df.sort_values(by='total_revenue', ascending=False).iloc[0]
    st.metric('Kategori', value=top_category.name)

with col2:
    total_orders= category_sum_df.total_orders.sum()
    st.metric('Total Pembelian', value=total_orders)

with col3:
    total_revenue = format_currency(category_sum_df.total_revenue.sum(), "BRL", locale='es_CO') 
    st.metric('Total Revenue', value=total_revenue)

fig, ax = plt.subplots(figsize=(16, 8))
sns.barplot(x='product_category_name_english', y='total_revenue', data=category_sum_df, color='skyblue')
ax.set_title('Kategori produk dengan penjualan terbanyak', fontsize=18)
ax.set_xlabel('Kategori Produk', fontsize=14)
ax.set_ylabel('Total Revenue', fontsize=14)

st.pyplot(fig)

st.subheader('Pola Pembelian Kategori Terlaris')

fig, ax = plt.subplots(figsize=(16,8))
sns.lineplot(data=monthly_sales_df,
             x='year_month',
             y='total_sales',
             hue='product_category_name_english',
             marker='o'
             )
ax.set_xlabel('Bulan', fontsize=18)
ax.set_ylabel('Total Penjualan', fontsize=18)
plt.xticks(rotation=45)

st.pyplot(fig)

st.subheader('Segmentasi Pelanggan Berdasarkan Pengeluaran')

fig, ax = plt.subplots(figsize=(18,6))
plt.pie(
    seller_segmentation_sum_df['seller_count'],
    labels=seller_segmentation_sum_df['spending_segment'],
    autopct='%1.2f%%',
    startangle=140,
    colors=sns.color_palette("viridis", len(seller_segmentation_sum_df))
)

st.pyplot(fig)

st.subheader('Rata-rata Pengeluaran Pelanggan Berdasarkan Segmentasi')

fig, ax = plt.subplots(figsize=(18, 12))
sns.barplot(data=avg_cust_segment_spent_df,
            x='spending_segment', y='avg_spending',
            hue='spending_segment',
            palette='viridis'
            )
ax.set_xlabel('Segmentasi pelanggan', fontsize=18)
ax.set_ylabel('Rata-rata pengeluaran', fontsize=18)

st.pyplot(fig)

st.subheader('Best Seller berdasarkan parameter RFM')

col1, col2, col3 = st.columns(3)

with col1:
    avg_recency = round(rfm_df.recency.mean(), 1)
    st.metric("Rata-rata Recency(hari)", value=avg_recency)

with col2:
    avg_frequency = round(rfm_df.frequency.mean(), 2)
    st.metric("Rata-rata Frequency", value=avg_frequency)

with col3:
    avg_monetary = format_currency(rfm_df.monetary.mean(), "BRL", locale='es_CO') 
    st.metric("Rata-rata Monetary", value=avg_monetary)

fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(16,8))

sns.barplot(y="recency", x="seller_id", data=rfm_df.sort_values(by="recency", ascending=True).head(5), ax=ax[0])
ax[0].set_ylabel(None)
ax[0].set_xlabel("seller_id", fontsize=18)
ax[0].set_title("By Recency (days)", loc="center", fontsize=20)
ax[0].tick_params(axis='y', labelsize=10)
ax[0].tick_params(axis='x', labelsize=10)

sns.barplot(y="frequency", x="seller_id", data=rfm_df.sort_values(by="frequency", ascending=False).head(5), ax=ax[1])
ax[1].set_ylabel(None)
ax[1].set_xlabel("seller_id", fontsize=18)
ax[1].set_title("By Frequency", loc="center", fontsize=20)
ax[1].tick_params(axis='y', labelsize=10)
ax[1].tick_params(axis='x', labelsize=10)

sns.barplot(y="monetary", x="seller_id", data=rfm_df.sort_values(by="monetary", ascending=False).head(5), ax=ax[2])
ax[2].set_ylabel(None)
ax[2].set_xlabel("seller_id", fontsize=18)
ax[2].set_title("By Monetary", loc="center", fontsize=20)
ax[2].tick_params(axis='y', labelsize=10)
ax[2].tick_params(axis='x', labelsize=10)

st.pyplot(fig)

st.caption('Copyright © Quby_z 2024')