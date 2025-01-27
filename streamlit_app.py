import streamlit as st
import sqlite3
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

def load_data():
    conn = sqlite3.connect('trading_history.db')
    
    # Trading History
    trades_df = pd.read_sql_query('''
        SELECT 
            timestamp,
            decision,
            reason,
            btc_balance,
            krw_balance,
            btc_avg_buy_price,
            btc_krw_price
        FROM trading_history
        ORDER BY timestamp DESC
    ''', conn)
    
    # Trading Reflection
    reflection_df = pd.read_sql_query('''
        SELECT *
        FROM trading_reflection
        ORDER BY reflection_date DESC
    ''', conn)
    
    conn.close()
    return trades_df, reflection_df

def main():
    st.title('Bitcoin Trading Dashboard')
    
    trades_df, reflection_df = load_data()
    
    # Convert timestamp strings to datetime
    trades_df['timestamp'] = pd.to_datetime(trades_df['timestamp'])
    reflection_df['reflection_date'] = pd.to_datetime(reflection_df['reflection_date'])
    
    # Sidebar filters
    st.sidebar.header('Filters')
    days = st.sidebar.slider('Last N days', 1, 90, 30)
    date_threshold = datetime.now() - timedelta(days=days)
    
    filtered_trades = trades_df[trades_df['timestamp'] > date_threshold]
    
    # Main dashboard
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader('Portfolio Value Over Time')
        portfolio_value = filtered_trades.apply(
            lambda x: x['btc_balance'] * x['btc_krw_price'] + x['krw_balance'],
            axis=1
        )
        fig_portfolio = px.line(
            x=filtered_trades['timestamp'],
            y=portfolio_value,
            title='Total Portfolio Value (KRW)'
        )
        st.plotly_chart(fig_portfolio)
    
    with col2:
        st.subheader('BTC Price History')
        fig_price = px.line(
            filtered_trades,
            x='timestamp',
            y='btc_krw_price',
            title='BTC Price (KRW)'
        )
        st.plotly_chart(fig_price)
    
    # Trading Activity
    st.subheader('Trading Activity')
    col3, col4, col5 = st.columns(3)
    
    with col3:
        total_trades = len(filtered_trades)
        st.metric('Total Trades', total_trades)
    
    with col4:
        buy_count = len(filtered_trades[filtered_trades['decision'] == 'buy'])
        st.metric('Buy Orders', buy_count)
    
    with col5:
        sell_count = len(filtered_trades[filtered_trades['decision'] == 'sell'])
        st.metric('Sell Orders', sell_count)
    
    # Trading History Table
    st.subheader('Recent Trading History')
    st.dataframe(
        filtered_trades[['timestamp', 'decision', 'reason', 'btc_krw_price']]
        .sort_values('timestamp', ascending=False)
    )
    
    # Trading Reflection Analysis
    st.subheader('Trading Reflections')
    if not reflection_df.empty:
        last_reflection = reflection_df.iloc[0]
        
        col6, col7 = st.columns(2)
        with col6:
            st.metric('Success Rate', f"{last_reflection['profit_loss_ratio']:.2f}%")
            st.text('Market Conditions')
            st.write(last_reflection['market_conditions'])
        
        with col7:
            st.metric('Total Trades Analyzed', last_reflection['total_trades'])
            st.text('Lessons Learned')
            st.write(last_reflection['lessons_learned'])
        
        st.text('Strategy Adjustments')
        st.write(last_reflection['strategy_adjustments'])
        
        # Show reflection history
        st.subheader('Reflection History')
        st.dataframe(
            reflection_df[['reflection_date', 'profit_loss_ratio', 'market_conditions', 'lessons_learned']]
            .sort_values('reflection_date', ascending=False)
        )

if __name__ == '__main__':
    main()