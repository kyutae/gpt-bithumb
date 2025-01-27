import base64
import os
from dotenv import load_dotenv
load_dotenv()
import python_bithumb
import pandas as pd
import numpy as np
import requests
from ta.momentum import RSIIndicator, StochRSIIndicator
from ta.trend import SMAIndicator, EMAIndicator, MACD
from ta.volatility import BollingerBands
from openai import OpenAI
import json
import time
from datetime import datetime
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from PIL import Image
import io
from datetime import timedelta
import sqlite3
import jwt 
import uuid
import hashlib
from urllib.parse import urlencode
import schedule
import logging

class TradingDatabase:
    def __init__(self, db_name="trading_history.db"):
        self.db_name = db_name
        self.setup_database()
    
    def setup_database(self):
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()
        
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS trading_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME NOT NULL,
            decision TEXT NOT NULL,
            reason TEXT,
            btc_balance REAL,
            krw_balance REAL,
            btc_avg_buy_price REAL,
            btc_krw_price REAL
        )
        ''')
        # 새로운 reflection 테이블 추가
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS trading_reflection (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            reflection_date DATETIME NOT NULL,
            period_start DATETIME NOT NULL,
            period_end DATETIME NOT NULL,
            total_trades INTEGER,
            profit_loss_ratio REAL,
            successful_indicators TEXT,
            failed_indicators TEXT,
            market_conditions TEXT,
            lessons_learned TEXT,
            strategy_adjustments TEXT
        )''')
        
        conn.commit()
        conn.close()

    def analyze_performance(self, period_days=7):
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()
        
        period_start = datetime.now() - timedelta(days=period_days)
        period_trades = cursor.execute('''
            SELECT * FROM trading_history 
            WHERE timestamp > ?
            ORDER BY timestamp
        ''', (period_start,)).fetchall()
        
        if not period_trades:
            return None
            
        # 수익률 계산
        initial_btc = period_trades[0][4]  # btc_balance
        initial_krw = period_trades[0][5]  # krw_balance
        final_btc = period_trades[-1][4]
        final_krw = period_trades[-1][5]
        final_btc_price = period_trades[-1][7]  # btc_krw_price
        
        initial_total = initial_btc * period_trades[0][7] + initial_krw
        final_total = final_btc * final_btc_price + final_krw
        profit_loss_ratio = ((final_total - initial_total) / initial_total) * 100
        
        # 성공/실패한 지표 분석
        decisions = [trade[2] for trade in period_trades]
        success_count = sum(1 for i in range(len(period_trades)-1) 
                          if (decisions[i] == 'buy' and period_trades[i+1][7] > period_trades[i][7]) or
                             (decisions[i] == 'sell' and period_trades[i+1][7] < period_trades[i][7]))
        
        return {
            'period_start': period_start,
            'period_end': datetime.now(),
            'total_trades': len(period_trades),
            'profit_loss_ratio': profit_loss_ratio,
            'success_rate': success_count / len(period_trades) if period_trades else 0
        }

    def generate_reflection(self, analysis_result):
        if not analysis_result:
            return None
                
        client = OpenAI()
        
        prompt = f"""
        Based on the following trading performance data:
        - Period: {analysis_result['period_start']} to {analysis_result['period_end']}
        - Total trades: {analysis_result['total_trades']}
        - Profit/Loss: {analysis_result['profit_loss_ratio']:.2f}%
        - Success rate: {analysis_result['success_rate']*100:.2f}%
        
        Please analyze and provide in JSON format:
        {{"market_conditions": "...",
        "successful_indicators": "...",
        "failed_indicators": "...",
        "lessons_learned": "...",
        "strategy_adjustments": "..."}}
        """
        
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        
        reflection = json.loads(response.choices[0].message.content)
        
        # DB에 반성 내용 저장
        with sqlite3.connect(self.db_name) as conn:
            cursor = conn.cursor()
            cursor.execute('''
            INSERT INTO trading_reflection (
                reflection_date, period_start, period_end, total_trades,
                profit_loss_ratio, successful_indicators, failed_indicators,
                market_conditions, lessons_learned, strategy_adjustments
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                datetime.now(),
                analysis_result['period_start'],
                analysis_result['period_end'],
                analysis_result['total_trades'],
                analysis_result['profit_loss_ratio'],
                reflection['successful_indicators'],
                reflection['failed_indicators'],
                reflection['market_conditions'],
                reflection['lessons_learned'],
                reflection['strategy_adjustments']
            ))
        
        return reflection
    
    def record_trade(self, decision, reason, btc_balance, krw_balance, 
                    btc_avg_buy_price, btc_krw_price):
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()
        
        cursor.execute('''
        INSERT INTO trading_history (
            timestamp, decision, reason, btc_balance, krw_balance,
            btc_avg_buy_price, btc_krw_price
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            datetime.now(),
            decision,
            reason,
            btc_balance,
            krw_balance,
            btc_avg_buy_price,
            btc_krw_price
        ))
        
        conn.commit()
        conn.close()

    def get_last_n_trades(self, n=5):
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()
        
        cursor.execute('''
        SELECT * FROM trading_history 
        ORDER BY timestamp DESC 
        LIMIT ?
        ''', (n,))
        
        trades = cursor.fetchall()
        conn.close()
        
        return trades

class BithumbAPI:
    def __init__(self, access_key, secret_key, decision, reason):
        self.decision = decision
        self.reason = reason
        self.access_key = access_key
        self.secret_key = secret_key
        self.api_url = 'https://api.bithumb.com'
        self.db = TradingDatabase()

    def _get_token_headers(self, params=None):
        payload = {
            'access_key': self.access_key,
            'nonce': str(uuid.uuid4()),
            'timestamp': round(time.time() * 1000)
        }
        
        if params:
            query = urlencode(params).encode()
            hash_obj = hashlib.sha512()
            hash_obj.update(query)
            query_hash = hash_obj.hexdigest()
            
            payload.update({
                'query_hash': query_hash,
                'query_hash_alg': 'SHA512'
            })

        jwt_token = jwt.encode(payload, self.secret_key)
        authorization_token = 'Bearer {}'.format(jwt_token)
        return {'Authorization': authorization_token}

    def get_account_info(self):
        try:
            response = requests.get(
                self.api_url + '/v1/accounts', 
                headers=self._get_token_headers()
            )
            if response.status_code == 200:
                return response.json()
            else:
                return f"Error: {response.status_code}, {response.text}"
        except Exception as err:
            return f"Exception occurred: {err}"

    def get_current_price(self, currency='BTC'):
        try:
            url = f"{self.api_url}/public/ticker/{currency}_KRW"
            response = requests.get(url)
            if response.status_code == 200:
                data = response.json()
                return float(data['data']['closing_price'])
            return None
        except Exception as e:
            print(f"Error getting current price: {e}")
            return None

    def print_balance_info(self, save_to_db=True):
        account_info = self.get_account_info()
        
        krw_balance = next((item['balance'] for item in account_info if item['currency'] == 'KRW'), '0')
        btc_info = next((item for item in account_info if item['currency'] == 'BTC'), None)
        
        if btc_info:
            btc_balance = float(btc_info['balance'])
            btc_avg_price = float(btc_info['avg_buy_price'])
            current_price = self.get_current_price()
            if current_price is None:
                current_price = btc_avg_price
            
            btc_in_krw = btc_balance * btc_avg_price
            
            print(f"=== 잔액 정보 ===")
            print(f"KRW 잔액: {float(krw_balance):,.2f} 원")
            print(f"BTC 잔액: {btc_balance:.8f} BTC")
            print(f"BTC 평균 매수가: {btc_avg_price:,.0f} 원")
            print(f"BTC 원화 환산액: {btc_in_krw:,.2f} 원")
            
            if save_to_db:
                self.db.record_trade(
                    decision=self.decision,
                    reason=self.reason,
                    btc_balance=btc_balance,
                    krw_balance=float(krw_balance),
                    btc_avg_buy_price=btc_avg_price,
                    btc_krw_price=current_price
                )
                
                # Print recent history
                print("\n=== Recent History ===")
                recent_trades = self.db.get_last_n_trades()
                for trade in recent_trades:
                    print(f"[{trade[1]}] Decision: {trade[2]}, BTC: {trade[4]:.8f}, KRW: {trade[5]:,.0f}")
    
def record_trading_decision(db, result, bithumb):
    """Record trading decision and balances to database"""
    try:
        # Get current balances and prices
        current_btc = float(bithumb.get_balance("BTC"))
        current_krw = float(bithumb.get_balance("KRW"))
        
        # BTC-KRW 형식으로 심볼 수정
        current_price = float(python_bithumb.get_current_price("BTC-KRW"))
        if current_price is None:
            print("Warning: Could not fetch current price, retrying...")
            time.sleep(1)
            current_price = float(python_bithumb.get_current_price("BTC-KRW"))
            
        avg_buy_price = float(bithumb.get_avg_buy_price("BTC"))
        
        # Record to database
        db.record_trade(
            decision=result['decision'],
            reason=result['reason'],
            btc_balance=current_btc,
            krw_balance=current_krw,
            btc_avg_buy_price=avg_buy_price,
            btc_krw_price=current_price
        )
        
        # Print recent trading history
        print("\n=== Recent Trading History ===")
        recent_trades = db.get_last_n_trades(5)
        for trade in recent_trades:
            print(f"[{trade[1]}] Decision: {trade[2]}, BTC: {trade[4]:.8f}, KRW: {trade[5]:,.0f}")
        
        # Print trading summary
        summary = db.get_trade_summary()
        print("\n=== Trading Summary ===")
        print(f"Total Trades: {summary['total_trades']}")
        print(f"Buys: {summary['total_buys']}")
        print(f"Sells: {summary['total_sells']}")
        print(f"Holds: {summary['total_holds']}")
        
    except Exception as e:
        print(f"Error in record_trading_decision: {e}")
        print("Failed to record trading decision to database")

def get_bitcoin_news():
    api_key = os.getenv("SERPAPI_KEY")
    search_url = "https://serpapi.com/search.json"
    params = {
        "q": "bitcoin",
        "tbm": "nws",
        "api_key": api_key
    }

    try:
        response = requests.get(search_url, params=params)
        response.raise_for_status()
        news_data = response.json()
        headlines_with_dates = [
            {"title": article['title'], "date": article.get('date')}
            for article in news_data.get('news_results', [])
        ]
        return headlines_with_dates
    except (requests.exceptions.RequestException, KeyError) as e:
        print(f"Error fetching Bitcoin news: {e}")
        return []

def get_fear_and_greed_index():
    try:
        response = requests.get('https://api.alternative.me/fng/')
        data = response.json()
        return {
            'value': int(data['data'][0]['value']),
            'classification': data['data'][0]['value_classification']
        }
    except Exception as e:
        print(f"Fear and Greed Index API 오류: {e}")
        return None

def add_custom_indicators(df):
    rsi_indicator = RSIIndicator(close=df['Close'])
    df['RSI'] = rsi_indicator.rsi()
    
    stoch_rsi = StochRSIIndicator(close=df['Close'])
    df['StochRSI'] = stoch_rsi.stochrsi()
    
    sma_20 = SMAIndicator(close=df['Close'], window=20)
    df['SMA_20'] = sma_20.sma_indicator()
    
    ema_50 = EMAIndicator(close=df['Close'], window=50)
    df['EMA_50'] = ema_50.ema_indicator()
    
    macd = MACD(close=df['Close'])
    df['MACD'] = macd.macd()
    df['MACD_signal'] = macd.macd_signal()
    
    bollinger = BollingerBands(close=df['Close'])
    df['BB_high'] = bollinger.bollinger_hband()
    df['BB_low'] = bollinger.bollinger_lband()
    
    return df

def capture_chart_as_base64(x, y, width, height):
    options = webdriver.ChromeOptions()
    options.add_argument('--start-maximized')
    options.add_argument('--disable-gpu')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    
    driver = webdriver.Chrome(options=options)
    
    try:
        driver.get('https://www.bithumb.com/react/trade/order/BTC-KRW')
        
        try:
            popup_button = WebDriverWait(driver, 10).until(
                EC.element_to_be_clickable((By.XPATH, 
                '//*[@id="popUpContainer"]/div/div/div/div/div/div/button/span'))
            )
            popup_button.click()
        except Exception as e:
            print(f"팝업 처리 중 오류 발생: {str(e)}")
        
        time.sleep(3)
        
        png = driver.get_screenshot_as_png()
        screenshot = Image.open(io.BytesIO(png))
        
        scaling_factor = screenshot.size[0] / driver.execute_script('return window.innerWidth')
        
        scaled_x = int(x * scaling_factor)
        scaled_y = int(y * scaling_factor)
        scaled_width = int(width * scaling_factor)
        scaled_height = int(height * scaling_factor)
        
        cropped_image = screenshot.crop((
            scaled_x, scaled_y,
            scaled_x + scaled_width,
            scaled_y + scaled_height
        ))
        
        new_size = (int(scaled_width/2), int(scaled_height/2))
        cropped_image = cropped_image.resize(new_size, Image.Resampling.LANCZOS)
        
        if cropped_image.mode == 'RGBA':
            cropped_image = cropped_image.convert('RGB')
        
        buffered = io.BytesIO()
        cropped_image.save(buffered, format="JPEG", quality=70, optimize=True)
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        return img_str
        
    except Exception as e:
        print(f"처리 중 오류가 발생했습니다: {str(e)}")
        return None
    
    finally:
        driver.quit()

def ai_trading():

    with open('trading_expert.txt', 'r', encoding='utf-8') as f:
        expert_knowledge = f.read()

    db = TradingDatabase()

    # 거래 실행 전에 이전 반성 내용 검토
    conn = sqlite3.connect(db.db_name)
    cursor = conn.cursor()
    latest_reflection = cursor.execute('''
        SELECT * FROM trading_reflection 
        ORDER BY reflection_date DESC LIMIT 1
    ''').fetchone()
    
    if latest_reflection:
        # GPT-4 프롬프트에 이전 반성 내용 추가
        expert_knowledge += f"\n\nPrevious Trading Reflection:\n"
        expert_knowledge += f"Market Conditions: {latest_reflection[8]}\n"
        expert_knowledge += f"Successful Indicators: {latest_reflection[6]}\n"
        expert_knowledge += f"Failed Indicators: {latest_reflection[7]}\n"
        expert_knowledge += f"Strategy Adjustments: {latest_reflection[10]}\n"

    df = python_bithumb.get_ohlcv("KRW-BTC", interval="day", count=30)
    
    df = df.rename(columns={
        'open': 'Open', 
        'close': 'Close', 
        'high': 'High', 
        'low': 'Low', 
        'volume': 'Volume'
    })
    
    for col in ['Open', 'Close', 'High', 'Low', 'Volume']:
        df[col] = df[col].astype(float)
    
    df_with_indicators = add_custom_indicators(df)
    fng_index = get_fear_and_greed_index()
    news_headlines = get_bitcoin_news()
    df_with_indicators.index = df_with_indicators.index.astype(str)
    
    chart_image = capture_chart_as_base64(
        x=700,
        y=270,
        width=950,
        height=380
    )

    client = OpenAI()
    response = client.chat.completions.create(
        model="gpt-4o",
        response_format={"type": "json_object"},
        stream=False,
        seed=123,
        temperature=0,
        top_p=1,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an expert in Bitcoin investing.\n\n"
                    f"Expert Trading Knowledge:\n{expert_knowledge}\n\n"
                    "Analyze the data using this expert knowledge along with:\n"
                    "- Chart data and technical indicators\n" 
                    "- Fear and Greed Index\n"
                    "- Latest Bitcoin news headlines\n"
                    "- Real-time chart image\n\n"
                    "Required JSON Response Format:\n"
                    "{\n"
                    "  \"decision\": \"buy|sell|hold\",\n"
                    "  \"reason\": \"Detailed explanation using expert knowledge\",\n"
                    "  \"risk_level\": \"low|medium|high\",\n"
                    "  \"confidence_score\": \"1-10 scale\",\n"
                    "  \"indicators\": {\n"
                    "    \"rsi\": \"RSI value and interpretation\",\n"
                    "    \"macd\": \"MACD signal and trend\",\n"
                    "    \"bollinger\": \"Position relative to bands\",\n"
                    "    \"volume_analysis\": \"Volume trend interpretation\"\n"
                    "  },\n"
                    "  \"market_sentiment\": {\n"
                    "    \"fear_greed\": \"Index interpretation\",\n"
                    "    \"news\": \"Overall news sentiment\",\n"
                    "    \"institutional_flow\": \"Institutional buying/selling pressure\"\n"
                    "  },\n"
                    "  \"trade_parameters\": {\n"
                    "    \"entry_price\": \"Suggested entry price range\",\n"
                    "    \"stop_loss\": \"Recommended stop loss level\",\n"
                    "    \"take_profit\": \"Target profit levels\",\n"
                    "    \"position_size\": \"Recommended position size %\"\n"
                    "  }\n"
                    "}"
                )
            },
            {
                "role": "user", 
                "content": json.dumps({
                    "chart_data": df_with_indicators.to_dict(),
                    "fear_and_greed_index": fng_index,
                    "news_headlines": news_headlines,
                    "chart_image": chart_image
                })
            }
        ]
    )

    result = json.loads(response.choices[0].message.content)
    
    if result["risk_level"] == "high" and result["confidence_score"] < 7:
        print("\n### Trade Skipped: High Risk, Low Confidence ###")
        return
        
    access = os.getenv("BITHUMB_ACCESS_KEY")
    secret = os.getenv("BITHUMB_SECRET_KEY")
    bithumb = python_bithumb.Bithumb(access, secret)

    # Record trading decision and balances
    record_trading_decision(db, result, bithumb)

    my_krw = bithumb.get_balance("KRW")
    my_btc = bithumb.get_balance("BTC")

    print("\n=== AI Trading Analysis ===")
    print(f"Decision: {result['decision'].upper()}")
    print(f"Risk Level: {result['risk_level'].upper()}")
    print(f"Confidence Score: {result['confidence_score']}/10")
    print(f"Reason: {result['reason']}")
    
    print("\nTrade Parameters:")
    for key, value in result['trade_parameters'].items():
        print(f"- {key.replace('_',' ').upper()}: {value}")
        
    print("\nTechnical Indicators:")
    for key, value in result['indicators'].items():
        print(f"- {key.upper()}: {value}")
    
    print("\nMarket Sentiment:")
    for key, value in result['market_sentiment'].items():
        print(f"- {key.replace('_',' ').upper()}: {value}")

    if result["decision"] == "buy":
        position_size = float(result["trade_parameters"]["position_size"].rstrip('%')) / 100
        trade_amount = my_krw * position_size * 0.997
        
        if trade_amount > 10000:
            print(f"\n### Buy Order Executed: {position_size*100}% of available KRW ###")
            #bithumb.buy_market_order("KRW-BTC", trade_amount)
        else:
            print("\n### Buy Order Failed: Insufficient KRW ###")
            
    elif result["decision"] == "sell":
        position_size = float(result["trade_parameters"]["position_size"].rstrip('%')) / 100
        current_price = python_bithumb.get_current_price("KRW-BTC")
        trade_amount = my_btc * position_size
        
        if trade_amount * current_price > 10000:
            print(f"\n### Sell Order Executed: {position_size*100}% of BTC holdings ###")
            #bithumb.sell_market_order("KRW-BTC", trade_amount)
        else:
            print("\n### Sell Order Failed: Insufficient BTC ###")
    else:
        print("\n### Hold Position ###")

    api = BithumbAPI(access, secret, result["decision"], result["reason"])
    api.print_balance_info()

    # 거래 후 성과 분석 및 반성
    analysis = db.analyze_performance(period_days=7)
    if analysis:
        reflection = db.generate_reflection(analysis)
        print("\n=== Trading Reflection ===")
        print(f"Success Rate: {analysis['success_rate']*100:.2f}%")
        print(f"Profit/Loss: {analysis['profit_loss_ratio']:.2f}%")
        print(f"Lessons Learned: {reflection['lessons_learned']}")
        print(f"Strategy Adjustments: {reflection['strategy_adjustments']}")

if __name__ == "__main__":
   schedule.every().day.at("09:00").do(ai_trading)
   schedule.every().day.at("21:00").do(ai_trading)
   
   logging.basicConfig(
       level=logging.INFO,
       format='%(asctime)s - %(levelname)s - %(message)s',
       handlers=[
           logging.FileHandler('trading_bot.log'),
           logging.StreamHandler()
       ]
   )
   
   logging.info("Trading bot started. Scheduled for 09:00 and 21:00")
   
   while True:
       schedule.run_pending()
       time.sleep(60)