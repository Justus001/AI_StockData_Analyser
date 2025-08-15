import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import requests
import json
from datetime import datetime, timedelta
import openai
from typing import Dict, List, Tuple, Optional
import warnings
import logging
from dataclasses import dataclass
from enum import Enum
import time
warnings.filterwarnings('ignore')

# Logging konfigurieren
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Konfiguration
st.set_page_config(
    page_title="AI Stock Analysis Pro Elite",
    page_icon="��",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# CSS für Premium-Design
st.markdown("""
<style>
    .main-header {
        font-size: 4rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
        animation: gradient 3s ease infinite;
    }
    @keyframes gradient {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    .analysis-container {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 50%, #4facfe 100%);
        padding: 2.5rem;
        border-radius: 25px;
        color: white;
        margin: 2rem 0;
        box-shadow: 0 20px 40px rgba(0,0,0,0.15);
        border: 3px solid rgba(255,255,255,0.3);
        backdrop-filter: blur(10px);
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.8rem;
        border-radius: 20px;
        color: white;
        margin: 0.8rem 0;
        box-shadow: 0 12px 30px rgba(0,0,0,0.2);
        transition: transform 0.3s ease;
    }
    .metric-card:hover {
        transform: translateY(-5px);
    }
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 35px;
        padding: 1.2rem 4rem;
        font-weight: bold;
        font-size: 1.3rem;
        box-shadow: 0 12px 35px rgba(0,0,0,0.25);
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        transform: translateY(-4px);
        box-shadow: 0 16px 40px rgba(0,0,0,0.35);
    }
    .ticker-input {
        background: rgba(255,255,255,0.15);
        border: 3px solid rgba(255,255,255,0.4);
        border-radius: 20px;
        padding: 1.2rem;
        color: white;
        font-size: 1.2rem;
        backdrop-filter: blur(10px);
    }
    .benchmark-card {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        margin: 0.5rem 0;
        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
    }
    .risk-indicator {
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: bold;
        text-align: center;
        margin: 0.5rem 0;
    }
    .risk-low { background: linear-gradient(135deg, #56ab2f 0%, #a8e6cf 100%); }
    .risk-medium { background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); }
    .risk-high { background: linear-gradient(135deg, #ff416c 0%, #ff4b2b 100%); }
    .risk-extreme { background: linear-gradient(135deg, #8B0000 0%, #DC143C 100%); }
</style>
""", unsafe_allow_html=True)

class RiskLevel(Enum):
    LOW = "Niedrig"
    MEDIUM = "Mittel"
    HIGH = "Hoch"
    EXTREME = "Extrem"

class RecommendationType(Enum):
    STRONG_BUY = "Stark Kaufen"
    BUY = "Kaufen"
    HOLD = "Halten"
    SELL = "Verkaufen"
    STRONG_SELL = "Stark Verkaufen"

@dataclass
class TradingSignal:
    recommendation: RecommendationType
    confidence: float
    risk_level: RiskLevel
    target_price: float
    stop_loss: float
    time_horizon: str
    reasoning: str
    risk_reward_ratio: float

@dataclass
class BenchmarkResult:
    ticker: str
    performance: float
    volatility: float
    sharpe_ratio: float
    max_drawdown: float
    beta: float
    correlation: float

class TechnicalIndicators:
    """Eigene Implementierung aller technischen Indikatoren"""
    
    @staticmethod
    def rsi(prices: pd.Series, period: int = 14) -> pd.Series:
        """Relative Strength Index berechnen"""
        try:
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi
        except Exception as e:
            logger.error(f"Fehler bei RSI-Berechnung: {e}")
            return pd.Series(index=prices.index, dtype=float)
    
    @staticmethod
    def macd(prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """MACD (Moving Average Convergence Divergence) berechnen"""
        try:
            ema_fast = prices.ewm(span=fast).mean()
            ema_slow = prices.ewm(span=slow).mean()
            macd_line = ema_fast - ema_slow
            signal_line = macd_line.ewm(span=signal).mean()
            histogram = macd_line - signal_line
            return macd_line, signal_line, histogram
        except Exception as e:
            logger.error(f"Fehler bei MACD-Berechnung: {e}")
            empty_series = pd.Series(index=prices.index, dtype=float)
            return empty_series, empty_series, empty_series
    
    @staticmethod
    def bollinger_bands(prices: pd.Series, period: int = 20, std_dev: float = 2) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Bollinger Bands berechnen"""
        try:
            middle = prices.rolling(window=period).mean()
            std = prices.rolling(window=period).std()
            upper = middle + (std * std_dev)
            lower = middle - (std * std_dev)
            return upper, middle, lower
        except Exception as e:
            logger.error(f"Fehler bei Bollinger Bands: {e}")
            empty_series = pd.Series(index=prices.index, dtype=float)
            return empty_series, empty_series, empty_series
    
    @staticmethod
    def stochastic(high: pd.Series, low: pd.Series, close: pd.Series, k_period: int = 14, d_period: int = 3) -> Tuple[pd.Series, pd.Series]:
        """Stochastic Oscillator berechnen"""
        try:
            lowest_low = low.rolling(window=k_period).min()
            highest_high = high.rolling(window=k_period).max()
            k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
            d_percent = k_percent.rolling(window=d_period).mean()
            return k_percent, d_percent
        except Exception as e:
            logger.error(f"Fehler bei Stochastic: {e}")
            empty_series = pd.Series(index=close.index, dtype=float)
            return empty_series, empty_series
    
    @staticmethod
    def williams_r(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """Williams %R berechnen"""
        try:
            highest_high = high.rolling(window=period).max()
            lowest_low = low.rolling(window=period).min()
            williams_r = -100 * ((highest_high - close) / (highest_high - lowest_low))
            return williams_r
        except Exception as e:
            logger.error(f"Fehler bei Williams %R: {e}")
            return pd.Series(index=close.index, dtype=float)
    
    @staticmethod
    def cci(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 20) -> pd.Series:
        """Commodity Channel Index berechnen"""
        try:
            typical_price = (high + low + close) / 3
            sma = typical_price.rolling(window=period).mean()
            mean_deviation = typical_price.rolling(window=period).apply(lambda x: np.mean(np.abs(x - x.mean())))
            cci = (typical_price - sma) / (0.015 * mean_deviation)
            return cci
        except Exception as e:
            logger.error(f"Fehler bei CCI: {e}")
            return pd.Series(index=close.index, dtype=float)
    
    @staticmethod
    def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """Average True Range berechnen"""
        try:
            high_low = high - low
            high_close = np.abs(high - close.shift())
            low_close = np.abs(low - close.shift())
            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            atr = true_range.rolling(window=period).mean()
            return atr
        except Exception as e:
            logger.error(f"Fehler bei ATR: {e}")
            return pd.Series(index=close.index, dtype=float)
    
    @staticmethod
    def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
        """On Balance Volume berechnen"""
        try:
            obv = pd.Series(index=close.index, dtype=float)
            obv.iloc[0] = volume.iloc[0]
            
            for i in range(1, len(close)):
                if close.iloc[i] > close.iloc[i-1]:
                    obv.iloc[i] = obv.iloc[i-1] + volume.iloc[i]
                elif close.iloc[i] < close.iloc[i-1]:
                    obv.iloc[i] = obv.iloc[i-1] - volume.iloc[i]
                else:
                    obv.iloc[i] = obv.iloc[i-1]
            
            return obv
        except Exception as e:
            logger.error(f"Fehler bei OBV: {e}")
            return pd.Series(index=close.index, dtype=float)
    
    @staticmethod
    def mfi(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, period: int = 14) -> pd.Series:
        """Money Flow Index berechnen"""
        try:
            typical_price = (high + low + close) / 3
            money_flow = typical_price * volume
            
            positive_flow = pd.Series(0.0, index=typical_price.index)
            negative_flow = pd.Series(0.0, index=typical_price.index)
            
            for i in range(1, len(typical_price)):
                if typical_price.iloc[i] > typical_price.iloc[i-1]:
                    positive_flow.iloc[i] = money_flow.iloc[i]
                elif typical_price.iloc[i] < typical_price.iloc[i-1]:
                    negative_flow.iloc[i] = money_flow.iloc[i]
            
            positive_mf = positive_flow.rolling(window=period).sum()
            negative_mf = negative_flow.rolling(window=period).sum()
            
            mfi = 100 - (100 / (1 + positive_mf / negative_mf))
            return mfi
        except Exception as e:
            logger.error(f"Fehler bei MFI: {e}")
            return pd.Series(index=close.index, dtype=float)
    
    @staticmethod
    def adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Average Directional Index berechnen"""
        try:
            # True Range
            high_low = high - low
            high_close = np.abs(high - close.shift())
            low_close = np.abs(low - close.shift())
            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            
            # Directional Movement
            up_move = high - high.shift()
            down_move = low.shift() - low
            
            plus_dm = pd.Series(0.0, index=high.index)
            minus_dm = pd.Series(0.0, index=high.index)
            
            plus_dm[(up_move > down_move) & (up_move > 0)] = up_move[(up_move > down_move) & (up_move > 0)]
            minus_dm[(down_move > up_move) & (down_move > 0)] = down_move[(down_move > up_move) & (down_move > 0)]
            
            # Smoothed values
            tr_smooth = true_range.rolling(window=period).mean()
            plus_dm_smooth = plus_dm.rolling(window=period).mean()
            minus_dm_smooth = minus_dm.rolling(window=period).mean()
            
            # DI values
            plus_di = 100 * (plus_dm_smooth / tr_smooth)
            minus_di = 100 * (minus_dm_smooth / tr_smooth)
            
            # ADX
            dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
            adx = dx.rolling(window=period).mean()
            
            return adx, plus_di, minus_di
        except Exception as e:
            logger.error(f"Fehler bei ADX: {e}")
            empty_series = pd.Series(index=close.index, dtype=float)
            return empty_series, empty_series, empty_series

class AIStockAnalysisProElite:
    def __init__(self):
        self.api_keys = self._load_api_keys()
        self.setup_openai()
        self.benchmark_tickers = ['^GSPC', '^DJI', '^IXIC', '^RUT']  # S&P 500, Dow, NASDAQ, Russell
        self.risk_free_rate = 0.045
        self.indicators = TechnicalIndicators()
        
    def _load_api_keys(self) -> Dict[str, str]:
        """API-Schlüssel laden mit Fehlerbehandlung"""
        try:
            return {
                "openai": st.secrets.get("OPENAI_API_KEY", "your-openai-api-key-here"),
                "alpha_vantage": st.secrets.get("ALPHA_VANTAGE_API_KEY", "your-alpha-vantage-key-here"),
                "news_api": st.secrets.get("NEWS_API_KEY", "your-news-api-key-here")
            }
        except Exception as e:
            logger.warning(f"Fehler beim Laden der API-Schlüssel: {e}")
            return {
                "openai": "your-openai-api-key-here",
                "alpha_vantage": "your-alpha-vantage-key-here",
                "news_api": "your-news-api-key-here"
            }
    
    def setup_openai(self):
        """OpenAI Client einrichten mit Fehlerbehandlung"""
        try:
            if self.api_keys["openai"] and self.api_keys["openai"] != "your-openai-api-key-here":
                openai.api_key = self.api_keys["openai"]
                logger.info("OpenAI API erfolgreich konfiguriert")
            else:
                logger.warning("OpenAI API-Schlüssel nicht konfiguriert")
        except Exception as e:
            logger.error(f"Fehler bei der OpenAI-Konfiguration: {e}")
            st.warning("OpenAI API-Schlüssel nicht konfiguriert. KI-Analysen werden nicht verfügbar sein.")
    
    def safe_api_call(self, func, *args, **kwargs):
        """Sichere API-Aufrufe mit Retry-Logik"""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.warning(f"API-Aufruf fehlgeschlagen (Versuch {attempt + 1}/{max_retries}): {e}")
                if attempt == max_retries - 1:
                    raise e
                time.sleep(2 ** attempt)  # Exponential backoff
    
    def get_fear_greed_index(self) -> pd.DataFrame:
        """Fear & Greed Index der letzten 120 Tage abrufen mit erweiterter Fehlerbehandlung"""
        try:
            # Mehrere API-Endpunkte versuchen
            apis = [
                "https://api.alternative.me/fng/",
                "https://fear-and-greed-index.p.rapidapi.com/v1/fear-and-greed-index"
            ]
            
            for api_url in apis:
                try:
                    if "rapidapi" in api_url:
                        headers = {"X-RapidAPI-Key": "your-rapidapi-key"}
                    else:
                        headers = {}
                    
                    response = requests.get(api_url, headers=headers, timeout=15)
                    response.raise_for_status()
                    data = response.json()
                    
                    if 'data' in data:
                        df = pd.DataFrame(data['data'])
                        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
                        df['value'] = pd.to_numeric(df['value'])
                        df['value_classification'] = df['value_classification'].astype(str)
                        df = df.sort_values('timestamp')
                        logger.info("Fear & Greed Index erfolgreich abgerufen")
                        return df
                except Exception as e:
                    logger.warning(f"API {api_url} fehlgeschlagen: {e}")
                    continue
            
            # Fallback: Simulierte Daten
            logger.info("Verwende simulierte Fear & Greed Daten")
            return self._generate_enhanced_fear_greed_data()
            
        except Exception as e:
            logger.error(f"Fehler beim Abrufen des Fear & Greed Index: {e}")
            return self._generate_enhanced_fear_greed_data()
    
    def _generate_enhanced_fear_greed_data(self) -> pd.DataFrame:
        """Verbesserte simulierte Fear & Greed Daten mit realistischen Mustern"""
        dates = pd.date_range(end=datetime.now(), periods=120, freq='D')
        
        # Realistische Marktzyklen simulieren
        np.random.seed(42)
        base_trend = np.sin(np.linspace(0, 4*np.pi, 120)) * 10
        volatility = np.random.normal(0, 8, 120)
        values = 50 + base_trend + volatility
        values = np.clip(values, 0, 100)
        
        # Klassifikationen basierend auf Werten
        classifications = []
        for val in values:
            if val >= 80:
                classifications.append("Extreme Greed")
            elif val >= 65:
                classifications.append("Greed")
            elif val >= 45:
                classifications.append("Neutral")
            elif val >= 30:
                classifications.append("Fear")
            else:
                classifications.append("Extreme Fear")
        
        return pd.DataFrame({
            'timestamp': dates,
            'value': values,
            'value_classification': classifications
        })
    
    def get_macroeconomic_data(self) -> Dict[str, float]:
        """Erweiterte makroökonomische Daten mit Fehlerbehandlung"""
        try:
            # Mehrere Datenquellen für Robustheit
            data_sources = {
                "treasury_yield": [("^TNX", "Close"), ("^TYX", "Close")],
                "vix": [("^VIX", "Close")],
                "usd_index": [("DX-Y.NYB", "Close"), ("UUP", "Close")],
                "gold_price": [("GC=F", "Close"), ("GLD", "Close")],
                "oil_price": [("BZ=F", "Close"), ("USO", "Close")],
                "btc_price": [("BTC-USD", "Close")],
                "euro_stoxx": [("^STOXX50E", "Close")],
                "japan_nikkei": [("^N225", "Close")],
                "china_shanghai": [("000001.SS", "Close")]
            }
            
            macro_data = {}
            
            for key, sources in data_sources.items():
                value = None
                for ticker, column in sources:
                    try:
                        data = yf.download(ticker, start=datetime.now() - timedelta(days=30), 
                                         end=datetime.now(), progress=False)
                        if not data.empty:
                            value = float(data[column].iloc[-1])
                            break
                    except Exception as e:
                        logger.warning(f"Fehler beim Abrufen von {ticker}: {e}")
                        continue
                
                # Fallback-Werte
                fallback_values = {
                    "treasury_yield": 4.5, "vix": 20.0, "usd_index": 100.0,
                    "gold_price": 2000.0, "oil_price": 80.0, "btc_price": 50000.0,
                    "euro_stoxx": 4000.0, "japan_nikkei": 30000.0, "china_shanghai": 3000.0
                }
                
                macro_data[key] = value if value is not None else fallback_values[key]
            
            logger.info("Makroökonomische Daten erfolgreich abgerufen")
            return macro_data
            
        except Exception as e:
            logger.error(f"Fehler beim Abrufen makroökonomischer Daten: {e}")
            return {
                "treasury_yield": 4.5, "vix": 20.0, "usd_index": 100.0,
                "gold_price": 2000.0, "oil_price": 80.0, "btc_price": 50000.0,
                "euro_stoxx": 4000.0, "japan_nikkei": 30000.0, "china_shanghai": 3000.0
            }
    
    def get_stock_data(self, ticker: str) -> Tuple[pd.DataFrame, Dict]:
        """Erweiterte Aktiendaten mit Fehlerbehandlung und Validierung"""
        try:
            stock = yf.Ticker(ticker)
            
            # Historische Daten (3 Jahre für bessere Analyse)
            hist = stock.history(period="3y")
            
            if hist.empty:
                raise ValueError(f"Keine historischen Daten für {ticker} verfügbar")
            
            # Datenqualität prüfen
            if len(hist) < 100:
                raise ValueError(f"Unzureichende Daten für {ticker}: {len(hist)} Tage")
            
            # Unternehmensinformationen
            info = stock.info
            
            # Finanzielle Kennzahlen
            financials = stock.financials
            balance_sheet = stock.balance_sheet
            cash_flow = stock.cashflow
            
            # Dividenden
            dividends = stock.dividends
            
            # Analystenbewertungen
            try:
                recommendations = stock.recommendations
            except:
                recommendations = None
            
            # Options-Chain für Volatilitätsanalyse
            try:
                options = stock.options
            except:
                options = []
            
            # Institutionelle Anleger
            try:
                institutional_holders = stock.institutional_holders
            except:
                institutional_holders = None
            
            logger.info(f"Aktiendaten für {ticker} erfolgreich abgerufen")
            return hist, {
                "info": info,
                "financials": financials,
                "balance_sheet": balance_sheet,
                "cash_flow": cash_flow,
                "dividends": dividends,
                "recommendations": recommendations,
                "options": options,
                "institutional_holders": institutional_holders
            }
            
        except Exception as e:
            logger.error(f"Fehler beim Abrufen der Aktiendaten für {ticker}: {e}")
            return pd.DataFrame(), {}
    
    def calculate_advanced_metrics(self, stock_data: pd.DataFrame) -> pd.DataFrame:
        """Erweiterte technische und fundamentale Metriken mit eigener Implementierung"""
        try:
            if stock_data.empty:
                return stock_data
            
            # Grundlegende Returns
            stock_data['Returns'] = stock_data['Close'].pct_change()
            stock_data['Log_Returns'] = np.log(stock_data['Close'] / stock_data['Close'].shift(1))
            
            # Volatilitätsmetriken
            stock_data['Volatility_20'] = stock_data['Returns'].rolling(window=20).std() * np.sqrt(252)
            stock_data['Volatility_60'] = stock_data['Returns'].rolling(window=60).std() * np.sqrt(252)
            
            # Moving Averages
            for period in [10, 20, 50, 100, 200]:
                stock_data[f'MA_{period}'] = stock_data['Close'].rolling(window=period).mean()
            
            # RSI mit verschiedenen Perioden
            for period in [14, 21]:
                stock_data[f'RSI_{period}'] = self.indicators.rsi(stock_data['Close'], period)
            
            # MACD
            macd, macd_signal, macd_hist = self.indicators.macd(stock_data['Close'])
            stock_data['MACD'] = macd
            stock_data['MACD_Signal'] = macd_signal
            stock_data['MACD_Histogram'] = macd_hist
            
            # Bollinger Bands
            bb_upper, bb_middle, bb_lower = self.indicators.bollinger_bands(stock_data['Close'])
            stock_data['BB_Upper'] = bb_upper
            stock_data['BB_Middle'] = bb_middle
            stock_data['BB_Lower'] = bb_lower
            stock_data['BB_Width'] = (bb_upper - bb_lower) / bb_middle
            
            # Stochastic Oscillator
            slowk, slowd = self.indicators.stochastic(stock_data['High'], stock_data['Low'], stock_data['Close'])
            stock_data['Stoch_K'] = slowk
            stock_data['Stoch_D'] = slowd
            
            # Williams %R
            stock_data['Williams_R'] = self.indicators.williams_r(stock_data['High'], stock_data['Low'], stock_data['Close'])
            
            # Commodity Channel Index
            stock_data['CCI'] = self.indicators.cci(stock_data['High'], stock_data['Low'], stock_data['Close'])
            
            # Average True Range (ATR)
            stock_data['ATR'] = self.indicators.atr(stock_data['High'], stock_data['Low'], stock_data['Close'])
            
            # On Balance Volume
            stock_data['OBV'] = self.indicators.obv(stock_data['Close'], stock_data['Volume'])
            
            # Money Flow Index
            stock_data['MFI'] = self.indicators.mfi(stock_data['High'], stock_data['Low'], stock_data['Close'], stock_data['Volume'])
            
            # Parabolic SAR (vereinfacht)
            stock_data['SAR'] = self._calculate_sar(stock_data['High'], stock_data['Low'], stock_data['Close'])
            
            # Advanced Momentum Indicators
            stock_data['ROC'] = self._calculate_roc(stock_data['Close'], 10)
            stock_data['MOM'] = self._calculate_momentum(stock_data['Close'], 10)
            
            # Volatility Indicators
            stock_data['NATR'] = self._calculate_natr(stock_data['High'], stock_data['Low'], stock_data['Close'])
            
            # Trend Indicators
            adx, plus_di, minus_di = self.indicators.adx(stock_data['High'], stock_data['Low'], stock_data['Close'])
            stock_data['ADX'] = adx
            stock_data['PLUS_DI'] = plus_di
            stock_data['MINUS_DI'] = minus_di
            
            # Fibonacci Retracements
            high = stock_data['High'].max()
            low = stock_data['Low'].min()
            diff = high - low
            stock_data['Fib_23.6'] = high - 0.236 * diff
            stock_data['Fib_38.2'] = high - 0.382 * diff
            stock_data['Fib_50.0'] = high - 0.500 * diff
            stock_data['Fib_61.8'] = high - 0.618 * diff
            stock_data['Fib_78.6'] = high - 0.786 * diff
            
            # Support und Resistance
            stock_data['Support_Level'] = stock_data['Low'].rolling(window=20).min()
            stock_data['Resistance_Level'] = stock_data['High'].rolling(window=20).max()
            
            # Price Channels
            stock_data['Donchian_High'] = stock_data['High'].rolling(window=20).max()
            stock_data['Donchian_Low'] = stock_data['Low'].rolling(window=20).min()
            stock_data['Donchian_Mid'] = (stock_data['Donchian_High'] + stock_data['Donchian_Low']) / 2
            
            # Advanced Risk Metrics
            stock_data['VaR_95'] = stock_data['Returns'].rolling(window=252).quantile(0.05)
            stock_data['CVaR_95'] = stock_data['Returns'].rolling(window=252).apply(
                lambda x: x[x <= x.quantile(0.05)].mean()
            )
            
            # Maximum Drawdown
            stock_data['Cumulative_Returns'] = (1 + stock_data['Returns']).cumprod()
            stock_data['Running_Max'] = stock_data['Cumulative_Returns'].expanding().max()
            stock_data['Drawdown'] = (stock_data['Cumulative_Returns'] - stock_data['Running_Max']) / stock_data['Running_Max']
            
            # Sharpe Ratio
            excess_returns = stock_data['Returns'] - (self.risk_free_rate / 252)
            stock_data['Sharpe_Ratio'] = excess_returns.rolling(window=252).mean() / stock_data['Returns'].rolling(window=252).std()
            
            # Sortino Ratio
            downside_returns = stock_data['Returns'].where(stock_data['Returns'] < 0, 0)
            stock_data['Sortino_Ratio'] = excess_returns.rolling(window=252).mean() / downside_returns.rolling(window=252).std()
            
            # Calmar Ratio
            stock_data['Calmar_Ratio'] = stock_data['Returns'].rolling(window=252).mean() / abs(stock_data['Drawdown'].rolling(window=252).min())
            
            # Beta (gegen S&P 500)
            beta = self._calculate_beta(stock_data)
            stock_data['Beta'] = beta
            
            # Information Ratio (gegen Benchmark)
            benchmark_returns = self._get_benchmark_returns(stock_data.index)
            if benchmark_returns is not None:
                active_returns = stock_data['Returns'] - benchmark_returns
                stock_data['Information_Ratio'] = active_returns.rolling(window=252).mean() / active_returns.rolling(window=252).std()
            
            # Jensen's Alpha
            if benchmark_returns is not None:
                stock_data['Jensens_Alpha'] = stock_data['Returns'] - (self.risk_free_rate / 252 + 
                    stock_data['Beta'] * (benchmark_returns - self.risk_free_rate / 252))
            
            # Treynor Ratio
            stock_data['Treynor_Ratio'] = excess_returns.rolling(window=252).mean() / stock_data['Beta']
            
            # Upside/Downside Capture
            if benchmark_returns is not None:
                upside_benchmark = benchmark_returns.where(benchmark_returns > 0, 0)
                downside_benchmark = benchmark_returns.where(benchmark_returns < 0, 0)
                upside_stock = stock_data['Returns'].where(stock_data['Returns'] > 0, 0)
                downside_stock = stock_data['Returns'].where(stock_data['Returns'] < 0, 0)
                
                stock_data['Upside_Capture'] = upside_stock.rolling(window=252).sum() / upside_benchmark.rolling(window=252).sum()
                stock_data['Downside_Capture'] = downside_stock.rolling(window=252).sum() / downside_benchmark.rolling(window=252).sum()
            
            # NPV-ähnliche Metrik (erweitert)
            stock_data['NPV_Metric'] = stock_data['Close'] * (1 + stock_data['Sharpe_Ratio'] * 0.1)
            
            # Risk-Adjusted Return
            stock_data['Risk_Adjusted_Return'] = stock_data['Returns'] / stock_data['Volatility_20']
            
            logger.info("Erweiterte Metriken erfolgreich berechnet")
            return stock_data
            
        except Exception as e:
            logger.error(f"Fehler bei der Berechnung erweiterter Metriken: {e}")
            return stock_data
    
    def _calculate_sar(self, high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
        """Vereinfachte Parabolic SAR Berechnung"""
        try:
            sar = pd.Series(index=close.index, dtype=float)
            sar.iloc[0] = low.iloc[0]
            
            for i in range(1, len(close)):
                if close.iloc[i] > sar.iloc[i-1]:
                    sar.iloc[i] = min(sar.iloc[i-1], low.iloc[i-1])
                else:
                    sar.iloc[i] = max(sar.iloc[i-1], high.iloc[i-1])
            
            return sar
        except:
            return pd.Series(index=close.index, dtype=float)
    
    def _calculate_roc(self, prices: pd.Series, period: int) -> pd.Series:
        """Rate of Change berechnen"""
        try:
            return ((prices - prices.shift(period)) / prices.shift(period)) * 100
        except:
            return pd.Series(index=prices.index, dtype=float)
    
    def _calculate_momentum(self, prices: pd.Series, period: int) -> pd.Series:
        """Momentum berechnen"""
        try:
            return prices - prices.shift(period)
        except:
            return pd.Series(index=prices.index, dtype=float)
    
    def _calculate_natr(self, high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
        """Normalized Average True Range berechnen"""
        try:
            atr = self.indicators.atr(high, low, close)
            return (atr / close) * 100
        except:
            return pd.Series(index=close.index, dtype=float)
    
    def _calculate_beta(self, stock_data: pd.DataFrame) -> pd.Series:
        """Beta gegen S&P 500 berechnen"""
        try:
            benchmark_returns = self._get_benchmark_returns(stock_data.index)
            if benchmark_returns is not None:
                # Rolling Beta über 252 Tage
                beta = pd.Series(index=stock_data.index, dtype=float)
                for i in range(252, len(stock_data)):
                    stock_ret = stock_data['Returns'].iloc[i-252:i]
                    bench_ret = benchmark_returns.iloc[i-252:i]
                    
                    if len(stock_ret.dropna()) > 50 and len(bench_ret.dropna()) > 50:
                        covariance = np.cov(stock_ret.dropna(), bench_ret.dropna())[0, 1]
                        variance = np.var(bench_ret.dropna())
                        if variance > 0:
                            beta.iloc[i] = covariance / variance
                
                return beta.fillna(1.0)
            return pd.Series(1.0, index=stock_data.index)
        except:
            return pd.Series(1.0, index=stock_data.index)
    
    def _get_benchmark_returns(self, dates) -> Optional[pd.Series]:
        """Benchmark-Returns für Berechnungen abrufen"""
        try:
            benchmark = yf.download("^GSPC", start=dates.min(), end=dates.max(), progress=False)
            if not benchmark.empty:
                return benchmark['Close'].pct_change()
        except:
            pass
        return None
    
    def get_news_and_conflicts(self, ticker: str) -> Tuple[List[Dict], List[Dict]]:
        """Erweiterte Nachrichten und Konflikte mit Sentiment-Analyse"""
        try:
            # Yahoo Finance News
            stock = yf.Ticker(ticker)
            news = stock.news
            
            # Nachrichten verarbeiten
            processed_news = []
            for item in news[:20]:  # Top 20 Nachrichten
                processed_news.append({
                    "title": item.get("title", ""),
                    "summary": item.get("summary", ""),
                    "published": item.get("published", ""),
                    "url": item.get("link", ""),
                    "sentiment": self._analyze_advanced_sentiment(item.get("title", "") + " " + item.get("summary", "")),
                    "impact_score": self._calculate_news_impact(item.get("title", "") + " " + item.get("summary", ""))
                })
            
            # Globale Ereignisse und Konflikte
            global_events = self._get_enhanced_global_events()
            
            logger.info("Nachrichten und globale Ereignisse erfolgreich abgerufen")
            return processed_news, global_events
            
        except Exception as e:
            logger.error(f"Fehler beim Abrufen der Nachrichten: {e}")
            return [], []
    
    def _analyze_advanced_sentiment(self, text: str) -> str:
        """Erweiterte Sentiment-Analyse mit Gewichtung"""
        positive_words = {
            "profit": 2, "growth": 2, "increase": 1, "positive": 2, "strong": 1, 
            "beat": 3, "rise": 1, "gain": 1, "up": 1, "higher": 1, "success": 2, 
            "win": 2, "advantage": 2, "opportunity": 2, "excellent": 3, "outstanding": 3,
            "record": 2, "breakthrough": 3, "innovation": 2, "leadership": 2
        }
        
        negative_words = {
            "loss": 2, "decline": 1, "decrease": 1, "negative": 2, "weak": 1, 
            "miss": 3, "fall": 1, "drop": 1, "down": 1, "lower": 1, "risk": 1, 
            "threat": 2, "concern": 1, "worry": 1, "crisis": 3, "bankruptcy": 4,
            "default": 3, "recession": 3, "depression": 4, "crash": 3
        }
        
        text_lower = text.lower()
        positive_score = sum(score for word, score in positive_words.items() if word in text_lower)
        negative_score = sum(score for word, score in negative_words.items() if word in text_lower)
        
        if positive_score > negative_score * 1.5:
            return "Sehr Positiv"
        elif positive_score > negative_score:
            return "Positiv"
        elif negative_score > positive_score * 1.5:
            return "Sehr Negativ"
        elif negative_score > positive_score:
            return "Negativ"
        else:
            return "Neutral"
    
    def _calculate_news_impact(self, text: str) -> float:
        """Berechnet den Impact-Score einer Nachricht"""
        impact_keywords = {
            "breaking": 3, "urgent": 3, "crisis": 4, "emergency": 4,
            "record": 2, "historic": 3, "unprecedented": 4, "shock": 3,
            "surge": 2, "plunge": 2, "rally": 2, "crash": 3
        }
        
        text_lower = text.lower()
        impact_score = sum(score for word, score in impact_keywords.items() if word in text_lower)
        
        # Normalisieren auf 0-10 Skala
        return min(impact_score, 10)
    
    def _get_enhanced_global_events(self) -> List[Dict]:
        """Erweiterte globale Ereignisse und Konflikte"""
        try:
            events = []
            events.append({
                "type": "Trade War",
                "description": "Zölle zwischen USA und China beeinflussen globale Lieferketten und Marktstabilität",
                "impact": "Negative",
                "severity": "High",
                "affected_sectors": ["Technology", "Manufacturing", "Retail"],
                "duration": "Ongoing",
                "market_impact": "Medium to High"
            })
            events.append({
                "type": "Geopolitical Tensions",
                "description": "Spannungen im Nahen Osten beeinflussen Ölpreise und Energiemärkte",
                "impact": "Mixed",
                "severity": "High",
                "affected_sectors": ["Energy", "Transportation", "Airlines"],
                "duration": "Ongoing",
                "market_impact": "High"
            })
            events.append({
                "type": "Central Bank Policy",
                "description": "Fed-Zinspolitik und quantitative Lockerung beeinflussen globale Märkte",
                "impact": "Mixed",
                "severity": "High",
                "affected_sectors": ["Financial", "Real Estate", "Consumer"],
                "duration": "Ongoing",
                "market_impact": "High"
            })
            events.append({
                "type": "Climate Policy",
                "description": "Klimawandel-Politik beeinflusst Energiesektor und nachhaltige Investitionen",
                "impact": "Mixed",
                "severity": "Medium",
                "affected_sectors": ["Energy", "Technology", "Transportation"],
                "duration": "Long-term",
                "market_impact": "Medium"
            })
            events.append({
                "type": "Technological Advancements",
                "description": "Neue Technologien verändern Branchen und schaffen Chancen",
                "impact": "Positive",
                "severity": "Medium",
                "affected_sectors": ["Technology", "Healthcare", "Finance"],
                "duration": "Ongoing",
                "market_impact": "Medium to High"
            })
            return events
        except Exception as e:
            logger.error(f"Fehler beim Abrufen globaler Ereignisse: {e}")
            return []