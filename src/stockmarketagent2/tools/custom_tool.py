from crewai.tools import BaseTool
from typing import Type
from pydantic import BaseModel, Field
import yfinance as yf
import requests
from bs4 import BeautifulSoup
import numpy as np
from sklearn.linear_model import LinearRegression

# ---------- Stock Data Tool ----------

class StockDataToolInput(BaseModel):
    """Input schema for StockDataTool."""
    stock_ticker: str = Field(..., description="The stock ticker symbol (e.g., AAPL, TSLA)")
    years: float = Field(..., description="How many years of data to retrieve (e.g., 0.5, 1, 3)")

class StockDataTool(BaseTool):
    name: str = "GetStockData"
    description: str = (
        "Retrieves historical stock data and computes short- and long-term SMAs based on years provided."
    )
    args_schema: Type[BaseModel] = StockDataToolInput

    def _run(self, stock_ticker: str, years: float) -> str:
        try:
            stock = yf.Ticker(stock_ticker)
            period = f"{int(years)}y" if years >= 1 else "6mo"
            hist = stock.history(period=period)

            if hist.empty:
                return f"No data returned from Yahoo Finance for {stock_ticker}."

            hist = hist.tail(200)
            latest_price = hist["Close"].iloc[-1]

            if years < 1:
                sma_short = hist["Close"].rolling(window=5).mean().iloc[-1]
                sma_long = hist["Close"].rolling(window=20).mean().iloc[-1]
            elif years < 3:
                sma_short = hist["Close"].rolling(window=20).mean().iloc[-1]
                sma_long = hist["Close"].rolling(window=50).mean().iloc[-1]
            else:
                sma_short = hist["Close"].rolling(window=50).mean().iloc[-1]
                sma_long = hist["Close"].rolling(window=200).mean().iloc[-1]

            short_dev = (latest_price - sma_short) / sma_short * 100
            long_dev = (latest_price - sma_long) / sma_long * 100

            return (
                f"Latest closing price for {stock_ticker}: ${latest_price:.2f}\n"
                f"Short-term SMA: ${sma_short:.2f} ({short_dev:.2f}% deviation)\n"
                f"Long-term SMA: ${sma_long:.2f} ({long_dev:.2f}% deviation)"
            )
        except Exception as e:
            return f"Could not retrieve stock data for {stock_ticker}. Error: {str(e)}"

# ---------- News Tool ----------

class StockNewsToolInput(BaseModel):
    """Input schema for StockNewsTool."""
    stock_ticker: str = Field(..., description="The stock ticker symbol to get news about.")

class StockNewsTool(BaseTool):
    name: str = "GetStockNews"
    description: str = "Scrapes recent financial news headlines and summaries for a given stock ticker symbol."
    args_schema: Type[BaseModel] = StockNewsToolInput

    def _run(self, stock_ticker: str) -> str:
        try:
            query = f"{stock_ticker} stock news"
            url = f"https://www.google.com/search?q={query}&tbm=nws"
            headers = {
                "User-Agent": "Mozilla/5.0"
            }

            response = requests.get(url, headers=headers)
            soup = BeautifulSoup(response.text, "html.parser")
            articles = soup.select("div.dbsr")

            news_summary = ""
            for article in articles[:3]:
                title = article.select_one("div.JheGif.nDgy9d").text
                link = article.a["href"]
                news_summary += f"- {title}\n{link}\n\n"

            return news_summary or "No recent news found."
        except Exception as e:
            return f"Error fetching news: {str(e)}"

# ---------- Forecast Price Tool ----------

class ForecastPriceToolInput(BaseModel):
    """Input schema for ForecastPriceTool."""
    stock_ticker: str = Field(..., description="Stock ticker symbol (e.g., AAPL)")
    years: int = Field(..., description="Number of years to forecast (e.g., 1, 3, 5)")

class ForecastPriceTool(BaseTool):
    name: str = "ForecastPrice"
    description: str = "Uses linear regression to forecast the stock price after the given number of years."
    args_schema: Type[BaseModel] = ForecastPriceToolInput

    def _run(self, stock_ticker: str, years: int) -> str:
        try:
            stock = yf.Ticker(stock_ticker)
            hist = stock.history(period="max")
            hist = hist.dropna()

            if hist.empty:
                return f"No historical data found for {stock_ticker}."

            prices = hist["Close"].values
            days = np.arange(len(prices)).reshape(-1, 1)

            model = LinearRegression()
            model.fit(days, prices)

            future_days = len(prices) + int(years * 252)  # Approx. trading days/year
            predicted_price = model.predict([[future_days]])[0]

            return f"Predicted price for {stock_ticker} after {years} years: ${predicted_price:.2f}"
        except Exception as e:
            return f"Could not forecast price for {stock_ticker}. Error: {str(e)}"


# ---------- Estimate ROI Tool ----------

class EstimateROIToolInput(BaseModel):
    """Input schema for EstimateROITool."""
    stock_ticker: str = Field(..., description="Stock ticker symbol (e.g., AAPL)")
    years: int = Field(..., description="Time horizon in years for ROI estimation")

class EstimateROITool(BaseTool):
    name: str = "EstimateROI"
    description: str = "Estimates the annualized ROI over the given number of years."
    args_schema: Type[BaseModel] = EstimateROIToolInput

    def _run(self, stock_ticker: str, years: int) -> str:
        try:
            stock = yf.Ticker(stock_ticker)
            hist = stock.history(period=f"{years}y")
            hist = hist.dropna()

            if hist.empty or len(hist) < 2:
                return f"Not enough data to estimate ROI for {stock_ticker}."

            start_price = hist["Close"].iloc[0]
            end_price = hist["Close"].iloc[-1]

            roi = (end_price / start_price) ** (1 / years) - 1
            return f"Estimated annualized ROI for {stock_ticker} over {years} years: {roi * 100:.2f}%"
        except Exception as e:
            return f"Could not estimate ROI for {stock_ticker}. Error: {str(e)}"

