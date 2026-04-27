import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ─── PAGE CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="FIN 330 Stock Dashboard", layout="wide")
st.title("FIN 330: Stock Analytics and Portfolio Dashboard")
st.write("Analyze individual stocks and evaluate a multi-asset portfolio using real Yahoo Finance data.")

# ─── SIDEBAR ───────────────────────────────────────────────────────────────────
st.sidebar.header("Navigation")
section = st.sidebar.radio("Choose a section:", ["Part 1: Stock Analysis", "Part 2: Portfolio Dashboard"])

# ══════════════════════════════════════════════════════════════════════════════
# PART 1: INDIVIDUAL STOCK ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

if section == "Part 1: Stock Analysis":

    st.header("Part 1: Individual Stock Analysis")

    # --- User inputs ---
    st.sidebar.subheader("Stock Settings")
    ticker = st.sidebar.text_input("Stock Ticker", "AAPL")

    # Run analysis button
    if st.sidebar.button("Run Stock Analysis"):

        # ── STEP 1: DATA COLLECTION ──────────────────────────────────────────
        st.subheader("Step 1: Data Collection")

        # Download 6 months of daily data
        stock = yf.Ticker(ticker)
        df = stock.history(period="6mo")

        if df.empty:
            st.error("No data found. Check the ticker symbol.")
            st.stop()

        # Use only the closing price
        close = df["Close"]

        st.success(f"6 months of data loaded for {ticker}")
        st.dataframe(df[["Open", "High", "Low", "Close", "Volume"]].tail(10))

        # ── STEP 2: TREND ANALYSIS ────────────────────────────────────────────
        st.subheader("Step 2: Trend Analysis")

        # Calculate moving averages
        df["MA20"] = close.rolling(window=20).mean()
        df["MA50"] = close.rolling(window=50).mean()

        current_price = close.iloc[-1]
        ma20 = df["MA20"].iloc[-1]
        ma50 = df["MA50"].iloc[-1]

        # Determine trend based on price vs moving averages
        if current_price > ma20 > ma50:
            trend = "Strong Uptrend"
        elif current_price < ma20 < ma50:
            trend = "Strong Downtrend"
        else:
            trend = "Mixed Trend"

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Current Price", f"${current_price:.2f}")
        col2.metric("20-Day MA", f"${ma20:.2f}")
        col3.metric("50-Day MA", f"${ma50:.2f}")
        col4.metric("Trend Signal", trend)

        # Plot closing price with moving averages
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(df.index, close, label="Close Price", linewidth=1.5)
        ax.plot(df.index, df["MA20"], label="20-Day MA", linestyle="--")
        ax.plot(df.index, df["MA50"], label="50-Day MA", linestyle="--")
        ax.set_title(f"{ticker} Price and Moving Averages")
        ax.set_xlabel("Date")
        ax.set_ylabel("Price ($)")
        ax.legend()
        st.pyplot(fig)

        # ── STEP 3: MOMENTUM (RSI) ────────────────────────────────────────────
        st.subheader("Step 3: Momentum (14-Day RSI)")

        # Calculate RSI manually
        delta = close.diff()                          # daily price changes
        gain = delta.clip(lower=0)                    # keep only gains
        loss = -delta.clip(upper=0)                   # keep only losses (make positive)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        rs = avg_gain / avg_loss                      # relative strength ratio
        df["RSI"] = 100 - (100 / (1 + rs))           # RSI formula

        rsi_value = df["RSI"].iloc[-1]

        # Interpret RSI reading
        if rsi_value > 70:
            rsi_signal = "Overbought (Possible Sell)"
        elif rsi_value < 30:
            rsi_signal = "Oversold (Possible Buy)"
        else:
            rsi_signal = "Neutral"

        col1, col2 = st.columns(2)
        col1.metric("RSI (14-Day)", f"{rsi_value:.2f}")
        col2.metric("RSI Signal", rsi_signal)

        # Plot RSI with overbought/oversold lines
        fig2, ax2 = plt.subplots(figsize=(12, 3))
        ax2.plot(df.index, df["RSI"], label="RSI", color="purple")
        ax2.axhline(70, color="red", linestyle="--", label="Overbought (70)")
        ax2.axhline(30, color="green", linestyle="--", label="Oversold (30)")
        ax2.set_title(f"{ticker} RSI (14-Day)")
        ax2.set_xlabel("Date")
        ax2.set_ylabel("RSI")
        ax2.legend()
        st.pyplot(fig2)

        # ── STEP 4: VOLATILITY ────────────────────────────────────────────────
        st.subheader("Step 4: Volatility (20-Day Annualized)")

        # Daily returns, then annualize: std * sqrt(252 trading days)
        daily_returns = close.pct_change()
        volatility = daily_returns.rolling(window=20).std().iloc[-1] * np.sqrt(252) * 100

        # Classify volatility level
        if volatility > 40:
            vol_level = "High"
        elif volatility >= 25:
            vol_level = "Medium"
        else:
            vol_level = "Low"

        col1, col2 = st.columns(2)
        col1.metric("Annualized Volatility", f"{volatility:.2f}%")
        col2.metric("Volatility Level", vol_level)

        # ── STEP 5: TRADING RECOMMENDATION ───────────────────────────────────
        st.subheader("Step 5: Trading Recommendation")

        # Simple rules-based recommendation combining all signals
        if trend == "Strong Uptrend" and rsi_value < 70:
            recommendation = "BUY"
            explanation = (
                f"{ticker} is in a strong uptrend (Price > 20MA > 50MA) "
                f"and RSI is not overbought ({rsi_value:.1f}). "
                f"Volatility is {vol_level.lower()} ({volatility:.1f}%). "
                "Conditions support a buy."
            )
        elif trend == "Strong Downtrend" or rsi_value > 70:
            recommendation = "SELL"
            explanation = (
                f"{ticker} shows a downtrend or overbought RSI ({rsi_value:.1f}). "
                f"Trend: {trend}. Volatility: {vol_level.lower()} ({volatility:.1f}%). "
                "Consider reducing exposure."
            )
        else:
            recommendation = "HOLD"
            explanation = (
                f"Mixed signals for {ticker}. Trend: {trend}. "
                f"RSI: {rsi_value:.1f} (Neutral). "
                f"Volatility: {vol_level.lower()} ({volatility:.1f}%). "
                "Wait for a clearer signal."
            )

        st.success(f"Recommendation: {recommendation}")
        st.write(explanation)

        # Download CSV of stock data
        csv = df.to_csv().encode("utf-8")
        st.download_button(
            label="Download Stock Data as CSV",
            data=csv,
            file_name=f"{ticker}_stock_analysis.csv",
            mime="text/csv"
        )


# ══════════════════════════════════════════════════════════════════════════════
# PART 2: PORTFOLIO PERFORMANCE DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════

elif section == "Part 2: Portfolio Dashboard":

    st.header("Part 2: Portfolio Performance Dashboard")

    # --- User inputs ---
    st.sidebar.subheader("Portfolio Settings")

    # Default example portfolio (5 stocks)
    default_tickers = "AAPL, MSFT, JPM, AMZN, NVDA"
    tickers_input = st.sidebar.text_input("5 Stock Tickers (comma-separated)", default_tickers)

    # Default equal weights
    default_weights = "0.20, 0.20, 0.20, 0.20, 0.20"
    weights_input = st.sidebar.text_input("Weights (must sum to 1.00)", default_weights)

    benchmark = st.sidebar.text_input("Benchmark ETF", "SPY")

    if st.sidebar.button("Run Portfolio Analysis"):

        # Parse tickers and weights from sidebar input
        tickers = [t.strip().upper() for t in tickers_input.split(",")]
        weights = [float(w.strip()) for w in weights_input.split(",")]

        # Validate inputs
        if len(tickers) != 5:
            st.error("Enter exactly 5 stock tickers.")
            st.stop()
        if len(weights) != 5:
            st.error("Enter exactly 5 weights.")
            st.stop()
        if abs(sum(weights) - 1.0) > 0.01:
            st.error(f"Weights must sum to 1.00. Current sum: {sum(weights):.2f}")
            st.stop()

        # ── STEP 1: PORTFOLIO SETUP ───────────────────────────────────────────
        st.subheader("Step 1: Portfolio Setup")

        weight_df = pd.DataFrame({"Ticker": tickers, "Weight": weights})
        st.dataframe(weight_df)

        # ── STEP 2 & 3: DATA COLLECTION ──────────────────────────────────────
        st.subheader("Step 2 & 3: Data Collection (1 Year)")

        # Download 1 year of data for all stocks and the benchmark
        all_tickers = tickers + [benchmark]
        raw = yf.download(all_tickers, period="1y", progress=False)["Close"]

        if raw.empty:
            st.error("Could not download data. Check tickers.")
            st.stop()

        st.success(f"Downloaded 1 year of data for: {', '.join(all_tickers)}")
        st.dataframe(raw.tail(5))

        # ── STEP 4: RETURN CALCULATIONS ───────────────────────────────────────
        st.subheader("Step 4: Return Calculations")

        # Daily percentage returns for each asset
        returns = raw.pct_change().dropna()

        # Portfolio daily return = weighted sum of individual returns
        portfolio_returns = returns[tickers].dot(weights)

        # Benchmark daily returns
        benchmark_returns = returns[benchmark]

        # Cumulative growth of $1 invested (for charting)
        portfolio_cumulative = (1 + portfolio_returns).cumprod()
        benchmark_cumulative = (1 + benchmark_returns).cumprod()

        # Plot cumulative returns
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(portfolio_cumulative.index, portfolio_cumulative, label="Portfolio", linewidth=2)
        ax.plot(benchmark_cumulative.index, benchmark_cumulative, label=benchmark, linestyle="--")
        ax.set_title("Portfolio vs Benchmark Cumulative Return")
        ax.set_xlabel("Date")
        ax.set_ylabel("Growth of $1")
        ax.legend()
        st.pyplot(fig)

        # ── STEP 5: PERFORMANCE METRICS ───────────────────────────────────────
        st.subheader("Step 5: Performance Metrics")

        # Total return: end value minus start, as a percentage
        total_return = (portfolio_cumulative.iloc[-1] - 1) * 100
        benchmark_total_return = (benchmark_cumulative.iloc[-1] - 1) * 100
        outperformance = total_return - benchmark_total_return

        # Annualized volatility: std of daily returns * sqrt(252 trading days)
        port_volatility = portfolio_returns.std() * np.sqrt(252) * 100
        bench_volatility = benchmark_returns.std() * np.sqrt(252) * 100

        # Sharpe ratio: excess return per unit of risk
        # Assumes a 0% risk-free rate for simplicity
        risk_free_rate = 0.0
        annualized_return = portfolio_returns.mean() * 252
        sharpe_ratio = (annualized_return - risk_free_rate) / (portfolio_returns.std() * np.sqrt(252))

        # Display all metrics in clean columns
        col1, col2, col3 = st.columns(3)
        col1.metric("Portfolio Total Return", f"{total_return:.2f}%")
        col2.metric("Benchmark Total Return", f"{benchmark_total_return:.2f}%")
        col3.metric("Outperformance", f"{outperformance:.2f}%")

        col4, col5, col6 = st.columns(3)
        col4.metric("Portfolio Volatility (Ann.)", f"{port_volatility:.2f}%")
        col5.metric("Benchmark Volatility (Ann.)", f"{bench_volatility:.2f}%")
        col6.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}")

        # ── STEP 6: INTERPRETATION ────────────────────────────────────────────
        st.subheader("Step 6: Interpretation")

        # Outperformance vs benchmark
        if outperformance > 0:
            st.write(f"The portfolio outperformed {benchmark} by {outperformance:.2f}%.")
        else:
            st.write(f"The portfolio underperformed {benchmark} by {abs(outperformance):.2f}%.")

        # Relative risk
        if port_volatility > bench_volatility:
            st.write(f"The portfolio was more risky than the benchmark ({port_volatility:.2f}% vs {bench_volatility:.2f}% volatility).")
        else:
            st.write(f"The portfolio was less risky than the benchmark ({port_volatility:.2f}% vs {bench_volatility:.2f}% volatility).")

        # Sharpe ratio interpretation
        if sharpe_ratio > 1:
            st.write(f"A Sharpe ratio of {sharpe_ratio:.2f} suggests good risk-adjusted returns.")
        elif sharpe_ratio > 0:
            st.write(f"A Sharpe ratio of {sharpe_ratio:.2f} suggests modest risk-adjusted returns.")
        else:
            st.write(f"A Sharpe ratio of {sharpe_ratio:.2f} suggests returns did not compensate for risk taken.")

        # Bar chart: individual stock total returns
        st.subheader("Individual Stock Returns")
        individual_returns = ((1 + returns[tickers]).cumprod().iloc[-1] - 1) * 100
        fig2, ax2 = plt.subplots(figsize=(8, 4))
        ax2.bar(individual_returns.index, individual_returns.values, color="steelblue")
        ax2.axhline(0, color="black", linewidth=0.8)
        ax2.set_title("Total Return by Stock (1 Year)")
        ax2.set_ylabel("Return (%)")
        ax2.set_xlabel("Ticker")
        st.pyplot(fig2)

        # Download portfolio returns as CSV
        combined = pd.DataFrame({
            "Portfolio": portfolio_returns,
            benchmark: benchmark_returns
        })
        csv = combined.to_csv().encode("utf-8")
        st.download_button(
            label="Download Portfolio Returns as CSV",
            data=csv,
            file_name="portfolio_returns.csv",
            mime="text/csv"
        )
