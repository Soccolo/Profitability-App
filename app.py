import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
from datetime import datetime
from scipy.optimize import brentq
from scipy.stats import norm, gaussian_kde
from scipy.interpolate import splrep, BSpline
from scipy import interpolate
import plotly.graph_objects as go
import warnings
import io

warnings.filterwarnings('ignore')

# Page config
st.set_page_config(page_title="Daily Profitability Calculator", layout="wide")
st.title("📈 Daily Profitability Calculator")

# Sidebar parameters
st.sidebar.header("⚙️ Model Parameters")

st.sidebar.subheader("Risk-Free Rate")
rfr_mode = st.sidebar.radio("Input method", ["Slider", "Manual"], key="rfr_mode", horizontal=True)
if rfr_mode == "Slider":
    risk_free_rate = st.sidebar.slider("Risk-Free Rate", 0.0, 0.15, 0.04, 0.005)
else:
    risk_free_rate = st.sidebar.number_input("Risk-Free Rate", min_value=0.0, max_value=1.0, value=0.04, step=0.001, format="%.4f")

st.sidebar.subheader("Liquidity Filters")
min_volume = st.sidebar.number_input("Minimum Volume", min_value=1, max_value=1000, value=20, step=1)
max_spread_ratio = st.sidebar.number_input("Max Spread Ratio", min_value=0.01, max_value=1.0, value=0.2, step=0.01)

st.sidebar.subheader("Capital")
free_capital = st.sidebar.number_input("Free Capital ($)", value=100.0, step=100.0)

st.sidebar.subheader("Risk Metrics")
var_confidence = st.sidebar.selectbox("VaR Confidence Level", [90, 95, 99], index=1)
show_percentiles = st.sidebar.multiselect(
    "Show Percentiles",
    options=[1, 5, 10, 25, 50, 75, 90, 95, 99],
    default=[5, 25, 50, 75, 95]
)

st.sidebar.subheader("Display Options")
show_unleveraged = st.sidebar.checkbox("Show unleveraged metrics", value=True)


# ============ FUNCTIONS ============

def filter_liquid_options(df, min_volume, max_spread_ratio):
    spread = df["ask"] - df["bid"]
    liquid_df = df[
        (df["volume"] >= min_volume) &
        (df["bid"] > 0) &
        ((df["ask"] - df["bid"]) / df["ask"] <= max_spread_ratio)
    ]
    return liquid_df


def call_bs_price(S, K, T, r, sigma):
    if T <= 0:
        return max(S - K, 0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return price


def implied_vol_call(price, S, K, T, r):
    if T <= 0:
        return np.nan

    def objective(sigma):
        return call_bs_price(S, K, T, r, sigma) - price

    try:
        implied_vol = brentq(objective, 1e-9, 5.0)
        return implied_vol
    except ValueError:
        return np.nan


def build_pdf(K_grid, iv_spline_tck, S, T, r):
    iv_vals = BSpline(*iv_spline_tck)(K_grid)
    call_prices = np.array([call_bs_price(S, K, T, r, iv) for K, iv in zip(K_grid, iv_vals)])
    first_derivative = np.gradient(call_prices, K_grid)
    second_derivative = np.gradient(first_derivative, K_grid)
    pdf_raw = np.exp(r * T) * second_derivative
    pdf_raw = np.clip(pdf_raw, 0, None)
    return K_grid, pdf_raw


def smooth_pdf(K_grid, pdf_raw):
    kde = gaussian_kde(K_grid, weights=pdf_raw)
    pdf_smooth = kde(K_grid)
    area = np.trapz(pdf_smooth, K_grid)
    if area > 0:
        pdf_smooth /= area
    return pdf_smooth


def union_of_lists(lists):
    union = set()
    for lst in lists:
        union.update(lst)
    return list(union)


def convolve_pdfs(x_lists, pdf_lists):
    x_result = np.array(x_lists[0])
    pdf_result = np.array(pdf_lists[0])

    for i in range(1, len(x_lists)):
        x_i = np.array(x_lists[i])
        pdf_i = np.array(pdf_lists[i])

        dx_result = np.mean(np.diff(x_result))
        dx_i = np.mean(np.diff(x_i))
        dx = min(dx_result, dx_i)

        x_result_uniform = np.arange(x_result.min(), x_result.max(), dx)
        x_i_uniform = np.arange(x_i.min(), x_i.max(), dx)

        f_result = interpolate.interp1d(x_result, pdf_result,
                                        bounds_error=False, fill_value=0)
        f_i = interpolate.interp1d(x_i, pdf_i,
                                   bounds_error=False, fill_value=0)

        pdf_result_uniform = f_result(x_result_uniform)
        pdf_i_uniform = f_i(x_i_uniform)

        pdf_conv = np.convolve(pdf_result_uniform, pdf_i_uniform) * dx

        x_min_new = x_result_uniform.min() + x_i_uniform.min()
        x_result = x_min_new + np.arange(len(pdf_conv)) * dx
        pdf_result = pdf_conv

    pdf_result = pdf_result / np.trapz(pdf_result, x_result)
    return x_result, pdf_result


def calculate_percentile(x_values, pdf_values, percentile):
    """Calculate the value at a given percentile from a PDF."""
    dx = np.diff(x_values)
    pdf_midpoints = (pdf_values[:-1] + pdf_values[1:]) / 2
    cdf = np.concatenate([[0], np.cumsum(pdf_midpoints * dx)])
    cdf = cdf / cdf[-1]
    
    target = percentile / 100.0
    idx = np.searchsorted(cdf, target)
    if idx == 0:
        return x_values[0]
    if idx >= len(x_values):
        return x_values[-1]
    
    x0, x1 = x_values[idx-1], x_values[idx]
    c0, c1 = cdf[idx-1], cdf[idx]
    if c1 == c0:
        return x0
    return x0 + (x1 - x0) * (target - c0) / (c1 - c0)


def calculate_probability_below(x_values, pdf_values, threshold):
    """Calculate probability that the value is below a threshold."""
    mask = x_values <= threshold
    if not np.any(mask):
        return 0.0
    x_below = x_values[mask]
    pdf_below = pdf_values[mask]
    return np.trapz(pdf_below, x_below)


def ticker_prediction(ticker_idx, stock_list, possible_expirations, expiration_idx):
    investment = stock_list['Value'].iloc[ticker_idx]
    ticker_symbol = stock_list['Stocks'].iloc[ticker_idx]
    ticker_data = yf.Ticker(ticker_symbol)

    current_price = ticker_data.history(period="1d")['Close'].iloc[-1]
    number_of_shares = investment / current_price

    selected_expiry = possible_expirations[expiration_idx]

    option_chain = ticker_data.option_chain(selected_expiry)
    calls_df = option_chain.calls[['strike', 'lastPrice', 'bid', 'ask', 'volume']].copy()

    filtered_calls_df = filter_liquid_options(calls_df, min_volume, max_spread_ratio)

    T = (pd.to_datetime(selected_expiry) - pd.Timestamp.today()).days / 365.0
    S = ticker_data.history().iloc[-1]['Close']

    filtered_calls_df['iv'] = filtered_calls_df.apply(
        lambda row: implied_vol_call(row['lastPrice'], S, row['strike'], T, risk_free_rate), axis=1)

    filtered_calls_df.dropna(subset=['iv'], inplace=True)

    if len(filtered_calls_df) < 4:
        return None, None

    strikes = filtered_calls_df['strike'].values
    ivs = filtered_calls_df['iv'].values

    iv_spline_tck = splrep(strikes, ivs, s=10, k=3)
    K_grid = np.linspace(strikes.min(), strikes.max(), 300)
    
    # Build raw PDF using Breeden-Litzenberger formula
    K_grid_for_pdf, pdf_raw = build_pdf(K_grid, iv_spline_tck, S, T, risk_free_rate)
    
    # Smooth and normalize the PDF (density over strike prices)
    pdf_smooth_over_K_grid = smooth_pdf(K_grid_for_pdf, pdf_raw)
    
    # Transform K_grid (strike prices) into investment values
    investment_grid_values = np.array([K * number_of_shares for K in K_grid_for_pdf])
    
    # Transform PDF from density per strike to density per investment value
    # P_Y(Y) = P_K(K) / number_of_shares (Jacobian transformation)
    pdf_smooth_over_investment_values = pdf_smooth_over_K_grid / number_of_shares

    return investment_grid_values, pdf_smooth_over_investment_values


# ============ MAIN APP ============

st.subheader("📁 Portfolio Input")

input_method = st.radio(
    "How would you like to input your portfolio?",
    ["Upload Excel File", "Manual Entry"],
    horizontal=True
)

# Info about leverage
with st.expander("ℹ️ About Leverage"):
    st.markdown("""
    **Value**: The total position size (including any leverage/margin)
    
    **Unleveraged Value**: Your actual capital at risk (without leverage)
    
    **Example**: If you have $1,000 and use 2x leverage to buy $2,000 worth of stock:
    - Value = $2,000
    - Unleveraged Value = $1,000
    
    If not using leverage, set both values equal.
    """)

stock_list = None

if input_method == "Upload Excel File":
    uploaded_file = st.file_uploader("📤 Upload your Stock Distribution Excel file", type=['xlsx', 'xls'])
    
    if uploaded_file is not None:
        stock_list = pd.read_excel(uploaded_file)
        
        st.write("**📊 Uploaded Portfolio:**")
        st.dataframe(stock_list, use_container_width=True)
        
        # Check required columns
        required_cols = ['Stocks', 'Value']
        if not all(col in stock_list.columns for col in required_cols):
            st.error("❌ Excel file must have 'Stocks' and 'Value' columns")
            stock_list = None
        else:
            # Add Unleveraged Value column if not present
            if 'Unleveraged Value' not in stock_list.columns:
                stock_list['Unleveraged Value'] = stock_list['Value']
                st.info("ℹ️ 'Unleveraged Value' column not found - assuming no leverage (set equal to Value)")
    else:
        st.info("👆 Please upload an Excel file with columns: 'Stocks', 'Value', and optionally 'Unleveraged Value'")
        
        st.write("**Example Format:**")
        example_df = pd.DataFrame({
            'Stocks': ['AAPL', 'MSFT', 'GOOGL'],
            'Value': [2000, 3000, 4000],
            'Unleveraged Value': [1000, 1500, 2000]
        })
        st.dataframe(example_df)

else:  # Manual Entry
    st.write("**✏️ Enter Your Portfolio:**")
    
    # Initialize session state for manual entries
    if 'manual_stocks' not in st.session_state:
        st.session_state.manual_stocks = pd.DataFrame({
            'Stocks': ['AAPL', 'MSFT'],
            'Value': [2000.0, 3000.0],
            'Unleveraged Value': [1000.0, 1500.0]
        })
    
    st.write("Edit the table below (click cells to edit, use + button to add rows):")
    
    edited_df = st.data_editor(
        st.session_state.manual_stocks,
        num_rows="dynamic",
        use_container_width=True,
        column_config={
            "Stocks": st.column_config.TextColumn(
                "Ticker Symbol",
                help="Enter stock ticker (e.g., AAPL, MSFT, GOOGL)",
                max_chars=10,
            ),
            "Value": st.column_config.NumberColumn(
                "Position Value ($)",
                help="Total position size (including leverage)",
                min_value=0,
                format="$%.2f",
            ),
            "Unleveraged Value": st.column_config.NumberColumn(
                "Unleveraged Value ($)",
                help="Actual capital at risk (without leverage)",
                min_value=0,
                format="$%.2f",
            )
        }
    )
    
    # Update session state
    st.session_state.manual_stocks = edited_df
    
    # Quick add section
    st.write("**Quick Add:**")
    quick_col1, quick_col2, quick_col3, quick_col4 = st.columns([2, 1.5, 1.5, 1])
    with quick_col1:
        new_ticker = st.text_input("Ticker", placeholder="e.g., GOOGL", key="new_ticker")
    with quick_col2:
        new_value = st.number_input("Value ($)", min_value=0.0, value=2000.0, step=100.0, key="new_value")
    with quick_col3:
        new_unlev = st.number_input("Unleveraged ($)", min_value=0.0, value=1000.0, step=100.0, key="new_unlev")
    with quick_col4:
        st.write("")
        if st.button("➕ Add"):
            if new_ticker.strip():
                new_row = pd.DataFrame({
                    'Stocks': [new_ticker.strip().upper()], 
                    'Value': [new_value],
                    'Unleveraged Value': [new_unlev]
                })
                st.session_state.manual_stocks = pd.concat([st.session_state.manual_stocks, new_row], ignore_index=True)
                st.rerun()
    
    # Validate and use the manual entries
    if len(edited_df) > 0 and edited_df['Stocks'].notna().any() and edited_df['Value'].notna().any():
        stock_list = edited_df.dropna(subset=['Stocks', 'Value'])
        stock_list = stock_list[stock_list['Stocks'].str.strip() != '']
        stock_list = stock_list[stock_list['Value'] > 0]
        stock_list = stock_list.reset_index(drop=True)
        
        # Fill missing Unleveraged Value with Value
        if 'Unleveraged Value' not in stock_list.columns or stock_list['Unleveraged Value'].isna().any():
            stock_list['Unleveraged Value'] = stock_list['Unleveraged Value'].fillna(stock_list['Value'])
        
        if len(stock_list) > 0:
            leverage_ratio = stock_list['Value'].sum() / stock_list['Unleveraged Value'].sum()
            st.success(f"✅ {len(stock_list)} stock(s) ready for analysis | Effective leverage: {leverage_ratio:.2f}x")
        else:
            stock_list = None
            st.warning("Please enter at least one valid stock with a positive value")
    else:
        st.warning("Please enter at least one stock and value")

# Continue with analysis if we have valid stock_list
if stock_list is not None and len(stock_list) > 0:
    tickers = list(stock_list['Stocks'])
    
    with st.spinner("Fetching expiration dates..."):
        expiration_dates = []
        for ticker in tickers:
            try:
                t = yf.Ticker(ticker)
                expirations = t.options
                expiration_dates.append(expirations)
            except Exception as e:
                st.warning(f"Could not fetch options for {ticker}: {e}")
                expiration_dates.append([])
        
        possible_expirations = union_of_lists(expiration_dates)
        possible_expirations = sorted(possible_expirations)
        
        # Remove 0DTE options
        if possible_expirations:
            first_exp_days = (pd.to_datetime(possible_expirations[0]) - pd.Timestamp.today()).days
            if first_exp_days <= 0:
                possible_expirations.pop(0)
                st.info("ℹ️ Removed 0DTE options from available expirations")
    
    if not possible_expirations:
        st.error("No valid expiration dates found for any ticker")
    else:
        st.subheader("📅 Select Expiration Date")
        
        exp_col1, exp_col2 = st.columns([2, 1])
        with exp_col1:
            selection_mode = st.radio(
                "Selection mode",
                ["Dropdown", "Manual Index"],
                horizontal=True
            )
        
        if selection_mode == "Dropdown":
            selected_expiry_str = st.selectbox(
                "Expiration Date",
                options=possible_expirations,
                index=0
            )
            target_expiration = possible_expirations.index(selected_expiry_str)
        else:
            with exp_col2:
                st.write("Available dates:")
                for i, exp in enumerate(possible_expirations[:10]):
                    st.caption(f"{i}: {exp}")
                if len(possible_expirations) > 10:
                    st.caption(f"... and {len(possible_expirations) - 10} more")
            
            target_expiration = st.number_input(
                "Expiration Index",
                min_value=0,
                max_value=len(possible_expirations) - 1,
                value=0,
                step=1
            )
        
        selected_exp_date = pd.to_datetime(possible_expirations[target_expiration])
        days_to_expiry = (selected_exp_date - pd.Timestamp.today()).days
        st.info(f"📆 Selected: **{possible_expirations[target_expiration]}** ({days_to_expiry} days to expiry)")
        
        if st.button("🚀 Calculate Distribution", type="primary"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            investment_grid_list = []
            pdf_smooth_list = []
            failed_tickers = []
            
            for i, ticker in enumerate(tickers):
                status_text.text(f"Processing {ticker}...")
                progress_bar.progress((i + 1) / len(tickers))
                
                try:
                    investment_grid, pdf_smooth = ticker_prediction(
                        i, stock_list, possible_expirations, target_expiration
                    )
                    if investment_grid is not None:
                        investment_grid_list.append(investment_grid)
                        pdf_smooth_list.append(pdf_smooth)
                    else:
                        failed_tickers.append(ticker)
                except Exception as e:
                    failed_tickers.append(ticker)
                    st.warning(f"Error processing {ticker}: {e}")
            
            status_text.text("Convolving PDFs...")
            
            if len(investment_grid_list) < 1:
                st.error("Could not process any tickers successfully")
            else:
                if failed_tickers:
                    st.warning(f"Skipped tickers (insufficient data): {', '.join(failed_tickers)}")
                
                # Handle single stock case properly
                if len(investment_grid_list) == 1:
                    investment_values_final = np.array([i + free_capital for i in investment_grid_list[0]])
                    pdf_values_final = np.array(pdf_smooth_list[0])
                else:
                    investment_values_final, pdf_values_final = convolve_pdfs(
                        investment_grid_list, pdf_smooth_list
                    )
                    investment_values_final = investment_values_final + free_capital
                
                status_text.text("Done!")
                progress_bar.progress(100)
                
                # Calculate statistics
                expected_value = np.trapz(investment_values_final * pdf_values_final, investment_values_final)
                
                # Leveraged values
                current_value_leveraged = sum(stock_list['Value']) + free_capital
                
                # Unleveraged values
                unleveraged_capital = sum(stock_list['Unleveraged Value']) + free_capital
                
                # Calculate gains and returns
                expected_gain = expected_value - current_value_leveraged
                expected_return_leveraged = (expected_gain / current_value_leveraged) * 100
                expected_return_unleveraged = (expected_gain / unleveraged_capital) * 100
                
                # Leverage ratio
                leverage_ratio = current_value_leveraged / unleveraged_capital
                
                # Calculate VaR
                var_percentile = 100 - var_confidence
                var_value = calculate_percentile(investment_values_final, pdf_values_final, var_percentile)
                var_loss = current_value_leveraged - var_value
                var_loss_pct_leveraged = (var_loss / current_value_leveraged) * 100
                var_loss_pct_unleveraged = (var_loss / unleveraged_capital) * 100
                
                # Calculate percentiles
                percentile_values = {}
                for p in show_percentiles:
                    percentile_values[p] = calculate_percentile(investment_values_final, pdf_values_final, p)
                
                # Probability of loss
                prob_loss = calculate_probability_below(investment_values_final, pdf_values_final, current_value_leveraged)
                prob_profit = 1 - prob_loss
                
                # Create Plotly figure
                fig = go.Figure()
                
                fig.add_trace(go.Scatter(
                    x=investment_values_final,
                    y=pdf_values_final,
                    mode='lines',
                    name='Implied PDF',
                    line=dict(color='cyan', width=3),
                    fill='tozeroy',
                    fillcolor='rgba(0, 255, 255, 0.1)'
                ))
                
                fig.add_vline(
                    x=current_value_leveraged,
                    line_dash="dash",
                    line_color="red",
                    line_width=2,
                    annotation_text=f"Current (Leveraged): ${current_value_leveraged:,.2f}",
                    annotation_position="top"
                )
                
                if show_unleveraged and leverage_ratio > 1.01:
                    fig.add_vline(
                        x=unleveraged_capital,
                        line_dash="dash",
                        line_color="yellow",
                        line_width=2,
                        annotation_text=f"Unleveraged Capital: ${unleveraged_capital:,.2f}",
                        annotation_position="top left"
                    )
                
                fig.add_vline(
                    x=expected_value,
                    line_dash="dash",
                    line_color="lime",
                    line_width=2,
                    annotation_text=f"Expected: ${expected_value:,.2f}",
                    annotation_position="top"
                )
                
                fig.add_vline(
                    x=var_value,
                    line_dash="dot",
                    line_color="orange",
                    line_width=2,
                    annotation_text=f"VaR {var_confidence}%: ${var_value:,.2f}",
                    annotation_position="bottom"
                )
                
                percentile_colors = {
                    1: 'rgba(255,0,0,0.3)', 5: 'rgba(255,100,0,0.3)', 
                    10: 'rgba(255,150,0,0.3)', 25: 'rgba(255,200,0,0.3)',
                    50: 'rgba(255,255,0,0.5)', 75: 'rgba(200,255,0,0.3)',
                    90: 'rgba(100,255,0,0.3)', 95: 'rgba(0,255,100,0.3)',
                    99: 'rgba(0,255,200,0.3)'
                }
                for p, val in percentile_values.items():
                    if p != 50:
                        fig.add_vline(
                            x=val,
                            line_dash="dot",
                            line_color=percentile_colors.get(p, 'gray'),
                            line_width=1,
                            opacity=0.7
                        )
                
                fig.update_layout(
                    title=f"Implied Probability Density for {possible_expirations[target_expiration]}",
                    xaxis_title="Portfolio Value ($)",
                    yaxis_title="Probability Density",
                    template="plotly_dark",
                    height=600,
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Summary Statistics
                st.subheader("📊 Summary Statistics")
                
                # Leverage info
                if leverage_ratio > 1.01:
                    st.info(f"⚡ **Leverage Ratio: {leverage_ratio:.2f}x** | Unleveraged Capital: ${unleveraged_capital:,.2f} → Position Size: ${current_value_leveraged:,.2f}")
                
                # Key metrics - Leveraged view
                st.write("**Leveraged View:**")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Position Value", f"${current_value_leveraged:,.2f}")
                with col2:
                    st.metric("Expected Value", f"${expected_value:,.2f}")
                with col3:
                    st.metric("Expected Return", f"{expected_return_leveraged:+.2f}%")
                with col4:
                    st.metric(f"VaR ({var_confidence}%)", f"${var_loss:,.2f}", 
                              delta=f"-{var_loss_pct_leveraged:.1f}%", delta_color="inverse")
                
                # Unleveraged view
                if show_unleveraged:
                    st.write("**Unleveraged View (Return on Actual Capital):**")
                    col1b, col2b, col3b, col4b = st.columns(4)
                    with col1b:
                        st.metric("Actual Capital", f"${unleveraged_capital:,.2f}")
                    with col2b:
                        st.metric("Expected Gain", f"${expected_gain:,.2f}")
                    with col3b:
                        st.metric("Expected Return", f"{expected_return_unleveraged:+.2f}%", 
                                  help="Return based on actual capital at risk")
                    with col4b:
                        st.metric(f"VaR ({var_confidence}%) on Capital", f"${var_loss:,.2f}", 
                                  delta=f"-{var_loss_pct_unleveraged:.1f}%", delta_color="inverse",
                                  help="VaR as percentage of actual capital")
                
                # Probabilities
                st.subheader("🎲 Probabilities")
                prob_col1, prob_col2 = st.columns(2)
                with prob_col1:
                    st.metric("Probability of Profit", f"{prob_profit*100:.1f}%")
                with prob_col2:
                    st.metric("Probability of Loss", f"{prob_loss*100:.1f}%")
                
                # Percentiles Table
                if show_percentiles:
                    st.subheader("📈 Percentiles")
                    percentile_data = {
                        'Percentile': [f"{p}th" for p in sorted(percentile_values.keys())],
                        'Portfolio Value': [f"${percentile_values[p]:,.2f}" for p in sorted(percentile_values.keys())],
                        'Return (Leveraged)': [f"{((percentile_values[p]/current_value_leveraged)-1)*100:+.2f}%" for p in sorted(percentile_values.keys())],
                    }
                    if show_unleveraged:
                        percentile_data['Return (Unleveraged)'] = [
                            f"{((percentile_values[p]-current_value_leveraged)/unleveraged_capital)*100:+.2f}%" 
                            for p in sorted(percentile_values.keys())
                        ]
                    percentile_df = pd.DataFrame(percentile_data)
                    st.dataframe(percentile_df, use_container_width=True, hide_index=True)
                
                # Download Section
                st.subheader("💾 Download Results")
                
                download_col1, download_col2, download_col3 = st.columns(3)
                
                with download_col1:
                    pdf_df = pd.DataFrame({
                        'Portfolio_Value': investment_values_final,
                        'Probability_Density': pdf_values_final
                    })
                    csv_pdf = pdf_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download PDF Data (CSV)",
                        data=csv_pdf,
                        file_name=f"implied_pdf_{possible_expirations[target_expiration]}.csv",
                        mime="text/csv"
                    )
                
                with download_col2:
                    summary_data = {
                        'Metric': [
                            'Position Value (Leveraged)', 
                            'Unleveraged Capital',
                            'Leverage Ratio',
                            'Expected Value', 
                            'Expected Gain',
                            'Expected Return (Leveraged %)', 
                            'Expected Return (Unleveraged %)',
                            f'VaR {var_confidence}% (Value)', 
                            f'VaR {var_confidence}% (Loss)',
                            f'VaR {var_confidence}% Loss (Leveraged %)',
                            f'VaR {var_confidence}% Loss (Unleveraged %)',
                            'Probability of Profit (%)', 
                            'Probability of Loss (%)'
                        ],
                        'Value': [
                            current_value_leveraged,
                            unleveraged_capital,
                            leverage_ratio,
                            expected_value, 
                            expected_gain,
                            expected_return_leveraged,
                            expected_return_unleveraged,
                            var_value, 
                            var_loss,
                            var_loss_pct_leveraged,
                            var_loss_pct_unleveraged,
                            prob_profit*100, 
                            prob_loss*100
                        ]
                    }
                    for p in sorted(percentile_values.keys()):
                        summary_data['Metric'].append(f'Percentile {p}th')
                        summary_data['Value'].append(percentile_values[p])
                    
                    summary_df = pd.DataFrame(summary_data)
                    csv_summary = summary_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Summary (CSV)",
                        data=csv_summary,
                        file_name=f"summary_stats_{possible_expirations[target_expiration]}.csv",
                        mime="text/csv"
                    )
                
                with download_col3:
                    dx = np.diff(investment_values_final)
                    pdf_midpoints = (pdf_values_final[:-1] + pdf_values_final[1:]) / 2
                    cdf = np.concatenate([[0], np.cumsum(pdf_midpoints * dx)])
                    cdf = cdf / cdf[-1]
                    
                    full_df = pd.DataFrame({
                        'Portfolio_Value': investment_values_final,
                        'PDF': pdf_values_final,
                        'CDF': cdf,
                        'Return_Leveraged_Percent': ((investment_values_final / current_value_leveraged) - 1) * 100,
                        'Return_Unleveraged_Percent': ((investment_values_final - current_value_leveraged) / unleveraged_capital) * 100
                    })
                    csv_full = full_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Full Data (CSV)",
                        data=csv_full,
                        file_name=f"full_distribution_{possible_expirations[target_expiration]}.csv",
                        mime="text/csv"
                    )
