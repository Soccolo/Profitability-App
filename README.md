# Daily Profitability Calculator

A Streamlit web app that calculates the implied probability distribution of your stock portfolio using options data, with support for leverage analysis.

## Features

- **File Upload or Manual Entry**: Input portfolio via Excel or interactive table
- **Leverage Support**: Track both leveraged position size and actual capital at risk
- **Interactive Parameters**: Adjust risk-free rate, liquidity filters, and capital via sidebar
- **Expiration Selection**: Choose from dropdown or enter index manually
- **0DTE Filtering**: Automatically removes same-day expiring options
- **Single Stock Support**: Works correctly with just one ticker
- **Risk Metrics**:
  - Value at Risk (VaR) at 90%, 95%, or 99% confidence
  - Customizable percentiles (1st through 99th)
  - Probability of profit/loss
  - Returns calculated both on leveraged and unleveraged basis
- **Interactive Plotly Chart**: Zoomable graph with PDF, current value, expected value, VaR, and percentile lines
- **CSV Downloads**: PDF data, summary statistics, and full distribution data

## Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## Usage

1. Upload an Excel file or use manual entry with columns:
   - `Stocks`: Ticker symbols (e.g., AAPL, MSFT, GOOGL)
   - `Value`: Total position size (including leverage)
   - `Unleveraged Value`: Actual capital at risk (optional - defaults to Value if not provided)

2. **Leverage Example**: 
   - You have $1,000 and use 2x margin to buy $2,000 of AAPL
   - Value = $2,000, Unleveraged Value = $1,000
   - Returns will be shown both ways:
     - Leveraged: % return on the $2,000 position
     - Unleveraged: % return on your actual $1,000 capital

3. Adjust parameters in the sidebar

4. Select an expiration date

5. Click "Calculate Distribution" to generate the implied PDF

6. Toggle "Show unleveraged metrics" in sidebar to compare leveraged vs unleveraged returns

## Deployment Options

### Streamlit Cloud (Free, easiest)
1. Push to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your repo and deploy

### Heroku
```bash
echo "web: streamlit run app.py --server.port=$PORT --server.address=0.0.0.0" > Procfile
heroku create your-app-name
git push heroku main
```

### Docker
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY app.py .
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0"]
```
