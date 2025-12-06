# Daily Profitability Calculator

A Streamlit web app that calculates the implied probability distribution of your stock portfolio using options data.

## Features

- **File Upload**: Drag & drop Excel files with your portfolio
- **Interactive Parameters**: Adjust risk-free rate, liquidity filters, and capital via sidebar
- **Expiration Selection**: Choose from dropdown or enter index manually
- **Risk Metrics**:
  - Value at Risk (VaR) at 90%, 95%, or 99% confidence
  - Customizable percentiles (1st, 5th, 10th, 25th, 50th, 75th, 90th, 95th, 99th)
  - Probability of profit/loss
- **Interactive Plotly Chart**: Zoomable, hoverable graph with PDF, current value, expected value, VaR, and percentile lines
- **CSV Downloads**:
  - PDF data (portfolio values + probability densities)
  - Summary statistics (expected value, VaR, percentiles)
  - Full data (PDF, CDF, and returns)

## Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## Usage

1. Upload an Excel file with two columns:
   - `Stocks`: Ticker symbols (e.g., AAPL, MSFT, GOOGL)
   - `Value`: Dollar amount invested in each stock

2. Adjust parameters in the sidebar:
   - Risk-free rate (slider or manual input)
   - Minimum volume and max spread ratio for liquidity filtering
   - Free capital
   - VaR confidence level
   - Which percentiles to display

3. Select an expiration date (dropdown or manual index)

4. Click "Calculate Distribution" to generate the implied PDF

5. Download results as CSV files

## Deployment Options

### Streamlit Cloud (Free, easiest)
1. Push to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your repo and deploy

### Heroku
```bash
# Create Procfile
echo "web: streamlit run app.py --server.port=$PORT --server.address=0.0.0.0" > Procfile

# Deploy
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
