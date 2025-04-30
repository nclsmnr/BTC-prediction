import os
import pandas as pd
import numpy as np
import yfinance as yf
from fredapi import Fred
import requests
import statsmodels.api as sm
from statsmodels.stats.stattools import durbin_watson
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.structural import UnobservedComponents
from sklearn.linear_model import RidgeCV, LassoCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error
import matplotlib.pyplot as plt

# ── UTILITIES ────────────────────────────────────────────────────────────────────
def download_monthly_close(ticker, start="2014-01-01"):
    """Scarica da yfinance il close mensile e allinea l'indice a fine mese."""
    df = yf.download(ticker, start=start, interval="1mo", progress=False)["Close"]
    # Resample per sicurezza su fine mese
    df = df.resample("M").last()
    return df.rename(ticker)

def make_stationary(series, signif=0.05, max_diff=2):
    """ADF test; differenzia fino a p-val < signif o max_diff."""
    s = series.copy()
    for _ in range(max_diff + 1):
        pval = adfuller(s.dropna())[1]
        if pval < signif:
            return s
        s = s.diff().dropna()
    return s

# ── 1. RACCOLTA DATI ──────────────────────────────────────────────────────────────
# 1.1 API key FRED
FRED_API_KEY = os.getenv("FRED_API_KEY")
if not FRED_API_KEY:
    raise RuntimeError("e112dce38460b509b97db2564f48810c")
fred = Fred(api_key=FRED_API_KEY)

# 1.2 Prezzo BTC e DXY
btc = download_monthly_close("BTC-USD").rename("BTC_Price")
dxy = download_monthly_close("DX-Y.NYB").rename("DXY")

# 1.3 Stock-to-Flow mensile
s2f = pd.read_csv(
    "https://charts.bitbo.io/stock-to-flow/btc_s2f_monthly.csv",
    parse_dates=["Date"], index_col="Date"
)["S2F"].resample("M").last()

# 1.4 Hash rate mensile
hr = pd.read_csv(
    "https://api.blockchain.info/charts/hash-rate?timespan=all&format=csv",
    parse_dates=["Date"], index_col="Date"
)["Value"].resample("M").mean().rename("HashRate")

# 1.5 Active addresses mensile
aa = pd.read_csv(
    "https://bitinfocharts.com/comparison/bitcoin-activeaddresses.csv",
    parse_dates=["Date"], index_col="Date"
)["Bitcoin Active Addresses"].resample("M").mean().rename("ActiveAddr")

# 1.6 CPI USA mensile
cpi = fred.get_series("CPIAUCSL").resample("M").last().rename("CPI")

# 1.7 Fed Funds mensile
fed = fred.get_series("FEDFUNDS").resample("M").last().rename("FedFunds")

# 1.8 ETF inflows mensile
etf = (
    pd.DataFrame(requests
        .get("https://sosovalue.com/api/etf/btc-spot/inflows?period=1y")
        .json()
    )
    .assign(Date=lambda df: pd.to_datetime(df["date"]))
    .set_index("Date")["inflow"]
    .resample("M").sum()
    .rename("ETF_Inflows")
)

# 1.9 Unione e pulizia
df = pd.concat([btc, s2f, hr, aa, cpi, fed, dxy, etf], axis=1).dropna()

# ── 2. DIAGNOSI E PRE-PROCESSING TIME-SERIES ────────────────────────────────────
# 2.1 Stazionarietà
df_stat = pd.DataFrame({col: make_stationary(df[col]) for col in df}).dropna()

# 2.2 Log-transform
df_stat["log_P"] = np.log(df_stat["BTC_Price"])
for col in ["S2F", "HashRate", "ActiveAddr", "ETF_Inflows"]:
    df_stat[f"log_{col}"] = np.log(df_stat[col] + (1 if col=="ETF_Inflows" else 0))

# ── 3. FEATURE ENGINEERING ──────────────────────────────────────────────────────
# 3.1 Lag delle esogene (t-1)
lags = {
    f"{col}_lag1": df_stat[col].shift(1)
    for col in ["log_S2F","log_HashRate","log_ActiveAddr","CPI","FedFunds","DXY","log_ETF_Inflows"]
}
df_feat = pd.concat([df_stat, pd.DataFrame(lags)], axis=1).dropna()

# 3.2 Interazioni
df_feat["int_S2F_x_HR"] = df_feat["log_S2F"] * df_feat["log_HashRate"]

# ── 4. MATRICI X e y ─────────────────────────────────────────────────────────────
features = [
    "log_S2F","log_HashRate","log_ActiveAddr","CPI","FedFunds","DXY","log_ETF_Inflows",
    "log_S2F_lag1","log_HashRate_lag1","log_ActiveAddr_lag1","CPI_lag1","FedFunds_lag1","DXY_lag1","log_ETF_Inflows_lag1",
    "int_S2F_x_HR"
]
X = sm.add_constant(df_feat[features])
y = df_feat["log_P"]

# ── 5. MULTICOLLINEARITÀ (VIF) ──────────────────────────────────────────────────
vif = pd.DataFrame({
    "feature": X.columns,
    "VIF": [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
})
print("VIF:\n", vif)

# ── 6. REGRESSIONI PENALIZZATE ───────────────────────────────────────────────────
tscv = TimeSeriesSplit(n_splits=5)
ridge = RidgeCV(alphas=[0.1,1,10], cv=tscv).fit(X, y)
lasso = LassoCV(alphas=np.logspace(-3,1,10), cv=tscv).fit(X, y)
print(f"Ridge α={ridge.alpha_:.3g}, R²={ridge.score(X,y):.3f}")
print(f"Lasso α={lasso.alpha_:.3g}, R²={lasso.score(X,y):.3f}")

# ── 7. OLS e diagnostica residui ────────────────────────────────────────────────
ols = sm.OLS(y, X).fit()
print(ols.summary())
dw = durbin_watson(ols.resid)
print(f"Durbin–Watson: {dw:.3f}")
from statsmodels.graphics.tsaplots import plot_acf
plot_acf(ols.resid, lags=20); plt.title("ACF residui OLS"); plt.show()
ols_hac = ols.get_robustcov_results(cov_type="HAC", maxlags=12)
print("OLS Newey–West:\n", ols_hac.summary())

# ── 8. ARIMAX ────────────────────────────────────────────────────────────────────
arimax = ARIMA(endog=y, exog=X, order=(1,0,1)).fit()
print(arimax.summary())

# ── 9. State‐Space (Kalman) ─────────────────────────────────────────────────────
ucm = UnobservedComponents(endog=y, level="local level", exog=X).fit()
print(ucm.summary())

# ── 10. MACHINE LEARNING ────────────────────────────────────────────────────────
def rolling_ml(model, X, y, splits=5):
    tscv = TimeSeriesSplit(n_splits=splits)
    yt, yp = [], []
    for tr, te in tscv.split(X):
        model.fit(X.iloc[tr], y.iloc[tr])
        yt += list(y.iloc[te])
        yp += list(model.predict(X.iloc[te]))
    return yt, yp

for name, mdl in [("RF", RandomForestRegressor(n_estimators=100)),
                  ("GB", GradientBoostingRegressor())]:
    yt, yp = rolling_ml(mdl, X, y)
    print(f"{name}: R²={r2_score(yt,yp):.3f}, RMSE={np.sqrt(mean_squared_error(yt,yp)):.3f}, MAPE={mean_absolute_percentage_error(yt,yp):.3f}")

# ── 11. CONFRONTO PREVISIONI ─────────────────────────────────────────────────────
best = ridge.predict(X)
df_plot = pd.DataFrame({
    "Real": np.exp(y),
    "Ridge": np.exp(best),
    "OLS": np.exp(ols.predict(X)),
    "ARIMAX": np.exp(arimax.predict(start=y.index[0], end=y.index[-1], exog=X))
}, index=y.index)

plt.figure(figsize=(10,6))
for col in df_plot.columns:
    plt.plot(df_plot[col], label=col, linestyle="--" if col!="Real" else "-")
plt.yscale("log")
plt.legend(); plt.title("BTC: Reale vs Predetto"); plt.show()




