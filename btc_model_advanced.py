import os
import pandas as pd
import numpy as np
import yfinance as yf
from pandas_datareader import data as pdr
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

# ── 1. RACCOLTA DATI ──────────────────────────────────────────────────────────────
# Legge la chiave da variabile d’ambiente
FRED_API_KEY = os.getenv("FRED_API_KEY")
if FRED_API_KEY is None:
    raise RuntimeError("Devi definire la variabile d’ambiente FRED_API_KEY")
fred = Fred(api_key=FRED_API_KEY)
yf.pdr_override()

# 1.1 Prezzo BTC mensile
btc = pdr.get_data_yahoo("BTC-USD", start="2014-01-01", interval="1mo")["Close"].rename("BTC_Price")

# 1.2 Stock-to-Flow mensile
s2f = pd.read_csv(
    "https://charts.bitbo.io/stock-to-flow/btc_s2f_monthly.csv",
    parse_dates=["Date"], index_col="Date"
)["S2F"]

# 1.3 Hash rate mensile
hr = pd.read_csv(
    "https://api.blockchain.info/charts/hash-rate?timespan=all&format=csv",
    parse_dates=["Date"], index_col="Date"
)["Value"].resample("M").mean().rename("HashRate")

# 1.4 Active addresses mensile
aa = pd.read_csv(
    "https://bitinfocharts.com/comparison/bitcoin-activeaddresses.csv",
    parse_dates=["Date"], index_col="Date"
)["Bitcoin Active Addresses"].resample("M").mean().rename("ActiveAddr")

# 1.5 CPI USA mensile
cpi = fred.get_series("CPIAUCSL").resample("M").last().rename("CPI")

# 1.6 Fed Funds mensile
fed = fred.get_series("FEDFUNDS").resample("M").last().rename("FedFunds")

# 1.7 DXY mensile
dxy = pdr.get_data_yahoo("DX-Y.NYB", start="2014-01-01", interval="1mo")["Close"].rename("DXY")

# 1.8 ETF inflows mensile
etf = (
    pd.DataFrame(requests
        .get("https://sosovalue.com/api/etf/btc-spot/inflows?period=1y")
        .json()
    )
    .assign(Date=lambda df: pd.to_datetime(df['date']))
    .set_index("Date")["inflow"]
    .resample("M").sum()
    .rename("ETF_Inflows")
)

# 1.9 Unione e pulizia
df = pd.concat([btc, s2f, hr, aa, cpi, fed, dxy, etf], axis=1).dropna()

# ── 2. DIAGNOSI E PRE‐PROCESSING TIME‐SERIES ───────────────────────────────────────
def make_stationary(series, signif=0.05, max_diff=2):
    """ADF test; differenzia fino a stazionarietà o max_diff."""
    s = series.copy()
    for d in range(max_diff + 1):
        pval = adfuller(s.dropna())[1]
        if pval < signif:
            return s
        s = s.diff().dropna()
    return s

df_stat = pd.DataFrame({col: make_stationary(df[col]) for col in df}).dropna()

# 2.2 Trasformazioni logaritmiche
df_stat["log_P"] = np.log(df_stat["BTC_Price"])
for col in ["S2F", "HashRate", "ActiveAddr", "ETF_Inflows"]:
    df_stat[f"log_{col}"] = np.log(df_stat[col] + (1 if col == "ETF_Inflows" else 0))

# ── 3. FEATURE ENGINEERING ────────────────────────────────────────────────────────
# 3.1 Lags
lags = {}
for col in ["log_S2F", "log_HashRate", "log_ActiveAddr", "CPI", "FedFunds", "DXY", "log_ETF_Inflows"]:
    lags[f"{col}_lag1"] = df_stat[col].shift(1)
df_feat = pd.concat([df_stat, pd.DataFrame(lags)], axis=1).dropna()

# 3.2 Interazioni
df_feat["int_S2F_x_HR"] = df_feat["log_S2F"] * df_feat["log_HashRate"]

# ── 4. MATRICE X, Y ───────────────────────────────────────────────────────────────
features = [
    "log_S2F", "log_HashRate", "log_ActiveAddr", "CPI", "FedFunds", "DXY", "log_ETF_Inflows",
    "log_S2F_lag1", "log_HashRate_lag1", "log_ActiveAddr_lag1", "CPI_lag1", "FedFunds_lag1", "DXY_lag1", "log_ETF_Inflows_lag1",
    "int_S2F_x_HR"
]
X = sm.add_constant(df_feat[features])
y = df_feat["log_P"]

# ── 5. MULTICOLLINEARITÀ & VIF ─────────────────────────────────────────────────────
vif = pd.DataFrame({
    "feature": X.columns,
    "VIF": [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
})
print("VIF:\n", vif)

# ── 6. REGRESSIONE PENALIZZATA ─────────────────────────────────────────────────────
tscv = TimeSeriesSplit(n_splits=5)
ridge = RidgeCV(alphas=[0.1, 1, 10], cv=tscv).fit(X, y)
lasso = LassoCV(alphas=np.logspace(-3, 1, 10), cv=tscv).fit(X, y)
print("Ridge α:", ridge.alpha, "R²:", ridge.score(X, y))
print("Lasso α:", lasso.alpha_, "R²:", lasso.score(X, y))

# ── 7. OLS & DIAGNOSTICS ─────────────────────────────────────────────────────────
ols = sm.OLS(y, X).fit()
print(ols.summary())
dw = durbin_watson(ols.resid)
print("Durbin–Watson:", dw)
from statsmodels.graphics.tsaplots import plot_acf
plot_acf(ols.resid, lags=20); plt.title("ACF residui OLS"); plt.show()
ols_hac = ols.get_robustcov_results(cov_type='HAC', maxlags=12)
print("OLS Newey–West:\n", ols_hac.summary())

# ── 8. ARIMAX ─────────────────────────────────────────────────────────────────────
arimax = ARIMA(endog=y, exog=X, order=(1,0,1)).fit()
print(arimax.summary())

# ── 9. STATE‐SPACE (Kalman) ───────────────────────────────────────────────────────
ucm = UnobservedComponents(endog=y, level='local level', exog=X).fit()
print(ucm.summary())

# ──10. MACHINE LEARNING ────────────────────────────────────────────────────────────
def rolling_ml(model, X, y, n_splits=5):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    y_true, y_pred = [], []
    for train_idx, test_idx in tscv.split(X):
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        y_true += list(y.iloc[test_idx])
        y_pred += list(model.predict(X.iloc[test_idx]))
    return y_true, y_pred

for name, mdl in [("RF", RandomForestRegressor(n_estimators=100)),
                  ("GB", GradientBoostingRegressor())]:
    yt, yp = rolling_ml(mdl, X, y)
    print(f"{name} R²:", r2_score(yt, yp),
          "RMSE:", np.sqrt(mean_squared_error(yt, yp)),
          "MAPE:", mean_absolute_percentage_error(yt, yp))

# ──11. CONFRONTO PREDIZIONI ───────────────────────────────────────────────────────
best_pred = ridge.predict(X)
df_plot = pd.DataFrame({
    "real": np.exp(y),
    "pred_ridge": np.exp(best_pred),
    "pred_ols": np.exp(ols.predict(X)),
    "pred_arimax": np.exp(arimax.predict(start=y.index[0], end=y.index[-1], exog=X))
}, index=y.index)

plt.figure(figsize=(10,6))
plt.plot(df_plot["real"], label="Prezzo reale")
plt.plot(df_plot["pred_ridge"], "--", label="Ridge")
plt.plot(df_plot["pred_ols"], "--", label="OLS")
plt.plot(df_plot["pred_arimax"], "--", label="ARIMAX")
plt.yscale("log"); plt.legend(); plt.title("Confronto modelli"); plt.show()



