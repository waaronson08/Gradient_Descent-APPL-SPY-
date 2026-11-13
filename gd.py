"""
Gradient descent on real price data.

Step 1: reproduce your toy problem (should land on w=1.5, b=1/3).
Step 2: same idea on stock returns: regress a stock's returns on the market's.
        slope w = beta, intercept b = alpha (daily).
Setup:  pip install numpy yfinance
"""
import numpy as np


def gradient_descent(x, y, lr=0.05, steps=5000):
    """Fit y_hat = w*x + b by minimizing mean squared error."""
    w, b = 0.0, 0.0
    history = []
    for _ in range(steps):
        err = w * x + b - y
        grad_w = 2 * np.mean(err * x)   # dL/dw
        grad_b = 2 * np.mean(err)       # dL/db
        w -= lr * grad_w
        b -= lr * grad_b
        history.append(np.mean(err ** 2))
    return w, b, history


# From the toy problem
x = np.array([1.0, 2.0, 3.0])
y = np.array([2.0, 3.0, 5.0])
w, b, _ = gradient_descent(x, y, lr=0.05, steps=5000)
print(f"Toy problem: w={w:.4f} (exact 1.5), b={b:.4f} (exact {1/3:.4f})")


# real data (stock vs. market)
def load_returns(stock="AAPL", market="SPY", start="2019-01-01"):
    try:
        import yfinance as yf
        px = yf.download([stock, market], start=start, auto_adjust=True,
                         progress=False)["Close"].dropna()
        rets = px.pct_change().dropna() * 100      # percent returns keep numbers ~1
        return rets[market].to_numpy(), rets[stock].to_numpy()
    except Exception as e:
        print(f"Download failed ({e}); using simulated data instead.")
        rng = np.random.default_rng(0)
        m = rng.normal(0.05, 1.0, 1500)
        s = 0.02 + 1.2 * m + rng.normal(0, 1.0, 1500)
        return m, s


mkt, stk = load_returns()

# Fit on the first 70% of days, evaluate on the last 30% (never test on training data)
split = int(len(mkt) * 0.7)
x_tr, y_tr = mkt[:split], stk[:split]
x_te, y_te = mkt[split:], stk[split:]

w, b, hist = gradient_descent(x_tr, y_tr, lr=0.05, steps=5000)

# Sanity check against the closed-form least-squares answer
w_ls, b_ls = np.polyfit(x_tr, y_tr, 1)
print(f"Gradient descent: beta={w:.4f}, alpha={b:.4f}% per day")
print(f"Closed form:      beta={w_ls:.4f}, alpha={b_ls:.4f}% per day")

mse_tr = np.mean((w * x_tr + b - y_tr) ** 2)
mse_te = np.mean((w * x_te + b - y_te) ** 2)
print(f"MSE train={mse_tr:.4f}  test={mse_te:.4f}")