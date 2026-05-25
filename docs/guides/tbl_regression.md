# Regression — `tbl_regression()`

Supported model libraries:

| Library | Examples | Native exponentiate |
|---|---|---|
| **statsmodels** | OLS, Logit, Probit, Poisson, NegativeBinomial, GLM | log-link models only |
| **lifelines** | CoxPHFitter, WeibullAFTFitter, LogNormalAFTFitter | yes (HRs) |
| **sklearn** | LinearRegression, LogisticRegression (binary), Lasso, Ridge | no CIs |

```python
# statsmodels logistic regression
import statsmodels.api as sm
X = sm.add_constant(df[['age', 'bmi']])
fit = sm.Logit(df['event'], X).fit(disp=False)
ps.tbl_regression(fit, exponentiate=True)   # column auto-labelled "OR"
```

## Multi-model side-by-side

Pass a list:

```python
ps.tbl_regression(
    [fit_unadjusted, fit_adjusted],
    exponentiate=True,
    model_labels=['Unadjusted', 'Adjusted'],
)
```

## lifelines Cox PH

```python
from lifelines import CoxPHFitter
cph = CoxPHFitter()
cph.fit(df, duration_col='time', event_col='event',
        formula='age + bmi + treatment')

ps.tbl_regression(cph)                      # HRs, CIs, p-values
```

## sklearn

```python
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression().fit(X, y)
ps.tbl_regression(clf)                      # point estimates; no CIs
```

A footnote warns when CIs aren't available.
