# Reading the results

Start with the primary risk-forecast comparison, then inspect trading consequences and secondary diagnostics. Forecast MSE concerns log realized variance; it is not a return or a measure of alpha. The primary holdout has 127 month-end origins, December 2015–June 2026. Trading results cover January 2016–July 2026.

| Published table | Question and reading |
|---|---|
| [primary_hypothesis.csv](../results/primary_hypothesis.csv) | Do cross-sectional features improve L1 risk forecasts? Positive MSE improvement favors the larger model. Both intervals include zero; Holm-adjusted p=0.050728 for RAW and RES. |
| [forecast_results.csv](../results/forecast_results.csv) | How do the principal forecast models compare in design and holdout? Each row names its benchmark; adjacent rows do not automatically define a tested comparison. |
| [market_state_contrast.csv](../results/market_state_contrast.csv) | Does the market block contribute more to raw than residual momentum? The primary contrast uses RAW_c and RES on a common eligible universe. Positive values are the hypothesized direction. |
| [strategy_results.csv](../results/strategy_results.csv) | Do forecast models improve net trading outcomes? Compare like samples and costs. Headline Sharpe is Lo-adjusted; annual turnover measures traded notional relative to NAV. |
| [sharpe_differences.csv](../results/sharpe_differences.csv) | How uncertain are Sharpe differences relative to historical-volatility scaling? Read the cost and benchmark columns alongside the interval. |
| [transport_comparison.csv](../results/transport_comparison.csv) | Does transport add beyond L1+L2 and six linear copula moments? This comparison does not establish value beyond every possible dependence model. |
| [quadratic_control.csv](../results/quadratic_control.csv) | Does transport add after squares and interactions of the copula moments? This is a secondary control, separate from the primary hypothesis. |
| [state_features.csv](../results/state_features.csv) | Do filtered states add to the design-selected feature prefix? RAW and RES use different prefixes. Within this secondary family, two states are the main specification and three are a check. |
| [state_persistence.csv](../results/state_persistence.csv) | Does filtering improve on emission-only probabilities? This non-nested comparison uses paired loss intervals and descriptive two-sided tests. |
| [baseline_forecasts.csv](../results/baseline_forecasts.csv) | How do established predictors perform across variance, return and tail targets? Target-specific benchmarks differ; tail results have few events and remain descriptive. |
| [placebo_audit.csv](../results/placebo_audit.csv) | Does adding persistent noise appear useful under the amended estimator? Read mean incremental R² and rejection rates by series and number of added predictors. |
| [robustness.csv](../results/robustness.csv) | How do pre-specified construction and data assumptions affect results? These are sensitivity comparisons, not replacements chosen for their favorable outcomes. |
| [cost_sensitivity.csv](../results/cost_sensitivity.csv) | What happens at alternative notional trading costs? These are position-ledger replays; flat costs omit market impact and stock borrowing. |
| [registered_forecasts_superseded.csv](../results/registered_forecasts_superseded.csv) | What did the original shared-penalty estimator report? Retained for transparency; these forecasts were superseded before holdout evaluation. |
| [registered_strategies_superseded.csv](../results/registered_strategies_superseded.csv) | What trading results accompanied that original estimator? Historical evidence, not another candidate selected using the holdout. |
| [registered_placebos_superseded.csv](../results/registered_placebos_superseded.csv) | How did the original estimator behave under added noise? This table preserves test rejection rates for 200 RAW replications at each block size; the amended audit is a separate table. |

RAW ranks formation-period total returns; RES ranks residual-momentum scores; RAW_c uses raw rankings on the residual-eligible universe. L1-own is the HAR own-risk benchmark. L1 adds established predictors, L2 adds cross-sectional features, CM adds linear copula moments, Q adds quadratic terms, and OT adds transport distances. HMMF denotes filtered state probabilities and HMME emission-only probabilities; the trailing number gives the state count.

BSC denotes historical-volatility scaling. DM-transfer and BDP-transfer calibrate their baseline parameters on long French momentum history and apply them to the study's stock portfolios. Their transferred construction is part of the interpretation.

The mse_improvement column is benchmark loss minus model loss. Its confidence interval uses squared log-variance units for the primary target. The r2_oos column is a fraction relative to the row's named benchmark; multiply by 100 to express a percentage. Use cw_holm_pvalue for the pre-specified primary decision, rather than rounding it to 0.05 or substituting an unadjusted p-value.

Where reported, break-even cost means equality of annual arithmetic net means with BSC, not equality of Sharpe ratios. Missing crossings are not zero cost. Deflated Sharpe values are descriptive and depend on the recorded population of distinct strategy specifications.

| Figure | What it shows |
|---|---|
| [Momentum books](../figures/momentum_books.png) | Gross unscaled account histories, with RAW's worst months marked. These supply risk context, not forecast evidence. |
| [Feature redundancy](../figures/feature_redundancy.png) | Design-period relationships between established and added features. The descriptive audit does not select or residualize forecasting inputs. |
| [State timeline](../figures/state_timeline.png) | Filtered probabilities and centroid labels using the fit available at each origin. Numbered states are statistical summaries. |
| [Forecast loss](../figures/forecast_loss.png) | Cumulative paired squared-loss differences against L1. The plot's common reference differs from some registered secondary benchmarks. |
| [Net equity](../figures/net_equity.png) | Accounts after the headline trading cost. Daily marks do not imply daily rebalancing. |

The primary result fails to establish incremental predictive value; it does not prove that the effect is zero. Favorable forecast point estimates do not establish net profitability. Read [research design](research_design.md) for dependence adjustments, sample limitations and the retained negative findings.
