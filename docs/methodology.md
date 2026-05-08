# Methodology

The engine uses a hybrid ensemble:

1. Polling component: weighted poll aggregation with method/population weights and
   pollster-level shrinkage.
2. Fundamentals component: historical lean, incumbency, economic/demographic context,
   turnout history, and campaign-finance proxy features.
3. Market component: public market-implied probabilities adjusted for spread and
   open interest.
4. Public-signal component: news/pageview/official-release features. These default
   to experimental until backtests prove value.
5. Simulation layer: race-level vote-share draws with correlated election error,
   producing winner, margin, turnout, recount, certification, and control outcomes.

This first implementation provides a deterministic, testable modeling contract over
fixture data. CmdStan/NumPyro backends can replace the polling component behind the
same artifact schema.

