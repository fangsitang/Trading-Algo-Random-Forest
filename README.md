# Investor Overview

## Fund Profile

Our fund seeks to deliver robust, risk-adjusted returns through a highly diversified portfolio spanning multiple asset classes and geographies. By leveraging advanced machine learning techniques, we identify and capitalize on asset momentum trends, enhancing our ability to capture growth opportunities. At the core of our strategy is a fundamentals-driven, top-down approach to asset allocation, ensuring that every investment decision is aligned with a long-term, data-informed view of market dynamics.

## Investment Strategy

Our trading strategy employs a multi-step approach that integrates technical analysis signals, machine learning models, and macroeconomic data to optimize asset allocation. 


## Fund Fact Sheet






Extra : To learn how we leverage machine learning, technical analysis and macroeconomic data, please refer to our [report] (https://github.com/fangsitang/Trading-Algo-Random-Forest/blob/bf3ad9d70a75e0b76c86fb454aa724c2de76731c/Rapport_ML_Trend-Following.pdf).


1) Firstly, we compute eight key technical indicators to inform our two models, using Ridge regression to forecast monthly returns and Random Forest to predict upward movement probabilities.
2) These predictions are converted into z-scores used to rank individual assets.
3) The final weights are adjusted based on the prevailing macroeconomic regime identified through K-means clustering. 
