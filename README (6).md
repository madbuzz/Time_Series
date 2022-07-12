
# Website traffic analysis
The project  is based on the website viewership
prediction and analysis.
The objective of the project is to predict the Overview page total viewership of the 
website from learning the previous data set.
ARIMA and LSTM model is used for the prediction.The lSTM model has been used to predict the 
next 30 days viewership of the total oveierview page.
## Appendix

Variables and its abbreaviation is detailed here

>OPD-Overview page views (desktop)

>OPM-Overview page views (mobile)

>OPT-Overview page views (Total)

>OUD-Overview unique visitors (desktop)

>OUM-Overview unique visitors (Mobile)

>OUT-Overview unique visitors (Total)

>LPD-Life page views (desktop)

>LPM-Life page views (Mobile)

>LPT-Life page views (Total)

>LUD-Life unique Visitors (desktop)

>LUM-Life unique Visitors (Mobile)

>LUT-Life unique Visitors (Total)

>JPVD-Jobs page views (desktop)

>JPVM-Jobs page views (Mobile)

>JPVT-Jobs page views (total)

>JUVD-Jobs unique visitors (desktop)

>JUVM-Jobs unique visitors (Mobile)

>JUVT-Jobs unique visitors (total)

>TPVD-Total page views (desktop)

>TPVM-Total page views (mobile)	

>TPVT-Total page views (total)	

>TUVD-Total unique visitors (desktop)

>TUVM-Total unique visitors (mobile)

>TUVT-Total unique visitors (total)






## Requirements 
>Pycharm community edition 
>Pandas
>Numpy 
>Matplotlib
>Seaborn
>Statsmodels(ARIMA)
## Screenshots

Overview Page total visualization (Moblie v/s Desktop)
![Photo](https://github.com/madbuzz/Time_Series/blob/main/OPT%20%7BMD%7D.png)

Plot of all variables
![Photo](https://github.com/madbuzz/Time_Series/blob/main/1.png)


## Prediction

Final prediction after model fiting 

![Photo](https://github.com/madbuzz/Time_Series/blob/main/Final_prediction.png)
## Code example

    import os
    import warnings
    warnings.filterwarnings('ignore')
    import numpy as np 
    import seaborn as sns
    import pandas as pd
    import matplotlib.pyplot as plt
    plt.style.use('fivethirtyeight') 
    # Above is a special style template for matplotlib, highly useful for visualizing time series data
    %matplotlib inline
    import statsmodels.api as sm
    from numpy.random import normal, seed
    from scipy.stats import norm
    from statsmodels.tsa.arima_model import ARMA
    from statsmodels.tsa.stattools import adfuller
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
    from statsmodels.tsa.arima_process import ArmaProcess
    from statsmodels.tsa.arima_model import ARIMA
    import math
    from sklearn.metrics import mean_squared_error
    model = ARIMA(df.OPT, order=(1,1,1))
    model_fit = model.fit(disp=0)
    print(model_fit.summary())
    residuals = pd.DataFrame(model_fit.resid)
    fig, ax = plt.subplots(1,2)
    residuals.plot(title="Residuals", ax=ax[0])
    residuals.plot(kind='kde', title='Density', ax=ax[1])
    plt.show(


## Acknowledgment

[Github](https://github.com/Manishms18/Air-Passengers-Time-Series-Analysis)

[Machine Learning plus](https://www.machinelearningplus.com/time-series/arima-model-time-series-forecasting-python/)### Sample input
The sample input of the code is the 
! [Photo]("C:\Users\Athul\Desktop\Input.png")