
# Web traffic analysis
The project  is based on the website viewership
prediction and analysis.

## Appendix

Variables and its abbreaviation is detailed here

1.OPD-Overview page views (desktop)
2.OPM-Overview page views (mobile)
3.OPT-Overview page views (Total)
4.OUD-Overview unique visitors (desktop)
5.OUM-Overview unique visitors (Mobile)
6.OUT-Overview unique visitors (Total)
7.LPD-Life page views (desktop)
8.LPM-Life page views (Mobile)
9.LPT-Life page views (Total)
10.LUD-Life unique Visitors (desktop)
11.LUM-Life unique Visitors (Mobile)
12.LUT-Life unique Visitors (Total)
13.JPVD-Jobs page views (desktop)
13.JPVM-Jobs page views (Mobile)
14.JPVT-Jobs page views (total)
15.JUVD-Jobs unique visitors (desktop)
16.JUVM-Jobs unique visitors (Mobile)
17.JUVT-Jobs unique visitors (total)
18.TPVD-Total page views (desktop)
19.TPVM-Total page views (mobile)	
20.TPVT-Total page views (total)	
21.TUVD-Total unique visitors (desktop)
22.TUVM-Total unique visitors (mobile)
23.TUVT-Total unique visitors (total)






## Requirements 
>Pycharm community edition 
>Pandas
>Numpy 
>Matplotlib
>Seaborn
>Statsmodels(ARIMA)##Acknowledgment
[Github]:(https://github.com/Manishms18/Air-Passengers-Time-Series-Analysis)
[Machine Learning plus]:https://www.machinelearningplus.com/time-series/arima-model-time-series-forecasting-python/
## Sample codes


'''code example

from statsmodels.tsa.stattools import adfuller
from numpy import log
result = adfuller(df.OPT.dropna())
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])



model = ARIMA(df.OPT, order=(1,1,2))
model_fit = model.fit()
print(model_fit.summary())
}
## Screenshots
![myimage-alt-tag](url-to-image)


Overview Page total visualization (Moblie v/s Desktop)
![Photo](https://github.com/madbuzz/Time_Series/blob/main/OPT%20%7BMD%7D.png)

Plot of all variables
![Photo](https://github.com/madbuzz/Time_Series/blob/main/1.png)


## Prediction

Final prediction after model fiting 

![Photo](https://github.com/madbuzz/Time_Series/blob/main/Final_prediction.png)