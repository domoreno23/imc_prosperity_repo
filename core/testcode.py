#ML model used to predict coefficients of macarons for fair price value
import pandas as pd 
from sklearn.linear_model import LinearRegression

df = pd.read_csv("/Users/damianmoreno/imc_prosperity/observations.csv")

df_diff = df.diff().dropna()
correlations = df_diff.corr()
print(correlations["macaron_price"])

features = ['sunlight', 'sugar', 'transport', 'import', 'export']
x = df[features]
y = df['macaron_price']

model = LinearRegression()
model.fit(x,y)

#weights
weights = model.coef_
intercept = model.intercept_

print(weights)
print(intercept)