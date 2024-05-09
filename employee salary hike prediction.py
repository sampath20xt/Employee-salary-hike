 

import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
import warnings
warnings.filterwarnings('ignore')
salary_data = pd.read_csv('Salary_Data.csv')
salary_data
salary_data.shape
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 30 entries, 0 to 29
Data columns (total 2 columns):
YearsExperience 30 non-null float64
Salary 30 non-null float64
dtypes: float64(2)
memory usage: 608.0 bytes
salary_data.isna().sum()
YearsExperience 0
Salary 0
dtype: int64
salary_data.describe()
plt.subplot(121)
plt.boxplot(salary_data['Salary'])
plt.title('Salary Hike')
plt.subplot(122)
plt.boxplot(salary_data['YearsExperience'])
plt.title('Years of Experience')
plt.show()
salary_data.corr()
sns.heatmap(data= salary_data, annot=True)
plt.show()
sns.regplot(x='YearsExperience', y='Salary', data= salary_data)
plt.title('YearsExperience Vs Salary', fontsize = 14)
plt.show()
sns.scatterplot(x ='YearsExperience', y ='Salary', data= salary_data)
plt.title('Homoscedasticity', fontweight = 'bold', fontsize = 16)
plt.show()
salary_data.var()
YearsExperience 8.053609e+00
Salary 7.515510e+08
dtype: float64
sns.displot(salary_data['YearsExperience'], bins = 10, kde = True)
plt.title('Before Transformation')
sns.displot(np.log(salary_data['YearsExperience']), bins = 10, kde = True )
plt.title('After Transformation')
plt.show() 
labels = ['Before Transformation','After Transformation'
sns.distplot(salary_data['YearsExperience'], bins = 10, kde = True)
sns.distplot(np.log(salary_data['YearsExperience']), bins = 10, kde = Tru e)
plt.legend(labels)
plt.show()
sm.qqplot(np.log(salary_data['YearsExperience']), line = 'r')
plt.title('No transformation')
sm.qqplot(np.sqrt(salary_data['YearsExperience']), line = 'r')
plt.title('Log transformation')
sm.qqplot(np.sqrt(salary_data['YearsExperience']), line = 'r')
plt.title('Square root transformation')
sm.qqplot(np.cbrt(salary_data['YearsExperience']), line = 'r')
plt.title('Cube root transformation')
plt.show()
sns.displot(salary_data['Salary'], bins = 10, kde = True)
plt.title('Before Transformation')
sns.displot(np.log(salary_data['Salary']), bins = 10, kde = True)
plt.title('After Transformation')
plt.show()
labels = ['Before Transformation','After Transformation']
sns.distplot(salary_data['Salary'], bins = 10, kde = True)
sns.distplot(np.log(salary_data['Salary']), bins = 10, kde = True)
plt.title('After Transformation')
plt.legend(labels)
plt.show()
sm.qqplot(salary_data['Salary'], line = 'r')
plt.title('No transformation')
sm.qqplot(np.log(salary_data['Salary']), line = 'r')
plt.title('Log transformation')
sm.qqplot(np.sqrt(salary_data['Salary']), line = 'r')
plt.title('Square root transformation')
sm.qqplot(np.cbrt(salary_data['Salary']), line = 'r')
plt.title('Cube root transformation')
plt.show()
linear_model = smf.ols('Salary~YearsExperience', data = salary_data).fit()
print('R-squared :',linear_model.rsquared.round(3)) #Overall Contribution of Predictors
print('Adj.R-squared :',linear_model.rsquared_adj.round(3)) #Overall Contribution of Predictors
print('AIC Value :',linear_model.aic.round(3)) #Error Impurity
print('BIC Value :',linear_model.bic.round(3)) #Error Impurity
R-squared : 0.957
Adj.R-squared : 0.955
AIC Value : 606.882
BIC Value : 609.685
linear_model1 = smf.ols('np.sqrt(Salary)~np.sqrt(YearsExperience)', data = salary_data).fit()
print('R-squared :',linear_model1.rsquared.round(3))
print('Adj.R-squared :',linear_model1.rsquared_adj.round(3))
print('AIC Value :',linear_model1.aic.round(3))
print('BIC Value :',linear_model1.bic.round(3))
R-squared : 0.942
Adj.R-squared : 0.94
AIC Value : 237.046 
BIC Value : 239.848
linear_model2 = smf.ols('np.cbrt(Salary)~np.cbrt(Years Experience)', data = salary_data).fit()
print('R-squared :',linear_model2.rsquared.round(3))
print('Adj.R-squared :',linear_model2.rsquared_adj.round(3))
print('AIC Value :',linear_model2.aic.round(3))
print('BIC Value :',linear_model2.bic.round(3))
linear_model3 = smf.ols('np.log(Salary)~np.log(Years Experience)', data = salary_data).fit()
print('R-squared :',linear_model3.rsquared.round(3))
print('Adj.R-squared :',linear_model3.rsquared_adj.round(3))
print('AIC Value :',linear_model3.aic.round(3))
print('BIC Value :',linear_model3.bic.round(3))
R-squared : 0.905
Adj.R-squared : 0.902
AIC Value : -42.417
BIC Value : -39.615
linear_model.params
linear_model.params
Intercept 25792.200199
YearsExperience 9449.962321
dtype: float64
print(linear_model.tvalues,'\n',linear_model.pvalues)
Intercept 11.346940
YearsExperience 24.950094
dtype: float64
Intercept 5.511950e-12
YearsExperience 1.143068e-20
dtype: float64
linear_model.rsquared, linear_model.rsquared_adj
(0.9569566641435086, 0.9554194021486339)
sm.qqplot(linear_model.resid, line = 'q')
plt.title('Normal Q-Q plot of residuals of Model without any data transformation')
plt.show()
def get_standardized_values( vals ):
return (vals - vals.mean())/vals.std()
plt.scatter(get_standardized_values(linear_model.fittedvalues), get_standardized_values(linear_model.resid))
plt.title('Residual Plot for Model without any data transformation')
plt.xlabel('Standardized Fitted Values')
plt.ylabel('Standardized Residual Values')
plt.show()
from sklearn.metrics import mean_squared_error
linear_model1_pred_y =np.square(linear_model1.predict(salary_data['YearsExperience']))
linear_model2_pred_y =pow(linear_model2.predict(salary_data['YearsExperience']),3)
linear_model3_pred_y =np.exp(linear_model3.predict(salary_data['YearsExperience']))

linear_model1_rmse =np.sqrt(mean_squared_error(salary_data['Salary'], linear_model1_pred_y))
linear_model2_rmse =np.sqrt(mean_squared_error(salary_data['Salary'], linear_model2_pred_y))
linear_model3_rmse =np.sqrt(mean_squared_error(salary_data['Salary'], linear_model3_pred_y))
print('Linear Model =', np.sqrt(linear_model.mse_resid),'\n' 'Linear Model1=', linear_model1_rmse,'\n' 'Linear
Model2=', linear_model2_rmse,'\n' 'Linear Model3=', linear_model3_rmse)
Linear Model = 5788.315051119395
Linear Model1= 5960.64709617431
Linear Model2= 6232.815455835849
Linear Model3= 7219.716974372793
rmse = {'Linear Model': np.sqrt(linear_model.mse_resid), 'Linear Model1': linear_model1_rmse, 'Linear Mode
l2': linear_model3_rmse, 'Linear Model3' : linear_model3_rmse}
min(rmse, key=rmse.get)
'Linear Model'
predicted = pd.DataFrame()
predicted['YearsExperience'] = salary_data.YearsExperience
predicted['Salary'] = salary_data.Salary
predicted['Predicted_Salary_Hike'] = pd.DataFrame(linear_model.predict(predicted.YearsExperience))