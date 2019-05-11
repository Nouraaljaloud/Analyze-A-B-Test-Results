# Analyze-A-B-Test-Results


Analyze A/B Test Results
You may either submit your notebook through the workspace here, or you may work from your local machine and submit through the next page. Either way assure that your code passes the project RUBRIC. Please save regularly.
This project will assure you have mastered the subjects covered in the statistics lessons. The hope is to have this project be as comprehensive of these topics as possible. Good luck!


Table of Contents
Introduction
Part I - Probability
Part II - A/B Test
Part III - Regression


Introduction
A/B tests are very commonly performed by data analysts and data scientists. It is important that you get some practice working with the difficulties of these
For this project, you will be working to understand the results of an A/B test run by an e-commerce website. Your goal is to work through this notebook to help the company understand if they should implement the new page, keep the old page, or perhaps run the experiment longer to make their decision.
As you work through this notebook, follow along in the classroom and answer the corresponding quiz questions associated with each question. The labels for each classroom concept are provided for each question. This will assure you are on the right track as you work through the project, and you can feel more confident in your final submission meeting the criteria. As a final check, assure you meet all the criteria on the RUBRIC.


Part I - Probability
To get started, let's import our libraries.

import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
%matplotlib inline
#We are setting the seed to assure you get the same answers on quizzes as we set up
random.seed(42)

1. Now, read in the ab_data.csv data. Store it in df. Use your dataframe to answer the questions in Quiz 1 of the classroom.
a. Read in the dataset and take a look at the top few rows here:

#read csv file 
df = pd.read_csv('ab_data.csv')
df.head()


# for row and column
df.shape
# 294478 rows and 5 column
(294478, 5)

c. The number of unique users in the dataset.
# unique user 
len(df['user_id'].unique())
290584

d. The proportion of users converted.
b1 = sum(df['converted']== 1)
b2 = 294478
def per (b1,b2):
    n1= b1*100
    return n1/b2
print (per (b1,b2))    
11.965919355605513


e. The number of times the new_page and treatment don't match.
new = df[df['landing_page']== 'new_page']
new2 = df[df['landing_page']== 'old_page']
treatment = sum(new.group !='treatment')
treatment2 = sum(new2.group =='treatment')
​
print(treatment + treatment2 )
3893


f. Do any of the rows have missing values?
# information about dataset 
df.info()

2. For the rows where treatment does not match with new_page or control does not match with old_page, we cannot be sure if this row truly received the new or old page. Use Quiz 2 in the classroom to figure out how we should handle these rows.
a. Now use the answer to the quiz to create a new dataset that meets the specifications from the quiz. Store your new dataframe in df2.
# use GitHub to find a best answer
drop = df[( (df.group == 'treatment')&(df.landing_page == 'old_page') )|( (df.group == 'control') &(df.landing_page == 'new_page') ) ].index
​
df2 = df.drop(drop)
# Double Check all of the correct rows were removed - this should be 0
df2[((df2['group'] == 'treatment') == (df2['landing_page'] == 'new_page')) == False].shape[0]

0
3. Use df2 and the cells below to answer questions for Quiz3 in the classroom.
a. How many unique user_ids are in df2?


len(df2['user_id'].unique())
290584

b. There is one user_id repeated in df2. What is it?
# to find duplicated
df2[df2.duplicated(['user_id'], keep=False)]['user_id']
1899    773192
2893    773192
Name: user_id, dtype: int64

c. What is the row information for the repeat user_id?
# get from duplicated function 
df2[df2['user_id'] == 773192]
user_id	timestamp	group	landing_page	converted

d. Remove one of the rows with a duplicate user_id, but keep your dataframe as df2.
# drob duplicated
#df2.drop_duplicates(inplace=True)
df2.drop([2893], inplace=True)

4. Use df2 in the cells below to answer the quiz questions related to Quiz 4 in the classroom.
a. What is the probability of an individual converting regardless of the page they receive?
df2.converted.mean()
0.11959708724499628

b. Given that an individual was in the control group, what is the probability they converted?
converted_control = df2[df2['group'] == 'control'].converted.mean()
print (converted_control)
0.1203863045

c. Given that an individual was in the treatment group, what is the probability they converted?
converted_treatment = df2[df2['group'] == 'treatment'].converted.mean()
print(converted_treatment)
0.118808065515

d. What is the probability that an individual received the new page?
new_page = sum(df2.landing_page == 'new_page')
new_page / df2.shape[0]
0.5000619442226688


e. Consider your results from parts (a) through (d) above, and explain below whether you think there is sufficient evidence to conclude that the new treatment page leads to more conversions.
there is not enough sufficient evidence to conclude that the new treatment page leads to more conversions.


Part II - A/B Test
Notice that because of the time stamp associated with each event, you could technically run a hypothesis test continuously as each observation was observed.
However, then the hard question is do you stop as soon as one page is considered significantly better than another or does it need to happen consistently for a certain amount of time? How long do you run to render a decision that neither page is better than another?
These questions are the difficult parts associated with A/B tests in general.

1. For now, consider you need to make the decision just based on all the data provided. If you want to assume that the old page is better unless the new page proves to be definitely better at a Type I error rate of 5%, what should your null and alternative hypotheses be? You can state your hypothesis in terms of words or in terms of 𝑝𝑜𝑙𝑑

H0: pnew - pold <= 0
H1: pnew - pold > 0 H0: The user receiving the new page is less than or equal to the user receiving the old page. H1: The user receiving the new page is greater than the user receiving the old page.

Use a sample size for each page equal to the ones in ab_data.csv. 

Perform the sampling distribution for the difference in converted between the two pages over 10,000 iterations of calculating an estimate from the null. 

Use the cells below to provide the necessary parts of this simulation. If this doesn't make complete sense right now, don't worry - you are going to work through the problems below to complete this problem. You can use Quiz 5 in the classroom to make sure you are on the right track.

a. What is the conversion rate for 𝑝𝑛𝑒𝑤
p
n
e
w
 under the null?
In [18]:

# mean 
Pnew = df2.converted.mean()
Pnew 
Out[18]:
0.11959708724499628
b. What is the conversion rate for 𝑝𝑜𝑙𝑑
p
o
l
d
 under the null? 

In [19]:

Pold = df2.converted.mean()
Pold
Out[19]:
0.11959708724499628
c. What is 𝑛𝑛𝑒𝑤
n
n
e
w
, the number of individuals in the treatment group?
In [20]:

Nnew = sum(df2.landing_page == 'new_page')
Nnew
Out[20]:
145310
d. What is 𝑛𝑜𝑙𝑑
n
o
l
d
, the number of individuals in the control group?
In [21]:

Nold = sum(df2.landing_page == 'old_page')
Nold
Out[21]:
145274
e. Simulate 𝑛𝑛𝑒𝑤
n
n
e
w
 transactions with a conversion rate of 𝑝𝑛𝑒𝑤
p
n
e
w
 under the null. Store these 𝑛𝑛𝑒𝑤
n
n
e
w
 1's and 0's in new_page_converted.
In [22]:

new_page_converted = np.random.choice([0, 1],Nnew, p=(Pnew, 1-Pnew))
f. Simulate 𝑛𝑜𝑙𝑑
n
o
l
d
 transactions with a conversion rate of 𝑝𝑜𝑙𝑑
p
o
l
d
 under the null. Store these 𝑛𝑜𝑙𝑑
n
o
l
d
 1's and 0's in old_page_converted.
In [23]:

old_page_converted = np.random.choice([0, 1],Nold, p=(Pold, 1-Pold))
g. Find 𝑝𝑛𝑒𝑤
p
n
e
w
 - 𝑝𝑜𝑙𝑑
p
o
l
d
 for your simulated values from part (e) and (f).
In [24]:

new_page_converted.mean() - old_page_converted.mean()
Out[24]:
-0.00085812637191773344
h. Create 10,000 𝑝𝑛𝑒𝑤
p
n
e
w
 - 𝑝𝑜𝑙𝑑
p
o
l
d
 values using the same simulation process you used in parts (a) through (g) above. Store all 10,000 values in a NumPy array called p_diffs.
In [25]:

new_simulation = np.random.binomial(Nnew, Pnew, 10000)/Nnew
old_simulation = np.random.binomial(Nold, Pold, 10000)/Nold
In [26]:

p_diffs = new_simulation - old_simulation
i. Plot a histogram of the p_diffs. Does this plot look like what you expected? Use the matching problem in the classroom to assure you fully understand what was computed here.
In [27]:

plt.hist(p_diffs)
Out[27]:
(array([   12.,    69.,   395.,  1181.,  2421.,  2759.,  2026.,   861.,
          236.,    40.]),
 array([-0.00459985, -0.0037333 , -0.00286675, -0.00200021, -0.00113366,
        -0.00026711,  0.00059943,  0.00146598,  0.00233253,  0.00319907,
         0.00406562]),
 <a list of 10 Patch objects>)

j. What proportion of the p_diffs are greater than the actual difference observed in ab_data.csv?
In [28]:

obs_diff = converted_treatment - converted_control 
In [29]:

p_diffs = np.array(p_diffs)
(p_diffs > obs_diff).mean()
Out[29]:
0.90639999999999998
k. Please explain using the vocabulary you've learned in this course what you just computed in part j. What is this value called in scientific studies? What does this value mean in terms of whether or not there is a difference between the new and old pages?
the data show, with error rate of 0.05, that the old page has a higher probability of convert rate than a new page.
l. We could also use a built-in to achieve similar results. Though using the built-in might be easier to code, the above portions are a walkthrough of the ideas that are critical to correctly thinking about statistical significance. Fill in the below to calculate the number of conversions for each page, as well as the number of individuals who received each page. Let n_old and n_new refer the the number of rows associated with the old page and new pages, respectively.
In [30]:

import statsmodels.api as sm
​
convert_old = sum(df2[df2['group'] == 'control'].converted)
convert_new = sum(df2[df2['group'] == 'treatment'].converted)
n_old = df2[df2['group'] == 'control'].converted.size 
n_new = df2[df2['group'] == 'treatment'].converted.size 
convert_old  ,convert_new, n_old, n_new 
/opt/conda/lib/python3.6/site-packages/statsmodels/compat/pandas.py:56: FutureWarning: The pandas.core.datetools module is deprecated and will be removed in a future version. Please use the pandas.tseries module instead.
  from pandas.core import datetools
Out[30]:
(17489, 17264, 145274, 145310)
m. Now use stats.proportions_ztest to compute your test statistic and p-value. Here is a helpful link on using the built in.
In [31]:

from scipy.stats import norm
​
z_score, p_value = sm.stats.proportions_ztest([convert_old, convert_new], [n_old, n_new], alternative='smaller')
z_score, p_value
Out[31]:
(1.3109241984234394, 0.90505831275902449)
n. What do the z-score and p-value you computed in the previous question mean for the conversion rates of the old and new pages? Do they agree with the findings in parts j. and k.?
z-score of 1.3116075339133115 does not exceed the critical value of 1.959963984540054, we fail to reject the null hypothesis that old page users has a better or equal converted rate than old page users. the converted rate for new page and old page have no difference. This result is the same as parts J. and K. result.
Part III - A regression approach
1. In this final part, you will see that the result you achieved in the A/B test in Part II above can also be achieved by performing regression.

a. Since each row is either a conversion or no conversion, what type of regression should you be performing in this case?
Logistic regression
b. The goal is to use statsmodels to fit the regression model you specified in part a. to see if there is a significant difference in conversion based on which page a customer receives. However, you first need to create in df2 a column for the intercept, and create a dummy variable column for which page each user received. Add an intercept column, as well as an ab_page column, which is 1 when an individual receives the treatment and 0 if control.
In [32]:

df2['intercept'] = 1
df2['ab_page'] = pd.get_dummies(df['group']) ['treatment']
df2.head()
Out[32]:
user_id	timestamp	group	landing_page	converted	intercept	ab_page
0	851104	2017-01-21 22:11:48.556739	control	old_page	0	1	0
1	804228	2017-01-12 08:01:45.159739	control	old_page	0	1	0
2	661590	2017-01-11 16:55:06.154213	treatment	new_page	0	1	1
3	853541	2017-01-08 18:28:03.143765	treatment	new_page	0	1	1
4	864975	2017-01-21 01:52:26.210827	control	old_page	1	1	0
c. Use statsmodels to instantiate your regression model on the two columns you created in part b., then fit the model using the two columns you created in part b. to predict whether or not an individual converts.
In [33]:

import statsmodels.api as sm
mod = sm.Logit(df2['converted'], df2[['intercept', 'ab_page']])
​
d. Provide the summary of your model below, and use it as necessary to answer the following questions.
In [34]:

summary = mod.fit()
summary.summary()
Optimization terminated successfully.
         Current function value: 0.366118
         Iterations 6
Out[34]:
Logit Regression Results
Dep. Variable:	converted	No. Observations:	290584
Model:	Logit	Df Residuals:	290582
Method:	MLE	Df Model:	1
Date:	Wed, 08 May 2019	Pseudo R-squ.:	8.077e-06
Time:	13:01:21	Log-Likelihood:	-1.0639e+05
converged:	True	LL-Null:	-1.0639e+05
LLR p-value:	0.1899
coef	std err	z	P>|z|	[0.025	0.975]
intercept	-1.9888	0.008	-246.669	0.000	-2.005	-1.973
ab_page	-0.0150	0.011	-1.311	0.190	-0.037	0.007
e. What is the p-value associated with ab_page? Why does it differ from the value you found in Part II?

Hint: What are the null and alternative hypotheses associated with your regression model, and how do they compare to the null and alternative hypotheses in Part II?
The p-value associated with ab_page in this regression model is 0.19 The null hypothesis associated with logistic regression is there is no relationship between the dependent and independent variables.this means there is no relationship between which page a user is shown and the conversion rate. part 2 is that conversion for a user receiving the new page is greater than conversion for a user receiving the old page. It differs from the value I found in Part II for the reason that they maintain different hypothesizes.
f. Now, you are considering other things that might influence whether or not an individual converts. Discuss why it is a good idea to consider other factors to add into your regression model. Are there any disadvantages to adding additional terms into your regression model?
Another factor can change the layout(color, picture )depend on the country. time to take to find the specific page
g. Now along with testing if the conversion rate changes for different pages, also add an effect based on which country a user lives in. You will need to read in the countries.csv dataset and merge together your datasets on the appropriate rows. Here are the docs for joining tables.
Does it appear that country had an impact on conversion? Don't forget to create dummy variables for these country columns - Hint: You will need two columns for the three dummy variables. Provide the statistical output as well as a written response to answer this question.
In [35]:

dfcou = pd.read_csv('countries.csv')
dfcou.head()
Out[35]:
user_id	country
0	834778	UK
1	928468	US
2	822059	UK
3	711597	UK
4	710616	UK
In [36]:

df3 = df2.merge(dfcou, on ='user_id')
df3.head()
Out[36]:
user_id	timestamp	group	landing_page	converted	intercept	ab_page	country
0	851104	2017-01-21 22:11:48.556739	control	old_page	0	1	0	US
1	804228	2017-01-12 08:01:45.159739	control	old_page	0	1	0	US
2	661590	2017-01-11 16:55:06.154213	treatment	new_page	0	1	1	US
3	853541	2017-01-08 18:28:03.143765	treatment	new_page	0	1	1	US
4	864975	2017-01-21 01:52:26.210827	control	old_page	1	1	0	US
In [37]:

df3[['CA', 'UK', 'US']] = pd.get_dummies(df3['country'])
logitc = sm.Logit(df3['converted'], df3[['intercept','ab_page','UK','US']])
result = logitc.fit()
result.summary()
Optimization terminated successfully.
         Current function value: 0.366113
         Iterations 6
Out[37]:
Logit Regression Results
Dep. Variable:	converted	No. Observations:	290584
Model:	Logit	Df Residuals:	290580
Method:	MLE	Df Model:	3
Date:	Wed, 08 May 2019	Pseudo R-squ.:	2.323e-05
Time:	13:01:22	Log-Likelihood:	-1.0639e+05
converged:	True	LL-Null:	-1.0639e+05
LLR p-value:	0.1760
coef	std err	z	P>|z|	[0.025	0.975]
intercept	-2.0300	0.027	-76.249	0.000	-2.082	-1.978
ab_page	-0.0149	0.011	-1.307	0.191	-0.037	0.007
UK	0.0506	0.028	1.784	0.074	-0.005	0.106
US	0.0408	0.027	1.516	0.130	-0.012	0.093
h. Though you have now looked at the individual factors of country and page on conversion, we would now like to look at an interaction between page and country to see if there significant effects on conversion. Create the necessary additional columns, and fit the new model.
Provide the summary results, and your conclusions based on the results.
In [38]:

df3['new_CA'] = df3['ab_page']*df3['CA']
df3['new_UK'] = df3['ab_page']*df3['UK']
df3['new_US'] = df3['ab_page']*df3['US']
df3.head()
Out[38]:
user_id	timestamp	group	landing_page	converted	intercept	ab_page	country	CA	UK	US	new_CA	new_UK	new_US
0	851104	2017-01-21 22:11:48.556739	control	old_page	0	1	0	US	0	0	1	0	0	0
1	804228	2017-01-12 08:01:45.159739	control	old_page	0	1	0	US	0	0	1	0	0	0
2	661590	2017-01-11 16:55:06.154213	treatment	new_page	0	1	1	US	0	0	1	0	0	1
3	853541	2017-01-08 18:28:03.143765	treatment	new_page	0	1	1	US	0	0	1	0	0	1
4	864975	2017-01-21 01:52:26.210827	control	old_page	1	1	0	US	0	0	1	0	0	0
In [39]:

logitN = sm.Logit(df3['converted'], df3[['intercept','ab_page','new_UK','new_US','UK','US']])
resultN = logitN.fit()
resultN.summary()
Optimization terminated successfully.
         Current function value: 0.366109
         Iterations 6
Out[39]:
Logit Regression Results
Dep. Variable:	converted	No. Observations:	290584
Model:	Logit	Df Residuals:	290578
Method:	MLE	Df Model:	5
Date:	Wed, 08 May 2019	Pseudo R-squ.:	3.482e-05
Time:	13:01:24	Log-Likelihood:	-1.0639e+05
converged:	True	LL-Null:	-1.0639e+05
LLR p-value:	0.1920
coef	std err	z	P>|z|	[0.025	0.975]
intercept	-2.0040	0.036	-55.008	0.000	-2.075	-1.933
ab_page	-0.0674	0.052	-1.297	0.195	-0.169	0.034
new_UK	0.0783	0.057	1.378	0.168	-0.033	0.190
new_US	0.0469	0.054	0.872	0.383	-0.059	0.152
UK	0.0118	0.040	0.296	0.767	-0.066	0.090
US	0.0175	0.038	0.465	0.642	-0.056	0.091
Finishing Up
P-values shown in the results.and in the case of the linear plot, the R-squared value is zero. when you have the zero for predicted R-square it means that your model is not able to predict any variation on response. and you definitely need to improve your data. we should keep old page, old page better then a new page.
Tip: Once you are satisfied with your work here, check over your report to make sure that it is satisfies all the areas of the rubric (found on the project submission page at the end of the lesson). You should also probably remove all of the "Tips" like this one so that the presentation is as polished as possible.
Directions to Submit
Before you submit your project, you need to create a .html or .pdf version of this notebook in the workspace here. To do that, run the code cell below. If it worked correctly, you should get a return code of 0, and you should see the generated .html file in the workspace directory (click on the orange Jupyter icon in the upper left).
Alternatively, you can download this report as .html via the File > Download as submenu, and then manually upload it into the workspace directory by clicking on the orange Jupyter icon in the upper left, then using the Upload button.
Once you've done this, you can submit your project by clicking on the "Submit Project" button in the lower right here. This will create and submit a zip file with this .ipynb doc and the .html or .pdf version you created. Congratulations!
In [40]:

from subprocess import call
call(['python', '-m', 'nbconvert', 'Analyze_ab_test_results_notebook.ipynb'])
Out[40]:
0
