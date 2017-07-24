# Cholesterol-Risk-Factors
Designed a cholesterol risk factor predictor using random forests, feature engineering and classifying patients based on cholesterol levels
dhar Meda,  Sharma,  Garrison
Health Data Mining
Due: 7/23/2017
Final Project Report:
TITLE: Evaluating Risk Factors to Predict High Cholesterol Levels
TOPIC: Evaluating Risk Factors to Predict High Cholesterol Levels 
➔	We will be evaluating high cholesterol risk by looking at its association with a patient’s nutritional intake, demographics and social aspects. 

HYPOTHESIS: Nutritional intake, socioeconomic status and demographics will affect the cholesterol levels of an individual.

Part A: Introduction and background of the problem; []   
With approximately 71 million people with High Cholesterol, it is a prevalent issue within the United States today. One’s cholesterol level is considered ‘borderline high risk’ from 200-240 mg/DL while anything above 240 mg/DL is ‘high risk’ for heart disease, a common cause of death within the country. The health condition is known as a ‘silent condition’ as it does not present symptoms to the outside world. It is important to note that in some instances high cholesterol levels are considered genetic, and thus are influenced by patient characteristics , such as Age, Gender and Race or Ethnicity, and others are environmental. The Nutrition Source at Harvard says that “the biggest influence on blood cholesterol level is the mix of fats and carbohydrates in your diet.” The goal of our research is to see if nutrition is in fact the highest risk factor, or if high cholesterol is more affected by social or health aspects. Based on prior research, we feel that nutritional intake will have the biggest impact on cholesterol levels.

In terms of how total cholesterol is actually calculated, it consists of the “the sum of high-density lipoprotein, HDL (good cholesterol), plus low-density lipoprotein, LDL (bad cholesterol), plus 20 percent of your triglycerides, a type of fat carried in your blood”. Often ‘Direct HDL-Cholesterol’ is also a common measurement observed. This solely measures HDL, which consists of “about 1/4 to 1/3 of blood cholesterol.” Whereas HDL, LDL, triglycerides and another type of cholesterol called Lp(a) are components making up total cholesterol count [ehow].



While these genetic patient characteristics are often uncontrollable elements, there are also controllable risk factors such as one’s Diet, Weight and their amount of Physical Activity:
●	Cholesterol can be described as a ‘waxy, fat-like substance made in the liver’. It is found in foods such as meat, eggs and dairy products.
●	LDL cholesterol levels can go up while HDL (the ‘good’ cholesterol) levels can go down when you are overweight
●	To raise HDL levels and lose weight, increased physical activity is essential

Part B: Data description; []
Our Dataset is the “Health and Nutrition” Dataset from the Center for Disease Control. The data was sourced from the National Health and Nutrition Examination Survey (NHANES), a survey meant to provide pertinent information regarding the current health status of United States citizens. The program, which began in the early 1960’s, uses surveys to specifically gather data on the nutritional status of both adults and children in the US, through both a home interview and a health examination, following written consent of all participants.The hope of these surveys is that it will provide enough information to influence and improve health within the US population. 

Most of the datasets created from these surveys are open to the public through the CDC, and are separated by topic: Demographics, Dietary, Examination, Laboratory, and Questionnaire. For our specific project, the desired variables were spread across multiple surveys, meaning we needed to create a dataset of our own, using the patient ID (PAT_ID) as our key. The dataset our team created includes 5658 sample accounts from individual citizens, and 33 variables sourced from multiple NHANES surveys. The variables we included consist of nutritional characteristics such as “# of frozen meals/ pizza in the past 30 days,” “# of ready-to-eat foods in the last 30 days,” “Amount of Milk Consumption in the past 30 days,” along with characteristics specifically pertaining to the health and social aspects of those filling out the survey. The health characteristics include variables such as the following; “Direct HDL Cholesterol,” “# with Hepatitis A Antibody,” and ”Pregnancy Status.” For the purpose of this Data Mining project, we will initially be working with given variables in our data set, to determine which nutritional, social, and health variables have the highest correlation to our health topic- “High Cholesterol- mg/dL.”

 
Variables in our created dataset:
1.	Patient ID
2.	Gender
3.	Age in Years at Screening
4.	Race/ Hispanic Origin w/ NH Asian
5.	Served Active Duty in US Armed Forces
6.	Served in a Foreign Country
7.	Country of Birth
8.	Education Level- Adults 20+
9.	Marital Status
10.	Pregnancy Status at Exam
11.	Direct HDL- Cholesterol (mg/dL)
12.	Direct HDL- Cholesterol (mmol/L)
13.	Second HDL
14.	Second HDL SI
15.	Third HDL
16.	Third HDL SI
17.	Hepatitis A Antibody
18.	SP ever had pain or discomfort in chest
19.	SP get it walking uphill or in a hurry
20.	Annual Household Income
21.	How Healthy is the Diet
22.	Past 30 day milk consumption
23.	How often drank milk age 5-12
24.	How often drank milk age 13-17
25.	How often drank milk age 18-35
26.	# meals from fast food or pizza place
27.	. of ready-to eat foods in past 30 days
28.	. of frozen meals/ pizza in past 30 days
29.	Eat at fast food/ pizza places
30.	Covered by health insurance
31.	Time when no insurance in past year?
32.	Total Cholesterol (mg/dL)
33.	Total Cholesterol (mmol/L)
 

Following creation of our dataset, further research had to be done to determine what some of the labels signify. While some were straightforward, such as “gender” or “country of birth,” other categorical variables required further clarification on the values. The Key is below for both continuous and categorical variables.

Key: For Categorical Variables
*How healthy it the Diet? [1-excellent, 2-very good, 3-good, 4-fair, 5-poor, 7-refused, 9-don’t know]
*Past 30 day milk consumption [0-never, 1-rarely <1x/week, 2-sometimes 1x/week, 3-often 1x/day, 4-varied, 7-refused, 9-don’t know]
*How often drank milk (age 5-12), (age 13-17), (age 18-35) [0-never, 1-rarely <1x/week, 2-sometimes 1x/week, 3-often 1x/day, 4-varied, 7-refused, 9-don’t know]
*Eat at fast food/ pizza places [1-yes, 2-no, 7-refused, 9-don’t know]
*SP ever had pain or discomfort in chest [1-yes, 2-no, 7-refused, 9-don’t know]
*SP get it walking uphill or in a hurry [1-yes, 2-no, 7-refused, 9-don’t know]
*Gender [1-male, 2-female]
*Race/ Hispanic origin w/ NH Asian [1-mexican american, 2-other hispanic, 3-non hispanic white, 4-non hispanic black, 5-non hispanic asian, 6-non hispanic asian, 7-other race incl. Multi-racial]
*Served active duty in US Armed Forces [1-yes, 2-no, 7-refused, 9-don’t know]
*Served in a foreign country [1-yes, 2-no, 7-refused, 9-don’t know]
*Country of birth [1-born in 50 US states or D.C, 2-others, 77-refused, 99-don’t know]
*Education level- Adults 20+ [1-less than 9th grade, 2- 9to11th grade or 12 w/ no diploma, 3-high school graduate/GED, 4- some college or AA degree, 5-college graduate or above, 7-refused, 9-don’t know]
*Marital Status [1-married, 2-widowed, 3-divorced, 4-separated, 5-never, 6-living with partner, 77-refused, 99-don’t know]
*Pregnancy status exam [1-yes positive pregnancy test, 2-not pregnant at exam, 3-cannot determine]
*Annual household income [1- $0 to $4,999, 2-$5,000 to $9,999, 3-$10,000 to $14,999, 4-$15,000 to $19,999, 5-$20,000 to $24,999, 6-$25,000 to $34,999, 7-$35,000 to $44,999, 8-$45,000 to $54,999, 9-$55,000 to $64,999, 10- $65,000 to $75,999, 12-$20,000 and over, 13-under $20,000, 14-$75,000 to $99,000, 15-$100,000 and over, 77-refused, 99-don’t know]
*Hepatitis A antibody [1-positive, 2-negative, 3-indeterminate]

Key: Continuous Variables
*# of meals from fast food or pizza place [1 to 21- range of values, 0-none, 5555-more than 21, 7777-refused, 9999-don’t know]
*# of ready-to-eat foods in past 30 days [1 to 180- range of values, 0-never, 7777-refused, 9999-don’t know]
*# of frozen meals/ pizza in past 30 days [1 to 180- range of values, 0-never, 7777-refused, 9999-don’t know]
*Age in years at screening [0 to 79- range of values, 80-80 years or older]
*Direct HDL-Cholesterol (mg/dL) [10 to 173 range of values]
*Direct HDL-Cholesterol (mmol/L) [0.26 to 4.47 range of values]
*Total Cholesterol (mg/mL) [69 to 813 range of values]
*Total Cholesterol (mmol/L) [1.78 to 21.02 range of values]
*Second HDL (mg/mL) [range of values]
*Second HDL (mmol/L) [range of values]
*Third HDL (mg/mL) [range of values]
*Third HDL (mmol/L) [range of values]
*Patient ID [sequential values]

Part C: Tool/algorithm description; [, , ]

Prediction Model: []
1)	Pre-Processing: We replaced all NA values in the dataset with the mean of each column. This allowed us to retain as much data as possible, rather than omitting columns. We were left with 5658 instances [patients]. In addition, we changed the outcome variable of “Total Cholesterol (mg/dL)” from a continuous variable to a categorical variable using risk levels we determined from our initial research. They are as follows

Risk Level #1: NO/ low  Risk <200 mg/dL == [value in dataset represented as 0]
Risk Level #2: Borderline High Risk <=200-240 mg/dL == [value in dataset represented as 1]
Risk Level #3: High Risk for Heart Disease >240 mg/dL == [value in dataset represented as 2]
 
2) 	Building Model:  We used a Random Forest Algorithm to build our Model. We tested our model on estimators listed in Part D: Model Building to determine accuracy.
3)   Feature Selection: We ran our Random Forest Model to determine how much each feature is influencing our predictive model outcome of “Cholesterol Risk Levels,” listed above, through accuracy value. We then chose the variables of the highest importance to our model through determination of an accuracy threshold. The important features were then used to rebuild our model to get a higher accuracy
3)	Rebuild Model: Our Prediction Model using the Random Forest Algorithm was rebuilt using our selected features.
4)  Testing: Use the testing data to score the model built using the training data.

Bivariate Analysis: [, ]: 
Analysis was performed on all variables included in our dataset to confirm significance of features impacting our outcome of “Total Cholesterol (mg/dL).”
1)	Testing: Used simple linear regression for continuous variables and a t-test for our categorical variables. Our Bivariate Analysis looks at the continuous outcome of “Total Cholesterol (mg/dL)” instead of the categorical outcome of “Risk Levels” used in our predictive model.
2)	Interpretation: Looked at the p-values of all 33 variables to determine significance of variable in relation to outcome. Chose the top variables.

Part D: Comprehensive experimental results and analysis; []
PREDICTION MODEL:
Preprocessing:
Based on our domain knowledge we put together 33 different features including nutritional, socioeconomic and demographics for predicting high risk cholesterol. We had 13.1% value for missing data within the entire dataset. The “NA” values within the dataset were replaced by the means of the respective columns, thus keeping the dataset with the original count of 5658 observations. Our outcome variable in our prediction model is “Total Cholesterol (mg/dl),” which was divided into three categories (0,1,2) based on the Cholesterol Risk Levels published by NHLBI (National Heart, Lung, and Blood Institute). Input data had variables like PAT_ID and other object data which were removed to prepare the data for building model. 
Model Building:
After the preprocessing of the data, we initially chose to use the random forests model on our 27 independent variables and “Cholesterol Risk Level” outcome. Random Forests resulted with an accuracy of 65% with 500 estimators at a random state of 42. As the classification problem is to measure the risk levels, precision has to be very high as not to miss any patients with elevated cholesterol (True Positives). However, accuracy is not the best measure in deciding whether random forest is the best model to measure our hypothesis. For the Random Forest Model, we saw a precision of 0.68 for the class of 0 (low risk level), 0.35 for the class of 1 (borderline high risk) and 0.18 for the class of 3 (high risk patients). In addition, we saw an average precision of 0.55 and a recall of 0.64, which is quite low for analysis of a healthcare problem. Below, Figure (1) shows the importance of different features that were used in the prediction of “Cholesterol Risk Levels.” We then went on to see how other classifiers worked using the same data to check for better accuracy.

 

Figure (1): Feature-importance for the prediction of “Cholesterol Risk Levels.” [Random Forest Classifier]
 

Evaluation and Model Selection:
In this step we have used six different classifiers namely Passive Aggressive Classifier, Gradient boosting classifier, Support vector machine, K-nearest neighbor, Dumb classifier and logistic regression to select the best classifier for our problem. Below are the accuracies for each model:
Passive Aggressive Classifier:                   0.54
Gradient Boosting Classifier:                      0.67
Support Vector Machine(SVM):                  0.66
Random Forest Classifier:                          0.64
K Nearest Neighbor Classifier:                   0.63
Logistic Regression:                                   0.66
Dump Classifier:                                         0.66

As discussed above, accuracy is not the best measure for deciding a model’s performance, we have plotted a confusion matrix to see each model's precision, recall and f-1 score. Below is Figure (2) showing confusion matrices of each algorithm.

Figure (2): Confusion Matrix of Various Classifiers


Based on the figure above, it is evident that the Gradient Boosting algorithm has the highest number of both true positives (3640) and the least number of false positives (1691) compared to the others. For our particular problem we have identified that the Gradient Boosting algorithm is the best model with an accuracy of 67%, precision of 60%, recall of 67% and f- score of 0.57. Below are the precision, recall and f- scores for all the six models.

Passive Aggressive Classifier:
              precision    recall  f1-score   support
          0       0.67      0.64      0.65      3746
          1       0.25      0.35      0.29      1343
          2       0.10      0.04      0.06       569
avg / total       0.52      0.51      0.51      5658


Gradient Boosting Classifier:
              precision    recall  f1-score   support
          0       0.68      0.97      0.80      3746
          1       0.44      0.09      0.15      1343
          2       0.46      0.03      0.06       569
avg / total       0.60      0.67      0.57      5658


Support vector machine(SVM):
              precision    recall  f1-score   support
          0       0.66      0.99      0.79      3746
          1       0.26      0.01      0.03      1343
          2       0.00      0.00      0.00       569
avg / total       0.50      0.66      0.53      5658


Random Forest Classifier:
              precision    recall  f1-score   support
          0       0.69      0.91      0.78      3746
          1       0.35      0.17      0.23      1343
          2       0.27      0.05      0.08       569
avg / total       0.57      0.65      0.58      5658


K Nearest Neighbor Classifier:
              precision    recall  f1-score   support
          0       0.68      0.88      0.77      3746
          1       0.32      0.18      0.23      1343
          2       0.24      0.03      0.05       569
avg / total       0.55      0.63      0.57      5658


Logistic Regression:
              precision    recall  f1-score   support
          0       0.67      0.99      0.80      3746
          1       0.30      0.01      0.02      1343
          2       0.17      0.00      0.00       569
avg / total       0.53      0.66      0.53      5658

Dump Classifier:
              precision    recall  f1-score   support
          0       0.66      1.00      0.80      3746
          1       0.00      0.00      0.00      1343
          2       0.00      0.00      0.00       569
avg / total       0.44      0.66      0.53      5658

Figure (3) below shows the variable importance vs relative importance graph for the Gradient Boosting classification algorithm. The feature importances did not change much compared to the Random Forests algorithm but Gradient Boosting’s performance was much better compared to any other classifier.

Figure (3): Feature-importance for the prediction of “Cholesterol Risk Levels.” [Gradient Boosting Classifier]



Optimization:
After evaluation and model selection, we have looked at predicted vs. actual plots and deviance vs. boosting iteration curves to determine different parameters like the training and testing split, number of estimators selection, type of loss and learning rate of the algorithm. Below is the predicted vs. actual scatter plot with levels 0,1 and 2 for both actual and predicted values distributed from 0- 2.5. As seen below in Figure (4) we can see that there is a higher number of predictions for 0 and 1 levels, whereas fewer predictions for the level 2 outcome due to skewed data.

Figure (4): Actual vs. Predicted Scatter Plot

Below Figure (5) shows Deviance vs. Boosting Iterations where the red line represents the testing error and blue line, the training error. The point at which both graphs meet is the maximum number of iterations we can use. In this case, the test error stays constant after 500 whereas the training error deviates (decreases) to 3000 due to overfitting. 



Figure (5): Deviance vs. Boosting Iterations


Optimized Parameters:

n_estimators: 500
max_depth: 3
learning_rate: 0.02
loss: ls(least square)

Considering we used a multi-class outcome, we compared against mean squared error values to select the best parameters. After adjusting the Gradient Boosting classifier using the optimized parameters above, we obtained the best, mean squared error value of 0.384 (compared to the previous values of 0.403, 0.4 and 0.483 when tweaking the parameters). 


BIVARIATE ANALYSIS

Testing: We tested all 27 final variables and their effect on “Total Cholesterol (mg/dL) to confirm that our Predictive Model had chosen the most significant variables in our dataset.

TABLE 1: BIVARIATE ANALYSIS: Association of variables with Total Cholesterol (mg/L) 



Part E: Results discussion and insights []
Results:
Our original hypothesis states that nutritional intake, socioeconomic status and demographics will affect the cholesterol levels of an individual.
In order to test our hypothesis, we carried out preprocessing as our first step. This resulted in obtaining a cleaner dataset, that was ready for multi-class classification. After building our model using the Random Forest Classifier, we gained 65% accuracy and a precision of 0.55. However, once compared with the results of using our dataset to create a model with other classifiers, we came to a different conclusion. Our final predictive model selection resulted in utilization of the Gradient Boosting Classifier, which delivered the highest accuracy and precision values.
After conducting our predictive model analysis, and choosing our classifier, we saw that Direct HDL, Second HDL, Third HDL, Annual Household Income and Age in years at screening, had a higher effect on our outcome, “Cholesterol Risk Level”--No/low risk level, borderline high risk level, and high risk level. As seen in Figure (3), these variables were top 5 out of the 27 predictors retrieved from our predictive model, which we attained after running the feature-importances argument. We expected attributes such as Second, Third and Direct HDL to have an impact as they directly contribute to ‘Total Cholesterol’ count. However, we didn’t expect the remaining two risk factors, pointing to both social aspects (Annual Household Income) and demographics (Age in year at screening), to be such high indicators. 
Following decision on our important features, we then conducted a bivariate analysis on the chosen 27 of our attributes to confirm whether our predictive model chose the most significant variables. We compared the same 27 variables for the bivariate analysis from our predictive model. This analysis incorporated the t-test for categorical variables and linear, simple regression for the continuous ones. In this case, normalizing was not required as we could apply the Central Limit Theorem (CLT) - our dataset consists of a sufficiently large sample size (N > 30), where the mean of all samples from the same population will be approximately equal to the mean of our population. 
Trial and Error:
Based on our p-values (included only top 15 variables in Table 1.), all variables were statistically significant except for No. of ready-to-eat foods in past 30 days, No. of frozen meals/ pizza in past 30 days and No. of meals from fast food or pizza place. In order to further improve our model accuracy, we eliminated these 3 non-significant features and rebuilt the model using the gradient boosting classifier. Attempting this elimination, did not improve accuracy and reduced it from 67%, and therefore we decided to keep all 27 variables for our final analysis.
Insights: 
We concluded that all 27 attributes were important features for our outcome as this had resulted in the highest accuracy. After testing our feature-elimination on both Random Forest and Gradient Boosting, removing non-significant variables was not helpful. The results from our final model, helped to prove our hypothesis that in fact nutritional intake, demographics and social aspects all significantly affect cholesterol levels of a patient as reflected in Table 2 below. Nutritional Intake accounts for 40% of the top 15 attributes (more than 50% of the chosen, 27 variables):

Table 2: Looks at Top 15 Attributes (including both controllable and uncontrollable risk factors)


Part F: Open problems/future research. []
While the results of our research showed the risk factors with the biggest impact on high cholesterol risk levels, we must mention the dataset we created can be used to test the correlation between other characteristics as well. As mentioned in the description of our data, our dataset includes social, nutritional and health elements, and while we chose cholesterol levels as our outcome, further research with the data can be done using any of the other variables as the outcome. For example, we could look to see if there’s a correlation between income and nutrition, or educational status and nutrition. We could even look at the connection between service in the armed forces with nutritional status or educational status. In addition, we could expand our dataset through the addition of variables from other NHANES datasets to look at another health topic. If we were to do further research on high cholesterol risk factors, variables we’d like to look at would be “smoking” and “blood pressure.” The American Heart Association states that both smoking status and high blood pressure are high risk factors, yet we were not able to find these variables in NHANES to add to our dataset. As a team, we agree that it would be interesting to look at these factor’s correlation to high cholesterol and whether or not they would be important elements to our prediction model. Lastly, if we were to focus on “Risk of Heart Disease” as an outcome, rather than “High Cholesterol,” we would aim to look at “Triglyceride levels”, another predictor of heart disease, as the outcome and use our dataset to see which variables are the biggest risk factors.

From our analysis, we have come to the conclusion that it would be beneficial to look at the effect of Public Health Initiatives on High Cholesterol Risk. Creation of our predictive model showed that both Annual Income, and Age at Screening have high measures of feature importance. As neither are elements that can be fixed by a medication or a change in diet, the best intervention would be awareness of the risk of high cholesterol, and education on how to lower it. 

Our data has helped to identify the most important predictors and risk factors, and our results could help to identify interventions to combat high cholesterol, and further prevent the risk of heart disease. Now that we’ve identified the risk factors and created a prediction model, we hope that our research impacts the community and that their awareness will aid in lowering the risk of heart disease caused by high cholesterol


NOTE: While different sections may have been written individually, our team met multiple times to discuss our project. We had vast discussions regarding our topic, algorithms, and the results of our models. This project was truly a team effort.




Sources:
1.	https://www.hsph.harvard.edu/nutritionsource/cholesterol/
2.	http://www.ehow.com/facts_6182128_direct-hdl_.html 
3.	http://www.statinusage.com/Pages/cholesterol-overview.aspx
4.	https://www.cdc.gov/cholesterol/family_history.htm
5.	https://www.cdc.gov/dhdsp/data_statistics/fact_sheets/fs_cholesterol.htm 
6.	http://www.webmd.com/cholesterol-management/guide/high-cholesterol-risk-factors 
7.	http://www.heart.org/HEARTORG/Conditions/Cholesterol/PreventionTreatmentofHighCholesterol/Prevention-and-Treatment-of-High-Cholesterol-Hyperlipidemia_UCM_001215_Article.jsp#.WXJ05dPyvUo

