# Causal Story of Ecommerce Churn-out

### Team Members
- Anisha Thakrar
- Dyuthi Vinod
- Kavya Gajjar

### Objective & Motivation

For sustainable growth, the primary focus in E-Commerce business is increasing the Customer Acquisition and Customer Retention. 
To achieve that it should have low Churn out rate. Fundamentally, churn occurs when a customer stops consuming from a company. A high churn rate equals a low retention rate. Churn affects the size of your customer base and has a big impact on your customer lifetime value.

The motivation behind selecting this project was because E-Commerce websites have gained a lot of traction in the recent years. Every individual today has entered into the world of online shopping, including us. Therefore, we thought it would be interesting to analyse the causal effect on such website and to actually see how certain variables affect the churn rate of customers. The main goal of the project  is to find the factors that cause the customers to churn out

### Dataset

- The dataset used here is a publicly available [Kaggle](https://www.kaggle.com/datasets/ankitverma2010/ecommerce-customer-churn-analysis-and-prediction) dataset
- It consists of 5630 rows and 20 columns
- The target variable here is **ChurnFlag** where if a customer has churned out, the value will be 1, else it is 0
- Treatment variables: SatisfactionFlag, ComplainFlag

*SatisfactionScore = 1 is SatisfactionFlag = 1; SatisfactionScore = [2,5]  is SatisfactionFlag = 0

### Assumptions

- Satisfaction Flag has two values True or False. If a customer is dissatisfied, they are more likely to churn out
- Complain is a sign of dissatisfaction, which might lead to churning out
- Higher value of days since last order indicates that customer hasn’t used the e-commerce website/app for a longer period of time (lower order count), hence more likely to churn out
- Lower order count in last month indicates less usage of the e-commerce website/app, hence lower % increase of orders from last year
- Increase in Percentage increase in orders from last year indicates the customer is active. Which means rate of churning out will be low
- Tenure is the length of relationship of the customer with the organization. Longer the tenure indicates more activity on the website, hence higher number of Order counts and higher Percentage increase in orders from last year
- City Tier; tier 1 might have higher number of order counts, which might lower the chance of churning out

### Video Link 

Video link to our presentation: https://youtu.be/C6TPtmsYvL4 
