# Causal Data Science Analysis of Airbnb Listings : Boston

## Authors
[Kaavya Gowthaman](http://www.linkedin.com/in/kaavya-gowthaman), [Prathwish Shetty](https://www.linkedin.com/in/prathwish/), [Sonal Jain](https://www.linkedin.com/in/sjain2212/)

## Abstract

In the past few years we have seen a shift in the way the hospitality industry works. People have started preferring local experiences and local hosts over fancy hotels. Airbnb has played a very crucial role in driving this change. 
There are a large group of people who use Airbnb to generate income. Our project aims to help prospective Airbnb hosts maximize their return on investment using Causal data science Analysis, by helping them identify properties that would generate maximum return on investement.

[See Video abstract](https://youtu.be/L632ONT1L54) 
  
  
## How to explore this project

A good starting point to explore this project is with the `Causal Data Analysis.ipynb` notebook, it details all the concepts and methods used in this analysis. We would recommend using *Google Colab* to explore the notebook,as there are some Google Colab specific code that renders an applet.

The project uses data from multiple sources and we have used various API's to collect the data. If you plan on collecting data for the same please refer to the follwing links to get the necessary API.
1. [Here Maps](https://developer.here.com/)
1. [Walkscore.com](https://www.walkscore.com/professional/walk-score-apis.php)
1. [greatschools.com](https://www.greatschools.org/api/request-api-key)

If the above API's are deprecated or you are unable to collect data, we have stored the data in a directory `data`.

## Directory Dictionary

```
├── home\n
│   ├── Causal_Data_Science_Notebook.ipynb   --- Main Notebook with the analysis
│   ├── model_tests_notebook.ipynb  		 --- R notebook with the tests used in the main notebook
│   ├── data
│       ├── listings.csv        			---data directly from Airbnb from Feb 13th 2020
│       ├── listings_manual.csv 			---listing with their price estimates, scrapped manually*
│       ├── listings_full.csv   			---Feature rich dataset with all collected features*
│   ├── images
```

