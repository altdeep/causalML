

# Airbnb Analysis using Causal Inference in Machine Learning

## Authors
[AbdulRehman](https://www.linkedin.com/in/abdulrehman1997/), [Jerry Franklin](https://www.linkedin.com/in/jerry-adams-franklin/) ,  [Sapna Sharma](https://www.linkedin.com/in/sapnasharma22/)

## Abstract

As more and more people prefer to stay at airbnb accommodation rather than staying at fancy hotels, the demand for airbnb has increased over time. Due to this demand more people are investing in houses to rent on Airbnb. Investors need to know which property to invest in to get high return on investment.


[See Video abstract](https://www.youtube.com/watch?v=wHgvCu7qzY0&feature=youtu.be) 
  
  
## How to explore this project

The main file is  `airbnb_analysis.ipynb` notebook. It contains step by step implementation of causal inference on ROI of any investment listed on airbnb 

Necessary api to collect the data are as follows :
1. [Here Maps](https://developer.here.com/)
1. [Walkscore.com](https://www.walkscore.com/professional/walk-score-apis.php)
1. [greatschools.com](https://www.greatschools.org/api/request-api-key)

The data collected by our team is in the folder `Data`
We have done the analysis and testing of the models in R and Python Languages.
Analysis.ipynb is the python notebook and
Global Markov & Failthfulness.ipynb is the code in R

#### Step 1: Data Gathereing
We filtered out data based on
- the property type (Condominiums, Apartments )
- only the properties which were available to rent as a whole 
#### Step 2: Featue Engineering
We used Google's [geo-coding API](https://developers.google.com/maps/documentation/geocoding/start?utm_source=google&utm_medium=cpc&utm_campaign=FY18-Q2-global-demandgen-paidsearchonnetworkhouseads-cs-maps_contactsal_saf&utm_content=text-ad-none-none-DEV_c-CRE_315916117595-ADGP_Hybrid+%7C+AW+SEM+%7C+BKWS+~+Google+Maps+Geocoding+API-KWID_43700039136946117-kwd-300650646186-userloc_9004054&utm_term=KW_google%20geocoding%20api-ST_google+geocoding+api&gclid=CjwKCAjwnIr1BRAWEiwA6GpwNYs9HqeKeAm07opBtifC1HqKtl2GTBfPIQz2365hvhJp4v2jhtcbxhoCVbIQAvD_BwE)
to get the addresses for the properties using their latitude and longitude. 

#### Step3: Feature Engineering(Zestimate)
We used [Zillow](https://www.zillow.com) to get the Zestimates (current estimated market price) for each of these properties using their addresses. 

The data from these two files were combined. 

### Step4: Causal DAG exploration and evaluation

In the file Global Markov & Failthfulness.ipynb we explore various DAG structures and evaluate each based on Global Markov and faithfulness properties, which can be found at the end of the file.

### Step5: Model Building using Pyro

In the file `airbnb_analysis.ipynb` we build our causal models in pyro, analyze the results.

## Directory Dictionary

```
├── home\n
│   ├── airbnb_analysis.ipynb   --- Main Notebook with the analysis
│   ├── model_tests_notebook.ipynb  		 --- R notebook with the tests used in the main notebook
│   ├── data
│       ├── listings.csv        			---data directly from Airbnb
│       ├── listings_manual.csv 			---listing with their price estimates, scrapped manually*
│       ├── listings_full.csv   			---Feature rich dataset with all collected features*
│   ├── images
```
