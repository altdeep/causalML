library(stringr)
library(tidyverse)
library(purrr)

cost_mean <- 8.5
cost_sd <- 1.5
cost_low_bound <- qnorm(1/3, cost_mean, cost_sd)
cost_high_bound <- qnorm(2/3, cost_mean, cost_sd)
get_levels <- function(mu, sigma){
    .levels <- unique(bnlearn::discretize(rnorm(1000000, mu, sigma)))
}

p_gene <- Vectorize(function(){
    out <- sample(c(T, F), 1, prob = c(.3, .7))
    return(out)
})
p_cost <- Vectorize(function(){
    return(rnorm(1, 8.5, 1.5))
})

p_smoking <- Vectorize(function(cost, gene){
    if(gene){
        if(cost < cost_low_bound){
            prob <- c(.05, .15, .8)
        }
        if((cost >= cost_low_bound) && (cost < cost_high_bound)){
            prob <- c(.2, .3, .5)
        }
        if(cost >= cost_high_bound){
            prob <- c(.2, .4, .4)
        }
    } else {
        if(cost < cost_low_bound){
            prob <- c(.2, .3, .5)
        }
        if((cost >= cost_low_bound) && (cost < cost_high_bound)){
            prob <- c(.2, .3, .5)
        }
        if(cost >= cost_high_bound){
            prob <- c(.3, .3, .4)
        }
    }
    out <- sample(c("Low", "Med", "High"), 1, prob = prob)
    return(out)
})

p_tar <- Vectorize(function(smoke){
    if(smoke == "Low"){
        prob <- c(.1, .9)
    } 
    if(smoke == "Med"){
        prob <- c(.5, .5)
    }
    if(smoke == "High"){
        prob <- c(.9, .1)
    }
    out <- sample(c("High", "Low"), 1, prob = prob)
    return(out)
})

p_lung_cancer <- Vectorize(function(tar, gene){
    if(gene){
        if(tar == "Low"){
            prob <- c(.1, .9)
        }
        if(tar == "Med"){
            prob <- c(.2, .8)
        }
        if(tar == "High"){
            prob <- c(.3, .7)
        }
    } else {
        if(tar == "Low"){
            prob <- c(.5, .5)
        }
        if(tar == "Med"){
            prob <- c(.7, .3)
        }
        if(tar == "High"){
            prob <- c(.9, .1)
        }
    }
    out <- sample(c(T, F), 1, prob = prob)
    return(out)
})

N <- 1000

data_set <- tibble(
    D = map_lgl(1:N, ~ p_gene()),
    C = map_chr(1:N, ~ p_cost()),
) %>%
    mutate(
        S = p_smoking(C, D),
        T = p_tar(S),
        L = p_lung_cancer(T, D)
    ) %>%
    select(-D) # Make D latent
    as.data.frame

write_csv(data_set, "/Users/robertness/Downloads/causalML/datasets/cigs_and_cancer.csv")
