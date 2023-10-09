##
## Bike Share EDA Code
##

## Libraries
library(tidyverse)
library(vroom)
library(DataExplorer)
library(dplyr)
library(skimr)
library(GGally)

## Read in the data
bike <- vroom("C:/Users/eaa54/Documents/School/STAT348/KaggleBikeShare/train.csv")

## EDA
glimpse(bike)
plot(bike$count, bike$weather)
boxplot(bike$weather)
table(bike$weather) #only 1 4 weather
skimr::skim(bike)
plot_missing(bike) #no missing data
plot_correlation(bike)
ggplot(bike, aes(x=temp, y=count, color=holiday)) +
  geom_point() +
  geom_smooth() #looking temperature trend
plot(y = bike$count, x = bike$windspeed)
