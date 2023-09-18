## Data Cleaning
library(tidyverse)
library(tidymodels)
library(vroom)

bike <- vroom("C:/Users/eaa54/Documents/School/STAT348/KaggleBikeShare/train.csv")
testdata <- vroom("C:/Users/eaa54/Documents/School/STAT348/KaggleBikeShare/test.csv")

#dplyr cleaning step
class(bike$holiday) #holiday is currently numeric - will want to change to factor
class(bike$season) #season numeric - change to factor
class(bike$workingday) #numeric change to factor
class(bike$weather) #same
class(bike$temp) #numeric
class(bike$atemp)
class(bike$humidity)
class(bike$windspeed)

#don't think I need date because weather covers what would be interesting from date
#take out registered and casual
#maybe keep time?
testdata$weather[testdata$weather == 4] <- 3
clean_bike <- bike %>%
  select(datetime, season, holiday, workingday, weather, temp, atemp, humidity, windspeed, count)
clean_bike$weather[bike$weather == 4] <- 3

#feature engineering step - get some of this
myrecipe <- recipe(count ~ ., data = clean_bike) %>%
  step_zv(all_predictors()) %>%
  step_lincomb() %>%
  step_mutate(feelslike = atemp-temp, wind2 = windspeed^2, holiday = factor(holiday), weather = factor(weather)) %>%
  step_num2factor(season, levels=c("1", "2", "3", "4")) %>%
  step_rm(temp, atemp)
 
bake(prep(myrecipe), new_data = NULL)

model <- linear_reg() %>% #type of model
  set_engine("lm") # engine - what R function to use to fit model

bike_workflow <- workflow() %>%
  add_recipe(myrecipe) %>%
  add_model(model) %>%
  fit(clean_bike) #feed training data

bike_predictions <- predict(bike_workflow,new_data=testdata)
bike_predictions$datetime <- as.character(format(testdata$datetime))
names(bike_predictions)[1] <- "count"
bike_predictions <- bike_predictions %>%
  mutate(count = pmax(0, count))

vroom_write(bike_predictions, file="bike_predictions.csv", delim=",")
