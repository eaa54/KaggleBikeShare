##KAGGLE Bike Share Competition Final##

#Load Libraries
library(tidyverse)
library(tidymodels)
library(vroom)
library(poissonreg)
library(glmnet)
library(rpart)
library(stacks)
library(bartMachine)

#Load Data
bike <- vroom("C:/Users/eaa54/Documents/School/STAT348/KaggleBikeShare/train.csv")
testdata <- vroom("C:/Users/eaa54/Documents/School/STAT348/KaggleBikeShare/test.csv")

#Log Transformation + Remove columns not in test set
bike <- bike %>%
  select(-casual, -registered) %>%
  mutate(count = log(count))

#Create Recipe
final_recipe <- recipe(count ~ ., data = bike) %>%
  step_zv(all_predictors()) %>%
  step_lincomb() %>%
  step_mutate(weather = ifelse(weather == 4, 3, weather)) %>% #not enough 4s, change to 3s
  step_mutate(holiday = factor(holiday), 
              weather = factor(weather),
              season = factor(season),
              workingday = factor(workingday)) %>%
  step_time(datetime, features = "hour") %>% #create cols for hour, year, and dow
  step_date(datetime, features = "year") %>%
  step_date(datetime, features = "dow") %>%
  step_rm(datetime) %>%
  step_mutate(datetime_hour = factor(datetime_hour), #make factors as suggested by Ty
              datetime_year = factor(datetime_year),
              datetime_dow = factor(datetime_dow)) %>%
  step_normalize(all_numeric_predictors()) %>% 
  step_dummy(all_nominal_predictors())

#Setup Stacking
#Split data for CV
folds <- vfold_cv(bike, v = 5, repeats = 1)

#Create Control Grid - for stacking
untuned <- control_stack_grid() #will use untuned in this stacked model
tuned <- control_stack_resamples() 

#Define Models for Stacking
##MODEL 1: Penalized Regression##
pen_mod <- linear_reg(penalty = tune(), mixture = tune()) %>%
  set_engine("glmnet") 

#Set Workflow
pen_wf <- workflow() %>%
  add_recipe(final_recipe) %>%
  add_model(pen_mod) %>%
  fit(bike)

#Create grid of values to tune over
tuning_grid <- grid_regular(penalty(),
                            mixture(),
                            levels = 10)

#Create all models with tuning parameters from grid
CV_results <- pen_wf %>%
  tune_grid(resamples=folds,
            grid=tuning_grid,
            metrics=metric_set(rmse),
            control=untuned) 

##MODEL 2: Random Forest##
rf_mods <- rand_forest(mtry = tune(), min_n=tune(),trees=1000) %>%
                      set_engine("ranger") %>% #specifies R function
                      set_mode("regression")

#Set Workflow
rf_wf <- workflow() %>%
  add_recipe(final_recipe) %>%
  add_model(rf_mods)

#Grid of values to tune over
tuning_grid <- grid_regular(mtry(c(1,8)),
                            min_n(),
                            levels = 5) #25 values

#Run Random Forest and get all models with tuning parameters specified in grid
rf_results <- rf_wf %>%
  tune_grid(resamples=folds,
            grid=tuning_grid,
            metrics=metric_set(rmse),
            control=untuned)

#MODEL 3: Poisson Regression
pois_mod <- poisson_reg() %>% 
  set_engine("glm") 

pois_wf <- workflow() %>%
  add_recipe(final_recipe) %>%
  add_model(pois_mod) %>%
  fit(bike) #feed training data


##Stacking##
#Stack Models Together
my_stack <- stacks() %>%
  add_candidates(rf_results) %>%
  add_candidates(CV_results) %>%
  add_candidates(pois_wf)
  
#Blend models
stack_mod <- my_stack %>%
  blend_predictions() %>%
  fit_members()

#Predict
preds <- stack_mod %>% predict(new_data=testdata)
preds <- exp(preds) #since we did log(count) at the beginning

#Final Formatting Steps
preds$datetime <- as.character(format(testdata$datetime)) #formatting for kaggle
names(preds)[1] <- "count" #change first column name to count
preds <- preds %>% mutate(count = pmax(0, count)) #round negative predictions to 0

#Write CSV
vroom_write(preds, file="final2_preds.csv", delim=",")


#TRY SPLITTING CASUAL AND REGISTERED
#Split data to predict casual and registered separately then add at end
bike_casual <- bike %>%
  select(-count) %>%
  mutate(count = bike$casual) %>%
  select(-casual, -registered)

bike_registered <- bike %>%
  select(-count) %>%
  mutate(count = bike$registered) %>%
  select(-casual, -registered)

#MAKE A CASUAL RECIPE
casual_recipe <- recipe(count ~ ., data = bike_casual) %>%
  step_zv(all_predictors()) %>%
  step_lincomb() %>%
  step_mutate(weather = ifelse(weather == 4, 3, weather)) %>% #not enough 4s, change to 3s
  step_mutate(holiday = factor(holiday), 
              weather = factor(weather),
              season = factor(season),
              workingday = factor(workingday)) %>%
  step_time(datetime, features = "hour") %>%
  step_date(datetime, features = "year") %>%
  step_date(datetime, features = "dow") %>%
  step_rm(datetime) %>%
  step_mutate(datetime_hour = factor(datetime_hour),
              datetime_year = factor(datetime_year),
              datetime_dow = factor(datetime_dow)) %>%
  step_normalize(all_numeric_predictors()) %>% 
  step_dummy(all_nominal_predictors())

#MAKE A REGISTERED RECIPE
registered_recipe <- recipe(count ~ ., data = bike_registered) %>%
  step_zv(all_predictors()) %>%
  step_lincomb() %>%
  step_mutate(weather = ifelse(weather == 4, 3, weather)) %>% #not enough 4s, change to 3s
  step_mutate(holiday = factor(holiday), 
              weather = factor(weather),
              season = factor(season),
              workingday = factor(workingday)) %>%
  step_time(datetime, features = "hour") %>% #make separate cols for hour, year and dow
  step_date(datetime, features = "year") %>%
  step_date(datetime, features = "dow") %>%
  step_rm(datetime) %>%
  step_mutate(datetime_hour = factor(datetime_hour),
              datetime_year = factor(datetime_year),
              datetime_dow = factor(datetime_dow)) %>%
  step_normalize(all_numeric_predictors()) %>% 
  step_dummy(all_nominal_predictors())

rf_mods <- rand_forest(mtry = tune(), 
                       min_n = tune(),
                       trees=500) %>%
            set_engine("ranger") %>% #specifies R function
            set_mode("regression")

#Set Workflow
rf_wf_casual <- workflow() %>%
  add_recipe(casual_recipe) %>%
  add_model(rf_mods)

#Grid of values to tune over
tuning_grid <- grid_regular(mtry(c(1,8)),
                            min_n(),
                            levels = 5) #25 values

#Run Random Forest and get all models with tuning parameters specified in grid
rf_results_casual <- rf_wf_casual %>%
  tune_grid(resamples=folds,
            grid=tuning_grid,
            metrics=metric_set(rmse),
            control=untuned)

#Set Workflow - Registered
rf_wf_reg <- workflow() %>%
  add_recipe(registered_recipe) %>%
  add_model(rf_mods)

rf_results_reg <- rf_wf_reg %>%
  tune_grid(resamples=folds,
            grid=tuning_grid,
            metrics=metric_set(rmse),
            control=untuned)

my_stack_cas <- stacks() %>%
  add_candidates(rf_results_casual)

#Blend models
stack_mod_cas <- my_stack_cas %>%
  blend_predictions() %>%
  fit_members()

#Predict Casual
preds_casual <- stack_mod_cas %>% predict(new_data=testdata)
preds_casual <- exp(preds_casual) #since we did log(count) at the beginning

#Stack Registered
my_stack_reg <- stacks() %>%
  add_candidates(rf_results_reg)

#Blend models
stack_mod_reg <- my_stack_reg %>%
  blend_predictions() %>%
  fit_members()

#Predict Registered
preds_reg <- stack_mod_reg %>% predict(new_data=testdata)
preds_reg <- exp(preds_reg) #since we did log(count) at the beginning

preds_split <- preds_reg+preds_casual

#Final Formatting Steps
preds_split$datetime <- as.character(format(testdata$datetime)) #formatting for kaggle
names(preds_split)[1] <- "count" #change first column name to count
preds_split <- preds_split %>% mutate(count = pmax(0, count)) #round negative predictions to 0

view(preds_split)

#Write CSV
vroom_write(preds_split, file="finalsplit_preds.csv", delim=",")
