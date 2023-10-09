## Data Cleaning
library(tidyverse)
library(tidymodels)
library(vroom)
bike <- vroom("C:/Users/eaa54/Documents/School/STAT348/KaggleBikeShare/train.csv")
testdata <- vroom("C:/Users/eaa54/Documents/School/STAT348/KaggleBikeShare/test.csv")

#change 4 to 3 in test data
testdata$weather[testdata$weather == 4] <- 3

#change 4 to 3 in training data and take out registered and casual
#make count a log count
clean_bike <- bike %>%
  select(datetime, season, holiday, workingday, weather, temp, atemp, humidity, windspeed, count) %>%
  mutate(count = log(count))
clean_bike$weather[bike$weather == 4] <- 3

#feature engineering step - this should be based on EDA
#whatever you want done to test and train, do within your recipe
myrecipe <- recipe(count ~ ., data = clean_bike) %>%
  step_zv(all_predictors()) %>%
  step_lincomb() %>%
  step_mutate(holiday = factor(holiday), weather = factor(weather)) %>%
  step_num2factor(season, levels=c("1", "2", "3", "4")) %>%
  step_time(datetime, features = "hour") %>%
  step_rm(datetime)
 
bake(prep(myrecipe), new_data = NULL)

model <- linear_reg() %>% #type of model
  set_engine("lm") #engine - what R function to use to fit model: in this case a linear model

bike_workflow <- workflow() %>%
  add_recipe(myrecipe) %>%
  add_model(model) %>%
  fit(clean_bike) #feed training data

#how to look at your model
summary(extract_fit_engine(bike_workflow))

bike_predictions <- predict(bike_workflow,new_data=testdata)
bike_predictions$datetime <- as.character(format(testdata$datetime))
names(bike_predictions)[1] <- "count"
bike_predictions <- bike_predictions %>%
  mutate(count = pmax(0, count))

#write as a csv
vroom_write(bike_predictions, file="bike_predictions.csv", delim=",")

########################
###Poisson Regression###
########################

#extra libraries needed
library(poissonreg)

pois_mod <- poisson_reg() %>% #type of model
  set_engine("glm") #"generalized linear model"

#create recipe - do things here that you want applied to training and testing data
myrecipe_poiss <- recipe(count ~ ., data = clean_bike) %>%
  step_zv(all_predictors()) %>%
  step_lincomb() %>%
  step_mutate(holiday = factor(holiday), weather = factor(weather)) %>%
  step_num2factor(season, levels=c("1", "2", "3", "4")) %>%
  step_time(datetime, features = "hour") %>%
  step_rm(datetime)

bike_workflow_p <- workflow() %>%
  add_recipe(myrecipe_poiss) %>%
  add_model(pois_mod) %>%
  fit(clean_bike) #feed training data

#exp predictions because I took log count at the beginning
bike_predict_poiss <- exp(predict(bike_workflow_p, testdata))

#extra step to get dates in the proper format for Kaggle
bike_predict_poiss$datetime <- as.character(format(testdata$datetime))

#change first column name to count
names(bike_predict_poiss)[1] <- "count"

#make sure there are no negative predictions - round them to zero if they exist
bike_predict_poiss <- bike_predict_poiss %>%
  mutate(count = pmax(0, count))

#write csv
vroom_write(bike_predict_poiss, file="bike_predictions_poiss.csv", delim=",")

##########################
###Penalized Regression###
##########################
#extra packages needed 
library(glmnet) #this is for doing elastic net penalties

#predict(preg_wf, new_data=testData
myrecipe_penalized <- recipe(count ~ ., data = clean_bike) %>%
  step_zv(all_predictors()) %>%
  step_lincomb() %>%
  step_mutate(holiday = factor(holiday), weather = factor(weather)) %>%
  step_num2factor(season, levels=c("1", "2", "3", "4")) %>%
  step_time(datetime, features = "hour") %>%
  step_rm(datetime) %>%
  step_normalize(all_numeric_predictors()) %>% #makes all means 0 and sd 1
  step_dummy(all_nominal_predictors())

#tried penalty=mixture=0.5 and got 1.12
#tried penalty=1,mixture=0.75 and got 1.37
#tried penalty=0.5,mixture=0.25 and got 1.07
#tried penalty=0.25,mixture=0.10 and got 1.05
#tried penalty=0.25,mixture=0 and got 1.049
#tried penalty=1,mixture=0 and got 1.046
#tried penalty-0.75,mixture=0 and got 1.046(slightly better)
penalized_mod <- poisson_reg(penalty=0.75, mixture=0) %>%
  set_engine("glmnet")

penalized_workflow <- workflow() %>%
  add_recipe(myrecipe_penalized) %>%
  add_model(penalized_mod) %>%
  fit(clean_bike)

penalized_preds <- exp(predict(penalized_workflow, new_data=testdata))

penalized_preds$datetime <- as.character(format(testdata$datetime))

#change first column name to count
names(penalized_preds)[1] <- "count"

#make sure there are no negative predictions - round them to zero if they exist
penalized_preds <- penalized_preds %>%
  mutate(count = pmax(0, count))

#write csv
vroom_write(penalized_preds, file="penalized_preds.csv", delim=",")

#################
##TUNING MODELS##
#################
library(tidymodels)
library(poissonreg) #if you want to do penalized, poisson regression

myrecipe_CV <- recipe(count ~ ., data = clean_bike) %>%
  step_zv(all_predictors()) %>%
  step_lincomb() %>%
  step_mutate(holiday = factor(holiday), weather = factor(weather)) %>%
  step_num2factor(season, levels=c("1", "2", "3", "4")) %>%
  step_time(datetime, features = "hour") %>%
  step_rm(datetime, holiday) %>%
  step_normalize(all_numeric_predictors()) %>% #makes all means 0 and sd 1
  step_dummy(all_nominal_predictors())

model_CV <- linear_reg() %>% #type of model
  set_engine("lm")

## Set Workflow
reg_wf <- workflow() %>%
add_recipe(myrecipe_CV) %>%
add_model(model_CV)

## Grid of values to tune over
tuning_grid <- grid_regular(penalty(),
                            mixture(),
                            levels = 10) ## L^2 total tuning possibilities

## Split data for CV
folds <- vfold_cv(clean_bike, v = 5, repeats=1) #k-fold

## Run the CV
CV_results <- reg_wf %>%
tune_grid(resamples=folds,
          grid=tuning_grid,
          metrics=metric_set(rmse, mae, rsq)) #Or leave metrics NULL5

## Plot Results (example)
collect_metrics(CV_results) %>% # Gathers metrics into DF
  filter(.metric=="rmse") %>%
  ggplot(data=., aes(x=penalty, y=mean, color=factor(mixture))) +
  geom_line()

## Find Best Tuning Parameters
bestTune <- CV_results %>%
  select_best("rmse")

## Finalize Workflow and fit it
final_wf <- reg_wf %>%
  finalize_workflow(bestTune) %>%
  fit(data=clean_bike)

## Predict
preds_CV <- exp(predict(final_wf, new_data=testdata))
preds_CV$datetime <- as.character(format(testdata$datetime))

#change first column name to count
names(preds_CV)[1] <- "count"

#make sure there are no negative predictions - round them to zero if they exist
preds_CV <- preds_CV %>%
  mutate(count = pmax(0, count))

#write csv
vroom_write(comp_preds, file="preds_CV.csv", delim=",")



##################
##RANDOM FORESTS##
##################
library(tidymodels)
library(tidyverse)

rf_mod <- rand_forest(mtry = tune(), min_n=tune(),trees=500) %>% #Type of model
                      set_engine("ranger") %>% # What R function to use
                      set_mode("regression")

myrecipe_rf <- recipe(count ~ ., data = clean_bike) %>%
  step_zv(all_predictors()) %>%
  step_lincomb() %>%
  step_mutate(holiday = factor(holiday), weather = factor(weather)) %>%
  step_num2factor(season, levels=c("1", "2", "3", "4")) %>%
  step_time(datetime, features = "hour") %>%
  step_rm(datetime, holiday) %>%
  step_normalize(all_numeric_predictors()) %>% #makes all means 0 and sd 1
  step_dummy(all_nominal_predictors())

## Set Workflow
rf_wf <- workflow() %>%
  add_recipe(myrecipe_rf) %>%
  add_model(rf_mod)

## Grid of values to tune over
tuning_grid <- grid_regular(penalty(),
                            mixture(),
                            levels = 5) ## L^2 total tuning possibilities

## Split data for CV
folds <- vfold_cv(clean_bike, v = 5, repeats=1) #k-fold

## Run the CV
rf_results <- rf_wf %>%
  tune_grid(resamples=folds,
            grid=tuning_grid,
            metrics=metric_set(rmse)) #Or leave metrics NULL

## Find Best Tuning Parameters
bestTune <- rf_results %>%
  select_best("rmse")

## Finalize Workflow and fit it
final_wf <- rf_wf %>%
  finalize_workflow(bestTune) %>%
  fit(data=clean_bike)

## Predict
preds_rf <- exp(predict(final_wf, new_data=testdata))
preds_rf$datetime <- as.character(format(testdata$datetime))

#change first column name to count
names(preds_rf)[1] <- "count"

#make sure there are no negative predictions - round them to zero if they exist
preds_rf <- preds_rf %>%
  mutate(count = pmax(0, count))

#write csv
vroom_write(rf_preds, file="preds_rf.csv", delim=",")


############
##STACKING##
############

#an ensemble method
#usually outperforms the individual base learners
#see best results when base models capture different aspects of the data

library(stacks)
library(tidyverse)
library(tidymodels)
library(vroom)

bike <- vroom("C:/Users/eaa54/Documents/School/STAT348/KaggleBikeShare/train.csv")
testdata <- vroom("C:/Users/eaa54/Documents/School/STAT348/KaggleBikeShare/test.csv")

#change 4 to 3 in test data
testdata$weather[testdata$weather == 4] <- 3

#change 4 to 3 in training data and take out registered and casual
#make count a log count
clean_bike <- bike %>%
  select(datetime, season, holiday, workingday, weather, temp, atemp, humidity, windspeed, count) %>%
  mutate(count = log(count))
clean_bike$weather[bike$weather == 4] <- 3

recipe_stack <- recipe(count ~ ., data = clean_bike) %>%
  step_zv(all_predictors()) %>%
  step_lincomb() %>%
  step_mutate(holiday = factor(holiday), weather = factor(weather)) %>%
  step_num2factor(season, levels=c("1", "2", "3", "4")) %>%
  step_time(datetime, features = "hour") %>%
  step_rm(datetime, holiday) %>%
  step_normalize(all_numeric_predictors()) %>% #makes all means 0 and sd 1
  step_dummy(all_nominal_predictors())

#Split data for CV
folds <- vfold_cv(clean_bike, v = 10, repeats = 1)

# Create a Control Grid
untuned <- control_stack_grid()

#Penalized Reg Model
preg_mod <- linear_reg(penalty=tune(),
                       mixture=tune()) %>%
            set_engine("glmnet")

#Set workflow
preg_wf <- workflow() %>%
  add_recipe(recipe_stack) %>%
  add_model(preg_mod)

preg_tune <- grid_regular(penalty(),
                          mixture(),
                          levels = 5)

preg_mods <- preg_wf %>%
  tune_grid(resamples=folds,
            grid=preg_tune,
            metrics=metric_set(rmse),
            control=untuned)

#Linear Regression
lin_reg <- 
  linear_reg() %>%
  set_engine("lm")

lin_wf <- workflow() %>%
  add_model(lin_reg) %>%
  add_recipe(recipe_stack)

lin_mods <- fit_resamples(
  lin_wf,
  resamples = folds,
  metrics = metric_set(rmse),
  control=control_stack_resamples())

#Random Forest


my_stack <- stacks() %>%
  add_candidates(preg_mods) %>%
  add_candidates(lin_mods)

stack_mod <- my_stack %>%
  blend_predictions() %>%
  fit_members()

preds <- stack_mod %>% predict(new_data=testdata)
preds <- exp(preds)
preds$datetime <- as.character(format(testdata$datetime))

#change first column name to count
names(preds)[1] <- "count"

#make sure there are no negative predictions - round them to zero if they exist
preds <- preds %>%
  mutate(count = pmax(0, count))

#write csv
vroom_write(preds, file="preds.csv", delim=",")
