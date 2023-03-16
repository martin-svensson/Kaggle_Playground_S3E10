# ====================================================================================================== #
# Description
#
#   Performance estimate on test set and submission
#
# Change log:
#   Ver   Date        Comment
#   1.0   16/03/23    Initial version
#
# ====================================================================================================== #
# ------------------------------------------------------------------------------------------------------ #
# LIBRARIES
# ------------------------------------------------------------------------------------------------------ #

library(data.table)
library(tidyverse)
library(magrittr)

library(tidymodels)

# ------------------------------------------------------------------------------------------------------ #
# IMPORT AND SOURCES
# ------------------------------------------------------------------------------------------------------ #

df_test <- fread("./Data/test.csv")

load("./Output/01_df_train.RData")
load("./Output/01_data_split.RData")

load("./Output/03_recipes.RData")

load("./Output/04_my_metric_set.RData")
load("./Output/04_workflow_set.RData")
load("./Output/04_tuning_results.RData")

# ------------------------------------------------------------------------------------------------------ #
# PROGRAM
# ------------------------------------------------------------------------------------------------------ #

grid_results_best <- 
  grid_results %>% 
  extract_workflow_set_result("recipe_1_xgboost") %>% 
  select_best(metric = "mn_log_loss")

# -- Performance estimate on test set

performance_est <- 
  grid_results %>% 
  extract_workflow(
    "recipe_1_xgboost"
  ) %>% 
  finalize_workflow(
    grid_results_best
  ) %>% 
  last_fit(
    split = data_split,
    metrics = my_metric_set
  )

# Performance
performance_est %>% 
  collect_metrics()

# Predictions
performance_est %>% 
  collect_predictions() %>% 
  ggplot(aes(x = .pred_1)) +
  geom_histogram() + 
  facet_wrap(~ Class)
  

# ---- Submission --------------------------------------------------------------

wflow_final_spec <- 
  wflow %>% 
  extract_workflow("recipe_1_xgboost") %>% 
  finalize_workflow(
    grid_results_best
  )

wflow_final_fit <- 
  wflow_final_spec %>% 
  fit(df_train)

subm_pred <- 
  wflow_final_fit %>% 
  predict(
    new_data = df_test,
    type = "prob"
  )

df_subm <- 
  df_test %>% 
  cbind(subm_pred) %>% 
  select(
    id,
    "Class" = .pred_1
  ) 

# ==== EXPORT ------------------------------------------------------------------------------------------ 

df_subm %>% 
  fwrite(file = "./Output/05_submission.csv") 
         
         