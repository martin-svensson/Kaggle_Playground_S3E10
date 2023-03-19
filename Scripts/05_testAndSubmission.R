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
library(probably)

# ------------------------------------------------------------------------------------------------------ #
# IMPORT AND SOURCES
# ------------------------------------------------------------------------------------------------------ #

df_test <- fread("./Data/test.csv")

load("./Output/01_output.RData")
load("./Output/03_output.RData")
load("./Output/04_output.RData")

# ------------------------------------------------------------------------------------------------------ #
# PROGRAM
# ------------------------------------------------------------------------------------------------------ #

best_wflow_id <- "recipe_3_xgboost"

grid_results_best <- 
  grid_results %>% 
  extract_workflow_set_result(best_wflow_id) %>% 
  select_best(metric = "mn_log_loss")


# ---- Performance estimate on test set ----------------------------------------

performance_est <- 
  grid_results %>% 
  extract_workflow(
    best_wflow_id
  ) %>% 
  finalize_workflow(
    grid_results_best
  ) %>% 
  last_fit(
    split = output_01$data_split,
    metrics = my_metric_set
  )

# Performance
performance_est %>% 
  collect_metrics()

# Predictions
df_test_preds <- 
  performance_est %>% 
  collect_predictions()
  
df_test_preds %>% 
  ggplot(aes(x = .pred_1)) +
  geom_histogram() + 
  facet_wrap(~ Class)
  
df_test_preds %>% 
  cal_plot_breaks(
    Class,
    .pred_1,
    num_breaks = 15,
    include_rug = FALSE,
    event_level = "second"
  )

# -- with calibration
cal_preds <- 
  output_04$model_isoreg(
    df_test_preds$.pred_1
  )

df_cal_combined <- 
  bind_rows(
    mutate(
      df_test_preds, 
      source = "original"
    ), 
    mutate(
      df_test_preds, 
      .pred_1 = cal_preds, 
      source = "cal"
    )
  )

df_cal_combined %>%
  group_by(source) %>%
  cal_plot_breaks(
    Class,
    .pred_1,
    num_breaks = 15,
    include_rug = FALSE,
    event_level = "second"
  )
# Calibration seems to work well, so we will use it in the submission as well

# ---- Submission --------------------------------------------------------------

wflow_final_spec <- 
  wflow %>% 
  extract_workflow(
    best_wflow_id
  ) %>% 
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

# -- Calibration
cal_switch <- FALSE

if (cal_switch) {
  
  cal_preds_subm <- 
    output_04$model_isoreg(
      subm_pred$.pred_1
    )
  
}

# -- Submission
if (cal_switch) {
  
  df_subm <- 
    df_test %>% 
    mutate(
      Class = cal_preds_subm
    ) %>% 
    select(
      id, 
      Class
    )
  
} else {
  
  df_subm <- 
    df_test %>% 
    cbind(subm_pred) %>% 
    select(
      id,
      "Class" = .pred_1
    ) 
  
}


# ==== EXPORT ------------------------------------------------------------------------------------------ 

df_subm %>% 
  fwrite(file = "./Output/05_submission.csv") 
         
         