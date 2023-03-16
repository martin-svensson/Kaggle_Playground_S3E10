# ====================================================================================================== #
# Description
#
#   Model comparison through resampling techniques (cross validation) and
#   workflow sets (combining feature engineering and models)
#
# Change log:
#   Ver   Date        Comment
#   1.0   14/03/23    Initial version
#
# ====================================================================================================== #
# ------------------------------------------------------------------------------------------------------ #
# LIBRARIES
# ------------------------------------------------------------------------------------------------------ #

library(data.table)
library(tidyverse)
library(magrittr)

library(tidymodels)
library(tictoc)

# ------------------------------------------------------------------------------------------------------ #
# IMPORT AND SOURCES
# ------------------------------------------------------------------------------------------------------ #

load("./Output/01_df_train.RData")
load("./Output/01_data_split.RData")
load("./Output/03_recipes.RData")

# ------------------------------------------------------------------------------------------------------ #
# PROGRAM
# ------------------------------------------------------------------------------------------------------ #

df_train <- training(data_split)

# ---- Model specifications ----------------------------------------------------

xgb_spec <- 
  boost_tree(
    tree_depth = tune(), 
    learn_rate = tune(), 
    loss_reduction = tune(), 
    min_n = tune(), 
    sample_size = tune(), 
    trees = tune()
  ) %>% 
  set_engine("xgboost") %>% 
  set_mode("classification")

logreg_spec <- 
  logistic_reg(
    penalty = tune(),
    mixture = tune()
  ) %>% 
  set_engine("glmnet") %>% 
  set_mode("classification")

# ---- Workflow set ------------------------------------------------------------

wflow <- 
  workflow_set(
    preproc = 
      list("recipe_1" = recipe_1), 
    models = 
      list(
        "xgboost" = xgb_spec,
        "logistic_reg" = logreg_spec
      )
  )


# ---- Tune models -------------------------------------------------------------

my_metric_set <- 
  metric_set(
    mn_log_loss,
    roc_auc
  )

grid_ctrl <- 
  control_grid(
    save_pred = TRUE,
    parallel_over = NULL, # automatically chooses betweeen "resamples" and "everything"
    save_workflow = FALSE
  )

tic("grid search")
grid_results <- 
  wflow %>% 
  workflow_map(
    seed = 976,
    resamples = 
      vfold_cv(
        df_train, 
        v = 4,
        strata = Class
      ),
    grid = 5,
    control = grid_ctrl,
    metrics = my_metric_set
  )
toc()


# ---- Compare models ----------------------------------------------------------

# Best model
grid_results %>% 
  autoplot(
    metric = c("mn_log_loss", "roc_auc"),
    select_best = TRUE
  )

# Parameter values for xgboost
grid_results %>% 
  autoplot(
    id = "recipe_1_xgboost",
    metric = "mn_log_loss"
  )

# ==== EXPORT ------------------------------------------------------------------------------------------ 

save(
  wflow,
  file = "./Output/04_workflow_set.RData"
)

save(
  my_metric_set,
  file = "./Output/04_my_metric_set.RData"
)

save(
  grid_results,
  file = "./Output/04_tuning_results.RData"
)
