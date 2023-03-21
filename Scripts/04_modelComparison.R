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
library(brulee) # multi layer nnets (torch)
library(finetune)
library(themis)
library(probably) # probability callibration

library(tictoc)
library(DataExplorer)

# ------------------------------------------------------------------------------------------------------ #
# IMPORT AND SOURCES
# ------------------------------------------------------------------------------------------------------ #

load("./Output/01_output.RData")
load("./Output/03_output.RData")

# ------------------------------------------------------------------------------------------------------ #
# PROGRAM
# ------------------------------------------------------------------------------------------------------ #

df_train <- training(output_01$data_split)

# ---- Model specifications ----------------------------------------------------

# -- xgboost: some parameters are determined based on earlier tuning 
#   (to ease computation for later tuning, even though potential interaction
#    effect between parameters may not be captured)
xgb_spec <- 
  boost_tree(
    tree_depth = 6, 
    learn_rate = 0.05, 
    loss_reduction = 0.01, 
    min_n = 30, 
    sample_size = 0.6, 
    trees = 750
  ) %>% 
  set_engine("xgboost") %>% 
  set_mode("classification")

# -- GAM
gam_spec <- 
  gen_additive_mod(
    select_features = TRUE,
    adjust_deg_free = tune()
  ) %>% 
  set_engine("mgcv") %>% 
  set_mode("classification")


# -- Neural net: turned out not to be competitive
nnet_spec <- 
  mlp(
    hidden_units = c(15, 5),
    dropout = tune(),
    epochs = tune(),
    learn_rate = tune(),
    activation = "relu"
  ) %>% 
  set_engine("brulee") %>% # allows multiple hidden layers
  set_mode("classification")
  

# -- Logistic regression: turned out not to be competitive
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
      list(
        "recipe_3" = output_03$recipe_3
      ), 
    models = 
      list(
        "gam" = gam_spec,
        "xgboost" = xgb_spec
      )
  )

# GAM: specify gam formula
wflow %<>% 
  update_workflow_model(
    id = "recipe_3_gam",
    spec = gam_spec,
    formula = Class ~ 
      s(Mean_Integrated, k = 20)
    + s(SD)
    + s(EK, k = 30)
    + s(Skewness)
    + s(Mean_DMSNR_Curve)
    + s(SD_DMSNR_Curve)
    + s(EK_DMSNR_Curve)
    + s(Skewness_DMSNR_Curve)
    # interactions
    + ti(EK, EK_DMSNR_Curve)
    + ti(SD, SD_DMSNR_Curve)
    + ti(Skewness, Skewness_DMSNR_Curve)
  )

# update parameter range for tuning
recipe_3_xgboost_param <- 
  wflow %>% 
  extract_workflow("recipe_3_xgboost") %>% 
  extract_parameter_set_dials() %>% 
  update(over_ratio = over_ratio(c(0.5, 1)))

wflow %<>%
  option_add(
    param_info = recipe_3_xgboost_param,
    id = "recipe_3_xgboost"
  ) 

# ---- Tune models -------------------------------------------------------------

# try race to speed it up

my_metric_set <- 
  metric_set(
    mn_log_loss,
    roc_auc
  )

grid_ctrl <- 
  control_race(
    save_pred = TRUE,
    parallel_over = NULL, # automatically chooses betweeen "resamples" and "everything"
    save_workflow = FALSE
  )

tic("grid search")
grid_results <- 
  wflow %>% 
  workflow_map(
    #"tune_race_anova", # faster, more crude, tuning compared to grid search
    seed = 976,
    resamples = 
      vfold_cv(
        df_train, 
        v = 5,
        strata = Class
      ),
    grid = 5,
    control = grid_ctrl,
    metrics = my_metric_set
  )
toc()


# ---- Compare models ----------------------------------------------------------

# -- Best model
grid_results %>% 
  autoplot(
    metric = c("mn_log_loss", "roc_auc"),
    select_best = TRUE
  ) + 
  geom_text(
    aes(
      y = mean,
      label = wflow_id
    ), 
    angle = 90, 
    nudge_x = 0.05,
    color = "black",
    size = 3
  ) +
  theme(legend.position = "none")

# -- Parameter values for xgboost
grid_results %>% 
  autoplot(
    id = "recipe_3_xgboost",
    metric = "mn_log_loss"
  )

# -- Analysis of residuals
grid_best_res <- 
  grid_results %>% 
  filter(
    wflow_id == "recipe_3_xgboost"
  ) %>% 
  .$result %>% 
  .[[1]] %>% # there is probably a better way to extract the results
  collect_predictions() %>% 
  mutate(
    res = abs((as.integer(Class) - 1) - .pred_1)
  )

grid_best_res %>% 
  ggplot(aes(x = res)) +
  geom_histogram() +
  facet_grid(~ Class)

# Inspect observations which were severely under- or overpredicted
df_large_res <- 
  grid_best_res %>% 
  filter(
    res > 0.7
  ) %>% 
  arrange(
    desc(res)
  ) 

df_train %>%
  dplyr::slice(df_large_res$.row) %>% 
  unique %>% 
  filter(Class == 0) %>% 
  select(!c(id, Class)) %>% 
  plot_histogram()

df_train %>%
  dplyr::slice(df_large_res$.row) %>% 
  unique %>% 
  select(!c(id)) %>% 
  plot_boxplot(by = "Class")
  # Compare to the EDA
  # For class == 1, the model incorrectly predicts observations with a low EK
  # very low Skewness and a large Mean_Integreated.  

# -- Probability Calibration
grid_best_res %>% 
  cal_plot_breaks(
    Class,
    .pred_1,
    num_breaks = 15,
    include_rug = FALSE,
    event_level = "second"
  )

# Fit the calibration model on all oof predictions
model_isoreg <- 
  isoreg(
    x = grid_best_res$.pred_1,
    y = as.integer(grid_best_res$Class) - 1
  ) %>% as.stepfun() # as.stepfun in order to predict on new data

cal_preds <- 
  model_isoreg(grid_best_res$.pred_1)

df_cal_combined <- 
  bind_rows(
    mutate(
      grid_best_res, 
      source = "original"
    ), 
    mutate(
      grid_best_res, 
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

# ==== EXPORT ------------------------------------------------------------------------------------------ 

output_04 <- 
  list(
    "wflow" = wflow,
    "my_metric_set" = my_metric_set,
    "grid_results" = grid_results,
    "model_isoreg" = model_isoreg
  )

save(
  output_04,
  file = "./Output/04_output.RData"
)
