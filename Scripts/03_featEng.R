# ====================================================================================================== #
# Description
#
#   Feature engineering: implementing findings from EDA
#
# Change log:
#   Ver   Date        Comment
#   1.0   13/03/23    Initial version
#
# ====================================================================================================== #
# ------------------------------------------------------------------------------------------------------ #
# LIBRARIES
# ------------------------------------------------------------------------------------------------------ #

library(data.table)
library(tidyverse)
library(magrittr)

library(tidymodels)
library(themis)

# ------------------------------------------------------------------------------------------------------ #
# IMPORT AND SOURCES
# ------------------------------------------------------------------------------------------------------ #

load("./Output/01_output.RData")

# ------------------------------------------------------------------------------------------------------ #
# PROGRAM
# ------------------------------------------------------------------------------------------------------ #

df_train <- training(output_01$data_split)

features <- 
  names(df_train)[2:9]

target <- 
  names(df_train)[10]

# --- Feature Engineering -----------------------------------------------------
# Create multiple different recipes if we want to include feature engineering
# in the model comparison (combine them in a workflow set)

# -- No interactions
recipe_1 <- 
  recipe(
    as.formula(paste0(target, "~", paste0(features, collapse = "+"))),
    data = df_train
  ) %>% 
  step_log(
    Mean_DMSNR_Curve, 
    SD_DMSNR_Curve
  ) %>%
  step_normalize(
    all_numeric()
  )

# -- All interactions
recipe_2 <- 
  recipe(
    as.formula(paste0(target, "~", paste0(features, collapse = "+"))),
    data = df_train
  ) %>% 
  step_log(
    Mean_DMSNR_Curve, 
    SD_DMSNR_Curve
  ) %>%
  step_normalize(
    all_numeric()
  ) %>% 
  step_interact(
    terms = ~ all_numeric():all_numeric()
  )

# -- hybrid sampling for class imbalance 
#    builds on recipe 1 since interactions do not seem to be significant
#    hybrid sampling works better than not handling class imbalance

recipe_3 <- 
  recipe(
    as.formula(paste0(target, "~", paste0(features, collapse = "+"))),
    data = df_train
  ) %>% 
  step_log(
    Mean_DMSNR_Curve, 
    SD_DMSNR_Curve
  ) %>%
  step_normalize(
    all_numeric()
  ) %>% 
  themis::step_bsmote(
    seed = 1553,
    over_ratio = tune()
  )
  
# ==== EXPORT ------------------------------------------------------------------------------------------ 

output_03 <- 
  list(
    "recipe_1" = recipe_1,
    "recipe_2" = recipe_2,
    "recipe_3" = recipe_3
  )  

save(
  output_03,
  file = "./Output/03_output.RData"
)

