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

# ------------------------------------------------------------------------------------------------------ #
# IMPORT AND SOURCES
# ------------------------------------------------------------------------------------------------------ #

load("./Output/01_df_train.RData")
load("./Output/01_data_split.RData")

# ------------------------------------------------------------------------------------------------------ #
# PROGRAM
# ------------------------------------------------------------------------------------------------------ #

df_train <- training(data_split)

features <- 
  names(df_train)[2:9]

target <- 
  names(df_train)[10]

# --- Feature Engineering -----------------------------------------------------
# Create multiple different recipes if we want to include feature engineering
# in the model comparison (combine them in a workflow set)

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
  ) %>% 
  step_interact(
    terms = ~ all_numeric():all_numeric()
  )
  
# ==== EXPORT ------------------------------------------------------------------------------------------ 

save(
  recipe_1,
  file = "./Output/03_recipes.RData"
)

