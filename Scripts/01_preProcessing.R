# ====================================================================================================== #
# Description
#
#   Pre-processing, including data split (train/test)
#
# Change log:
#   Ver   Date        Comment
#   1.0   12/03/23    Initial version
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
# PROGRAM
# ------------------------------------------------------------------------------------------------------ #

df_train <- fread("./Data/train.csv")

df_train %>% str
df_train %>% summary

# -- Target to factor (tidymodels works with factors, not integers)

df_train[
  ,
  Class := as.factor(Class)
]

# -- Data split
# The classes are imbalanced, so we split accordingly

set.seed(752)

data_split <- 
  df_train %>% 
  initial_split(
    prop = 0.8,
    strata = Class
  )

# ==== EXPORT ------------------------------------------------------------------------------------------ 

output_01 <- 
  list(
    "df_train" = df_train,
    "data_split" = data_split
  )

save(
  output_01,
  file = "./Output/01_output.RData"
)
