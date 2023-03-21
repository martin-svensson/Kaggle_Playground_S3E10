# ====================================================================================================== #
# Description
#
#   GAM just takes way too long to fit (and therefore tune), so we fit a 
#   GAM on the training data here to be used in and ensemble (potentially)
#
# Change log:
#   Ver   Date        Comment
#   1.0   20/03/23    Initial version
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

load("./Output/01_output.RData")

# ------------------------------------------------------------------------------------------------------ #
# PROGRAM
# ------------------------------------------------------------------------------------------------------ #

test_rec <- 
  recipe(
    Class ~ 
      Mean_Integrated
    + SD
    + EK
    + Skewness
    + Mean_DMSNR_Curve
    + SD_DMSNR_Curve
    + EK_DMSNR_Curve
    + Skewness_DMSNR_Curve,
    data = df_train
  )

test_wflow <- 
  workflow() %>% 
  add_recipe(test_rec) %>% 
  add_model(
    gen_additive_mod(
      mode = "classification",
      select_features = TRUE,
      adjust_deg_free = 1
    ),
    formula = Class ~ 
      s(Mean_Integrated, k = 8)
    + s(SD)
    + s(EK, k = 10)
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

tic("test gam")
gam_fit <- 
  test_wflow %>% 
  fit(df_train)
toc()


# ==== EXPORT ------------------------------------------------------------------------------------------ 

save(
  gam_fit, 
  file = "./Output/04a_gam_fit.RData"
)
