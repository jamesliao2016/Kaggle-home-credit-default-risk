library(tidyverse)
library(xgboost)
library(magrittr)
set.seed(0)

#---------------------------
cat("Loading data...\n")
tr <- read_csv("../input/application_train.csv") 
te <- read_csv("../input/application_test.csv")

bureau <- read_csv("../input/bureau.csv") %>% 
  mutate_if(is.character, funs(factor(.) %>% as.integer())) 

cred_card_bal <-  read_csv("../input/credit_card_balance.csv") %>% 
  mutate_if(is.character, funs(factor(.) %>% as.integer())) 

pos_cash_bal <- read_csv("../input/POS_CASH_balance.csv") %>% 
  mutate_if(is.character, funs(factor(.) %>% as.integer())) 

prev <- read_csv("../input/previous_application.csv") %>% 
  mutate_if(is.character, funs(factor(.) %>% as.integer())) 

#---------------------------
cat("Preprocessing...\n")

avg_bureau <- bureau %>% 
  group_by(SK_ID_CURR) %>% 
  summarise_all(funs(mean(., na.rm = TRUE))) %>% 
  mutate(buro_cnt = bureau %>%  
           group_by(SK_ID_CURR) %>% 
           count() %$% n)

avg_cred_card_bal <- cred_card_bal %>% 
  group_by(SK_ID_CURR) %>% 
  summarise_all(funs(mean(., na.rm = TRUE))) %>% 
  mutate(card_cnt = cred_card_bal %>%  
           group_by(SK_ID_CURR) %>% 
           count() %$% n)

avg_pos_cash_bal <- pos_cash_bal %>% 
  group_by(SK_ID_CURR) %>% 
  summarise_all(funs(mean(., na.rm = TRUE))) %>% 
  mutate(pos_cnt = pos_cash_bal %>%  
           group_by(SK_ID_PREV, SK_ID_CURR) %>% 
           group_by(SK_ID_CURR) %>% 
           count() %$% n)

avg_prev <- prev %>% 
  group_by(SK_ID_CURR) %>% 
  summarise_all(funs(mean(., na.rm = TRUE))) %>% 
  mutate(prev_cnt = prev %>%  
           group_by(SK_ID_CURR) %>% 
           count() %$% n)

tri <- 1:nrow(tr)
y <- tr$TARGET

tr_te <- tr %>% 
  select(-TARGET) %>% 
  bind_rows(te) %>%
  left_join(avg_bureau, by = "SK_ID_CURR") %>% 
  left_join(avg_cred_card_bal, by = "SK_ID_CURR") %>% 
  left_join(avg_pos_cash_bal, by = "SK_ID_CURR") %>% 
  left_join(avg_prev, by = "SK_ID_CURR") %>% 
  mutate_if(is.character, funs(factor(.) %>% as.integer())) %>% 
  data.matrix()

rm(tr, te, prev, avg_prev, bureau, avg_bureau, 
   cred_card_bal, avg_cred_card_bal, pos_cash_bal, avg_pos_cash_bal); gc()

#---------------------------
cat("Preparing data...\n")
dtest <- xgb.DMatrix(data = tr_te[-tri, ])
tr_te <- tr_te[tri, ]
tri <- caret::createDataPartition(y, p = 0.9, list = F) %>% c()
dtrain <- xgb.DMatrix(data = tr_te[tri, ], label = y[tri])
dval <- xgb.DMatrix(data = tr_te[-tri, ], label = y[-tri])
cols <- colnames(tr_te)

rm(tr_te, y, tri); gc()

myeta=0.025
mydepth=6
mychild=19
mygamma=0
mysubsamp=0.8
mycolsamp=0.632
myalpha=0
mylambda=0.05
myround=2000
#---------------------------
cat("Training model...\n")
p <- list(objective = "binary:logistic",
          booster = "gbtree",
          eval_metric = "auc",
          nthread = 8,
          eta = myeta,
          max_depth = mydepth,
          min_child_weight = mychild,
          gamma = mygamma,
          subsample = mysubsamp,
          colsample_bytree = mycolsamp,
          alpha = myalpha,
          lambda = mylambda,
          nrounds = myround)

m_xgb <- xgb.train(p, dtrain, p$nrounds, list(val = dval), print_every_n = 50, early_stopping_rounds = 200)

xgb.importance(cols, model=m_xgb) %>% 
  xgb.plot.importance(top_n = 30)


filename=paste0("tidy_xgb_eta",myeta,"_depth",mydepth,"_child",mychild,
                "_gamma",mygamma,"_subsamp",mysubsamp,"_colsamp",mycolsamp,
                "_alpha",myalpha,"_lambda",mylambda,"_AUC",round(m_xgb$best_score,5),".csv")
#---------------------------
read_csv("../input/sample_submission.csv") %>%  
  mutate(SK_ID_CURR = as.integer(SK_ID_CURR),
         TARGET = predict(m_xgb, dtest)) %>%
  write_csv(filename)

