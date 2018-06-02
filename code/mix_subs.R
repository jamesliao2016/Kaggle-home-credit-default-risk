require(data.table)

# date_string="20180530"
date_string="20180531"
# sub_prefix='first_submission_MacOSX_seed'
sub_prefix='lightgbm_xgb'
subdir='../subs/'
# files=dir(subdir,pattern=paste0(sub_prefix,'[0-9]*.csv'))
files=c('tidy_xgb_eta0.01_depth6_child19_gamma0_subsamp0.8_colsamp0.632_alpha0_lambda0.05_0.77888.csv', 'first_submission_MacOSX_seed_mix_20180530_7subs_simple_mean_sub.csv')
mix_method='simple_mean'  # simple_mean, rank_mean, logit_mean, 
mix_method='logit_mean'
mix_method='rank_mean'
nfile=5
nfile=length(files)

all_sub=NULL
weight_sum=0
# for (myfile_idx in range(length(files)))
for (myfile_idx in range(nfile)) {
  myfile=files[myfile_idx]
  myfile_full=paste0(subdir,myfile)
  mysub=fread(myfile_full)
  myweight=1
  mysub$RANK=rank(mysub$TARGET)/nrow(mysub)
  mysub$LOGIT=qlogis(mysub$TARGET)
  if (is.null(all_sub)) {
    all_sub=mysub
    all_sub$TARGET=all_sub$TARGET*myweight
    all_sub$RANK=all_sub$RANK*myweight
    all_sub$LOGIT=all_sub$LOGIT*myweight
  } else {
    all_sub$TARGET=all_sub$TARGET+mysub$TARGET*myweight
    all_sub$RANK=all_sub$RANK+mysub$RANK*myweight
    all_sub$LOGIT=all_sub$LOGIT+mysub$LOGIT*myweight
  }
  weight_sum=weight_sum+myweight
}

for (mix_method in c('simple_mean','rank_mean','logit_mean')) {
  temp_sub=all_sub[,.(SK_ID_CURR,TARGET)]
  if (mix_method=="logit_mean") {
    temp_sub$TARGET=all_sub$LOGIT/weight_sum
  } else if (mix_method == "rank_mean") {
    temp_sub$TARGET=all_sub$RANK/weight_sum
  } else {
    temp_sub$TARGET=all_sub$TARGET/weight_sum
  }
  sub_filename=paste0(sub_prefix,'_mix_',date_string,'_',nfile,'subs_',mix_method,'_sub.csv')
  fwrite(temp_sub, file=sub_filename, row.names=F, quote=F)
}
