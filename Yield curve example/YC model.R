#### Purpose: Fit deep regression networks to BoE yield curve data
#### Author: Ronald Richman
#### License: MIT
#### Data: The data were sourced from the HMD by downloading the relevant text files

### disable GPU
Sys.setenv("CUDA_VISIBLE_DEVICES" = -1)  

require(data.table)
require(dplyr)
require(ggplot2)
require(data.table)
require(reshape2)
require(HMDHFDplus)
require(gnm)
require(stringr)
require(ggpubr)
require(lubridate)
require(bestNormalize)

dat = fread("c:/r/BoE_ycs.csv")

### get training data into format required for Keras

dat %>% setkey(years,term, curve)
dat[,id:=1:.N]

val = dat %>% sample_frac(0.1)

x = list(date_fact = dat[!id %in% val$id]$date_fact %>% as.array,
         term_fact = dat[!id %in% val$id]$term_std %>% as.array,
         curve_fact = dat[!id %in% val$id]$curve_fact %>%  as.array)

y = list(outputs = dat[!id %in% val$id]$yield_std %>% as.array)       

x_val = list(date_fact = dat[id %in% val$id]$date_fact %>% as.array,
             term_fact = dat[id %in% val$id]$term_std %>% as.array,
             curve_fact = dat[id %in% val$id]$curve_fact %>%  as.array)

y_val = list(outputs = dat[id %in% val$id]$yield_std %>% as.array) 


x_all = list(date_fact = dat$date_fact %>% as.array,
             term_fact = dat$term_std %>% as.array,
             curve_fact = dat$curve_fact %>%  as.array)

y_all = list(outputs = dat$yield_std %>% as.array)  


### train embedding model to reproduce yield curves

require(keras)

############### Build embedding layers
date_fact <- layer_input(shape = c(1), dtype = 'int32', name = 'date_fact') 
term_fact <- layer_input(shape = c(1), dtype = 'float32', name = 'term_fact')
curve_fact <- layer_input(shape = c(1), dtype = 'int32', name = 'curve_fact')

### date_embed is the state vector \theta

date_embed = date_fact %>% 
  layer_embedding(input_dim = 20000, output_dim = 3,input_length = 1 ) %>%
  layer_flatten(name = 'date_embed')

curve_embed = curve_fact %>% 
  layer_embedding(input_dim = 3, output_dim = 5,input_length = 1 ) %>%
  layer_flatten(name = 'curve_embed')

embeds = list(date_embed,curve_embed,term_fact) %>% 
  layer_concatenate()

outputs = embeds %>% 
  layer_dense(units = 128, activation = "tanh") %>% 
  layer_batch_normalization() %>% 
  layer_dropout(rate = 0.05) %>% 
  layer_dense(units = 128, activation = "tanh") %>% 
  layer_batch_normalization() %>% 
  layer_dropout(rate = 0.05) %>% 
  layer_dense(units = 1, activation = "sigmoid", name = 'outputs')

model <- keras_model(inputs = c(date_fact,term_fact, curve_fact), 
                     outputs = c(outputs))

adam = optimizer_adam(lr=0.01)
lr_callback = callback_reduce_lr_on_plateau(factor=.90, patience = 2, verbose=1, cooldown = 1, min_lr = 0.00005)
model_callback = callback_model_checkpoint(filepath = "c:/r/term_mod.h5", verbose = 1,save_best_only = TRUE)

model %>% compile(
  optimizer = adam,
  loss = "mse")
 
# fit = model %>% fit(
#   x = x,
#   y = y,
#   epochs = 100,
#   batch_size = 1024,verbose = 1, shuffle = T,validation_data = list(x_val, y_val),
#   callbacks = list(lr_callback,model_callback))

### file has been provided on GitHub

model = load_model_hdf5(paste0("c:/r/","term_mod",".h5"), compile = T)

preds = model %>% predict(x=x_all, batch_size = 16000)

dat[,preds := preds]

dat[,pred_yield :=(max-min)*preds+min ]

require(patchwork)

p1 = dat[years == ymd("2000-01-31")] %>% melt(measure.vars=c("pred_yield","yield")) %>% 
  ggplot(aes(x=term, y= value))+
  geom_point(aes(colour=variable, shape = curve))+
  ggtitle("31 Janaury 2000")+ theme_pubr()

p2 = dat[years == ymd("2019-12-31")] %>% melt(measure.vars=c("pred_yield","yield")) %>% 
  ggplot(aes(x=term, y= value))+geom_point(aes(colour=variable, shape = curve))+
  ggtitle("31 December 2019")+ theme_pubr()


p1|p2

ggsave("c:/r/fitted_yc.pdf", width = 15, height = 15)

### examine embeddings
date_embed = (model$layers[[3]] %>% get_weights())[[1]] %>% data.table()
date_embed %>% setnames(names(date_embed), paste0("Theta_", 1:3))
date_embed = date_embed[2:10407]
date_embed[,Date := dat[,unique(years)]]

date_embed[Date %in% dat[month_end == T, unique(years)]] %>% melt(id.vars = "Date") %>%data.table %>%  
  ggplot(aes(x=Date, y=value))+geom_line()+facet_wrap(~variable)+theme_pubr()

ggsave("c:/r/fitted_theta.pdf", width = 15, height = 15)

### forecasts

### examine embeddings

date_embed = (model$layers[[3]] %>% get_weights())[[1]] %>% data.table()
date_embed = date_embed[2:10407]
date_embed[,Date := dat[,unique(years)]]

### use pre 2019 as train
look_back = 10
train_dat = date_embed[Date < (ymd("2019-01-01") - days(18))]
### use post 2018 as test
test_dat = date_embed[Date >= (ymd("2019-01-01") - days(18))]

samples = train_dat[,.N] - look_back

date_fact_series_train =array( dim = c(samples,look_back,3))
date_fact_last_train =array( dim = c(samples,3))
forecast_target_train =array( dim = c(samples,3))

for (samp in 1:samples){
  date_fact_series_train[samp,,] = as.matrix(train_dat[(samp):(samp+look_back-1),c(1:3),with=F])
  date_fact_last_train[samp,] = as.matrix(train_dat[(samp+look_back-1),c(1:3),with=F])
  forecast_target_train[samp,] = as.matrix(train_dat[(samp+look_back),c(1:3),with=F])
}

x_series = list(date_fact_series = date_fact_series_train,date_last_series= date_fact_last_train)
y_series = list(outputs_series = forecast_target_train)


samples = test_dat[,.N] - look_back

date_fact_series_test =array( dim = c(samples,look_back,3))
date_fact_last_test =array( dim = c(samples,3))
forecast_target_test =array( dim = c(samples,3))

for (samp in 1:samples){
  date_fact_series_test[samp,,] = as.matrix(test_dat[(samp):(samp+look_back-1),c(1:3),with=F])
  date_fact_last_test[samp,] = as.matrix(test_dat[(samp+look_back-1),c(1:3),with=F])
  forecast_target_test[samp,] = as.matrix(test_dat[(samp+look_back),c(1:3),with=F])
}

x_series_test = list(date_fact_series = date_fact_series_test, date_last_series=date_fact_last_test)
y_series_test = list(outputs_series = forecast_target_test)

train_res = list()
test_res = list()

mod_name = "term_mod_forecasts_cnn_32"
for (i in 1:10){
  
  print(i)
  
  ##### build time based model
  date_fact_series <- layer_input(shape = c(look_back,3), dtype = 'float32', name = 'date_fact_series')
  date_last_series <- layer_input(shape = c(3), dtype = 'float32', name = 'date_last_series') 
  
  features = date_fact_series %>%
    layer_conv_1d(filters = 32, kernel_size = 5) %>%
    layer_average_pooling_1d(pool_size = 5, strides = 2) %>%
    layer_flatten() %>%
    layer_dropout(rate = 0.15)
  
  date_fact_series_embed_average = features %>%
    layer_dense(units = 3, activation = "linear", kernel_regularizer = regularizer_l1(l = 0.00001))
  
  
  outputs_series = list(date_last_series, date_fact_series_embed_average) %>%  layer_add(name = 'outputs_series')
  
  
  
  model_series <- keras_model(inputs = c(date_fact_series,date_last_series ), 
                              outputs = outputs_series)
  
  ### run model
  adam = optimizer_adam(lr=0.001)
  lr_callback = callback_reduce_lr_on_plateau(factor=.90, patience = 5, verbose=1, cooldown = 1, min_lr = 0.00005)
  model_callback = callback_model_checkpoint(filepath = paste0("c:/r/",mod_name, i,".h5"), 
                                             verbose = 1,save_best_only = TRUE)
  
  pinball_loss = function(y_true, y_pred, tau=0.5){
    err = y_true - y_pred
    k_mean(k_maximum(tau * err, (tau - 1) * err), axis=-1)
  }
  
  model_series %>% compile(
    optimizer = adam,
    loss = "mse")
  
  # fit = model_series %>% fit(
  #   x = x_series,
  #   y = y_series,
  #   epochs = 50,
  #   batch_size = 32,verbose = 1, shuffle = T,
  #   validation_split = 0.05,
  #   callbacks = list(lr_callback,model_callback))
  # 
  adam = optimizer_adam(lr=0.001)
  
  model_series %>% compile(
    optimizer = adam,
    loss = "mse")
  # 
  # fit = model_series %>% fit(
  #   x = x_series,
  #   y = y_series,
  #   epochs = 50,
  #   batch_size = 32,verbose = 1, shuffle = T,
  #   validation_split = 0.05,
  #   callbacks = list(lr_callback,model_callback))
  
  model_series = load_model_hdf5(paste0("c:/r/",mod_name, i,".h5"), compile = F)
  
  model_series %>% compile(
    optimizer = adam,
    loss = "mse")
  
  preds = model_series %>% predict(x_series, batch_size = 1024)
  
  train_dat[(look_back+1):.N,paste0("preds",1:3) := data.table(preds)]
  
  preds_test = model_series %>% predict(x_series_test, batch_size = 1024)
  
  test_dat[(look_back+1):.N,paste0("preds",1:3) := data.table(preds_test)]
  
  train_res[[i]] = train_dat %>% copy
  test_res[[i]] = test_dat %>% copy
  
  
}

for(i in 1:10){
  train_res[[i]][,run:=i]
  test_res[[i]][,run:=i]
}

train_dat = train_res %>% rbindlist()
test_dat = test_res %>% rbindlist()

train_dat[,set:="train"]
test_dat[,set:="test"]

all_dat = rbind(train_dat, test_dat)


test_dat[,paste0("lag_V",1:3):=lapply(.SD, lag),by=run, .SDcols = 1:3]

test_dat[!is.na(preds1), .(forecast_mean = mean(c((V1-preds1)^2,(V2-preds2)^2,(V3-preds3)^2)),
                           forecast_sd = sd(c((V1-preds1)^2,(V2-preds2)^2,(V3-preds3)^2)),
                           naive_mean = mean(c((V1-lag_V1)^2,(V2-lag_V2)^2,(V3-lag_V3)^2)),
                           naive_sd = sd(c((V1-lag_V1)^2,(V2-lag_V2)^2,(V3-lag_V3)^2))), by = run]


temp = train_res[[1]] %>% copy
temp = temp[,c(1:4), with=F]
temp = cbind(temp,
             train_dat[,.(preds1  = mean(preds1),preds2 = mean(preds2), preds3 = mean(preds3)), by = .(Date)][,c(-1)])
train_dat = temp %>% copy

temp = test_res[[1]] %>% copy
temp = temp[,c(1:4), with=F]
temp = cbind(temp,
             test_dat[,.(preds1  = mean(preds1),preds2 = mean(preds2), preds3 = mean(preds3)), by = .(Date)][,c(-1)])
test_dat = temp %>% copy

test_dat[,paste0("lag_V",1:3):=lapply(.SD, lag), .SDcols = 1:3]

forecast_res = test_dat[!is.na(preds1), .(Type = "Forecast", `Average MSE` = mean(c((V1-preds1)^2,(V2-preds2)^2,(V3-preds3)^2))*10^4,
                                          `Standard Deviation of Residuals` = sd(c((V1-preds1)^2,(V2-preds2)^2,(V3-preds3)^2))*10^4)]

naive_res = test_dat[!is.na(preds1), .(Type = "Naive", `Average MSE` = mean(c((V1-lag_V1)^2,(V2-lag_V2)^2,(V3-lag_V3)^2))*10^4,
                                       `Standard Deviation of Residuals` = sd(c((V1-lag_V1)^2,(V2-lag_V2)^2,(V3-lag_V3)^2))*10^4)]

rbind(forecast_res, naive_res ) %>% 
  xtable::xtable(, type="latex") %>% print(file = "c:/r/yc_forecast_final_cnn32.tex")

