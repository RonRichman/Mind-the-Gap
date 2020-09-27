#### Purpose: Fit deep regression network to all the HMD and related data
#### Author: Ronald Richman
#### License: MIT
#### Data: The data were sourced from the HMD by downloading the relevant text files

require(data.table)
require(dplyr)
require(ggplot2)
require(reshape2)
require(ChainLadder)
require(ggpubr)
require(patchwork)
require(scales)
require(MortalityLaws)

### Run the file munge_HMD.R to produce the following data file in csv form.

all_mort  = fread("c:/r/all_mx_sub.csv")
all_mort[,region := paste0(Country,"_",Sub_Nat)]
all_mort[,type:=ifelse(Sub_Nat == "National", "National", "Sub-National")]

#### Fit LC baseline to all countries with sufficient data

# Only take regions with 10 years of data or more

regions_LC = all_mort[Year<2000][,.N/100,keyby = .(region,Sex)][V1>10]$region %>% unique

# The following code loops through each country and sex separately and fits the LC model:
# This is done using the SVD procedure as per the original formulation in the LC paper.

# Forecasts are with a random walk with drift. 

results=list()
i=0

for (region_select in regions_LC){
  for (sex in c("Male", "Female")) {
    
    i=i+1        
    
    print(i)        
    print(region_select)
    print(sex)
    
    train=all_mort[Year<2000][region == region_select][Sex == sex]
    test=all_mort[Year>=2000][region == region_select][Sex == sex]
    
    x = train[,unique(Age)] %>% length()
    t = train[,unique(Year)] %>% length()
    t_forecast = test[,unique(Year)] %>% length()
    
    ### fit via SVD
    
    train[,ax:= mean(logmx), by = (Age)]
    train[,mx_adj:= logmx-ax]
    test[,ax:= mean(logmx), by = (Age)]
    test[,mx_adj:= logmx-ax]
    
    rates_mat = train %>% dcast.data.table(Age~Year, value.var = "mx_adj", sum) %>% as.matrix()
    rates_mat=rates_mat[,-1]
    svd_fit = svd(rates_mat)
    
    ax =train[,unique(ax)]
    bx =svd_fit$u[,1]*svd_fit$d[1]
    k =svd_fit$v[,1]
    
    c1 = mean(k)
    c2 = sum(bx)
    ax = ax+c1*bx
    bx = bx/c2
    k = (k-c1)*c2
    
    forecast_k  =k %>% forecast::rwf(t_forecast, drift = T)
    k_forecast = forecast_k$mean
    
    fitted = (ax+(bx)%*%t(k)) %>% melt
    train$pred_LC_svd = fitted$value %>% exp
    fitted_test = (ax+(bx)%*%t(k_forecast)) %>% melt
    test$pred_LC_svd =   fitted_test$value %>% exp
    
    results[[i]] = rbind(train, test)
  }
}

all_mort = rbindlist(results)
metrics= all_mort[Year>1999,.(LC_SVD=sum((mx-pred_LC_svd)^2)/.N), keyby = .(Country,Sub_Nat,Sex)] %>% 
  melt(id.vars = c("Country", "Sub_Nat","Sex"))
metrics[,min_mse:=min(value), by = .(Country,Sub_Nat,Sex)]
metrics[,best:=ifelse(value==min_mse,T,F)]
metrics[,Model:=variable]

metrics[,type:=ifelse(Sub_Nat == "National", "National", "Sub-National")]
metrics[order(min_mse)] %>% 
  ggplot(aes(x=reorder(interaction(Country, Sub_Nat),(min_mse), sum), y = (min_mse)))+
  geom_col(aes(fill = type))

metrics[!is.na(value) & !is.infinite(value),
        .(`Average MSE` = mean(value)*10^4, 
          `Median MSE` = median(value)*10^4),
        keyby = type] %>% 
  xtable::xtable(, type="latex") %>% print(file = "c:/r/lc_results_subnat.tex")

#### Fit neural network 
all_mort=all_mort[!is.na(logmx) & mx>0]

scale_min_max = function(dat,dat_test)  {
  min_dat = min(dat)
  max_dat = max(dat)
  dat_scaled=(dat-min_dat)/(max_dat-min_dat)
  dat_scaled_test = (dat_test-min_dat)/(max_dat-min_dat)
  return(list(train = dat_scaled, test = dat_scaled_test, min = min_dat, max=max_dat))
}

train = all_mort[Year < 2000]
test = all_mort[Year >= 2000]

scaled = scale_min_max(train$logmx, test$logmx)
train$mx_scale = scaled$train
test$mx_scale = scaled$test

year_scale = scale_min_max(train$Year,test$Year)
train$Year = year_scale[[1]]
test$Year = year_scale[[2]]

#### Build data for the neural network 

get_keras_data = function(dat) {
  dat = dat[,c("Year","Age", "Country_fact", "Sex_fact","Sub_Nat_fact", "mx_scale"),with=F]
  x = list(Year      = dat$Year,
           Age = dat$Age, Country = dat$Country_fact, Sex=dat$Sex_fact,
           SubNat = dat$Sub_Nat_fact)
  
  y = (main_output= dat$mx_scale)
  return(list(x = x, y = y))
}

country_dim = train[,max(Country_fact)]+1
region_dim = train[,max(Sub_Nat_fact)]+1

#train

train_list = get_keras_data(train)
test_list = get_keras_data(test)

# define network
all_dat_list = list()

mod_name = "tanh_128_embd_5_5_10_10_batchnorm_dropout_04_05"

# fit the network ten times

for (i in 1:10) {
print(i)
require(keras)
k_clear_session()

############### Build embedding layers
Year <- layer_input(shape = c(1), dtype = 'float32', name = 'Year') 
Age <- layer_input(shape = c(1), dtype = 'int32', name = 'Age')
Country <- layer_input(shape = c(1), dtype = 'int32', name = 'Country')
Sex <- layer_input(shape = c(1), dtype = 'int32', name = 'Sex')
SubNat <- layer_input(shape = c(1), dtype = 'int32', name = 'SubNat')

Age_embed = Age %>% 
  layer_embedding(input_dim = 100, output_dim = 5,input_length = 1, name = 'Age_embed') %>%
  keras::layer_flatten()


Sex_embed = Sex %>% 
  layer_embedding(input_dim = 2, output_dim = 5,input_length = 1, name = 'Sex_embed') %>%
  keras::layer_flatten()

Country_embed = Country %>% 
  layer_embedding(input_dim = country_dim, output_dim = 10,input_length = 1, name = 'Country_embed') %>%
  keras::layer_flatten()

SubNat_embed = SubNat %>% 
  layer_embedding(input_dim = region_dim, output_dim = 10,input_length = 1, name = 'SubNat_embed') %>%
  keras::layer_flatten()

feats <- list(Year,Age_embed,Sex_embed,Country_embed,SubNat_embed) %>% layer_concatenate() %>% layer_batch_normalization() %>% layer_dropout(rate = 0.04)

middle=feats%>% 
  layer_dense(units = 128, activation = 'tanh') %>% 
  layer_batch_normalization() %>% 
  layer_dropout(0.05) %>% 
  
  layer_dense(units = 128, activation = 'tanh') %>% 
  layer_batch_normalization() %>% 
  layer_dropout(0.05) %>% 
  
  layer_dense(units = 128, activation = 'tanh') %>% 
  layer_batch_normalization() %>% 
  layer_dropout(0.05) %>% 
  
  layer_dense(units = 128, activation = 'tanh') %>% 
  layer_batch_normalization() %>% 
  layer_dropout(0.05)

# skip connection

main_output =
  
  layer_concatenate(list(feats, middle)) %>% 
  layer_dense(units = 128, activation = 'tanh') %>% 
  layer_batch_normalization() %>% 
  layer_dropout(0.05) %>%
  layer_dense(units = 1, activation = 'sigmoid', name = 'main_output')

model <- keras_model(
  inputs = c(Year,Age,Country,Sex,SubNat), 
  outputs = c(main_output))

adam = optimizer_adam(lr=0.0005)
lr_callback = callback_reduce_lr_on_plateau(factor=.80, patience = 5, verbose=1, cooldown = 5, min_lr = 0.00005)
model_callback = callback_model_checkpoint(filepath = paste0("c:/r/", mod_name,"_",i, ".mod"), verbose = 1,save_best_only = TRUE)

model %>% compile(
  optimizer = adam,
  loss = "mse")
 
fit = model %>% fit(
  x = train_list$x,
  y = train_list$y,
  epochs = 50,
  batch_size = 4096,verbose = 1, shuffle = T,
  validation_split = 0.05,
  callbacks = list(lr_callback,model_callback))

# learning rate restart

adam = optimizer_adam(lr=0.0005)
model %>% compile(
  optimizer = adam,
  loss = "mse")

fit = model %>% fit(
  x = train_list$x,
  y = train_list$y,
  epochs = 50,
  batch_size = 4096,verbose = 1, shuffle = T,
  validation_split = 0.05,
  callbacks = list(lr_callback,model_callback))

model = load_model_hdf5(paste0("c:/r/", mod_name,"_",i, ".mod"))

train$mx_deep_reg_full = model %>% predict(train_list$x, batchsize = 32000)
train[,mx_deep_reg_full:=exp(mx_deep_reg_full*(scaled$max-scaled$min)+scaled$min)]

test$mx_deep_reg_full = model %>% predict(test_list$x, batchsize = 32000)
test[,mx_deep_reg_full:=exp(mx_deep_reg_full*(scaled$max-scaled$min)+scaled$min)]

deep =rbind(train,test
) %>% setkey(Country,Sub_Nat, Sex, Year,Age)

all_mort %>% setkey(Country,Sub_Nat, Sex, Year,Age)
all_mort[,mx_deep_reg_full:=deep[,c("mx_deep_reg_full"),with=F]]
all_mort[,run:=i]
all_mort[,modname := mod_name]

all_dat_list[[i]] = all_mort %>% copy()

}

all_dat = all_dat_list %>% rbindlist()
all_dat[Year>1999,.(
  LC_SVD=sum((mx-pred_LC_svd)^2)/.N,
  DEEP=sum((mx-mx_deep_reg_full)^2)/.N), keyby = .(run)]

all_dat %>% fwrite(paste0("c:/r/", mod_name, ".csv"))

temp = all_dat_list[[1]] %>% copy
temp$pred = all_dat[,.(pred = mean(mx_deep_reg_full)), by = .(Country, Sub_Nat, Sex, Age, Year, type)]$pred
all_dat = temp

#### Compare performance
metrics = all_dat[Year>1999,.(
  LC_SVD=sum((mx-pred_LC_svd)^2)/.N,
  DEEP=sum((mx-pred)^2)/.N), keyby = .(type, Country,Sub_Nat,Sex)] %>% 
  melt(id.vars = c("type", "Country","Sub_Nat", "Sex")) 

metrics[,min_mse:=min(value), by = .(type,Country,Sub_Nat,Sex)]
metrics[,best:=ifelse(value==min_mse,T,F)]
metrics[best !=T,min_mse_2:=min(value), by = .(type,Country,Sub_Nat,Sex)]
metrics[best !=T,best_2:=ifelse(value==min_mse_2,T,F)]

metrics[,Model:=variable]
metrics[!is.na(value) & !is.infinite(value),
        .(`Average MSE` = mean(value)*10^4, 
          `Median MSE` = median(value)*10^4,
          `Best Performance`=sum(best==T)),
        keyby = .(Model, type)] %>% 
  xtable::xtable(, type="latex") %>% print(file = "c:/r/DEEP_results_final.tex")

metrics[,Region:=ifelse(Sub_Nat == "National", "National", Country)]

metrics[order(min_mse)] %>% 
  ggplot(aes(x=reorder(interaction(Country, Sub_Nat),log(min_mse), sum), y = log(min_mse)))+
  geom_col(aes(fill = Region)) + facet_wrap(~Model) + theme_pubr()+ 
  theme(axis.text.x=element_blank(),
        axis.ticks.x=element_blank())+  labs(x = "Regions, ordered by MSE", y="Log of MSE")

ggsave("c:/r/mse_compare.pdf")
