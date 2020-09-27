#### Purpose: Build the training data for the YC example
#### Author: Ronald Richman
#### License: MIT
#### Data: The data were sourced from the BoE by downloading the relevant text files

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

# Note that these files are available on GitHub

dat = fread("c:/r/uk_ycs.csv", header=T)
dat = dat %>% melt(id.vars = "years") %>% data.table()
dat[,curve:="nominal"]

dat_rl = fread("c:/r/uk_yc_rl.csv", header=T)
dat_rl = dat_rl %>% melt(id.vars = "years") %>% data.table()
dat_rl[,curve:="real"]

dat=rbind(dat, dat_rl)

### remove zero entries
dat = dat[value != 0]

dat[,years:=ymd(years)]

# remove zero entries
dat[,agg:=sum(value), by = years]
dat = dat[agg!= 0]

dat[,term:=variable]
dat[,yield:=value]
dat$variable = NULL
dat$value = NULL

### find month end curves

dat[, ym:= paste0(year(years), month(years))]
dat[,day:=as.integer(day(years))]
dat[,max_day := max(day), by = ym]
dat[,month_end := ifelse(day == max_day,T,F)]

### standardize variables for keras
dat[,date_fact:=as.integer(as.factor(years))]
dat[,term_fact:=as.integer(as.factor(term))]
dat[,curve_fact:=as.integer(as.factor(curve))]

# check distribution of yields
dat$yield %>% hist
dat[,mean:=mean(yield)]
dat[,sd:=sd(yield)]
dat[,z_score:= (yield-mean)/sd]
dat$z_score %>% hist

dat[,min:=min(yield), by= curve]
dat[,max:=max(yield), by= curve]
dat[,yield_std := (yield-min)/(max-min)]
dat$yield_std %>% hist

dat[,term:=as.double(as.character(term))]
dat[,min_term:=min(term)]
dat[,max_term:=max(term)]
dat[,term_std := (term-min_term)/(max_term-min_term)]
dat$term_std %>% hist

month_end_data = dat[month_end == T]

dat %>% fwrite("c:/r/BoE_ycs.csv")
