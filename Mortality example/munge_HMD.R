#### Purpose: Munge HMD and associated data
#### Author: Ronald Richman
#### License: MIT

require(data.table)
require(dplyr)
require(ggplot2)
require(reshape2)
require(ChainLadder)
require(ggpubr)
require(patchwork)
require(scales)
require(MortalityLaws)

### Place the files for mx from the HMD in this directory and the run the code (https://www.mortality.org/hmd/zip/by_statistic/death_rates.zip)
dir = "C:/R/death_rates"

all_files = list.files(dir)

all_mort = list()
i = 0
for (file in all_files){
  i=i+1
  dat = HMDHFDplus::readHMD(paste0(dir,"/",file)) %>% data.table
  dat[,Country :=str_replace(file, ".Mx_1x1.txt", "")]
  all_mort[[i]] = dat
}

all_mort = rbindlist(all_mort)
all_mort %>% fwrite("c:/r/allmx.csv")

### Get Australia HMD data

dat_aus = ReadAHMD("mx")
dat_aus = dat_aus$data %>% data.table()
dat_aus[, Country := "AUS"]
dat_aus[, Sub_Nat := country]
dat_aus$country = NULL
dat_aus %>% fwrite("c:/r/allmx_oz")

### Get Canada HMD data

dat_can = ReadCHMD("mx")
dat_can = dat_can$data %>% data.table()
dat_can[, Country := "CAN"]
dat_can[, Sub_Nat := country]
dat_can$country = NULL
dat_can = dat_can[Sub_Nat != "CAN"]
dat_can %>% fwrite("c:/r/allmx_can")

### Get Canada HMD data

dat_jpn = ReadJMD("mx")
dat_jpn = dat_jpn$data %>% data.table()
dat_jpn[, Country := "JPN"]
dat_jpn[, Sub_Nat := region]
dat_jpn$region = NULL
dat_jpn = dat_jpn[Sub_Nat != "Japan"]
dat_jpn %>% fwrite("c:/r/allmx_jpn")

### Get USA HMD data - Place the files for mx from the HMD in this directory and the run the code (https://usa.mortality.org/uploads/lifetables/lifetables.zip)

dir = "C:/R/usahmd/lifetables/states"

all_files = list.files(dir)

all_mort = list()
i = 0
for (file in all_files){
  i=i+1
  dat = fread(paste0(dir,"/",file,"/",file,"_fltper_1x1.csv")) %>% data.table
  all_mort[[i]] = dat
  i=i+1
  dat2 = fread(paste0(dir,"/",file,"/",file,"_mltper_1x1.csv")) %>% data.table
  all_mort[[i]] = dat2
}

all_mort = rbindlist(all_mort)
all_mort[,Country:= "USA"]
all_mort[,Sub_Nat:= PopName]
all_mort$PopName = NULL
all_mort[,Sex:= ifelse(Sex=="f", "Female", "Male")]
all_mort %>% fwrite("c:/r/allmx_usa.csv")

### read all files
all_mort= fread("c:/r/allmx.csv")
all_mort[,Sub_Nat:="National"]
all_mort$OpenInterval = NULL
all_mort_aus= fread("c:/r/allmx_oz")
all_mort_can= fread("c:/r/allmx_can")
all_mort_jpn= fread("c:/r/allmx_jpn")
all_mort = rbindlist(list(all_mort, all_mort_aus, all_mort_can, all_mort_jpn))


all_mort_usa= fread("c:/r/allmx_usa.csv")
all_mort_usa = all_mort_usa[,c(1:4,12:13)] %>% dcast.data.table(Year+Age+Country+Sub_Nat~Sex, value.var = "mx")
all_mort_usa[,Total:=(Male+Female)/2]

all_mort=rbindlist(list(all_mort, all_mort_usa), use.names = T)
all_mort[,Age:=as.integer(Age)]

all_mort%>% setkey(Country, Sub_Nat, Year, Age)

#### Setup file

all_mort = all_mort[Year>1949 & Age<100]
all_mort = all_mort[,c(1,2,3,4,6,7),with=F]
all_mort = all_mort %>% melt(id.vars=c("Year" ,"Age", "Country", "Sub_Nat")) %>% data.table

all_mort[,mx:=value]
all_mort$value = NULL

all_mort[,Sex:=variable]
all_mort$variable = NULL

all_mort[,Country_fact:=as.integer(as.factor(Country))-1]
all_mort[,Sub_Nat_fact:=as.integer(as.factor(Sub_Nat))-1]
all_mort[,Sex_fact:=as.integer(as.factor(Sex))-1]

### Impute missing mortality values using overall mean
imputed = all_mort[mx > 0 & !is.na(mx)][,.(imputed = mean(mx)), keyby = .(Sex, Year, Age)]
all_mort %>% setkey(Sex, Year, Age)
all_mort = merge(all_mort, imputed)
all_mort[,imputed_flag := ifelse(mx==0|is.na(mx),"TRUE", "FALSE")]
all_mort[imputed_flag == T, mx:= imputed]
all_mort[,logmx:=log(mx)]
all_mort[imputed_flag == T]

all_mort %>% fwrite("c:/r/all_mx_sub.csv")

### Check properties of file

all_mort[,unique(Country)] %>% length()
all_mort[,unique(Sub_Nat)] %>% length()
all_mort[,unique(interaction(Country, Sub_Nat))]
all_mort[,unique(Year)] %>% length()