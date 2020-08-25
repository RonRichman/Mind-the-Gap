#### Purpose: Fit predictive models to the triangle in Taylor and Ashe (1983)
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

#### Helper functions - Poisson Deviance

poisson_deviance <- function(true, predicted) {
  predicted = predicted+10e-100
  return(2  * sum(ifelse(
    true == 0 , 0 , true * log(true / predicted)
  ) - (true - predicted)))
}

### Fit a simple GLM model to triangle from Taylor and Ashe (1983), which is stored
### in the ChainLadder package as GenIns

triangle = GenIns %>% cum2incr() %>% as.LongTriangle() %>% data.table()
triangle[,origin := as.integer(origin)]
triangle[,dev := as.integer(dev)]
triangle[,calendaryear := origin + dev-1]

ay = triangle[,unique(origin)]
dy = triangle[,unique(dev)]

### Build a data frame that contains entries for all accident and development years

test_model = expand.grid(ay,dy) %>% data.table()
test_model %>% setnames(names(test_model), names(triangle)[1:2])
test_model[,calendaryear := origin + dev-1]

### Predict using GLM model

glm_model = glm(value~ as.factor(origin) + as.factor(dev)-1, 
                family = poisson(link = "log"), data = triangle)

triangle$preds_glm = (glm_model %>% predict(type = "response")) 
test_model$predictions = glm_model %>% predict(newdata = test_model, type = "response")

poisson_deviance(triangle$value, triangle$preds_glm)/triangle[,.N]

### Apply Chain-ladder

CL = GenIns %>% MackChainLadder()

test_model[calendaryear > 10, sum(predictions)] - (CL %>% summary)$Totals[4,1]

### Fit the GLM using the R Keras package

require(keras)

### Define the network

input <- layer_input(shape = c(19), dtype = 'float32', name = 'input') 

middle = input %>% layer_dense(units = 10, activation = "relu")
z = middle %>% layer_dense(units = 1, activation = "exponential", name = 'z', 
                           use_bias = F)

modelz <- keras_model(inputs = list(input), outputs = c(z))

y = input %>% 
  layer_dense(units = 1, activation = "exponential", name = 'y', 
              use_bias = F)

### Define a Keras model

model <- keras_model(inputs = list(input), outputs = c(y))

adam = optimizer_adam(lr=0.1)

model %>% compile(
  optimizer = adam,
  loss = "poisson")

### Derive data for Keras
train_mat = (model.matrix(~as.factor(origin) + as.factor(dev) - 1, data = triangle)) %>% 
  array(dim = c(55,19))
test_mat = (model.matrix(~as.factor(origin) + as.factor(dev) - 1, data = test_model)) %>%
  array(dim = c(test_model[,.N],19))

train_dat = list(input = train_mat, y = triangle$value)
test_dat = list(input = test_mat)

### Train the model

fit = model %>% fit(
  x = train_dat$input,
  y = train_dat$y,
  epochs = 1000,
  batch_size = 55,verbose = 1, shuffle = T)

### Derive predictions and test IBNR

triangle$preds_NN = (model %>% predict(train_dat$input))
poisson_deviance(triangle$value, triangle$preds_NN)/triangle[,.N]

test_model$preds_NN = (model %>% predict(test_dat$input))

test_model[calendaryear>10, sum(preds_NN)] - (CL %>% summary)$Totals[4,1]

### Fit a GLM using Keras Embeddings
require(keras)

### Define the network

ay <- layer_input(shape = c(1), dtype = 'int32', name = 'ay') 
ay_fact = ay %>% layer_embedding(input_dim = 20, output_dim = 1) %>% layer_flatten()

dy <- layer_input(shape = c(1), dtype = 'int32', name = 'dy') 
dy_fact = dy %>% layer_embedding(input_dim = 20, output_dim = 1) %>% layer_flatten()

y = list(ay_fact,dy_fact) %>% layer_add() %>% 
  layer_dense(units = 1, activation = "exponential", name = 'y', 
              use_bias = F, weights = list(array(c(1), dim = c(1,1))), trainable = F)

### Define a Keras model

model <- keras_model(inputs = list(ay,dy), outputs = c(y))

adam = optimizer_adam(lr=0.1)

model %>% compile(
  optimizer = adam,
  loss = "poisson")

### Derive data for Keras
train_dat = list(input = list(ay = triangle$origin, dy = triangle$dev), y = triangle$value)
test_dat = list(input = list(ay = test_model$origin, dy = test_model$dev))

### Train the model

fit_embed = model %>% fit(
  x = train_dat$input,
  y = train_dat$y,
  epochs = 1000,
  batch_size = 55,verbose = 1, shuffle = T)

### Derive predictions and test IBNR

triangle$preds_NN = (model %>% predict(train_dat$input))
poisson_deviance(triangle$value, triangle$preds_NN)/triangle[,.N]

test_model$preds_NN = (model %>% predict(test_dat$input))

test_model[calendaryear>10, sum(preds_NN)] - (CL %>% summary)$Totals[4,1]

### test efficiency 

to_plot = data.table(epoch = 1:1000, dummy_coded = fit$metrics$loss, embeddings = fit_embed$metrics$loss)

to_plot[1:1000] %>% melt(id.vars = "epoch") %>% 
  ggplot(aes(x=epoch, y=value))+geom_line(aes(colour = variable, group= variable))+theme_pubr()

### Fit a shallow NN using Keras
require(keras)

### Define the network

ay <- layer_input(shape = c(1), dtype = 'int32', name = 'ay') 
ay_fact = ay %>% layer_embedding(input_dim = 20, output_dim = 2) %>% layer_flatten()

dy <- layer_input(shape = c(1), dtype = 'int32', name = 'dy') 
dy_fact = dy %>% layer_embedding(input_dim = 20, output_dim = 2) %>% layer_flatten()

y = list(ay_fact,dy_fact) %>% layer_concatenate() %>% 
  layer_dense(units = 4, activation = "tanh") %>% 
  layer_dense(units = 1, activation = "sigmoid", name = 'y')

### Define a Keras model

model <- keras_model(inputs = list(ay,dy), outputs = c(y))

adam = optimizer_adam(lr=0.1)

model %>% compile(
  optimizer = adam,
  loss = "poisson")

### Derive data for Keras
train_dat = list(input = list(ay = triangle$origin, dy = triangle$dev),
                 y = (triangle$value - min(triangle$value))/(max(triangle$value) - min(triangle$value)))
                                  
test_dat = list(input = list(ay = test_model$origin, dy = test_model$dev))

### Train the model

fit_embed = model %>% fit(
  x = train_dat$input,
  y = train_dat$y,
  epochs = 1000,
  batch_size = 55,verbose = 1, shuffle = T)

### Derive predictions and test IBNR

triangle$preds_NN = (model %>% predict(train_dat$input))*(max(triangle$value) - min(triangle$value))+min(triangle$value)

poisson_deviance(triangle$value, triangle$preds_NN)/triangle[,.N]
poisson_deviance(triangle$value, triangle$preds_glm)/triangle[,.N]

test_model$preds_NN = (model %>% predict(test_dat$input))*(max(triangle$value) - min(triangle$value))+min(triangle$value)

test_model[calendaryear>10, sum(preds_NN)] - (CL %>% summary)$Totals[4,1]

### Plots

NN = test_model %>% as.triangle(value = "preds_NN") %>% incr2cum() %>% as.LongTriangle() %>% data.table()
NN[,description:="NN"]
GLM = test_model %>% as.triangle(value = "predictions") %>% incr2cum() %>% as.LongTriangle() %>% data.table()
GLM[,description:="GLM"]
data_triang = GenIns %>%  as.LongTriangle() %>% data.table()
data_triang[,description:="Data"]

to_plot = rbind(NN,GLM, data_triang)

data_vs_glm = to_plot[description %in% c("Data", "GLM")] %>% ggplot(aes(x=dev, y = value))+ 
  geom_point(aes(colour=description, shape=description, group = description))+
  facet_wrap(~origin)+theme_pubr()+ theme(axis.text.x = element_text(angle = 90, hjust = 1, size=9))+
  ggtitle("Data versus Fitted GLM")+ scale_y_continuous(labels = comma)

data_vs_NN = to_plot[description %in% c("Data", "NN")] %>% ggplot(aes(x=dev, y = value))+ 
  geom_point(aes(colour=description, shape=description, group = description))+
  facet_wrap(~origin)+theme_pubr()+ theme(axis.text.x = element_text(angle = 90, hjust = 1, size = 9))+
  ggtitle("Data versus Fitted NN")+ scale_y_continuous(labels = comma)

data_vs_glm | data_vs_NN