# -------------------------------------------------------------------------------
#     III Encontro Nacional dos Estudantes de Atuária - ENEAT - UNIFAL - MG
# Minicurso: Modelagem Preditiva - Uma Abordagem de Statistical Machine Learning
#                    Ministrante: Lucas Pereira Lopes
#             Parte I - Modelos de Regressão - 24/04/2018
# -------------------------------------------------------------------------------

# =====================================
#              PACOTES
# =====================================

ipak <- function(pkg){
  new.pkg <- pkg[!(pkg %in% installed.packages()[, "Package"])]
  if (length(new.pkg)) 
    install.packages(new.pkg, dependencies = TRUE)
  sapply(pkg, require, character.only = TRUE)
}

packages <- c("readr","ggplot2","Metrics","readxl","glmnet","gridExtra","data.table","tm",
              "FNN","randomForest","rpart","rpart.plot",
              "caret","pROC","ROCR","e1071","MASS","rattle","dplyr","xgboost",
              "lasso2")
ipak(packages)

# -------------------------------------
#         Modelos Paramétricos
# -------------------------------------
rm(list=ls())

library(readr)
NAICExpense <-  read_csv("C:/Users/icsa/Desktop/Minicurso - Modelagem Preditiva - III ENEAT/Dados/NAICExpense.csv")
NAICExpense <- na.omit(NAICExpense)

# To study expenses, we examine a random sample of 500 insurance companies from the 
# National Association of Insurance Commissioners (NAIC) database of over 3,000 companies
# 2005 annual reports - EUA

sample <- sample.int(n = nrow(NAICExpense), size = floor(.7*nrow(NAICExpense)), replace = F)
treino <- NAICExpense[sample, -c(1)]
teste  <- NAICExpense[-sample, -c(1)]

attach(treino)

# Regressão Linear
ajuste = lm(EXPENSES ~ .,
            data = treino)

attach(teste)

valoresPrevistos= predict(ajuste, newdata = teste[,-5])

resultado = cbind(valoresPrevistos,teste[,5])

library(ggplot2)
ggplot(resultado,aes(x=resultado[,1], y=resultado[,2])) + 
  geom_point()+
  geom_smooth(method=lm)+
  labs(title="Predito vs. Observado - Regressão Linear",
       x="Preditos", y = "Observados")

library(Metrics)

(risco_estimado = sum((teste[,5]-valoresPrevistos)^2)/nrow(teste))
(risco_estimado2 = mse(resultado[,2],resultado[,1])) #mse(actual, predicted)

# -----------------------------------------
#  Modelos Paramétricos em Dimensões Altas
# -----------------------------------------
rm(list=ls())

# Exemplo 1: Residential Building Data Set

library(readxl)
Residential_Building_Data_Set <- read_excel("C:/Users/icsa/Desktop/Minicurso - Modelagem Preditiva - III ENEAT/Dados/Residential-Building-Data-Set.xlsx")
Residential_Building_Data_Set <- na.omit(Residential_Building_Data_Set)

# Attribute Information: Totally 105: 8 project physical and financial variables, 
# 19 economic variables and indices in 5 time lag numbers (5*19 = 95), and two output 
# variables that are construction costs and sale prices

sample <- sample.int(n = nrow(Residential_Building_Data_Set), size = floor(.7*nrow(Residential_Building_Data_Set)), replace = F)
treino <- data.frame(Residential_Building_Data_Set[sample, ])
teste  <- data.frame(Residential_Building_Data_Set[-sample,])

#attach(treino)

# Stepwise via AIC

nulo = lm(treino[,108]~1,data=treino[,-108])
completo = lm(treino[,108]~.,data=treino[,-108])
final = step(nulo, scope=list(lower=nulo, upper=completo),data=treino,
       direction="forward",trace=FALSE)

#Se colocarmos o argumento trace=TRUE, os passos intermediários dos procedimentos serão exibidos. 
# É escolhido de acordo com o AIC

#attach(teste)

valoresPrevistos= predict(final, newdata = teste[,-108])

resultado = as.data.frame(cbind(valoresPrevistos,teste[,108]))

library(ggplot2)
(grafico_stepwise <- ggplot(resultado,aes(x=resultado[,1], y=resultado[,2])) + 
  geom_point()+
  geom_smooth(method=lm)+
  labs(title="Predito vs. Observado - Stepwise",
       x="Preditos", y = "Observados"))

(erro_lm = mse(teste[,108],valoresPrevistos)) #mse(actual, predicted)

# Lasso
treino <- as.matrix(treino)
teste  <- as.matrix(teste)

library(glmnet)
ajuste = glmnet(treino[,-108], treino[,108], alpha=1) # ajusta o modelo
# calcula coeficientes para diferentes lambdas, a escolha do grid e automatica mas pode ser mudada
validacaoCruzada = cv.glmnet(treino[,-108], treino[,108], alpha=1) # validacao cruzada
plot(validacaoCruzada) # plota lambda vs risco estimada
lambdaOtimo = validacaoCruzada$lambda.min # retorna

# melhor lambda

coefficients(ajuste,s = lambdaOtimo) # melhor lambda

plot(ajuste,xvar="lambda",label=TRUE)

preditos = predict(ajuste,newx=teste[,-108],s = lambdaOtimo) # prediz Y para cada linha de xNovo usando melhor lambda

resultado2 = as.data.frame(cbind(preditos,teste[,108]))

library(ggplot2)
(grafico_lasso = ggplot(resultado2,aes(x=resultado2[,1], y=resultado2[,2])) + 
  geom_point()+
  geom_smooth(method=lm)+
  labs(title="Predito vs. Observado - Regressão Lasso",
       x="Preditos", y = "Observados"))

(erro_lasso = mse(teste[,108],preditos)) #mse(actual, predicted)

#A regressão ridge pode ser ajustada usando-se essas mesmas funções, mas usamos alpha=0 ao invés de alpha=1.

# Ridge

ajuste = glmnet(treino[,-108], treino[,108], alpha=0) # ajusta o modelo
# calcula coeficientes para diferentes lambdas, a escolha do grid e automatica mas pode ser mudada
validacaoCruzada = cv.glmnet(treino[,-108], treino[,108], alpha=0) # validacao cruzada
plot(validacaoCruzada) # plota lambda vs risco estimada
lambdaOtimo = validacaoCruzada$lambda.min # retorna

# melhor lambda

coefficients(ajuste,s = lambdaOtimo) # melhor lambda

plot(ajuste,xvar="lambda",label=TRUE)

preditos = predict(ajuste,newx=teste[,-108],s = lambdaOtimo) # prediz Y para cada linha de xNovo usando melhor lambda

resultado3 = as.data.frame(cbind(preditos,teste[,108]))

library(ggplot2)
(grafico_ridge <- ggplot(resultado3,aes(x=resultado3[,1], y=resultado3[,2])) + 
  geom_point()+
  geom_smooth(method=lm)+
  labs(title="Predito vs. Observado - Regressão Ridge",
       x="Preditos", y = "Observados"))

(erro_ridge = mse(teste[,108],preditos)) #mse(actual, predicted)

resultado_final = matrix(c(erro_lm,erro_lasso,erro_ridge),nrow=3)
row.names(resultado_final) <- c("Stepwise","Lasso","Ridge")
colnames(resultado_final) <- c("Risco Estimado")
resultado_final

library(gridExtra)
grid.arrange(grafico_stepwise,grafico_lasso,grafico_ridge,ncol=3)

# Exemplo 2: Cancer de Próstata
rm(list=ls())

library(glmnet)
library(lasso2)
data(Prostate)
dados <- Prostate

# These data come from a study that examined the correlation between the level 
# of prostate specific antigen and a number of clinical measures in men who were 
# about to receive a radical prostatectomy. It is data frame with 97 rows and 9 columns.

# treinamento - Normalizar
dados[,1:8] = apply(dados[,1:8], 2, scale, center = TRUE,
                    scale = TRUE)
dados = dados[,1:9]

# conjunto treinamento e Validação
sample <- sample.int(n = nrow(dados), size = floor(.7*nrow(dados)), replace = F)
treino <- as.data.frame(dados[sample, ])
teste  <- as.data.frame(dados[-sample,])

attach(treino)

# Mínimos Quadrados
ajuste_mq = lm(lpsa ~ ., data = treino[,-9])
predito_mq = predict.lm(ajuste_mq,
                        newdata = as.data.frame(teste[,-9]))

resultado = as.data.frame(cbind(predito_mq,teste[,9]))

library(ggplot2)
(grafico_mq <- ggplot(resultado,aes(x=resultado[,1], y=resultado[,2])) + 
  geom_point()+
  geom_smooth(method=lm)+
  labs(title="Predito vs. Observado - Regressão Linear",
       x="Preditos", y = "Observados"))

(erro_mq = mse(teste[,9],predito_mq)) #mse(actual, predicted)

# Regressão Ridge

treino <- as.matrix(treino)
teste  <- as.matrix(teste)

vc_ridge = cv.glmnet(treino[,-9], treino[,9], alpha = 0)
ajuste_ridge = glmnet(treino[,-9], treino[,9], alpha = 0)

plot(ajuste_ridge,xvar="lambda",label=TRUE)

predito_ridge = predict(ajuste_ridge, s = vc_ridge$lambda.1se,
                        newx = teste[,-9])

resultado2 = as.data.frame(cbind(predito_ridge,teste[,9]))

library(ggplot2)
(grafico_ridge <- ggplot(resultado2,aes(x=resultado2[,1], y=resultado2[,2])) + 
  geom_point()+
  geom_smooth(method=lm)+
  labs(title="Predito vs. Observado - Regressão Ridge",
       x="Preditos", y = "Observados"))

(erro_ridge = mse(teste[,9],predito_ridge)) #mse(actual, predicted)

# Regressão LASSO

vc_lasso = cv.glmnet(treino[,-9], treino[,9], alpha = 1)
ajuste_lasso = glmnet(treino[,-9], treino[,9], alpha = 1)
predito_lasso = predict(ajuste_lasso, s = vc_lasso$lambda.1se,
                        newx = teste[,-9])

plot(ajuste_lasso,xvar="lambda",label=TRUE)

resultado3 = as.data.frame(cbind(predito_lasso,teste[,9]))

library(ggplot2)
(grafico_lasso <- ggplot(resultado3,aes(x=resultado3[,1], y=resultado3[,2])) + 
  geom_point()+
  geom_smooth(method=lm)+
  labs(title="Predito vs. Observado - Regressão Lasso",
       x="Preditos", y = "Observados"))

(erro_lasso = mse(teste[,9],predito_lasso)) #mse(actual, predicted)


resultado_final = matrix(c(erro_mq,erro_lasso,erro_ridge),nrow=3)
row.names(resultado_final) <- c("MQ","Lasso","Ridge")
colnames(resultado_final) <- c("Risco Estimado")
resultado_final

library(gridExtra)
grid.arrange(grafico_mq,grafico_ridge,grafico_lasso,ncol=3)

# Coeficientes

coef(ajuste_mq)
coef(ajuste_ridge,s=vc_ridge$lambda.1se)
coef(ajuste_lasso,s=vc_lasso$lambda.1se)

# Exemplo 3: Amazon Fine Food Reviews
rm(list=ls())

library(data.table)
library(tm)
library(glmnet)
library(readr)
dados <- read_csv("C:/Users/icsa/Desktop/Minicurso - Modelagem Preditiva - III ENEAT/Dados/Reviews.csv")

# dataset are derived from the customers reviews in Amazon Commerce Website for authorship identification. 

# seleciona 70.000 observações
set.seed(1)
selecao = sample(nrow(dados), 70000)
dados = dados[selecao,]

# indica observações de treinamento
tr = sample.int(70000, 50000, replace = F)
corp = VCorpus(VectorSource(dados$Text))
dtm = DocumentTermMatrix(corp,
                         control = list(tolower = TRUE,
                                        stemming = FALSE,
                                        removeNumbers = TRUE,
                                        removePunctuation = TRUE,
                                        removeStripwhitespace = TRUE,
                                        weighting = weightTf,
                                        bounds=list(global=c(50, Inf))))

dtmMatrix = sparseMatrix(i = dtm$i, j = dtm$j, x = dtm$v,
                         dimnames = list(NULL, dtm$dimnames[[2]]),
                         dims = c(dtm$nrow, dtm$ncol))

dim(dtmMatrix)

# Mínimos Quadrados
ajuste_mq = glmnet(dtmMatrix[tr,], dados$Score[tr], alpha = 0,
                   lambda = 0)

predito_mq = predict(ajuste_mq, newx = dtmMatrix[-tr,])

library(Metrics)
(erro_mq = mse(dados$Score[-tr],predito_mq)) #mse(actual, predicted)

# Ridge
vc_ridge = cv.glmnet(dtmMatrix[tr,], dados$Score[tr],
                     alpha = 0)

ajuste_ridge = glmnet(dtmMatrix[tr,], dados$Score[tr],
                      alpha = 0)

predito_ridge = predict(ajuste_ridge, s = vc_ridge$lambda.1se,
                        newx = dtmMatrix[-tr,])

(erro_ridge = mse(dados$Score[-tr],predito_ridge)) #mse(actual, predicted)

# LASSO
vc_lasso = cv.glmnet(dtmMatrix[tr,], dados$Score[tr],
                     alpha = 1)

ajuste_lasso = glmnet(dtmMatrix[tr,], dados$Score[tr],
                      alpha = 1)

predito_lasso = predict(ajuste_lasso, s = vc_lasso$lambda.1se,
                        newx = dtmMatrix[-tr,])

(erro_lasso = mse(dados$Score[-tr],predito_lasso)) #mse(actual, predicted)

resultado_final = matrix(c(erro_mq,erro_lasso,erro_ridge),nrow=3)
row.names(resultado_final) <- c("MQ","Lasso","Ridge")
colnames(resultado_final) <- c("Risco Estimado")
resultado_final

# Importância das palavras nas notas

coef(ajuste_ridge,s=vc_ridge$lambda.1se) # Ridge

# Lasso

df = coef(ajuste_lasso,s=vc_lasso$lambda.1se)  # Coeficientes do lasso
negativos = sort(df[order(df,decreasing=TRUE),])[1:(length(df)/100)] #1% das palavras que impactam negativamente na avaliação do cliente
positivos = sort(df[order(df,decreasing=TRUE),],decreasing=TRUE)[1:(length(df)/100)] #1% das palavras que impactam positivamente na avaliação do cliente

negativos = as.data.frame(cbind(names(negativos),as.numeric(negativos)))
positivos = as.data.frame(cbind(names(positivos),as.numeric(positivos)))

# Voltar pro Slide =)

# -------------------------------------
#       Métodos Não Paramétricos
# -------------------------------------
rm(list=ls())

# Exemplo 1: Face data

dados <- read.table("C:/Users/icsa/Desktop/Minicurso - Modelagem Preditiva - III ENEAT/Dados/dadosFacesAltaResolucaob.txt")
  
# Há 698 observações e 4096 covariáveis. As covariáveis contém os pixels relativos a essa imagem
# (cada rosto), que possui dimensão 64 por 64, ou seja, cada covariável corresponde a um píxel de
# uma imagem.

# Utilizando o comando image, plote 5 imagens deste banco.

par(mfrow=c(2,3))

individuo1 <- matrix(t(dados[1,-1]),64,64)
individuo2 <- matrix(t(dados[2,-1]),64,64)
individuo3 <- matrix(t(dados[3,-1]),64,64)
individuo4 <- matrix(t(dados[4,-1]),64,64)
individuo5 <- matrix(t(dados[5,-1]),64,64)

image(individuo1)
image(individuo2)
image(individuo3)
image(individuo4)
image(individuo5)

# conjunto treinamento e Validação

# Divisão dos dados

fracaotreino <- 0.70
fracaovalidacao <- 0.15
fracaoteste <- 0.15

tamanhoamostratreino <- floor(fracaotreino * nrow(dados[-1,]))
tamanhoamostravalidacao <- floor(fracaovalidacao * nrow(dados[-1,]))
tamanhoamostrateste       <- floor(fracaoteste * nrow(dados[-1,]))

indicestreino <- sort(sample(seq_len(nrow(dados[-1,])), size=tamanhoamostratreino))
indicesnaotreino <- setdiff(seq_len(nrow(dados[-1,])), indicestreino)
indicesvalidacao  <- sort(sample(indicesnaotreino, size=tamanhoamostravalidacao))
indicesteste        <- setdiff(indicesnaotreino, indicesvalidacao)

dadostreino   <- dados[indicestreino, ]
ydadostreino <- dadostreino[,1]
dadosvalidacao <- dados[indicesvalidacao, ]
dadosteste       <- dados[indicesteste, ]

# K-nn

library(FNN)

k<-seq(1:nrow(dadosvalidacao))

knn <- function(dadostreino,dadosvalidacao,ydadostreino,k){
  ajustados <- matrix(NA,nrow = nrow(dadosvalidacao),ncol = length(k))
  for(i in 1:length(k)){
    ajuste = knn.reg(train=dadostreino,test=dadosvalidacao,y=ydadostreino,
                     k=i)
    ajustados[,i] = ajuste$pred
  }
  return(ajustados)
}

# Fazendo predição
y.pred.train <- knn(dadostreino[,-1],dadosvalidacao[,-1],dadostreino[,1],k)

# Risco Estimado
mse.train <- apply((y.pred.train - dadosvalidacao[,1])^2 , 2, mean)

# Plotar o Risco Estimado em função de K
plot(mse.train , type='l' , xlab='k' , ylab='MSE', col=1 , lwd=2)

k.melhor = k[which.min(mse.train)]

# Agora uso o conjunto de teste!

ajuste = knn.reg(train=dadostreino[,-1],test=dadosteste[,-1],y=dadostreino[,1],
                 k=4)

predVal=ajuste$pred

resultado = as.data.frame(cbind(predVal,dadosteste[,1]))

library(ggplot2)
(grafico_knn <- ggplot(resultado,aes(x=resultado[,1], y=resultado[,2])) + 
  geom_point()+
  geom_smooth(method=lm)+
  labs(title="Predito vs. Observado - K-nn",
       x="Preditos", y = "Observados"))

library(Metrics)
(erro_knn = mse(dadosteste[,1],predVal)) #mse(actual, predicted)

# Bagging

treino <- rbind(dadostreino,dadosvalidacao)
teste <- dadosteste

library(randomForest)
ajuste = randomForest(x=treino[,-1],
                      y=treino[,1],
                      mtry=ncol(treino)-1,
                      importance = TRUE)

preditos <- predict(ajuste,teste[,-1])

resultado2 = as.data.frame(cbind(preditos,teste[,1]))

library(ggplot2)
(grafico_bagging <- ggplot(resultado2,aes(x=resultado2[,1], y=resultado2[,2])) + 
  geom_point()+
  geom_smooth(method=lm)+
  labs(title="Predito vs. Observado - Bagging",
       x="Preditos", y = "Observados"))

(erro_bagg = mse(teste[,1],preditos)) #mse(actual, predicted)

# Florestas

library(randomForest)
m=sqrt(ncol(treino))
ajuste = randomForest(x=treino[,-1],
                      y=treino[,1],
                      mtry=m,
                      importance = TRUE)


preditos <- predict(ajuste,teste[,-1])

resultado3 = as.data.frame(cbind(preditos,teste[,1]))

library(ggplot2)
(grafico_flo <- ggplot(resultado3,aes(x=resultado3[,1], y=resultado3[,2])) + 
  geom_point()+
  geom_smooth(method=lm)+
  labs(title="Predito vs. Observado - Florestas",
       x="Preditos", y = "Observados"))


(erro_flo = mse(teste[,1],preditos)) #mse(actual, predicted)

resultado_final = matrix(c(erro_knn,erro_bagg,erro_flo),nrow=3)
row.names(resultado_final) <- c("Knn","Bagging","Florestas")
colnames(resultado_final) <- c("Risco Estimado")
resultado_final

library(gridExtra)
grid.arrange(grafico_knn,grafico_bagging,grafico_flo,ncol=3)

# Outros Exemplos:

rm(list=ls())

# Bagging

# Data: an experiment on the cold tolerance of the grass species Echinochloa crus-galli.

library(randomForest)
ajuste = randomForest(x=CO2[,! colnames(CO2) %in% c("uptake")],
                      y=CO2[,"uptake"],
                      mtry=ncol(CO2) -1,
                      importance = TRUE)
varImpPlot(ajuste)


# Florestas

library(randomForest)
data(CO2)
m=sqrt(ncol(CO2))
ajuste = randomForest(x=CO2[,! colnames(CO2) %in% c("uptake")],
                      y=CO2[,"uptake"],
                      mtry=m,
                      importance = TRUE)
varImpPlot(ajuste)

rm(list=ls())

# Árvore

library(rpart)
library(rpart.plot)
data("mtcars")

#data = The data was extracted from the 1974 Motor Trend US magazine, and comprises 
# fuel consumption and 10 aspects of automobile design and performance for 32 
# automobiles (1973-74 models) (mpg	 Miles/(US) gallon)

# Ajustar a árvore:
fit <- rpart(mpg ~ .,method="anova", data=mtcars)
# poda:
melhorCp=fit$cptable[which.min(fit$cptable[,"xerror"]),"CP"]
# cp é uma medida de complexidade da árvore, essencialmente proporcional ao número de folhas presentes. Este código
# escolhe o melhor cp via validação cruzada.

pfit<- prune(fit,
             cp=melhorCp)

# plotar árvore podada
rpart.plot(pfit, type = 4, extra = 1)

# Fim! Até amanhã com modelos de classificação! 
#                       Obrigado.
# ---------------------------------------------------------------
