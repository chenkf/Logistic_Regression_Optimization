###########################################################################################
## test data 1 - 5000 observations with 8 predictors generated from different distributions
###########################################################################################

# create predictors
set.seed(191)
x1=rpois(5000,5)
x2 = rnorm(5000)
x3 = rnorm(5000)
x4 = rnorm(5000,6)
x5 = rnorm(5000,3)
x6 = rpois(5000,3)
x7 = rbinom(5000,3,0.5)
x8 = rbinom(5000,10, 0.6)

# after simulated - scale X 
x1 = scale(x1, center = T, scale =T)
x2 = scale(x2, center = T, scale =T)
x3 = scale(x3, center = T, scale =T)
x4 = scale(x4, center = T, scale =T)
x5 = scale(x5, center = T, scale =T)
x6 = scale(x6, center = T, scale =T)
x7 = scale(x7, center = T, scale = T)
x8 = scale(x8, center = T, scale = T)

# creat y
z1 = 1 + 3 * x1 + 1 * x2 + 2 * x3 - 0.4 * x4 - 1.2 * x5 + 0.5 * x6 + 2 * x7 + 2 * x8    
pr1 = 1/(1+exp(-z1))         # pass through an inv-logit function
y1 = rbinom(5000,1,pr1)      # bernoulli response variable

#now feed it to glm:
df = data.frame(y=y1, x1=x1, x2=x2, x3 = x3, x4 = x4, x5 = x5, x6 = x6, x = x7, x8 = x8)
logit1 = glm(y1~x1+x2 + x3 + x4 + x5 + x6 + x7 + x8 ,data=df,family="binomial",  maxit = 100)

summary(logit1)

# Deviance Residuals: 
#   Min       1Q   Median       3Q      Max  
# -3.1109  -0.2414   0.0211   0.2831   3.2094  
# 
# Coefficients:
#   Estimate Std. Error z value Pr(>|z|)    
#   (Intercept)  1.11726    0.06097  18.326  < 2e-16 ***
#   x1           3.10394    0.10616  29.238  < 2e-16 ***
#   x2           0.96819    0.06022  16.078  < 2e-16 ***
#   x3           2.13788    0.08122  26.322  < 2e-16 ***
#   x4          -0.36124    0.05367  -6.731 1.69e-11 ***
#   x5          -1.17870    0.06178 -19.079  < 2e-16 ***
#   x6           0.51243    0.05546   9.240  < 2e-16 ***
#   x7           2.09640    0.08137  25.764  < 2e-16 ***
#   x8           1.99802    0.07871  25.385  < 2e-16 ***
#   ---
#   Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
# 
# (Dispersion parameter for binomial family taken to be 1)
# 
# Null deviance: 6817.3  on 4999  degrees of freedom
# Residual deviance: 2384.2  on 4991  degrees of freedom
# AIC: 2402.2
# 
# Number of Fisher Scoring iterations: 7

# confidence interval 
confint(logit1)
#                2.5 %     97.5 %
# (Intercept)  0.9994762  1.2385592
# x1           2.9008393  3.3171491
# x2           0.8518539  1.0879917
# x3           1.9820942  2.3005940
# x4          -0.4671526 -0.2566905
# x5          -1.3017511 -1.0594753
# x6           0.4047062  0.6221842
# x7           1.9403513  2.2594439
# x8           1.8469753  2.1556230

# s.e.
sqrt(diag(vcov(logit1))) 
# (Intercept)          x1          x2          x3          x4          x5          x6          x7          x8 
# 0.06096755  0.10616145  0.06021692  0.08121961  0.05366943  0.06178139  0.05545961  0.08136958  0.07870742

### add an intercept tot C++ code ####
incept = rep(1,5000)
df = cbind(df[,1],incept, df[,2:9])

# export data
write.table(na.omit(df), "df.txt", row.names = F, col.names = F,sep="\t")


########################################################################
## data 2 - 1000 observations with 4 predictors from normal distribution
########################################################################

set.seed(66)
dt2_x1 = rnorm(1000)            
dt2_x2 = rnorm(1000)
dt2_x3 = rnorm(1000)

dt2_x1 = scale(dt2_x1, center = T, scale = T)
dt2_x2 = scale(dt2_x2, center = T, scale = T)
dt2_x3 = scale(dt2_x3, center = T, scale = T)

z2 = 1 + 2*dt2_x1 + 3*dt2_x2 + 4*dt2_x3        
pr2 = 1/(1+exp(-z2))         # pass through an inv-logit function
y2 = rbinom(1000,1,pr2)      # bernoulli response variable

#now feed it to glm:
df2 = data.frame(y=y2,x1=dt2_x1,x2=dt2_x2, x3 = dt2_x3)
summary(df2)
logit2 = glm( y~x1+x2+x3,data=df2,family="binomial") 
summary(logit2)

# Deviance Residuals:
#   Min        1Q    Median        3Q       Max
# -2.40463  -0.22207   0.01732   0.25753   2.78393
# 
# Coefficients:
#   Estimate Std. Error z value Pr(>|z|)
# (Intercept)   1.0707     0.1365   7.843 4.41e-15 ***
#   x1            1.9699     0.1830  10.762  < 2e-16 ***
#   x2            2.8217     0.2271  12.427  < 2e-16 ***
#   x3            3.8607     0.2856  13.519  < 2e-16 ***
#   ---
#   Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
# 
# (Dispersion parameter for binomial family taken to be 1)
# 
# Null deviance: 1364.90  on 999  degrees of freedom
# Residual deviance:  467.24  on 996  degrees of freedom
# AIC: 475.24
# 
# Number of Fisher Scoring iterations: 7

confint(logit2)
#             2.5 %   97.5 %
# (Intercept) 0.8006839 1.340071
# x1          1.5960485 2.315402
# x2          2.3877756 3.286182
# x3          3.4545899 4.619564

# s.e.
sqrt(diag(vcov(logit2))) 
# (Intercept)          x1          x2          x3 
# 0.1373236   0.1831515   0.2287150   0.2965097

## add an intercept tot C++ code ####
incept2 = rep(1,1000)
df2 = cbind(df2[,1],incept2, df2[,2:4])

# export data
write.table(na.omit(df2), "df2.txt", row.names = F, col.names = F,sep="\t")

#################################################################################
## test data 3 - 100,000 observations with 3 predictors from normal distribution
#################################################################################

set.seed(666)
x1_dt3 = rnorm(100000)           
x2_dt3 = rnorm(100000)
x3_dt3 = rnorm(100000)
x4_dt3 = rnorm(100000)

x1_dt3 = scale(x1_dt3, center = T, scale = T)
x2_dt3 = scale(x2_dt3, center = T, scale = T)
x3_dt3 = scale(x3_dt3, center = T, scale = T)
x4_dt3 = scale(x4_dt3, center = T, scale = T)


z3 = 1 + 5*x1_dt3 + 1*x2_dt3 + 4*x3_dt3 - 0.5 * x4_dt3      
pr3 = 1/(1+exp(-z3))         # pass through an inv-logit function
y3 = rbinom(100000,1,pr3)      # bernoulli response variable

#now feed it to glm:
df3 = data.frame(y=y3,x1=x1_dt3,x2=x2_dt3, x3 = x3_dt3, x4 = x4_dt3)
logit3 = glm( y~x1 + x2 + x3 + x4,data=df3,family="binomial")
summary(logit3)

# Deviance Residuals: 
#   Min       1Q   Median       3Q      Max  
# -3.9210  -0.1316   0.0055   0.1708   4.5703  
# 
# Coefficients:
#   Estimate Std. Error z value Pr(>|z|)    
#  (Intercept)  0.99726    0.01485   67.15   <2e-16 ***
#   x1           5.01089    0.03918  127.90   <2e-16 ***
#   x2           1.00955    0.01504   67.12   <2e-16 ***
#   x3           4.00563    0.03224  124.25   <2e-16 ***
#   x4          -0.50238    0.01362  -36.89   <2e-16 ***
#   ---
#   Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
# 
# (Dispersion parameter for binomial family taken to be 1)
# 
# Null deviance: 137167  on 99999  degrees of freedom
# Residual deviance:  37838  on 99995  degrees of freedom
# AIC: 37848
# 
# Number of Fisher Scoring iterations: 8

confint(logit3)
#              2.5 %     97.5 %
# (Intercept)  0.9682472  1.0264612
# x1           4.9346139  5.0881962
# x2           0.9801736  1.0391340
# x3           3.9428456  4.0692239
# x4          -0.5291173 -0.4757383

sqrt(diag(vcov(logit3)))
# (Intercept)          x1          x2          x3          x4 
# 0.01485041  0.03917889  0.01504081  0.03223915  0.01361700 

## add intercept for the data to Kaifeng
incpt3 = rep(1,100000)
df3 = cbind(df3[,1],incpt3, df3[,2:5])

#export data 
write.table(na.omit(df3), "df3.txt", row.names = F, col.names = F,sep="\t")


###########################################################################
### test data 4: data 1,000,000 with 4 predictors from normal distributions
###########################################################################

set.seed(666)
x1_dt4 = rnorm(1000000)           
x2_dt4 = rnorm(1000000)
x3_dt4 = rnorm(1000000)
x4_dt4 = rnorm(1000000)
x5_dt4 = rnorm(1000000)

x1_dt4 = scale(x1_dt4, center = T, scale = T)
x2_dt4 = scale(x2_dt4, center = T, scale = T)
x3_dt4 = scale(x3_dt4, center = T, scale = T)
x4_dt4 = scale(x4_dt4, center = T, scale = T)
x5_dt4 = scale(x5_dt4, center = T, scale = T)

# it looks like big beta will make it has perfect separation 
z4 = 1 + 3*x1_dt4 + 1*x2_dt4 + 2*x3_dt4 + 1.5 * x4_dt4         
pr4 = 1/(1+exp(-z4))         # pass through an inv-logit function
y4 = rbinom(1000000,1,pr4)      # bernoulli response variable

df4 = data.frame(y=y4,x1=x1_dt4,x2=x2_dt4, x3 = x3_dt4, x4 = x4_dt4)
logit4 = glm( y~x1 + x2 + x3 + x4,data=df4,family="binomial")
summary(logit4)

# Deviance Residuals: 
#   Min       1Q   Median       3Q      Max  
# -4.1777  -0.3164   0.0569   0.3763   4.2167  
# 
# Coefficients:
#   Estimate Std. Error z value Pr(>|z|)    
# (Intercept) 1.000231   0.003811   262.5   <2e-16 ***
#   x1          3.005914   0.006727   446.9   <2e-16 ***
#   x2          0.998923   0.003890   256.8   <2e-16 ***
#   x3          2.002737   0.005140   389.6   <2e-16 ***
#   x4          1.499239   0.004451   336.8   <2e-16 ***
#   ---
#   Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
# 
# (Dispersion parameter for binomial family taken to be 1)
# 
# Null deviance: 1353966  on 999999  degrees of freedom
# Residual deviance:  562936  on 999995  degrees of freedom
# AIC: 562946
# 
# Number of Fisher Scoring iterations: 7

confint(logit4)
#             2.5 %   97.5 %
# (Intercept) 0.9927675 1.007707
# x1          2.9927514 3.019119
# x2          0.9913057 1.006554
# x3          1.9926763 2.012826
# x4          1.4905255 1.507974

sqrt(diag(vcov(logit4)))
# (Intercept)          x1          x2          x3          x4 
# 0.003811107 0.006726501 0.003889855 0.005140413 0.004451333

### add an intercept 
incpt4 = rep(1, nrow(df4))
df4 = cbind(df4[,1], incpt4, df4[,2:5])

# export data
write.table(na.omit(df4), "df4.txt", row.names = F, col.names = F,sep="\t")


############################################
## test data 5: wine data - real data 
############################################

# read in data - attached in the submitted files 
wine <- read.delim("wine.txt", header=FALSE)

wine_logit = glm(wine$V1 ~. - wine$V2, data = wine[,-2], family = "binomial")
summary(wine_logit)

# Call:
#   glm(formula = wine7$V1 ~ . - wine7$V2, family = "binomial", data = wine7[, 
#                                                                            -2])
# 
# Deviance Residuals: 
#   Min       1Q   Median       3Q      Max  
# -2.0061  -0.7663  -0.1988   0.7041   2.3699  
# 
# Coefficients:
#   Estimate Std. Error z value Pr(>|z|)    
# (Intercept)  -0.3536     0.2347  -1.507    0.132    
# V3            0.2711     0.2524   1.074    0.283    
# V4           -1.6428     0.3447  -4.766 1.88e-06 ***
# V5            0.2500     0.2485   1.006    0.314    
# V6            1.2599     0.2998   4.203 2.64e-05 ***
#   ---
#   Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
# 
# (Dispersion parameter for binomial family taken to be 1)
# 
# Null deviance: 179.11  on 129  degrees of freedom
# Residual deviance: 116.65  on 125  degrees of freedom
# AIC: 126.65
# 
# Number of Fisher Scoring iterations: 5

confint(wine_logit)
#             2.5 %      97.5 %
# (Intercept) -0.8286851  0.09822462
# V3          -0.2261055  0.77779950
# V4          -2.3925040 -1.03211952
# V5          -0.2418919  0.74407281
# V6           0.7170417  1.90176979

# export data 
write.table(na.omit(wine), "wine.txt", row.names = F, col.names = F,sep="\t")
