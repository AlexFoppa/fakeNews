getwd()
setwd("C:\\Users\\alfop\\Desktop\\Complemento\\R Advanced")

fin <- read.csv("P3-Future-500-The-Dataset.csv", na.strings = c(""))
head(fin)
str(fin)
summary(fin)

#factoriza??o
fin$ID <- factor(fin$ID)
fin$Inception <- factor(fin$Inception)

#sub e gsub

fin$Expenses <- gsub(" Dollars","", fin$Expenses)
fin$Expenses <- gsub(",","", fin$Expenses)
fin$Expenses <- gsub("\\$","", fin$Expenses)

fin$Revenue <- gsub(" Dollars","", fin$Revenue)
fin$Revenue <- gsub(",","", fin$Revenue)
fin$Revenue <- gsub("\\$","", fin$Revenue)

fin$Growth <- gsub("\\%","", fin$Growth)

fin$Revenue <- as.numeric(fin$Revenue)
fin$Growth <- as.numeric(fin$Growth)
fin$Expenses <- as.numeric(fin$Expenses)

head(fin, 24)
fin[!complete.cases(fin),]

fin[which(fin$Revenue == 9746272),]
fin[is.na(fin$Expenses),]

fin_backup <- fin

#remove missing data

fin <- fin[!is.na(fin$Industry),]
rownames(fin) <- 1:nrow(fin)
rownames(fin) <- NULL

#factual analysis

fin[is.na(fin$State) & fin$City=="New York","State"]<-"NY"
fin[is.na(fin$State) & fin$City=="San Francisco","State"]<-"CA"

#Median inputation

median_empl_retail <- median(fin[fin$Industry =="Retail","Employees"], na.rm = TRUE)
fin[is.na(fin$Employees) & fin$Industry=="Retail","Employees"] <- median_empl_retail

median_finan_retail <- median(fin[fin$Industry =="Financial Services","Employees"], na.rm = TRUE)
fin[is.na(fin$Employees) & fin$Industry=="Financial Services","Employees"] <- median_finan_retail


median_growth_constr <- median(fin[fin$Industry =="Construction","Growth"], na.rm = TRUE)
fin[is.na(fin$Growth) & fin$Industry=="Construction","Growth"] <- median_growth_constr

median_rev_constr <- median(fin[fin$Industry =="Construction","Revenue"], na.rm = TRUE)
fin[is.na(fin$Revenue) & fin$Industry=="Construction","Revenue"] <- median_rev_constr

median_expens_constr <- median(fin[fin$Industry =="Construction","Expenses"], na.rm = TRUE)
fin[is.na(fin$Expenses) & fin$Industry=="Construction","Expenses"] <- median_expens_constr

fin[is.na(fin$Profit), "Profit"] <- fin[is.na(fin$Profit), "Revenue"] - fin[is.na(fin$Profit), "Expenses"] 

fin[is.na(fin$Expenses), "Expenses"] <- fin[is.na(fin$Expenses), "Revenue"] - fin[is.na(fin$Expenses), "Profit"]

p<- ggplot(data = fin)
p + geom_point(aes(x=Revenue, y = Expenses,
                   colour = Industry, size = Profit))


d <- ggplot(data = fin, aes(x=Revenue, y = Expenses,
                        colour = Industry))
d + geom_point() +
    geom_smooth(fill = NA, size = 1.2)
  

f <- ggplot(data = fin, aes(x=Industry, y = Growth,
                            colour = Industry))
f + geom_jitter() + 
  geom_boxplot(size = 1, alpha = 0.5, outlier.color = NA)
