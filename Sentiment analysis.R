# Install and load required libraries
library(tm)
library(caret)
library(glmnet)
library(dplyr)

# Load data
gs_data <- read.csv("C:/Users/anish/Downloads/AmazonReview - AmazonReview.csv")
gs_data <- gs_data %>% 
  mutate(BinarySentiment = ifelse(Sentiment == 1, 0, 1))

#data set has columns "review" and "Binarysentiment"
colnames(gs_data) <- c("Review", "BinarySentiment")

# Preprocess the data
corpus <- Corpus(VectorSource(gs_data$Review))
corpus <- tm_map(corpus, content_transformer(tolower))
corpus <- tm_map(corpus, removePunctuation)
corpus <- tm_map(corpus, removeNumbers)
corpus <- tm_map(corpus, removeWords, stopwords("en"))
corpus <- tm_map(corpus, stripWhitespace)

# Create a document-term matrix
dtm <- DocumentTermMatrix(corpus)

# Convert the document-term matrix to a data frame
dtm_df <- as.data.frame(as.matrix(dtm))
dtm_df$BinarySentiment <- as.factor(gs_data$BinarySentiment)

# Split the data into training and testing sets
set.seed(123)
train_index <- createDataPartition(dtm_df$BinarySentiment, p = 0.8, list = FALSE)
train_data <- dtm_df[train_index, ]
test_data <- dtm_df[-train_index, ]

# Build a logistic regression model
model <- glmnet(as.matrix(train_data[, -ncol(train_data)]), as.factor(train_data$BinarySentiment), family = "binomial")
model=glm(BinarySentiment~.,data = train_data, family = binomial("logit"))
summary(model)
# Make predictions on the testing set
predictions <- predict(model, newx = as.matrix(test_data[, -ncol(test_data)]), type = "response")
predicted_labels <- ifelse(predictions > 0.5, "Positive", "Negative")

# Evaluate the model
confusion_matrix <- table(predicted_labels, test_data$BinarySentiment)
accuracy <- sum(diag(confusion_matrix)) / sum(confusion_matrix)
print(confusion_matrix)
print(paste("Accuracy:", accuracy))
