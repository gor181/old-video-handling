---
title_meta  : Chapter 1
title       : What is Machine Learning
description : In this first chapter, you get your first intro to machine learning. After learning the true fundamentals of machine learning, you'll experiment with the techniques that are explained in more detail in future chapters.
attachments :
  slides_link: https://s3.amazonaws.com/assets.datacamp.com/course/intro_to_ml/slides/ch1_slides_new.pdf
free_preview: TRUE

--- type:VideoExercise xp:50 key:bcc46f3ba5
## Machine Learning: What's the challenge?

*** =video_link
```{r,eval=FALSE}
//player.vimeo.com/video/163565150
```

*** =video_stream
```{r,eval=FALSE}
https://player.vimeo.com/external/163565150.hd.mp4?s=7f21c9bf7b28967cfa8afb132999fa83788d26cc&profile_id=119
```

*** =video_hls
//videos.datacamp.com/transcoded/682_intro_to_ml/v2/hls-ch1_1.master.m3u8

*** =projector_key
3afe8f444733273e81bcff04474de367

*** =skills
6

--- type:NormalExercise xp:100 key:6da2a1b0c7
## Acquainting yourself with the data

As a first step, you want to find out some properties of the dataset with which you'll be working. More specifically, you want to know more about the dataset's number of observations and variables.

In this exercise, you'll explore the [`iris`](http://www.rdocumentation.org/packages/datasets/functions/iris) dataset. If you want to learn more about it, you can click on it or type `?iris` in the console.

Your job is to extract the number of observations and variables from [`iris`](http://www.rdocumentation.org/packages/datasets/functions/iris). This dataset is readily available in R (in the [`datasets`](http://www.rdocumentation.org/packages/datasets) package that's loaded by default).

*** =instructions
- Use the two ways presented in the video to find out the number of observations and variables of the [`iris`](http://www.rdocumentation.org/packages/datasets/functions/iris) data set: [`str()`](http://www.rdocumentation.org/packages/utils/functions/str) and [`dim()`](http://www.rdocumentation.org/packages/base/functions/dim). Can you interpret the results?
- Call [`head()`](http://www.rdocumentation.org/packages/utils/functions/head) and [`tail()`](http://www.rdocumentation.org/packages/utils/functions/tail) on [`iris`](http://www.rdocumentation.org/packages/datasets/functions/iris) to reveal the first and last observations in the [`iris`](http://www.rdocumentation.org/packages/datasets/functions/iris) dataset.
- Finally, call the [`summary()`](http://www.rdocumentation.org/packages/base/functions/summary) function to generate a summary of the dataset. What does the printout tell you?

*** =hint
- The [`str()`](http://www.rdocumentation.org/packages/utils/functions/str) function gives you an overview of the different variables of the data.
- The [`dim()`](http://www.rdocumentation.org/packages/base/functions/dim) function tells you the number of observations and variables respectively.
- The [`summary()`](http://www.rdocumentation.org/packages/base/functions/summary) function returns several measures for each variable. Such as the maximum observed value, the mean and many more!

*** =pre_exercise_code
```{r}
# no pec
```

*** =sample_code
```{r}
# iris is available from the datasets package

# Reveal number of observations and variables in two different ways



# Show first and last observations in the iris data set



# Summarize the iris data set

```

*** =solution
```{r}
# iris is available from the datasets package

# Reveal number of observations and variables in two different ways
str(iris)
dim(iris)

# Show first and last observations in the iris data set
head(iris)
tail(iris)

# Summarize the iris data set
summary(iris)
```

*** =sct
```{r}
test_function("str", "object",
              not_called_msg = "You forgot to call the function <code>str()</code> with argument <code>iris</code>.",
              incorrect_msg = "Did you set the argument in <code>str()</code> to <code>iris</code>?")
test_function("dim", "x",
              not_called_msg = "You forgot to call the function <code>dim()</code> with argument <code>iris</code>.",
              incorrect_msg = "Did you set the argument in <code>dim()</code> to <code>iris</code>?")
test_function("head", "x",
              not_called_msg = "You forgot to call the function <code>head()</code> with argument <code>iris</code>.",
              incorrect_msg = "Did you set the argument in <code>head()</code> to <code>iris</code>?")
test_function("tail", "x",
              not_called_msg = "You forgot to call the function <code>tail()</code> with argument <code>iris</code>.",
              incorrect_msg = "Did you set the argument in <code>tail()</code> to <code>iris</code>?")
test_function("summary", "object",
              not_called_msg = "You forgot to call the function <code>summary()</code> with argument <code>iris</code>.",
              incorrect_msg = "Did you set the argument in <code>summary()</code> to <code>iris</code>?")
test_error()
success_msg("Fantastic! The functions that you've used here are very useful and important to get to know your data before actually getting your hands dirty with some machine learning techniques. Head over to the next exercise.")
```

*** =skills
1,6

--- type:PureMultipleChoiceExercise xp:50 key:98cc2b12f5
## What is, what isn't?

Part of excelling at machine learning is knowing when you're dealing with a machine learning problem in the first place. Machine learning is more than simply computing averages or performing some data manipulation. It actually involves making predictions about observations based on previous information.

Which of the following statements uses a machine learning model?

(1) Determine whether an incoming email is spam or not.
(2) Obtain the name of last year's Giro d'Italia champion.
(3) Automatically tagging your new Facebook photos.
(4) Select the student with the highest grade on a statistics course.

*** =possible_answers
- (1) and (2)
- (3) and (4)
- [(1) and (3)]
- (2) and (4)

*** =hint
Remember that machine learning requires predicting or estimating some variable from existing observations.


*** =feedback

- Incorrect, try again. Although spam filtering is clearly a Machine Learning problem, obtaining the name of last year's Giro champion can be simply extracted by sorting a list of competitors. Try one more time!
- Darn, you almost got it. While photo recognition is a common example of a classification problem, identifying the top scoring student can be solved by sorting a list of scores. Try again!
- Great job! Both spam detection and photo recognition are common examples of classification. Go on to the next exercise to learn more!
- Incorrect. Neither selecting last year's Giro champion nor the top scoring student are machine learning problems. In these cases, all you need is to sort your data to get an answer!


*** =skills
6

--- type:PureMultipleChoiceExercise xp:50 key:2c27bd069e
## What is, what isn't? (2)

Not sure whether you got the difference between basic data manipulation and machine learning? Have a look at the statements below and identify the one which is __not__ a machine learning problem.

*** =possible_answers
- Given a viewer's shopping habits, recommend a product to purchase the next time she visits your website.
- Given the symptoms of a patient, identify her illness.
- Predict the USD/EUR exchange rate for February 2016.
- [Compute the mean wage of 10 employees for your company.]

*** =hint
Calculating an average is a form of basic data manipulation.


*** =feedback

- Incorrect, try again. Recommender systems are typical machine learning implementations to anticipate users preferences.
- Incorrect, try again. Identifying illnesses out of symptoms can be tackled as a classification problem.
- Incorrect, try again. Forecasting is a typical regression problem: You take historical data and try to predict future values of that type of data.
- Good Job! Computing a mean is not machine learning!


*** =skills
1,6

--- type:NormalExercise xp:100 key:b43373217c
## Basic prediction model

Let's get down to a bit of coding! Your task is to examine this course's first prediction model. You'll be working with the [`Wage`](http://www.rdocumentation.org/packages/ISLR/functions/Wage) dataset. It contains the wage and some general information for workers in the mid-Atlantic region of the US.

As we briefly discussed in the video, there could be a relationship between a worker's `age` and his `wage`. Older workers tend to have more experience on average than their younger counterparts, hence you could expect an increasing trend in wage as workers age. So we built a linear regression model for you, using [`lm()`](http://www.rdocumentation.org/packages/stats/functions/lm): `lm_wage`. This model predicts the wage of a worker based only on the worker's age.

With this linear model `lm_wage`, which is built with data that contain information on workers' age and their corresponding wage, you can predict the wage of a worker given the age of that worker. For example, suppose you want to predict the wage of a 60 year old worker. You can use the [`predict()`](http://www.rdocumentation.org/packages/stats/functions/predict) function for this. This generic function takes a model as the first argument. The second argument should be some unseen observations as a data frame. [`predict()`](http://www.rdocumentation.org/packages/stats/functions/predict) is then able to predict outcomes for these observations.

**Note:** At this point, the workings of [`lm()`](http://www.rdocumentation.org/packages/stats/functions/lm) are not important, you'll get a more comprehensive overview of regression in chapter 4.

*** =instructions
- Take a look at the code that builds `lm_wage`, which models the `wage` by the `age` variable.
- See how the data frame `unseen` is created with a single column, `age`, containing a single value, 60.
- Predict the average wage at age 60 using [`predict()`](http://www.rdocumentation.org/packages/stats/functions/predict): you have to pass the arguments `lm_wage` and `unseen`. Print the result of your function call to the console (don't assign it to a variable). Can you interpret the result?

*** =hint
- The [`lm()`](http://www.rdocumentation.org/packages/stats/functions/lm) receives a formula such as `y ~ x` as an argument. Here, `y` is the variable that you try to model using the `x` variable. More theoretically, `y` is the response variable and `x` is the predictor variable.
- The [`predict()`](http://www.rdocumentation.org/packages/stats/functions/predict) function requires a model object, such as a linear model, as the first argument. The second argument should be a dataset to which you'd like to apply your model.

*** =pre_exercise_code
```{r, eval=FALSE}
library(ISLR)
```

*** =sample_code
```{r}
# The Wage dataset is available

# Build Linear Model: lm_wage (coded already)
lm_wage <- lm(wage ~ age, data = Wage)

# Define data.frame: unseen (coded already)
unseen <- data.frame(age = 60)

# Predict the wage for a 60-year old worker

```

*** =solution
```{r}
# The Wage dataset is available

# Build Linear Model: lm_wage (coded already)
lm_wage <- lm(wage ~ age, data = Wage)

# Define data.frame: unseen (coded already)
unseen <- data.frame(age = 60)

# Predict the wage for a 60-year old worker
predict(lm_wage, unseen)
```

*** =sct
```{r, eval=FALSE}
test_data_frame("lm_wage", columns = "coefficients", incorrect_msg = "code>lm_wage</code> was already defined for you. Do not change or remove it.")

msg <- "<code>unseen</code> was already defined for you. Do not change or remove it."
test_object("unseen", undefined_msg = msg, incorrect_msg = msg)

test_function("predict", "object", eval = FALSE,
              incorrect_msg = "Make sure to pass <code>lm_wage</code> as the first argument to <code>predict()</code>.")
test_output_contains("predict(lm_wage, unseen)",
                     incorrect_msg = "Have another look at the <code>predict()</code> function. The first argument is <code>lm_wage</code>, the second one is <code>unseen</code>.")

test_error()
success_msg("Well done! Based on the linear model that was estimated from the <code>Wage</code> dataset, you predicted the average wage for a 60 year old worker to be around 124 USD a day. That's not bad! Head over to the next video to receive a short introduction on several machine learning techniques!")
```

*** =skills
1,6

--- type:VideoExercise xp:50 key:fd29eec2d2
## Classification, Regression, Clustering

*** =video_link
```{r,eval=FALSE}
//player.vimeo.com/video/163565149
```

*** =video_stream
```{r,eval=FALSE}
https://player.vimeo.com/external/163565149.hd.mp4?s=2c61a199cba26c645c066f743400a20ad5b6b61b&profile_id=119
```

*** =video_hls
//videos.datacamp.com/transcoded/682_intro_to_ml/v1/hls-ch1_2.master.m3u8

*** =projector_key
03e0ceccde7e443f09ea1284e3d7ce9f

*** =skills
6

--- type:PureMultipleChoiceExercise xp:50 key:47bc012c89
## Classification, regression or clustering?

As you saw in the video, you can solve quite a few problems using classification, regression or clustering. Which of the following questions can be answered using a classification algorithm?

*** =possible_answers
- How does the exchange rate depend on the [GDP](https://en.wikipedia.org/wiki/Gross_domestic_product)?
- [Does a document contain the handwritten letter S?]
- How can I group supermarket products using purchase frequency?

*** =hint
Classification requires tagging observations into pre-defined labels. Most classification problems aim to predict whether an item belongs or doesn't belong to a given pre-specified category.


*** =feedback

- Incorrect, try again. Dependence of exchange rate on GDP looks more like a time series analysis, the trend line is often found using Regression.
- Fantastic Job! Identifying whether a letter is or isn't present in a handwritten text is a perfect example of a classification problem. You know in advance there are two possible categories: Either a scribble is the Letter S or it is not! It's time to tackle a - on classification problem: Spam Filtering.
- Nope, try again. Clustering is more suitable here, to identify potential categories out of a group of observations, without knowing what the expected categories are in advance.



*** =skills
1,6

--- type:NormalExercise xp:100 key:6b0c4851ed
## Classification: Filtering spam

Filtering spam from relevant emails is a typical machine learning task. Information such as word frequency, character frequency and the amount of capital letters can indicate whether an email is spam or not.

In the following exercise you'll work with the dataset `emails`, which is loaded in your workspace (Source: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Spambase)). Here, several emails have been labeled by humans as spam (1) or not spam (0) and the results are found in the column `spam`. The considered feature in `emails` to predict whether it was spam or not is `avg_capital_seq`. It is the average amount of sequential capital letters found in each email.

In the code, you'll find a crude spam filter we built for you, `spam_classifier()` that uses `avg_capital_seq` to predict whether an email is spam or not. In the function definition, it's important to realize that `x` refers to `avg_capital_seq`. So where the `avg_capital_seq` is greater than 4, `spam_classifier()` predicts the email is spam (1), if `avg_capital_seq` is inclusively between 3 and 4, it predicts not spam (0), and so on. This classifier's methodology of predicting whether an email is spam or not seems pretty random, but let's see how it does anyways!

Your job is to inspect the `emails` dataset, apply `spam_classifier` to it, and compare the predicted labels with the true labels. If you want to play some more with the `emails` dataset, you can download it [here](http://s3.amazonaws.com/assets.datacamp.com/course/intro_to_ml/emails_small.csv). And if you want to learn more about writing functions, consider taking the [Writing Functions in R course](https://www.datacamp.com/courses/writing-functions-in-r) taught by Hadley and Charlotte Wickham.

*** =instructions
- Check the dimensions of this dataset. Use [`dim()`](http://www.rdocumentation.org/packages/base/functions/dim).
- Inspect the definition of `spam_classifier()`. It's a simple set of statements that decide between spam and no spam based on a single input vector.
- Pass the `avg_capital_seq` column of `emails` to `spam_classifier()` to determine which emails are spam and which aren't. Assign the resulting outcomes to `spam_pred`.
- Compare the vector with your predictions, `spam_pred`, to the true spam labels in `emails$spam` with the `==` operator. Simply print out the result. This can be done in one line of code! How many of the emails were correctly classified?

*** =hint
- Use `spam_classifier(emails$avg_capital_seq)` to make predictions using the classifier. Assign the result of this call to `spam_pred`.
- The logical operator `==` tests for each pair of elements whether they are equal, `TRUE` means the pair is equal. It returns a logical vector.

*** =pre_exercise_code
```{r, eval=FALSE }
emails <- read.csv(url("http://s3.amazonaws.com/assets.datacamp.com/course/intro_to_ml/emails_small.csv"))
emails_original <- emails
```

*** =sample_code
```{r, eval=FALSE}
# The emails dataset is already loaded into your workspace

# Show the dimensions of emails


# Inspect definition of spam_classifier()
spam_classifier <- function(x){
  prediction <- rep(NA, length(x)) # initialize prediction vector
  prediction[x > 4] <- 1
  prediction[x >= 3 & x <= 4] <- 0
  prediction[x >= 2.2 & x < 3] <- 1
  prediction[x >= 1.4 & x < 2.2] <- 0
  prediction[x > 1.25 & x < 1.4] <- 1
  prediction[x <= 1.25] <- 0
  return(prediction) # prediction is either 0 or 1
}

# Apply the classifier to the avg_capital_seq column: spam_pred


# Compare spam_pred to emails$spam. Use ==

```

*** =solution
```{r, eval=FALSE}
# The emails dataset is already loaded into your workspace

# Show the dimensions of emails
dim(emails)

# Inspect definition of spam_classifier()
spam_classifier <- function(x){
  prediction <- rep(NA, length(x)) # initialize prediction vector
  prediction[x > 4] <- 1
  prediction[x >= 3 & x <= 4] <- 0
  prediction[x >= 2.2 & x < 3] <- 1
  prediction[x >= 1.4 & x < 2.2] <- 0
  prediction[x > 1.25 & x < 1.4] <- 1
  prediction[x <= 1.25] <- 0
  return(prediction) # prediction is either 0 or 1
}

# Apply the classifier to the avg_capital_seq column: spam_pred
spam_pred <- spam_classifier(emails$avg_capital_seq)

# Compare spam_pred to emails$spam. Use ==
spam_pred == emails$spam
```

*** =sct
```{r, eval=FALSE}
test_error()

test_output_contains("dim(emails_original)", incorrect_msg = "Are you sure you used <code>dim()</code> correctly to show the dimensions of <code>emails</code>?")

msg <- "Do not change or remove the definition of the <code>spam_classifier()</code> function."
test_object("spam_classifier", undefined_msg = msg, incorrect_msg = msg)
test_object("spam_pred",
            undefined_msg = "Make sure to assign the results of the classification to <code>spam_pred</code>.",
            incorrect_msg = "Have you performed the classification correctly using <code>spam_classifier()</code> and assigned the result to <code>spam_pred</code>? Pass the <code>avg_capital_seq</code> column of <code>emails</code> to <code>spam_classifier()</code>.")
test_output_contains("spam_pred == emails$spam", incorrect_msg = "Did you compare <code>spam_pred</code> to <code>emails$spam</code>, using the logical operator <code>==</code>?")

success_msg("Good job! It looks like `spam_classifier()` correctly filtered the spam 13 out of 13 times! Sadly, the classifier we gave you was made to perfectly filter all 13 examples. **If you were to use it on a new set of emails, the results would be far less satisfying**. In chapter 3, you'll learn more about techniques to classify the data, but without cheating!")
```

*** =skills
1,6

--- type:NormalExercise xp:100 key:913db6f7ce
## Regression: LinkedIn views for the next 3 days

It's time for you to make another prediction with regression! More precisely, you'll analyze the number of views of your LinkedIn profile. With your growing network and your data science skills improving daily, you wonder if you can predict how often your profile will be visited in the future based on the number of days it's been since you created your LinkedIn account.

The instructions will help you predict the number of profile views for the next 3 days, based on the views for the past 3 weeks. The `linkedin` vector, which contains this information, is already available in your workspace.

*** =instructions
- Create a vector `days` with the numbers from 1 to 21, which represent the previous 3 weeks of your `linkedin` views. You can use the [`seq()`](http://www.rdocumentation.org/packages/base/functions/seq) function, or simply `:`.
- Fit a linear model that explains the LinkedIn views. Use the [`lm()`](http://www.rdocumentation.org/packages/stats/functions/lm) function such that `linkedin` ( number of views) is a function of `days` (number of days since you made your account). As an example, `lm(y ~ x)` builds a linear model such that `y` is a function of `x`, or more colloquially, `y` is based on `x`. Assign the resulting linear model to `linkedin_lm`.
- Using this linear model, predict the number of views for the next three days (days 22, 23 and 24). Use [`predict()`](http://www.rdocumentation.org/packages/stats/functions/predict) and the predefined `future_days` data frame. Assign the result to `linkedin_pred`.
- See how the remaining code plots both the historical data and the predictions. Try to interpret the result.

*** =hint
- To build a vector with the integers 1 to 21, you can use `1:21`. You can also use `seq(length(linkedin))`.
- Remember that [`predict()`](http://www.rdocumentation.org/packages/stats/functions/predict) in this case receives a linear model and the `data.frame` of the future days, which is given.

*** =pre_exercise_code
```{r, eval=FALSE}
linkedin <- c(5, 7, 4, 9, 11, 10, 14,
              17, 13, 11, 18, 17, 21, 21,
              24, 23, 28, 35, 21, 27, 23)
```

*** =sample_code
```{r}
# linkedin is already available in your workspace

# Create the days vector


# Fit a linear model called on the linkedin views per day: linkedin_lm


# Predict the number of views for the next three days: linkedin_pred
future_days <- data.frame(days = 22:24)
linkedin_pred <-

# Plot historical data and predictions
plot(linkedin ~ days, xlim = c(1, 24))
points(22:24, linkedin_pred, col = "green")
```

*** =solution
```{r}
# linkedin is already available in your workspace

# Create the days vector
days <- 1:21

# Fit a linear model called on the linkedin views per day: linkedin_lm
linkedin_lm <- lm(linkedin ~ days)

# Predict the number of views for the next three days: linkedin_pred
future_days <- data.frame(days = 22:24)
linkedin_pred <- predict(linkedin_lm, future_days)

# Plot historical data and predictions
plot(linkedin ~ days, xlim = c(1, 24))
points(22:24, linkedin_pred, col = "green")
```

*** =sct
```{r, eval=FALSE}

msg <- "Do not remove or override the definition of the <code>linkedin</code> vector."
test_object("linkedin", undefined_msg = msg, incorrect_msg = msg)
test_object("days")
test_object("linkedin_lm")
msg <- "Do not change or remove the definition of the <code>future_days</code> data frame."
test_object("future_days", undefined_msg = msg, incorrect_msg = msg)
test_object("linkedin_pred")
msg <- "Do not remove or change the <code>plot()</code> or <code>points()</code> function."
test_function("plot", not_called_msg = msg)
test_function("points", not_called_msg = msg)
test_error()
success_msg("Fantastic effort! Wow, based on this model, your profile views will skyrocket! In the next exercise, you will explore a completely different machine learning technique: clustering.")
```

*** =skills
1,6

--- type:NormalExercise xp:100 key:c59c997198
## Clustering: Separating the iris species

Last but not least, there's clustering. This technique tries to group your objects. It does this without any prior knowledge of what these groups could or should look like. For clustering, the concepts of _prior knowledge_ and _unseen observations_ are less meaningful than for classification and regression.

In this exercise, you'll group irises in 3 distinct clusters, based on several flower characteristics in the [`iris`](http://www.rdocumentation.org/packages/datasets/functions/iris) dataset. It has already been chopped up in a data frame `my_iris` and a vector `species`, as shown in the sample code on the right.

The clustering itself will be done with the [`kmeans()`](http://www.rdocumentation.org/packages/stats/functions/kmeans) function. How the algorithm actually works, will be explained in the last chapter. For now, just try it out to gain some intuition!

**Note:** In problems that have a random aspect (like this problem with [`kmeans()`](http://www.rdocumentation.org/packages/stats/functions/kmeans)), the [`set.seed()`](http://www.rdocumentation.org/packages/base/functions/Random) function will be used to enforce reproducibility. If you fix the seed, the random numbers that are generated (e.g. in `kmeans()`) are always the same.

*** =instructions
- Use the [`kmeans()`](http://www.rdocumentation.org/packages/stats/functions/kmeans) function. The first argument is `my_iris`; the second argument is 3, as you want to find three clusters in `my_iris`. Assign the result to a new variable, `kmeans_iris`.
- The actual species of the observations is stored in `species`. Use [`table()`](http://www.rdocumentation.org/packages/base/functions/table) to compare it to the groups that the clustering came up with. These groups can be found in the `cluster` attribute of `kmeans_iris`.
- Inspect the code that generates a plot of `Petal.Length` against `Petal.Width` and colors by cluster.

*** =hint
- For the help file, click on [`kmeans()`](http://www.rdocumentation.org/packages/stats/functions/kmeans).
- For the second instruction, use `table(species, kmeans_iris$cluster)`. Have a look at the documentation of [`table()`](http://www.rdocumentation.org/packages/base/functions/table) if you're not familiar with this function.

*** =pre_exercise_code
```{r, eval=FALSE}
# no pec
```

*** =sample_code
```{r, eval=FALSE}
# Set random seed. Don't remove this line.
set.seed(1)

# Chop up iris in my_iris and species
my_iris <- iris[-5]
species <- iris$Species

# Perform k-means clustering on my_iris: kmeans_iris


# Compare the actual Species to the clustering using table()


# Plot Petal.Width against Petal.Length, coloring by cluster
plot(Petal.Length ~ Petal.Width, data = my_iris, col = kmeans_iris$cluster)
```

*** =solution
```{r}
# Set random seed. Don't remove this line.
set.seed(1)

# Chop up iris in my_iris and species
my_iris <- iris[-5]
species <- iris$Species

# Perform k-means clustering on my_iris: kmeans_iris
kmeans_iris <- kmeans(my_iris, 3)

# Compare the actual species to the clustering using table()
table(species, kmeans_iris$cluster)

# Plot Petal.Width against Petal.Length, coloring by cluster
plot(Petal.Length ~ Petal.Width, data = my_iris, col = kmeans_iris$cluster)
```

*** =sct
```{r, eval=FALSE}
msg0 <- "Do not mess with the random seed."
test_function("set.seed","seed", incorrect_msg = msg0, not_called_msg = msg0)

msg1 <- "Do not remove or change the definition of <code>my_iris</code> and <code>species</code>."
test_object("my_iris", undefined_msg = msg1, incorrect_msg = msg1)
test_object("species", undefined_msg = msg1, incorrect_msg = msg1)

test_object("kmeans_iris",
            incorrect_msg = "Be sure to use the <code>kmeans()</code> function with two arguments: <code>my_iris</code> and <code>3</code>.")
test_or(test_output_contains("table(species, kmeans_iris$cluster)"),
        test_output_contains("table(kmeans_iris$cluster, species)"),
        incorrect_msg = "Compare the actual species, in the <code>species</code> vector to the clusters in <code>kmeans_iris$cluster</code> with <code>table()</code>.")

msg1 <- "Don't change the predefined function call of <code>plot()</code>."
test_function("plot", args = c("formula", "data","col"), incorrect_msg = msg1, not_called_msg = msg1)
test_error()
success_msg("Great Job! Did you see those clusters? The <code>table()</code> function that the groups the clustering came up with, largely correspond to the actual species of the different observations. Now that you've tried regression, classification and clustering problems, it's time to delve a little into the differences between these three techniques.")
```

*** =skills
1,6

--- type:VideoExercise xp:50 key:0ff4bc2a0e
## Supervised vs. Unsupervised

*** =video_link
```{r,eval=FALSE}
//player.vimeo.com/video/163565147
```

*** =video_stream
```{r,eval=FALSE}
https://player.vimeo.com/external/163565147.hd.mp4?s=e7c8b7e683fc0b2ec7444722aab7776225f1be33&profile_id=119
```

*** =video_hls
//videos.datacamp.com/transcoded/682_intro_to_ml/v2/hls-ch1_3.master.m3u8

*** =projector_key
5f02e41d464a8f4b2452edbc162e566e

*** =skills
6


--- type:NormalExercise xp:100 key:bf3fac22f7
## Getting practical with supervised learning

Previously, you used [`kmeans()`](http://www.rdocumentation.org/packages/stats/functions/kmeans) to perform clustering on the [`iris`](http://www.rdocumentation.org/packages/datasets/functions/iris) dataset. Remember that you created your own copy of the dataset, and dropped the `Species` attribute? That's right, you removed the labels of the observations.

In this exercise, you will use the same dataset. But instead of dropping the `Species` labels, you will use them do some supervised learning using recursive partitioning! Don't worry if you don't know what that is yet. Recursive partitioning (a.k.a. decision trees) will be explained in Chapter 3.

*** =instructions
- Take a look at the [`iris`] (http://www.rdocumentation.org/packages/datasets/functions/iris) dataset, using [`str()`](http://www.rdocumentation.org/packages/utils/functions/str) and [`summary()`](http://www.rdocumentation.org/packages/base/functions/summary).
- The code that builds a supervised learning model with the [`rpart()`](http://www.rdocumentation.org/packages/rpart/functions/rpart) function from the `rpart` package is already provided for you. This model trains a decision tree on the `iris` dataset.
- Use the [`predict()`](http://www.rdocumentation.org/packages/stats/functions/predict) function with the `tree` model as the first argument. The second argument should be a data frame containing observations of which you want to predict the label. In this case, you can use the predefined `unseen` data frame. The third argument should be `type = "class"`. Simply print out the result of this prediction step.

*** =hint
The [`predict()`](http://www.rdocumentation.org/packages/stats/functions/predict) function can be used as follows: `predict(tree, unseen, type = "class")`.

*** =pre_exercise_code
```{r, eval=FALSE}
library(rpart)
```

*** =sample_code
```{r}
# Set random seed. Don't remove this line.
set.seed(1)

# Take a look at the iris dataset



# A decision tree model has been built for you
tree <- rpart(Species ~ Sepal.Length + Sepal.Width + Petal.Length + Petal.Width,
              data = iris, method = "class")

# A dataframe containing unseen observations
unseen <- data.frame(Sepal.Length = c(5.3, 7.2),
                     Sepal.Width = c(2.9, 3.9),
                     Petal.Length = c(1.7, 5.4),
                     Petal.Width = c(0.8, 2.3))

# Predict the label of the unseen observations. Print out the result.

```

*** =solution
```{r}
# Set random seed. Don't remove this line.
set.seed(1)

# Take a look at the iris dataset
str(iris)
summary(iris)

# A decision tree model has been built for you
tree <- rpart(Species ~ Sepal.Length + Sepal.Width + Petal.Length + Petal.Width,
              data = iris, method = "class")

# A dataframe containing unseen observations
unseen <- data.frame(Sepal.Length = c(5.3, 7.2),
                     Sepal.Width = c(2.9, 3.9),
                     Petal.Length = c(1.7, 5.4),
                     Petal.Width = c(0.8, 2.3))

# Predict the label of the unseen observations. Print out the result.
predict(tree, unseen, type = "class")
```

*** =sct
```{r, eval=FALSE}
msg0 <- "Do not mess with the random seed."
test_function("set.seed","seed", incorrect_msg = msg0, not_called_msg = msg0)

msg1 <- "Be sure to call <code>str()</code> with the <code>iris</code> dataset as argument."
test_function("str", "object",
              incorrect_msg = msg1,
              not_called_msg = msg1)
msg2 <- "Be sure to call <code>summary()</code> with the <code>iris</code> dataset as argument."
test_function("summary", "object",
              incorrect_msg = msg2,
              not_called_msg = msg2)
msg3 <- "Don't remove or change the predefined code."
test_object("tree", incorrect_msg = msg3, undefined_msg = msg3)
test_object("unseen", incorrect_msg = msg3, undefined_msg = msg3)
test_output_contains("predict(tree, unseen, type=\"class\")",
                     incorrect_msg = "Be sure to call the <code>predict()</code> function with 3 arguments: <code>tree</code>, <code>unseen</code> and <code>type=\"class\"</code>. Simply print out the result; don't assign it to a new variable.")

test_error()
success_msg("That was quite an astonishing effort, great job! In chapter 3, you'll learn more about classification along with some performance measures to assess your predictions! Go to the next exercise to understand how to go about an unsupervised learning problem.")
```


*** =skills
1,6

--- type:NormalExercise xp:100 key:5464003bf1
## How to do unsupervised learning (1)

In this exercise, you will group cars based on their horsepower and their weight. You can find the types of car and corresponding attributes in the `cars` data frame, which has been derived from the [`mtcars`](http://www.rdocumentation.org/packages/datasets/functions/mtcars) dataset. It's available in your workspace.

To cluster the different observations, you will once again use [`kmeans()`](http://www.rdocumentation.org/packages/stats/functions/kmeans).

In short, your job is to cluster the cars in 2 groups, but don't forget to explore the dataset first!

*** =instructions
- Explore the dataset using [`str()`](http://www.rdocumentation.org/packages/utils/functions/str) and [`summary()`](http://www.rdocumentation.org/packages/base/functions/summary).
- Use [`kmeans()`](http://www.rdocumentation.org/packages/stats/functions/kmeans) with two arguments to group the cars into _two_ clusters based on the contents of the `cars` data frame. Assign the result to `km_cars`.
- Print out the `cluster` element of `km_cars`; it shows which cars belong to which clusters.

*** =hint
- You should use [`kmeans()`](http://www.rdocumentation.org/packages/stats/functions/kmeans) as follows: `kmeans(cars, 2)`. Assign it to `km_cars`.
- Print out the `cluster` element of `km_cars` by using the `$` operator.

*** =pre_exercise_code
```{r, eval=FALSE}
cars <- as.data.frame(cbind(mtcars$wt, mtcars$hp))
colnames(cars) <- c("wt", "hp")
rownames(cars) <- attributes(mtcars)$row.names
```

*** =sample_code
```{r}
# The cars data frame is pre-loaded

# Set random seed. Don't remove this line.
set.seed(1)

# Explore the cars dataset



# Group the dataset into two clusters: km_cars


# Print out the contents of each cluster

```

*** =solution
```{r}
# The cars data frame is pre-loaded

# Set random seed. Don't remove this line.
set.seed(1)

# Explore the cars dataset
str(cars)
summary(cars)

# Group the dataset into two clusters: km_cars
km_cars <- kmeans(cars, 2)

# Print out the contents of each cluster
km_cars$cluster
```

*** =sct
```{r, eval=FALSE}
msg0 <- "Do not mess with the random seed."
test_function("set.seed","seed", incorrect_msg = msg0, not_called_msg = msg0)

msg1 <- "Be sure to call <code>str()</code> with the <code>cars</code> dataset as argument."
test_function("str", "object",
              incorrect_msg = msg1,
              not_called_msg = msg1)
msg2 <- "Be sure to call <code>summary()</code> with the <code>cars</code> dataset as argument."
test_function("summary", "object",
              incorrect_msg = msg2,
              not_called_msg = msg2)

test_object("km_cars",
            undefined_msg = "Be sure to define <code>km_cars</code> using <code>kmeans()</code> and two arguments.",
            incorrect_msg = "Your definition of <code>km_cars</code> is incorrect. Be sure to define <code>km_carsars</code> using <code>kmeans()</code> and two arguments.")

test_output_contains("print(km_cars$cluster)", incorrect_msg = "Don't forget to explicitly print out the cluster partitioning, <code>km_cars$cluster</code>.")

test_error()
success_msg("Good job! You can see, for example, that the Ferrari Dino is in cluster 2, while the Fiat X1-9 is grouped in cluster 1. However, if you would like a more comprehensive overview of the results, you should definitely visualize them! Head over the next exercise and find out how!")
```

*** =skills
1,6

--- type:NormalExercise xp:100 key:a7be585b58
## How to do unsupervised learning (2)

In the previous exercise, you grouped the cars based on their horsepower and their weight. Now let's have a look at the outcome!

An important part in machine learning is understanding your results. In the case of clustering, visualization is key to interpretation! One way to achieve this is by plotting the features of the cars and coloring the points based on their corresponding cluster.

In this exercise you'll summarize your results in a comprehensive figure. The dataset `cars` is already available in your workspace; the code to perform the clustering is already available.

*** =instructions
- Finish the [`plot()`](http://www.rdocumentation.org/packages/graphics/functions/plot.default.html) command by coloring the cars based on their cluster. Do this by setting the `col` argument to the cluster partitioning vector: `km_cars$cluster`.
- Print out the clusters' _centroids_, which are kind of like the centers of each cluster. They can be found in the `centers` element of `km_cars`.
- Replace the `___` in [`points()`](http://www.rdocumentation.org/packages/graphics/functions/points) with the clusters' centroids. This will add the centroids to your earlier plot. To learn about the other parameters that have been defined for you, have a look at the [graphical parameters documentation](http://www.rdocumentation.org/packages/graphics/functions/par).

*** =hint
- If given a vector with a length equal to the number of plotted observations, the `col` argument will color each observation based on the corresponding element in the given vector. In this case, the coloring vector `km_cars$cluster` only contains 1's and 2's, which correspond to the colors "black" and "red" respectively. Hence, the objects in cluster one will be colored black, while those in cluster two will be colored red.
- The graphical parameter [`pch`](http://www.rdocumentation.org/packages/graphics/functions/points) sets the points' symbol, in this case, a filled square.
- The graphical parameter [`bg`](http://www.rdocumentation.org/packages/graphics/functions/points) sets the fill color of the points. Only applicable for `pch` symbols 21 through 25.
- The graphical parameter [`cex`](http://www.rdocumentation.org/packages/graphics/functions/points) sets the size of the points' symbol. In this case, it enlarges the symbol by 100%.

*** =pre_exercise_code
```{r, eval=FALSE}
set.seed(1)
cars <- as.data.frame(cbind(mtcars$wt, mtcars$hp))
colnames(cars) <- c("wt", "hp")
rownames(cars) <- attributes(mtcars)$row.names
km_cars <- kmeans(cars, 2)
```

*** =sample_code
```{r}
# The cars data frame is pre-loaded

# Set random seed. Don't remove this line
set.seed(1)

# Group the dataset into two clusters: km_cars
km_cars <- kmeans(cars, 2)

# Add code: color the points in the plot based on the clusters
plot(cars)

# Print out the cluster centroids


# Replace the ___ part: add the centroids to the plot
points(___, pch = 22, bg = c(1, 2), cex = 2)
```

*** =solution
```{r}
# The cars data frame is pre-loaded

# Set random seed. Don't remove this line
set.seed(1)

# Group the dataset into two clusters: km_cars
km_cars <- kmeans(cars, 2)

# Add code: color the points in the plot based on the clusters
plot(cars, col = km_cars$cluster)

# Print out the cluster centroids
km_cars$centers

# Replace the ___ part: add the centroids to the plot
points(km_cars$centers, pch = 22, bg = c(1, 2), cex = 2)
```

*** =sct
```{r, eval=FALSE}
msg0 <- "Do not mess with the random seed."
test_function("set.seed","seed", incorrect_msg = msg0, not_called_msg = msg0)
msg <- "Do not change the predefined variables or objects."
test_object("km_cars",
            undefined_msg = msg,
            incorrect_msg = msg)

test_function("plot", "x",
              not_called_msg = "Don't remove the <code>plot()</code> code!",
              incorrect_msg = "Don't change the argument <code>cars</code> in the <code>plot()</code> function.")

test_function("plot", "col",
              not_called_msg = "Did you color the points based on the corresponding clusters? Set the parameter <code>col</code> to <code>km_cars$cluster</code> when calling <code>plot()</code>.",
              incorrect_msg = "Did you color the points based on the corresponding clusters? Set the parameter <code>col</code> to <code>km_cars$cluster</code> when calling <code>plot()</code>.")

test_output_contains("print(km_cars$centers)", incorrect_msg = "You didn't correctly print out the centers of <code>km_cars</code>. If you assign the centers to a new variable, they don't get printed out!")

test_error()
success_msg("Great! The cluster centroids are typically good representations of all the observations in a cluster. They are often used to <i>summarize</i> your clusters. Head over to the next exercise!")
```

*** =skills
1,6

--- type:PureMultipleChoiceExercise xp:50 key:252947ec5a
## Tell the difference

Wow, you've come a long way in this chapter. You've now acquainted yourself with 3 machine learning techniques. Let's see if you understand the difference between these techniques. Which ones are supervised, and which ones aren't?

From the following list, select the supervised learning problems:

(1) Identify a face on a list of Facebook photos. You can train your system on tagged Facebook pictures.
(2) Given some features, predict whether a fruit has gone bad or not. Several supermarkets provided you with their previous observations and results.
(3) Group DataCamp students into three groups. Students within the same group should be similar, while those in different groups must be dissimilar.

*** =possible_answers
- only (1) and (3) are supervised.
- (1), (2) and (3) are supervised.
- [only (1) and (2) are supervised.]
- only (3) is supervised.

*** =hint
Remember that supervised learning arises when you have some expectation about the output. Unsupervised learning, in contrast, does not depend on previous observations of how a particular problem was solved.


*** =feedback

- Oops! One of those is not supervised! Ask yourself the question: which problems would require you to train on a labeled dataset?
- Almost! One of those is not supervised! Ask yourself the question: which problems would require you to train on a labeled dataset?
- Nice! Grouping the students into three different groups will require features of the students, but you don't require a label. You just want to measure if the students are alike, based on their features.
- Oh ow! That's not right! Ask yourself the question: which problems would require you to train on a labeled dataset?


*** =skills
6
