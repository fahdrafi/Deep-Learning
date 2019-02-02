# Databricks notebook source
# MAGIC %md
# MAGIC Chapter 3
# MAGIC ---------
# MAGIC This notebook contains code for chapter 3

# COMMAND ----------

from keras.datasets import imdb

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

# COMMAND ----------

