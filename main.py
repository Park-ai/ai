import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import os

from sklearn.linear_model import LinearRegression
from sklearn import linear_model
from scipy import stats

from bokeh.plotting import figure, show, output_file, output_notebook
import pandas_bokeh
from bokeh.io import show, output_notebook
from bokeh.plotting import figure
pandas_bokeh.output_notebook()
pd.set_option('plotting.backend', 'pandas_bokeh')
from bokeh.models import HoverTool, ColumnDataSource

from bokeh.plotting import *


def calculate_dev(actual, prediction):
  """
  Calculate the standard deviation between actual data and prediction data

  Input:
    actual: actual training data
    prediction: prediction data from linear regression

  Return:
    calculated stadard deviation between actual and prediction data
  """
  
  deviation = math.sqrt((sum(actual - prediction)**2) / len(actual))
    
  return deviation 


class Plotting:

  def __init__(self, train_data, ideal_data, test_data):
    self.train_data = train_data
    self.ideal_data = ideal_data
    self.test_data = test_data


  def plot_test_data(self):
    """
    Plot the test data and save it to the image folder
    
    Args:
      None
    
    Retrns:     
      saved png image file
    """
    if not os.path.exists("./image"):
      os.makedirs("./image")
    
    fig = plt.figure()
    plt.scatter(self.test_data["x"], self.test_data["y"])
    plt.title("Distribution of Test Data")
    fig.savefig('image/test_data.png')

  
  def plot_train_data(self):
    """
    Plot the train data and save it to the image folder
    
    Args:
      None
    
    Retrns: 
      saved png image file
    """    
    
    fig = plt.figure()
    fig, axs = plt.subplots(len(self.train_data.columns[1:]), figsize=(8, 15))

    columns = list(self.train_data.columns[1:])

    for i in range(len(columns)):
        axs[i].scatter(self.train_data["x"], self.train_data[columns[i]])
        axs[i].set_title(f"Distribution of Traing Function: {columns[i]}")  
    
    fig.savefig('image/train_data.png')

  
  def plot_ideal_data(self):
    """
    Plot the ideal data and save it to the image folder
    
    Args:
      None
    
    Retrns: 
      saved png image file
    """
    
    fig = plt.figure()

    fig, axs = plt.subplots(10,5, figsize=(30, 55))

    columns = list(self.ideal_data.columns[1:])

    for i in range(len(columns)):
        axs[i//5, i%5].scatter(self.ideal_data["x"], self.ideal_data[columns[i]])
        axs[i//5, i%5].set_title(f"Distribution of ideal function: {columns[i]}")  
        
    fig.savefig('image/ideal_data.png')



class Program(Plotting):

  def __init__(self, train_data, ideal_data, test_data):
    super().__init__(train_data, ideal_data, test_data)


  def find_ideal_functions(self):
    """
    Find the four ideal functions that has the least deviation to the training function.
    
    Args:
      None
    
    Retrns: 
      list: for each training fuction, match the ideal function that has the least deviations
      [['y1', 'y42'], ['y2', 'y41'], ['y3', 'y11'], ['y4', 'y48']]
    """

    train_ideal_match = []

    for train_col in self.train_data.columns[1:]:

      dev_dict = {}
      for ideal_col in self.ideal_data.columns[1:]:
        dev_dict[ideal_col] = sum((self.train_data[train_col] - self.ideal_data[ideal_col])**2)

      min_dev = min(dev_dict.values())
      min_ideal_col = [k for k, v in dev_dict.items() if v==min_dev]

      train_ideal_match.append([train_col, min_ideal_col[0]])

    return  train_ideal_match


  def compare_train_ideal(self):
    """
    Plot four train and ideal functions that has the least deviations.

    Args:
      None

    Returns:
      four train-ideal images on html format
    """

    train_ideal_match = self.find_ideal_functions()

    fig = figure(width=700, height=500, background_fill_color="#fafafa", tools="hover",
              toolbar_location="above", toolbar_sticky=False, title = "Compare Train and Ideal Functions")

    fig.scatter(x=self.train_data["x"], y=self.train_data[train_ideal_match[0][0]], color="#53777a", alpha=0.8, legend_label="training data")
    fig.scatter(x=self.train_data["x"], y=self.train_data[train_ideal_match[1][0]], color="#53777a", alpha=0.8)
    fig.scatter(x=self.train_data["x"], y=self.train_data[train_ideal_match[2][0]], color="#53777a", alpha=0.8)
    fig.scatter(x=self.train_data["x"], y=self.train_data[train_ideal_match[3][0]], color="#53777a", alpha=0.8)

    fig.scatter(x=self.ideal_data["x"], y=self.ideal_data[train_ideal_match[0][1]], color="#F78888", alpha=0.8, legend_label="ideal data")
    fig.scatter(x=self.ideal_data["x"], y=self.ideal_data[train_ideal_match[1][1]], color="#F78888", alpha=0.8)
    fig.scatter(x=self.ideal_data["x"], y=self.ideal_data[train_ideal_match[2][1]], color="#F78888", alpha=0.8)
    fig.scatter(x=self.ideal_data["x"], y=self.ideal_data[train_ideal_match[3][1]], color="#F78888", alpha=0.8)

    fig.grid.grid_line_color = None
    fig.xaxis.axis_label='X-axis'
    fig.yaxis.axis_label='Y-axis'

    show(fig)


  def plot_ideal_test(self):
    """
    Plot the four ideal function to the test data
    
    Args:
      None
    
    Retrns: 
      test and four ideal images on html format
    """

    train_ideal_match = self.find_ideal_functions()

    fig = figure(width=700, height=500, background_fill_color="#fafafa", tools="hover",
              toolbar_location="above", toolbar_sticky=False, title="Compare Ideal functions to Test Data")

    fig.scatter(x=self.test_data["x"], y=self.test_data["y"], color="#F78888", alpha=0.8, legend_label="test data")

    fig.scatter(x=self.ideal_data["x"], y=self.ideal_data[train_ideal_match[0][1]], color="#626262", alpha=0.8, legend_label="ideal data")
    fig.scatter(x=self.ideal_data["x"], y=self.ideal_data[train_ideal_match[1][1]], color="#626262", alpha=0.8)
    fig.scatter(x=self.ideal_data["x"], y=self.ideal_data[train_ideal_match[2][1]], color="#626262", alpha=0.8)
    fig.scatter(x=self.ideal_data["x"], y=self.ideal_data[train_ideal_match[3][1]], color="#626262", alpha=0.8)

    fig.grid.grid_line_color = None
    fig.xaxis.axis_label='X-axis'
    fig.yaxis.axis_label='Y-axis'

    show(fig)


  def calculate_deviation(self):
    """
    Create four linear regressions with four ideal x and y data. With each linear regression, calcualte deviation from test data and from train data
    
    Args:
      None
    
    Retrns: 
      Dataframe: ideal function, train function, deviation from test data and deviation from train data
    """

    train_ideal_match = self.find_ideal_functions()

    ideal_functions = []
    train_functions = []
    test_deviations = []
    train_deviations = []


    for i in range(len(train_ideal_match)):
      x = self.ideal_data[['x']]
      y = self.ideal_data[train_ideal_match[i][1]]

      train_functions.append(train_ideal_match[i][0])
      ideal_functions.append(train_ideal_match[i][1])

      # linear regression with ideal functions
      regr = linear_model.LinearRegression()
      regr.fit(x, y)

      # regression prediction on test data
      test_prediction = regr.predict(self.test_data[["x"]])
      test_deviation = calculate_dev(self.test_data["y"], test_prediction)
      test_deviations.append(test_deviation)

      # regression prediction on train data
      train_prediction = regr.predict(self.train_data[["x"]])
      train_deviation = calculate_dev(self.train_data[train_ideal_match[i][0]], train_prediction)
      train_deviations.append(train_deviation)

    dev_df = pd.DataFrame(columns = ["train_functions", "ideal_functions", "test_deviations", "train_deviations"])
    dev_df["ideal_functions"] = ideal_functions
    dev_df["train_functions"] = train_functions
    dev_df["test_deviations"] = test_deviations
    dev_df["train_deviations"] = train_deviations


    return dev_df


  def recreate_test_data(self):
    """
    1. Select the ideal function according to the creteria that the existing maximum deviation of the calculated regression
    does not exceed the largest deviation betwen training dataset and ideal function chosen for it 
    by more than factor sqrt(2).
    2. Append the selected ideal function that meets the creteria, and the deviation. 
    
    Args:
      None
    
    Retrns:
      Created test data with the selected ideal function and deviation
    """    
    dev_df = self.calculate_deviation()

    condition_filtering = dev_df[dev_df["test_deviations"] < dev_df["train_deviations"] + np.sqrt(2)]

    for index, row in self.test_data.iterrows():
      self.test_data.at[index, 'dev_y'] = round(condition_filtering["test_deviations"].values[0], 2)
      self.test_data.at[index, 'no_ideal_func'] = condition_filtering["ideal_functions"].values

    recreated_test_data = self.test_data
    recreated_test_data.to_csv("data/recreted_test_data.csv", index=False)

    return recreated_test_data



if __name__ == "__main__":
    train_data = pd.read_csv("data/train.csv")
    ideal_data = pd.read_csv("data/ideal.csv")
    test_data = pd.read_csv("data/test.csv")

    plotting = Plotting(train_data, ideal_data, test_data)
  
    program = Program(train_data, ideal_data, test_data)

    plotting.plot_test_data()
    plotting.plot_train_data()
    plotting.plot_ideal_data()

    train_ideal_match = program.find_ideal_functions()
    print(train_ideal_match)

    program.compare_train_ideal()
    program.plot_ideal_test()
    
    dev_df = program.calculate_deviation()
    print(dev_df)

    created_test_data = program.recreate_test_data()
    print(created_test_data)
