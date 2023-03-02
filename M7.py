import numpy as np
import scipy as sp 
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit
import math
from sklearn.metrics import mean_squared_error
from scipy.optimize import minimize, least_squares
from scipy.stats import norm
from scipy import stats
from matplotlib.pyplot import figure

df = pd.read_csv("BEE529 Dataset BonnevilleChinook_2010to2015.csv")

#scrub
# print("Do we have any NaN Values:", df.isnull().values.any()) #do we have any nan values -- yes
# print("How many NaN values do we Have:", df.isnull().sum().sum()) #how many -- 147

# we have 147 NaN values which means at most we could have 147 rows with at least 1 NaN value. Since we have 2197 rows of data
# I feel comfortable with removing all rows with NaN and retaining the integrity of the data. 

df.dropna(inplace=True) # remove rows with at least 1 NaN

#assign columns as variables
count = df["Unnamed: 2"].values
outflow = df["Unnamed: 3"].values
temp = df["Unnamed: 4"].values
turbidity = df["Unnamed: 5"].values

# remove index 0 which is a column string
count = np.delete(count,0)
outflow = np.delete(outflow, 0)
temp = np.delete(temp,0)
turbidity = np.delete(turbidity,0)

# need to convert entries in numpy array to float.  Pandas defaults to importing all values as strings when hitting a NaN. This is an computationally inefficient way to 
# convert the arrays especially if we were using a larger data sets. If they were larger I may important to pandas, understand the data and then re-read the data 
# without NaN so pandas imports values as int and I wouldn't need to iterate across array again do this second conversion.

count = count.astype(np.float64)
outflow = outflow.astype(np.float64)
temp = temp.astype(np.float64)
turbidity = turbidity.astype(np.float64)

yObs = count 
xObs = temp


from scipy.optimize import curve_fit
# Lets define our model
def our_model(X, a, b, c):
    return a * X + b*X**2 + c

numMonteCarloRuns = 1_000_000
xObsStd = 1
yObsStd = 2

# Run the Monte Carlo simulation
yModels = {1:[],2:[],3:[],4:[],5:[],6:[],7:[],8:[],9:[],10:[],11:[],12:[],13:[],14:[],15:[],16:[],17:[],18:[],19:[],20:[],21:[],22:[],23:[],24:[],25:[]}
beta_0s = []
beta_1s = []
beta_2s = []
for i in np.arange(numMonteCarloRuns):
    # Add the random noise
    rand_xObs = xObs + norm.rvs(scale=xObsStd, loc=0, size=xObs.shape)
    # Check for negative values
    rand_xObs = np.abs(rand_xObs)
    rand_yObs = yObs + norm.rvs(scale=yObsStd, loc=0, size=yObs.shape)
    # Fit the model:
    popt, pcov = curve_fit(our_model, rand_xObs, rand_yObs, maxfev = 10**4)
    beta_0s.append(popt[0])
    beta_1s.append(popt[1])
    beta_2s.append(popt[2])
    for key in yModels:
        yModels[key].append(our_model(int(key), popt[0], popt[1], popt[2]))

# plot histograms at each temperature of interest
rows, cols = 5,5

fig, ax = plt.subplots(rows, cols, sharex=True, sharey=True)
fig.set_size_inches(10,10)
counter = 1
for row in range(rows):
    for col in range(cols):
        ax[row,col].hist(yModels[counter], bins =25,  density=True, histtype='stepfilled', color = "black", label = str(counter) + " C")
        ax[row,col].annotate(str(counter) + " C", xy=(0.7,0.9),xycoords='axes fraction',
             fontsize=14)
        counter += 1
plt.tight_layout()
plt.show()

rows, cols = 5,5

fig, ax = plt.subplots(rows, cols)
fig.set_size_inches(10,10)
counter = 1
for row in range(rows):
    for col in range(cols):
        ax[row,col].hist(yModels[counter], bins = 25, density=True, histtype='stepfilled', color = "black", label = str(counter) + " C")
        ax[row,col].annotate(str(counter) + " C", xy=(0.7,0.9),xycoords='axes fraction',
             fontsize=14)
        counter += 1
plt.tight_layout()
plt.show()

avgs = []
stds = []
for key in yModels:
    model_array = np.array(yModels[key])
    avgs.append(np.mean(model_array))
    stds.append(np.std(model_array))

rows, cols= 1,2
fig, ax = plt.subplots(rows,cols)
fig.set_size_inches(10,10)
ax[0].hist(beta_1s, bins= 50, density=True, histtype='stepfilled', color = "black")
ax[0].annotate("Beta_1", xy=(0.7,0.9),xycoords='axes fraction', fontsize=14)
ax[1].hist(beta_2s,bins = 50, density=True, histtype='stepfilled', color = "black")
ax[1].annotate("Beta_2", xy=(0.7,0.9),xycoords='axes fraction', fontsize=14)
plt.show()