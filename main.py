"""
This class is meant for replicating the results of the p-BRP, p-ARP and p-MBRP models of Schouten (2022), further
explanation can be found in the Bachelor thesis of Sternfeld, A. (2022).
"""
import itertools
import math
import time
from scipy import stats, integrate
import numpy as np  # numpy
import gurobipy as gp

# Initiate the model, and specify the model type
model = gp.Model()
modelType = "pARP"  # Choose between pARP, pBRP and pMBRP

# Specify the cycle for BRP and MBRP
m = 1

# Set the max age and the number of periods
maxAge = 45
numPeriods = 12

# Costs parameters
cpbar = 10  # Average preventive costs
cfbar = 50  # Average corrective costs
delta = 0  # Deviation due to different period of year
cpdeviation = delta * cpbar
cfdeviation = delta * cfbar

# Weibull parameters for deterioration process
alpha = 12
beta = 2

# Define state space components
if modelType == "pBRP" or modelType == "pMBRP":
    periods = np.arange(1, m * numPeriods + 1)
else:
    periods = np.arange(1, numPeriods + 1)
ages = np.arange(0, maxAge + 1)
stateSpace = itertools.product(periods, ages)
working = [age for age in ages if age != 0]  # states where a component is working


def actionSpace(age):
    """
    Function for obtaining all possible actions at a certain age
    :param age: age that we consider
    :return: all possible actions
    """
    if age == 0 or age == maxAge:  # If age is 0 or max age, maintenance must be performed
        return np.array([1])
    return np.array([0, 1])


def weibullFailure(x):
    """
    Probability of failure at age x
    :param x: age
    :return: probability of failure
    """
    return 1 - math.exp(-(x / alpha) ** beta + ((x - 1) / alpha) ** beta)


def transProb(begState, endState, action):
    """
    Function for obtaining the transition probabilities
    :param begState: beginning stage
    :param endState: ending stage
    :param action: action that is taken
    :return: transition probability
    """
    begPer = begState[0]
    begAge = begState[1]
    endPer = endState[0]
    endAge = endState[1]
    if not endPer % numPeriods == (begPer + 1) % numPeriods:  # If end period and beginning period don't align, return 0
        return 0
    if action == 0:  # If no maintenance is performed
        if endAge == begAge + 1 and begAge != 0 and begAge != maxAge:
            return 1 - weibullFailure(endAge)
        elif endAge == 0 and begAge != 0 and begAge != maxAge:
            return weibullFailure(begAge + 1)
        else:
            return 0
    else:  # If maintenance is performed
        if endAge == 1:
            return 1 - weibullFailure(1)
        elif endAge == 0:
            return weibullFailure(1)
        else:
            return 0


# Make variables
x = model.addVars(periods, ages, {0, 1}, lb=0, vtype=gp.GRB.CONTINUOUS, name="x")

# For pBRP, also define variables y as denoted in Schouten (2022)
if modelType == "pBRP" or modelType == "pMBRP":
    y = model.addVars(periods, vtype=gp.GRB.BINARY, name="Period")

# For pMBRP, also define variables z as denoted in Schouten (2022)
if modelType == "pMBRP":
    z = model.addVars(periods, ages, vtype=gp.GRB.BINARY, name="z")
    t = model.addVars(periods, vtype=gp.GRB.INTEGER, name="t")

# Add constraints, dependent on the model type.
if modelType == "pARP":
    model.addConstrs((sum(x[p1, t1, a] for a in actionSpace(t1)) - sum(
        transProb([p2, t2], [p1, t1], a) * x[p2, t2, a] for p2 in periods for t2 in
        ages for a in actionSpace(t2)) == 0 for p1 in periods for t1 in ages), name='7b')
    model.addConstrs((sum(x[p1, t1, a] for t1 in ages for a in actionSpace(t1)) == 1 / numPeriods
                      for p1 in periods), name='7c')
if modelType == "pBRP":
    model.addConstrs((sum(x[p1, t1, a] for a in actionSpace(t1)) - sum(
        transProb([p2, t2], [p1, t1], a) * x[p2, t2, a] for p2 in periods for t2 in
        ages for a in actionSpace(t2)) == 0 for p1 in periods for t1 in ages), name='10b')
    model.addConstrs(x[p1, t1, 0] + y[p1] <= 1 for p1 in periods for t1 in ages if t1 > 0)
    model.addConstrs(x[p1, t1, 1] - y[p1] <= 0 for p1 in periods for t1 in ages if t1 > 0)
    model.addConstrs((sum(x[p1, t1, a] for t1 in ages for a in actionSpace(t1)) == 1 / (m * numPeriods)
                      for p1 in periods), name='10e')

if modelType == "pMBRP":
    model.addConstrs((sum(x[p1, t1, a] for a in actionSpace(t1)) - sum(
        transProb([p2, t2], [p1, t1], a) * x[p2, t2, a] for p2 in periods for t2 in
        ages for a in actionSpace(t2)) == 0 for p1 in periods for t1 in ages), name='12b')
    model.addConstrs((sum(x[p1, t1, a] for t1 in ages for a in actionSpace(t1)) == 1 / numPeriods
                      for p1 in periods), name='12c')
    model.addConstrs((z[p, age] - y[p] <= 0 for p in periods for age in ages), name='12d')
    model.addConstrs((z[p, age1] - z[p, age2] <= 0 for p in periods for age1 in ages for age2 in ages if age1 < age2),
                     name='12e')
    model.addConstrs((t[p1] + p2 * y[p2] + m * numPeriods * y[p2] <= m * numPeriods + p1 for p1 in periods for
                      p2 in periods if p2 < p1), name='12f')
    model.addConstrs((t[p1] + p2 * y[p2] <= m * numPeriods + p1 for p1 in periods for p2 in periods if p2 > p1),
                     name='12g')
    model.addConstrs((maxAge * y[p] - maxAge * z[p, age] - t[p] <= maxAge - 1 - age for p in periods for age in ages),
                     name='12h')
    model.addConstrs((maxAge * z[p, age] + t[p] <= maxAge + age for p in periods for age in ages), name='12i')
    # Relation between x and z
    model.addConstrs(maxAge * z[p, t] + x[p, t, 0] <= maxAge for p in periods for t in ages if t != 0)
    model.addConstrs(x[p, t, 1] - maxAge * z[p, t] <= 0 for p in periods for t in ages if t != 0)
    model.addConstrs(z[p, t] == 0 for p in periods for t in ages if t == 0)


def prevCost(period):
    """
    Function for obtaining costs of preventive maintenance
    :param period: the period maintenance is performed in
    :return: preventive maintenance costs
    """
    phi = -2 * math.pi / numPeriods
    return cpbar + cpdeviation * math.cos((2 * math.pi * period) / numPeriods + phi)


def corCost(period):
    """
    Function for obtaining costs of corrective maintenance
    :param period: the period maintenance is performed in
    :return: corrective maintenance costs
    """
    phi = -2 * math.pi / numPeriods
    return cfbar + cfdeviation * math.cos((2 * math.pi * period) / numPeriods + phi)


# Making the objective function
model.setObjective((sum(prevCost(i1) * x[i1, i2, 1] for i1 in periods for i2 in working) + sum(
    corCost(i1) * x[i1, 0, 1] for i1 in periods)), gp.GRB.MINIMIZE)

# Optimize model
start = time.time()  # Set time before model is optimized
model.optimize()
end = time.time()  # Record time after model is optimized
print("time taken to run: ", end - start)
count = 0
even = 0
dec_vars = np.zeros((len(periods), len(ages), 2))

# Obtain the decision variables of the p-ARP model and store them in an array that is more clear
if modelType == "pARP":
    for v in model.getVars():
        i1 = v.index // ((maxAge + 1) * 2) + 1
        i2 = count
        a = v.index % 2
        even = v.index - 1
        if even % 2 == 0:
            count = (count + 1) % (maxAge + 1)
        # store the results
        dec_vars[(i1 - 1, i2, a)] = v.X

    # Store all policies with positive decision variables
    R = [(i1, i2, a) for i1 in periods for i2 in ages for a in {0, 1} if dec_vars[(i1 - 1, i2, a)] > 0]

    # Create set with all possible states and actions
    all = [(i1, i2, a) for i1 in periods for i2 in ages for a in {0, 1}]

    # Iteratively add action/state combos that transition to the states in R
    while len(R) < (maxAge + 1) * numPeriods:
        for (i1, i2, action) in all:
            for (j1, j2, a) in R:
                if (i1, i2, 0) not in R:
                    if (i1, i2, 1) not in R:
                        if transProb([i1, i2], [j1, j2], action) > 0:
                            R.append((i1, i2, action))
                if len(R) > (maxAge + 1) * numPeriods:
                    break
            if len(R) > (maxAge + 1) * numPeriods:
                break
        if len(R) > (maxAge + 1) * numPeriods:
            break

    # Sort the final policy variables x[i1,i2,a] for easy visualization
    R.sort()
    # print(R)

    # Create subset of R that we can search to find critical ages
    pairs = [r for r in R if r[2] != 0 if r[1] != 0]

    # Make array for critical ages
    critical_ages = np.zeros(len(periods))

    # Fill the array with critical ages
    for (i1, i2, a) in pairs:
        if critical_ages[(i1 - 1)] == 0:
            critical_ages[(i1 - 1)] = i2
    print("The critical ages are: ", critical_ages)

# For pBRP model, get the maintenance months
if modelType == "pBRP":
    for x in model.getVars():
        if x.VarName.find('Period') != -1:
            if x.X > 0:
                print(x.VarName, " here we maintain ", x.X)

# For pMBRP model, get the maintenance months
if modelType == "pMBRP":
    for x in model.getVars():
        if x.VarName.find('t') != -1 or x.VarName.find('Period') != -1:
            if x.X > 0:
                print(x.VarName, " here we maintain ", x.X)

# Print costs
print("The average annual costs are: ", model.objval * 12)
