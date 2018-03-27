# mean, m = 9
# L = L/m = 1/9
# PDF = l*e**(-l*x)
# CDF = 1 - e**(-l*x)
# NOTE: Random variable x (Number of dry days) is a discrete random variable

import math

def CDF(x):
    L = 1/9
    return 1 - math.exp(-L*x);

eps = 10**-5
#x>=13
Probability13 = CDF(float('inf')) - CDF(13)

#x<=2 && x>=0
Probability2 = CDF(2)

print("Probability of 13 or more dry days:", round(Probability13, 4))
print("Probability of 2 or less dry days:", round(Probability2, 4))
