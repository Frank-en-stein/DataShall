import pandas as pd
import scipy.stats as stat
import numpy as np
import matplotlib.pyplot as plt
import math, threading

test = { "testA": [58,49.7,51.4,51.8,57.5,52.4,47.8,45.7,51.7,46,50.4,61.9,49.6,61.6,54,54.9,49.7, 
           47.9,59.8,52.3,48.4,49.1,53.7,48.4,47.6,50.8,58.2,59.8,42.7,47.8,51.4,50.9,49.4, 
           64.1,51.7,48.7,48.3,46.1,47.3,57.7,41.8,51.5,46.9,42,50.5,46.3,44,59.3,52.8],
         
         "testB": [56.1,51.5,52.8,52.5,57.4,53.86,48.5,49.8,53.9,49.3,51.8,60,51.4,60.2,53.8,52, 
           49,49.7,59.9,51.2,51.6,49.3,53.8,50.7,50.8,49.8,59,56.6,47.7,47.2,50.9,53.3, 
           50.6,60.1,50.6,50,48.5,47.8,47.8,55.1,44.9,51.9,50.3,44.3,52,49,46.2,59,52]
         }
def r_to_z(r):
    return math.log((1 + r) / (1 - r)) / 2.0

def z_to_r(z):
    e = math.exp(2 * z)
    return((e - 1) / (e + 1))

def r_confidence_interval(r, alpha, n):
    z = r_to_z(r)
    se = 1.0 / math.sqrt(n - 3)
    z_crit = stat.norm.ppf(1 - alpha/2)  # 2-tailed z critical value

    lo = z - z_crit * se
    hi = z + z_crit * se

    return (z_to_r(lo), z_to_r(hi))

############## Above code from problem 2-i ################
np.random.seed(123)
bootstrap_distribution = [[] for i in range(10)]
original = pd.DataFrame(test)

def bootstrap_rand_pear_coeff():
    bootstrapped = np.array(original.values)
    bootstrapped = bootstrapped[np.random.choice(bootstrapped.shape[0], size = bootstrapped.shape[0], replace=True)]
    bootstrapped = pd.DataFrame(bootstrapped, columns = original.columns)
    #print(bootstrapped)
    #### now calculate pearson coeff r from randomly bootstrapped sample
    pearson_coeff = bootstrapped['testA'].corr(bootstrapped['testB'], method='pearson')  
    return pearson_coeff

def runner(idx):
    bootstrap_distribution[idx] = [bootstrap_rand_pear_coeff() for i in range(1000)]

#run 10 threads   
t = [threading.Thread(target=runner, args=(i,)) for i in range(10)]
for i in range(10):
    t[i].start()
    
############################# NEW ########################################
df = pd.DataFrame(test)
N = max(df.shape)
pearson_coeff = df['testA'].corr(df['testB'], method='pearson')
print('95% (direct) confidence interval', r_confidence_interval(pearson_coeff, .05, N))

plt.subplots_adjust(wspace = .30)
plt.subplot(1, 2, 1)
plt.style.use('seaborn-deep')

plt.hist([test['testA'], test['testB']], 10, label=['test A', 'test B'], normed=True)
plt.legend(loc='upper right')
plt.xlabel('test scores')
plt.ylabel('count')
###########################################################################

for i in range(10):
    t[i].join()
bootstrap_distribution = [x for y in bootstrap_distribution for x in y]
lower, higher = np.percentile(bootstrap_distribution, q=[2.5, 97.5])
print("95% Bootstrapped confidence interval: ", lower, higher)

plt.subplot(1, 2, 2)
plt.ylabel('count')
plt.hist(bootstrap_distribution, 25, normed=True)
plt.xlabel('bootstrap r')
xx = np.linspace(0.85, 1, 1000)
hh = stat.gaussian_kde(bootstrap_distribution)
plt.axvline(lower, color='b', linestyle='dashed', linewidth=1, label='2.5%')
plt.axvline(higher, color='b', linestyle='dashed', linewidth=1, label='97.5%')
plt.plot(xx, hh(xx))
plt.show()

