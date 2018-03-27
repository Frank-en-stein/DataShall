# DataShall Assignment Part 1
##### Complete repository (including part 2): https://github.com/Frank-en-stein/DataShall
## 1-(i)
> minimize: 60x0 + 40x1 + 50x2
> Subject to: -20x0 - 10x1 - 10x2 <= -350
>             -10x0 - 10x1 - 20x2 <= -400
~~~python
from scipy.optimize import linprog

maxim = [60, 40, 50]
coef = [[-20, -10, -10], [-10, -10, -20]]
cons = [-350, -400]

res = linprog(maxim, A_ub=coef, b_ub=cons, bounds=((0,1000),(0,1000),(0,1000)))
print('Minimum cost: ', res.fun)
print('Orders from supplier1: ', res.x[0])
print('Orders from supplier2: ', res.x[1])
print('Orders from supplier3: ', res.x[2])

#output:
#Minimum cost:  1350.0
#Orders from supplier1:  10.0
#Orders from supplier2:  0.0
#Orders from supplier3:  15.0
~~~

## 1-(ii)
> mean, m = 9
> L = L/m = 1/9
> PDF = l*e**(-l*x)
> CDF = 1 - e**(-l*x)
> NOTE: Random variable x (Number of dry days) is a discrete random variable
~~~python
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

#output
#Probability of 13 or more dry days: 0.2359
#Probability of 2 or less dry days: 0.1993
~~~

## 1(iii)
~~~python
import numpy as np, matplotlib.pyplot as plot
np.seterr(divide='ignore')

x = np.arange(-10,10,.25)
y = 0
try:
    y = -1/((x+2)**2) + 4
except:
    pass
plot.plot(x,y)
plot.show()
~~~

![alt text](https://github.com/Frank-en-stein/DataShall/blob/master/Figure_1iii.png?raw=true)

## 2-(i)
> first the code calculated the pearson correlation coefficient (r)
> Then Fischer Z transformation is used to find the 95% confidence interval
```python
import pandas as pd
import scipy.stats as stat
import math

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

df = pd.DataFrame(test)
N = max(df.shape)
pearson_coeff = df['testA'].corr(df['testB'], method='pearson')
print('Pearson Correlation Coefficient, r:', pearson_coeff)
print('95% confidence interval', r_confidence_interval(pearson_coeff, .05, N))

#output
#Pearson Correlation Coefficient, r: 0.9492478327802262
#95% confidence interval (0.911300309675357, 0.9712052221714117)
```

## 2-(ii)
```python
import pandas as pd
import scipy.stats as stat
import numpy as np
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
print("processing...")    
t = [threading.Thread(target=runner, args=(i,)) for i in range(10)]
for i in range(10):
    t[i].start()
for i in range(10):
    t[i].join()
bootstrap_distribution = [x for y in bootstrap_distribution for x in y]
lower, higher = np.percentile(bootstrap_distribution, q=[2.5, 97.5])
print("95% Bootstrapped confidence interval: ", lower, higher)

#output:
#processing...
#95% Bootstrapped confidence interval:  0.9163576753888669 0.9685917315496418
```

## 2-(iii)
```python
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

#output:
#95% (direct) confidence interval (0.911300309675357, 0.9712052221714117)
#95% Bootstrapped confidence interval:  0.9163576753888669 0.9685917315496418
```

![alt text](https://github.com/Frank-en-stein/DataShall/blob/master/Figure_2iii.png?raw=true)

## 4
>complexity analysis line by line:
>7     > list initialization: Constant complexity, O(1)
>8-11  > loop N = Length of string "Text" - 2 times:
>        9,11 contains conditional check performed in constant complecity, O(1)
>        11 contains list append operation which is also constant time operation, O(1)
>        So, 8-11 has a complexity of O(1) * (N-2) = O(N-2) approximately, O(N)
>12    > joining N element list into a string performed in linear time O(N)
>13    > printing N element string in linear time, O(N)
>--------------------------------------------------------------------------------------
>TOTAL = O(1) + O(N) + O(N) + O(N) = O(1) + 3*O(N)
>Ignoring contants, Approximately, O(N) 

```python
Text = "Dude!!!! And I thought I knew a lotttt. Phewwwww!\
I won’t back down. At least I understand now Daaata Science \
is much more than what we are taught in MOOOCs. That is allllright. \
I won’t get demotivated. I’ll work harder and in noooo time, I’ll \
get better & be backkk next time."

result = [Text[0], Text[1]]
for i in range(2, len(Text)):
    if Text[i] == Text[i-1] and Text[i] == Text[i-2]:
        continue
    result.append(Text[i])
result = ''.join(result)
print(result)

#output: Dude!! And I thought I knew a lott. Pheww!I won’t back down. At least I understand now Daata Science is much more than what we are taught in MOOCs. That is allright. I won’t get demotivated. I’ll work harder and in noo time, I’ll get better & be backk next time.
```
