#!/usr/bin/env python
# coding: utf-8

# In[5]:


theta_0,tau = 0,1; alpha,beta = 2,1/2

import pymc as pm
normal_gamma_toy_model = pm.Model()
with normal_gamma_toy_model:
    theta = pm.Normal("theta", mu=0, sigma=1)
    phi = pm.Gamma("phi", alpha=1, beta=1)
    x_obs = pm.Normal("likelihood", mu=theta, sigma=1/phi**0.5, observed=x)


# In[8]:


with normal_gamma_toy_model:
    MH = pm.Metropolis([theta, phi], S=np.array([0.1]), tune=False, tune_interval=0)
    idata_MH = pm.sample(step=MH)


# In[9]:


with normal_gamma_toy_model:
idata_HMC = pm.sample() 


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




