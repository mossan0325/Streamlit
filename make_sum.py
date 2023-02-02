#!/usr/bin/env python
# coding: utf-8

# In[8]:


import os
import pandas as pd
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

model = AutoModelForSeq2SeqLM.from_pretrained('./transformers/summary_ja/')    
tokenizer = AutoTokenizer.from_pretrained('sonoisa/t5-base-japanese') 


# In[9]:


def make_sum(text):
  inputs = tokenizer(text, return_tensors="pt", max_length=512,truncation=True)
  outputs = model.generate(inputs["input_ids"], max_length=40, min_length=10,num_beams=4, early_stopping=True)
  #print(tokenizer.decode(outputs[0], skip_special_tokens=True)) 
  return tokenizer.decode(outputs[0], skip_special_tokens=True)


# In[ ]:




