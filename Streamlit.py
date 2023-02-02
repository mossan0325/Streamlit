#!/usr/bin/env python
# coding: utf-8

# In[10]:


import streamlit as st
import numpy as np
import time


# In[ ]:


import os
import pandas as pd
import torch
import transformers
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

model = AutoModelForSeq2SeqLM.from_pretrained('./transformers/summary_ja/')    
tokenizer = AutoTokenizer.from_pretrained('sonoisa/t5-base-japanese') 


# In[ ]:


def make_sum(text):
  inputs = tokenizer(text, return_tensors="pt", max_length=512,truncation=True)
  outputs = model.generate(inputs["input_ids"], max_length=40, min_length=10,num_beams=4, early_stopping=True)
  #print(tokenizer.decode(outputs[0], skip_special_tokens=True)) 
  return tokenizer.decode(outputs[0], skip_special_tokens=True)


# In[8]:


with st.form("my_form", clear_on_submit=False):
    origin = st.text_area('文章を入力してください。')
    submitted = st.form_submit_button("要約")


# In[9]:


if submitted:
    with st.spinner("思考中..."):
        summary = make_sum(origin)
        time.sleep(3)
    st.write("#### 要約文\n", summary)
    st.write("#### 元の文章\n", origin)


# In[ ]:




