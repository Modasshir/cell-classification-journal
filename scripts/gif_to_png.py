
# coding: utf-8

# In[5]:


from PIL import Image
from glob import glob
import pandas as pd
import shutil
import os
import progressbar


# In[6]:

if os.path.exists('../png_images/'):
    shutil.rmtree('../png_images/')
os.makedirs('../png_images/')
df = pd.DataFrame()
df['source'] = glob('../images/*.gif')


# In[7]:


df['dest'] = df.source.map(lambda x: x.replace('images', 'png_images'))
df.dest = df.dest.map(lambda x: x.replace('gif', 'png'))


# In[8]:


bar = progressbar.ProgressBar(maxval=df.source.shape[0])
bar.start()
for i in range(df.source.shape[0]):
    im = Image.open(df.source.iloc[i], mode='r')
    im.save(df.dest.iloc[i])
    bar.update(i)
bar.finish()


# In[ ]:
