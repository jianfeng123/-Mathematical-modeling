#### 生成世界地图
import json
import pygal
import matplotlib.pyplot as plt
import pygal
from pygal_maps_world.i18n import COUNTRIES
from pygal_maps_world.maps import World
from pygal.style import RotateStyle
from pygal.style import LightColorizedStyle
from pygal.style import LightColorizedStyle as LCS, RotateStyle as RS
import pandas as pd
## 读取国家对应的数据
data0 = pd.read_csv('world.csv', header=None)
### 数据格式如下
### country count
###    a    num1    
###    b    num2    
######将国家生成code码，并生成字典格式
###    code  count
###    a_c   num1
###    b_c   num2
cc_populations = {}
for i in range(len(data0)):
    cc_populations[get_country_code(data0.loc[i,0])] = data0.loc[i,1]
# 根据所持数据把国家分成5类
cc_pops_1,cc_pops_2,cc_pops_3,cc_pops_4, cc_pops_5  = {},{},{},{},{}
for cc,pop in cc_populations.items():
	if pop < 10:
		cc_pops_1[cc] = pop
	elif pop < 100:
		cc_pops_2[cc] = pop
	elif pop < 1000:
		cc_pops_3[cc] = pop
	elif pop < 5000:
		cc_pops_4[cc] = pop
	else:
		cc_pops_5[cc] = pop
# 看看每组都包含多少个国家
print(len(cc_pops_1), len(cc_pops_2), len(cc_pops_3), len(cc_pops_4), len(cc_pops_5))
### 绘制最后结果
wm_style=pygal.style.RotateStyle('#3399AA',base_style=pygal.style.LightColorizedStyle)
wm = World(style=wm_style)
wm.title = ('Total number of terrorist attacks in countries')
wm.add('0-10', cc_pops_1)
wm.add('10-100', cc_pops_2)
wm.add('100-1000', cc_pops_3)
wm.add('1000-5000', cc_pops_4)
wm.add('5000-30000', cc_pops_5)
wm.render_to_file('world.svg')
