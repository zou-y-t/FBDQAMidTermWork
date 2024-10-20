#获取https://www.cnfin.com/in/jrjfb/index_9.shtml的响应
import requests
from bs4 import BeautifulSoup
import tqdm
import pandas as pd

def getHtml(url):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'}
    response = requests.get(url, headers=headers)
    response.encoding = response.apparent_encoding
    return response.text

def parseHtml(html):
    soup = BeautifulSoup(html, 'html.parser')
    items = soup.find_all('div', class_='ui-zxlist-item')
    data_list = []
    for item in items:
        title = item.find('h3').get_text(strip=True)
        description = item.find('p').get_text(strip=True)
        date = item.find('div', class_='ui-publish').get_text(strip=True).split()[0]
        data_list.append((title, description, date))
    return data_list

df=pd.DataFrame(columns=['date','title','content','position','loss_ratio'])
pd1=pd.read_csv('dateDataPosition.csv')
df['date']=pd1['date']
df['title']=[[] for i in range(len(df))]
df['content']=[[] for i in range(len(df))]
df['position']=pd1['position']
df['loss_ratio']=pd1['loss_ratio']
for n in tqdm.tqdm(range(27, 100)):
    url = 'https://www.cnfin.com/in/jrjfb/index_'+str(n)+'.shtml'
    html = getHtml(url)
    data_list = parseHtml(html)
    # 若data_list的date在df['date']中，则将data_list的title和content加入df中
    for data in data_list:
        if data[2] in df['date'].values:
            index=df[df['date']==data[2]].index[0]
            df['title'][index].append(data[0])
            df['content'][index].append(data[1])

print(df)
df.to_csv('dateDataPositionContent.csv',index=False)