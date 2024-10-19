import requests
from bs4 import BeautifulSoup
import pandas as pd
import tqdm

def getHtml(url):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
    }
    response = requests.get(url, headers=headers)
    response.encoding = response.apparent_encoding
    return response.text

def parseHtml(html):
    soup = BeautifulSoup(html, 'html.parser')
    data_list = []
    
    # Find all the relevant <p> tags
    items = soup.find_all('p', class_='list_word_p_1')
    for item in items:
        title = item.find('a').get_text(strip=True)
        date = item.find('em').get_text(strip=True)
        content_tag = item.find_next_sibling('p', class_='list_word_p_2')
        content = content_tag.get_text(strip=True) if content_tag else ''
        
        # Replace unwanted characters
        content = content.replace('\n', ' ').replace('\u3000', ' ').replace('\u3000', ' ')
        
        data_list.append({'title': title, 'date': date, 'content': content})
    
    return data_list

df=pd.DataFrame(columns=['date','title','content','position','loss_ratio'])
pd1=pd.read_csv('result.csv')
df['date']=pd1['date']
df['title']=[[] for i in range(len(df))]
df['content']=[[] for i in range(len(df))]
df['position'] = range(1, len(df)+1)
df['loss_ratio']=pd1['loss_ratio']

years = {
    '2023': 88,
    '2022': 78,
    '2021': 92
}

for year, pages in years.items():
    for page in tqdm.tqdm(range(1,pages+1)):
        if page == 1:
            url = 'https://www.cei.cn//defaultsite/s/column/4028c7ca-37115425-0137-115610a2-0078_'+year+'.html?articleListType=1&coluOpenType=2'
        else:
            url = 'https://www.cei.cn//defaultsite/s/column/4028c7ca-37115425-0137-115610a2-0078_'+year+'_'+str(page)+'.html?articleListType=1&coluOpenType=2'
        html = getHtml(url)
        data_list = parseHtml(html)
        for data in data_list:
            if data['date'] in df['date'].values:
                index=df[df['date']==data['date']].index[0]
                df['title'][index].append(data['title'])
                df['content'][index].append(data['content'])
        
    
    print(year+' done')

print(df)
df.to_csv('stockDateDataPositionContent.csv',index=False)