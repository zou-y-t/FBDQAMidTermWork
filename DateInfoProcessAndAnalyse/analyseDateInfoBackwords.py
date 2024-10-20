import pandas as pd
import jieba
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE

# 读取数据
df3 = pd.read_csv('stockDateDataPositionContent.csv')
# 删除content为[]的行
df3 = df3[df3['content'] != '[]']

# 定义好坏标签
df3['label'] = df3['position'].apply(lambda x: '好' if x < 300 else '坏')

# 对 description 列进行分词
df3['contents'] = df3['content'].apply(lambda x: ' '.join(eval(x)))
df3['description_cut'] = df3['contents'].apply(lambda x: ' '.join(jieba.cut(x)))

# 数据分层抽取
X_train, X_test, y_train, y_test = train_test_split(
    df3['description_cut'], df3['label'], test_size=0.3, stratify=df3['label'], random_state=42
)

# 特征向量化
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# 使用 SMOTE 进行过采样
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train_vec, y_train)

# 构建随机森林分类器
classifier = RandomForestClassifier(class_weight={'好': 2, '坏': 1}, random_state=42)
classifier.fit(X_train_res, y_train_res)

# 预测
y_pred = classifier.predict(X_test_vec)

# 输出分类报告
print(classification_report(y_test, y_pred))

# 打印显示哪些特征对好有正向作用
feature_importances = classifier.feature_importances_
feature_names = vectorizer.get_feature_names_out()
feature_importances_df = pd.DataFrame(
    {
        'feature': feature_names,
        'importance': feature_importances,
    }
)
print(feature_importances_df.sort_values('importance', ascending=False).head(10))

# 打印显示哪些特征对坏有正向作用
feature_importances_df = pd.DataFrame(
    {
        'feature': feature_names,
        'importance': feature_importances,
    }
)
print(feature_importances_df.sort_values('importance', ascending=True).head(10))