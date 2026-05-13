import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('AmesHousing.csv')

cat_none = ['Alley', 'Pool QC', 'Fence', 'Misc Feature', 'Bsmt Qual', 'Bsmt Cond',
            'Bsmt Exposure', 'BsmtFin Type 1', 'BsmtFin Type 2', 'Garage Type',
            'Garage Finish', 'Garage Qual', 'Garage Cond', 'Fireplace Qu', 'Mas Vnr Type']

for col in cat_none:
    if col in df: df[col] = df[col].fillna('None')

num_zero = ['Bsmt Full Bath', 'Garage Area', 'Bsmt Half Bath', 'Mas Vnr Area',
            'Garage Cars', 'BsmtFin SF 1', 'BsmtFin SF 2', 'Bsmt Unf SF', 'Total Bsmt SF']

for col in num_zero:
    if col in df: df[col] = df[col].fillna(0)

if 'Lot Frontage' in df and 'Neighborhood' in df:
    df['Lot Frontage'] = df.groupby('Neighborhood')['Lot Frontage'].transform(lambda x: x.fillna(x.median()))

for col in df.select_dtypes(include=np.number).columns:
    if df[col].isnull().any(): df[col] = df[col].fillna(df[col].median())
for col in df.select_dtypes(include='object').columns:
    if df[col].isnull().any(): df[col] = df[col].fillna(df[col].mode()[0])


df['Age_At_Sale'] = (df['Yr Sold'] - df['Year Built']).clip(lower=0)
df['Years_Since_Remod'] = (df['Yr Sold'] - df['Year Remod/Add']).clip(lower=0)

target = 'SalePrice'
X = df.drop(columns=[target])
y = df[target]

cat_features = X.select_dtypes(include='object').columns.tolist()
num_features = X.select_dtypes(include=np.number).columns.tolist()

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), num_features),

    ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_features)
])

X_proc = preprocessor.fit_transform(X)

feature_names = num_features + list(preprocessor.named_transformers_['cat'].get_feature_names_out(cat_features))


X_tr, X_te, y_tr, y_te = train_test_split(X_proc, y, test_size=0.2, random_state=40)

ridge = Ridge().fit(X_tr, y_tr) #.fit(X_tr, y_tr) — обучаем: показываем признаки и правильные ответы, модель ищет зависимость.
coeffs = pd.Series(ridge.coef_, index=feature_names)
top10 = coeffs.abs().nlargest(10)
print("Топ-10 признаков:", top10.index.tolist())


def get_rmse(X, y):
    X1, X2, y1, y2 = train_test_split(X, y, test_size=0.2, random_state=40)
    pred = Ridge().fit(X1, y1).predict(X2)

    return r2_score(y2, pred), np.sqrt(mean_squared_error(y2, pred))


r2_b, rmse_b = get_rmse(X_proc, y)

iso = IsolationForest(contamination=0.02, random_state=40)
df_num = df[num_features]
df['anom'] = iso.fit_predict(df_num)
q3 = df['Gr Liv Area'].quantile(0.75)
large = df[df['Gr Liv Area'] > q3]
iqr = large['SalePrice'].quantile(0.75) - large['SalePrice'].quantile(0.25)

low = large['SalePrice'].quantile(0.25) - 1.5 * iqr
df['susp'] = (df['Gr Liv Area'] > q3) & (df['SalePrice'] < low)
mask = (df['anom'] == -1) | df['susp']
df_clean = df[~mask].copy()
print(f"Удалено аномалий: {mask.sum()} из {len(df)}")

X_cl = df_clean.drop(columns=[target, 'anom', 'susp'])
y_cl = df_clean[target]
X_cl_proc = preprocessor.transform(X_cl)
r2_a, rmse_a = get_rmse(X_cl_proc, y_cl)
print(f"До: R^2={r2_b:.4f}, RMSE={rmse_b:.0f} | После: R^2={r2_a:.4f}, RMSE={rmse_a:.0f}")

sc = StandardScaler()
X_km = sc.fit_transform(df_clean[num_features])

km = KMeans(n_clusters=5, random_state=40, n_init=10)

df_clean['segment'] = km.fit_predict(X_km)

print("Сегменты:\n", df_clean.groupby('segment')[['SalePrice', 'Gr Liv Area', 'Overall Qual']].mean().round(1))


pca = PCA(n_components=0.95)
X_pca = pca.fit_transform(df_clean[num_features])
Xp_tr, Xp_te, yp_tr, yp_te = train_test_split(X_pca, y_cl, test_size=0.2, random_state=40)
r2_pca = r2_score(yp_te, Ridge().fit(Xp_tr, yp_tr).predict(Xp_te))
print(f"PCA: {X_pca.shape[1]} компонент, R^2={r2_pca:.4f}")

print(f"\nКризис 2008: 2007=${df['SalePrice'][df['Yr Sold']==2007].mean():.0f} -> 2008=${df['SalePrice'][df['Yr Sold']==2008].mean():.0f}")
spring = df['SalePrice'][df['Mo Sold'].isin([3,4,5])].mean()
winter = df['SalePrice'][df['Mo Sold'].isin([12,1,2])].mean()
print(f"Сезонность: весна=${spring:.0f} vs зима=${winter:.0f} ({(spring-winter)/winter*100:.1f}%)")

plt.figure(figsize=(6,4))
sns.scatterplot(x='Gr Liv Area', y='SalePrice', data=df, alpha=0.3, s=10)
plt.title('Price vs Gr Liv Area')
plt.tight_layout()
plt.show()
