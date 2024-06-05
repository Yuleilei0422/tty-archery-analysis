import pandas as pd
import numpy as np
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
import joblib
import streamlit as st

# 整理數據
data = {
    'Temperature': [28, 27, 27, 27, 28, 28, 28, 28, 28, 27, 27, 27],
    'Humidity': [77, 77, 77, 78, 77, 77, 76, 78, 79, 82, 82, 82],
    'WindSpeed': [0.6, 0.5, 0.4, 0.4, 0.4, 0.4, 0.6, 0.5, 0.4, 0.8, 0.6, 0.6],
    'Scores': [49, 45, 49, 45, 38, 48, 48, 41, 44, 46, 41, 33]
}

df = pd.DataFrame(data)

# 設置特徵和目標變量
X = df[['Temperature', 'Humidity', 'WindSpeed']]
y = df['Scores']

# 分割數據集為訓練集和測試集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 使用PolynomialFeatures進行特徵擴展
poly = PolynomialFeatures(degree=2, include_bias=False)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

# Lasso回歸模型
lasso = make_pipeline(StandardScaler(), Lasso(alpha=0.1))
lasso.fit(X_train_poly, y_train)

# 保存模型
joblib.dump(lasso, 'lasso_model.pkl')
joblib.dump(poly, 'poly_features.pkl')

def predict_score(temperature, humidity, windspeed):
    # 載入模型和特徵擴展
    lasso_model = joblib.load('lasso_model.pkl')
    poly_features = joblib.load('poly_features.pkl')
    
    # 構建輸入數據
    input_data = np.array([[temperature, humidity, windspeed]])
    
    # 轉換新的數據前，將 X 轉換為一個帶有特徵名稱的 pandas DataFrame
    input_data_df = pd.DataFrame(input_data, columns=['Temperature', 'Humidity', 'WindSpeed'])

    # 使用PolynomialFeatures進行特徵擴展
    input_data_poly = poly_features.transform(input_data_df)
    
    # 預測
    predicted_score = lasso_model.predict(input_data_poly)
    return predicted_score[0]

# Streamlit 應用
st.title("Martin's Score Prediction System")

temperature = st.number_input('Temperature', min_value=25, max_value=35)
humidity = st.number_input('Humidity', min_value=75, max_value=85)
windspeed = st.number_input('Wind Speed', min_value=0.0, max_value=1.0)

if st.button('Predict'):
    predicted_score = predict_score(temperature, humidity, windspeed)
    st.write(f'Predicted Score: {predicted_score}')
