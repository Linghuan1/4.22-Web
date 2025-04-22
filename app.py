# app.py

import streamlit as st
import pandas as pd
import pickle
import numpy as np
import os

# --- 配置页面 ---
st.set_page_config(
    page_title="风力发电量预测器",  # 页面标题
    page_icon="🌬️",                 # 页面图标 (风的表情符号)
    layout="wide",                 # 页面布局 ('centered' 或 'wide')
    initial_sidebar_state="expanded" # 侧边栏状态 ('auto', 'expanded', 'collapsed')
)

# --- 全局变量 ---
MODEL_FILENAME = 'XGBoost_best_model.pkl' 
MODEL_PATH = MODEL_FILENAME

# 这里的特征列表必须与模型训练时使用的特征完全一致
REQUIRED_FEATURES = ['月', '日', '时', '分', '测风塔70米风速(m/s)', 
'测风塔50米风速(m/s)', '测风塔30米风速(m/s)', '测风塔10米风速(m/s)'] # <--- 确认这个列表是正确的


# --- 加载模型 ---
@st.cache_resource # 使用缓存加载模型，提高性能
def load_model(path):
    """加载pickle格式的模型文件"""
    try:
        with open(path, 'rb') as file:
            model = pickle.load(file)
        print(f"模型 {path} 加载成功") # 控制台输出加载成功信息
        return model
    except FileNotFoundError:
        st.error(f"错误: 模型文件 {path} 未找到。请确保模型文件在同一目录下，并且文件名正确。") # 在网页上显示错误信息
        print(f"错误: 文件 {path} 未找到") # 控制台输出错误信息
        return None
    except Exception as e:
        st.error(f"加载模型时出错: {str(e)}") # 在网页上显示通用错误信息
        print(f"加载模型时出错: {str(e)}") # 控制台输出错误信息
        return None

model = load_model(MODEL_PATH) # 加载模型

# --- Streamlit 界面 ---
st.title("🌬️ 风力发电量预测器") # 应用主标题
st.markdown(f"使用训练好的 **{MODEL_FILENAME.split('_best_model.pkl')[0]}** 模型，根据输入的特征预测未来15分钟的风力发电量 (kWh)") # 应用描述

# --- 输入区域 ---
st.sidebar.header("⚙️ 请输入预测所需特征") # 侧边栏标题

# 创建一个字典来存储用户输入
input_features = {}

# 为每个需要的特征创建输入控件
st.sidebar.subheader("时间特征")
input_features['年'] = st.sidebar.number_input("年:", min_value=2019, max_value=2030, value=2023, step=1)
input_features['月'] = st.sidebar.slider("月:", min_value=1, max_value=12, value=6, step=1)
input_features['日'] = st.sidebar.slider("日:", min_value=1, max_value=31, value=15, step=1)
input_features['时'] = st.sidebar.slider("时 (24小时制):", min_value=0, max_value=23, value=12, step=1)
input_features['分'] = st.sidebar.selectbox("分:", options=[0, 15, 30, 45], index=0) # 假设数据是15分钟间隔

st.sidebar.subheader("气象特征")
# 这里需要根据 REQUIRED_FEATURES 动态添加输入，或者手动添加（如果特征固定）
# 手动添加示例 (确保名称与 REQUIRED_FEATURES 中的完全一致):
if '测风塔70米风速(m/s)' in REQUIRED_FEATURES:
    input_features['测风塔70米风速(m/s)'] = st.sidebar.number_input("测风塔70米风速(m/s):", min_value=0.0, value=5.0, step=0.1, format="%.1f")
if '测风塔50米风速(m/s)' in REQUIRED_FEATURES:
    input_features['测风塔50米风速(m/s)'] = st.sidebar.number_input("测风塔50米风速(m/s):", min_value=0.0, value=4.5, step=0.1, format="%.1f")
if '测风塔30米风速(m/s)' in REQUIRED_FEATURES:
    input_features['测风塔30米风速(m/s)'] = st.sidebar.number_input("测风塔30米风速(m/s):", min_value=0.0, value=4.0, step=0.1, format="%.1f")
if '测风塔10米风速(m/s)' in REQUIRED_FEATURES:
    input_features['测风塔10米风速(m/s)'] = st.sidebar.number_input("测风塔10米风速(m/s):", min_value=0.0, value=3.5, step=0.1, format="%.1f")
if '温度(°C)' in REQUIRED_FEATURES:
    input_features['温度(°C)'] = st.sidebar.number_input("温度(°C):", min_value=-20.0, max_value=50.0, value=15.0, step=0.1, format="%.1f")
if '气压(hPa)' in REQUIRED_FEATURES: # 如果气压被选入
     input_features['气压(hPa)'] = st.sidebar.number_input("气压(hPa):", min_value=800.0, max_value=1100.0, value=875.0, step=0.1, format="%.1f")
if '湿度(%)' in REQUIRED_FEATURES: # 如果湿度被选入
     input_features['湿度(%)'] = st.sidebar.slider("湿度(%):", min_value=0.0, max_value=100.0, value=60.0, step=0.1, format="%.1f")
# ... 如果有其他选入的特征，也在这里添加输入控件 ...


# --- 预测按钮和结果显示 ---
if st.sidebar.button("🚀 预测发电量", type="primary"): # 添加预测按钮
    if model is not None: # 确保模型已加载
        # 1. 准备输入数据
        # 确保所有 REQUIRED_FEATURES 都有输入值
        missing_inputs = [feature for feature in REQUIRED_FEATURES if feature not in input_features]
        if missing_inputs:
            st.error(f"错误：缺少以下特征的输入控件：{', '.join(missing_inputs)}")
        else:
            # 2. 转换为DataFrame，并确保特征顺序正确
            try:
                input_df = pd.DataFrame([input_features])
                input_df = input_df[REQUIRED_FEATURES] # 按照训练时的顺序排列特征

                # 3. 进行预测
                prediction = model.predict(input_df)
                predicted_value = prediction[0] # 获取预测结果 (假设模型输出单个值)
                # 对预测结果进行合理性处理（例如，发电量不能为负）
                predicted_value = max(0, predicted_value)

                # 4. 显示预测结果
                st.subheader("📈 预测结果") # 结果区域标题
                st.metric(label="预测发电量 (kWh)", value=f"{predicted_value:.4f}") # 使用metric显示结果，保留4位小数
                st.success("预测完成！") # 显示成功信息

                # 添加一些解释性文本
                st.markdown(f"""
                ---
                **说明:**
                *   ⚡️ 该预测结果基于输入的特征和训练好的 **{MODEL_FILENAME.split('_best_model.pkl')[0]}** 模型。
                *   🕒 预测的是接下来 **15 分钟** 时间段内的总发电量。
                """)

            except KeyError as e:
                st.error(f"输入数据准备错误: 缺少特征 {str(e)} 或顺序不匹配。请检查 `REQUIRED_FEATURES` 列表。") # 处理特征名称不匹配错误
                print(f"KeyError during prediction: {e}")
                print(f"Input DataFrame columns: {input_df.columns.tolist()}")
                print(f"Required features: {REQUIRED_FEATURES}")
            except Exception as e:
                st.error(f"预测过程中发生错误: {str(e)}") # 处理其他预测错误
                print(f"Prediction error: {e}")

    else:
        st.error("模型未能成功加载，无法进行预测。请检查模型文件路径和完整性。") # 如果模型未加载，提示用户

# --- 页脚信息  ---
st.sidebar.markdown("---")
st.sidebar.info(f"💬 模型: {MODEL_FILENAME.split('_best_model.pkl')[0]} | 数据: 风电场气象与发电数据")