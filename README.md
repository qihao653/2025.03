# 2025.03
# 基于随机森林的江苏滨海湿地分类
pip install rasterio numpy scikit-learn matplotlib pandas geopandas tqdm
# ===================== 江苏省滨海湿地随机森林分类代码 =====================
# 开发环境：Pycharm + Python3.8+
# 核心功能：遥感影像特征计算、随机森林模型训练、地物分类、精度评价、结果输出
# ==========================================================================
import rasterio
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, cohen_kappa_score, classification_report
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
# ===================== 1. 核心参数配置=====================
# 1.1 分类体系与编码（6类地物）
CLASS_DICT = {
    0: "背景",
    1: "水体",
    2: "植被",
    3: "建筑",
    4: "裸土",
    5: "岸滩",
    6: "海上浮筏与水生植物"
}
CLASS_NUM = len(CLASS_DICT) - 1  # 排除背景的分类数量
# 1.2 随机森林模型参数（论文设定）
RF_PARAMS = {
    "n_estimators": 100,  # 决策树数量，与论文一致
    "criterion": "gini",  # Gini指数分裂，与论文一致
    "random_state": 42,  # 固定随机种子，结果可复现
    "n_jobs": -1,  # 调用全部CPU核心，加速计算
    "oob_score": True,  # 袋外分数验证
    "verbose": 1
}
# 1.3 波段配置
# 输入影像波段顺序：[Blue, Green, Red, NIR, SWIR1, SWIR2]
BAND_NAMES = ["Blue", "Green", "Red", "NIR", "SWIR1", "SWIR2"]
BAND_INDEX = {name: i for i, name in enumerate(BAND_NAMES)}
# 1.4 文件路径配置
# 时空融合后的遥感影像路径（单期/多期tif）
IMAGE_PATH = r"./vsdf_fusion_image/202209_202211_fusion.tif"
# 样本ROI路径（shp格式，ENVI/ArcGIS绘制的训练样本，需包含class字段对应分类编码）
SAMPLE_SHP_PATH = r"./roi/sample_roi.shp"
# 分类结果输出路径
OUTPUT_PATH = r"./classification_result"
# 精度报告输出路径
ACCURACY_PATH = r"./accuracy_report"
# 创建输出文件夹
os.makedirs(OUTPUT_PATH, exist_ok=True)
os.makedirs(ACCURACY_PATH, exist_ok=True)
# ===================== 2. 遥感指数计算=====================
def calculate_spectral_indices(img_array):
  
    
    Blue = img_array[BAND_INDEX["Blue"], :, :].astype(np.float32)
    Green = img_array[BAND_INDEX["Green"], :, :].astype(np.float32)
    Red = img_array[BAND_INDEX["Red"], :, :].astype(np.float32)
    NIR = img_array[BAND_INDEX["NIR"], :, :].astype(np.float32)
    SWIR1 = img_array[BAND_INDEX["SWIR1"], :, :].astype(np.float32)
    SWIR2 = img_array[BAND_INDEX["SWIR2"], :, :].astype(np.float32)
    
    epsilon = 1e-8
    
    NDVI = (NIR - Red) / (NIR + Red + epsilon)
    
    alpha = 0.1
    beta = 0.9
    G_NDVI = (NIR - (alpha * Green + beta * Red)) / (NIR + (alpha * Green + beta * Red) + epsilon)
    
    MNDWI = (Green - SWIR1) / (Green + SWIR1 + epsilon)  # 改进型水体指数，区分水体与岸滩
    NDBI = (SWIR1 - NIR) / (SWIR1 + NIR + epsilon)  # 建筑指数，区分建筑与裸土
    NDWI = (Green - NIR) / (Green + NIR + epsilon)  # 归一化水体指数
    
    feature_list = [Blue, Green, Red, NIR, SWIR1, SWIR2, NDVI, G_NDVI, MNDWI, NDBI, NDWI]
    feature_array = np.stack(feature_list, axis=0)
    
    feature_array = np.nan_to_num(feature_array, nan=0, posinf=0, neginf=0)
    return feature_array
# ===================== 3. 样本数据提取 =====================
def extract_sample_data(feature_array, sample_shp, transform):
   
    import geopandas as gpd
    from rasterio.features import geometry_mask
    
    gdf = gpd.read_file(sample_shp)
   
    if "class" not in gdf.columns:
        raise ValueError("样本shp文件必须包含'class'字段，对应分类编码！")
    
    X = []
    y = []
   
    for idx, row in tqdm(gdf.iterrows(), total=len(gdf), desc="提取样本特征"):
        class_label = row["class"]
        geometry = row["geometry"]
        
        mask = geometry_mask(
            [geometry],
            out_shape=feature_array.shape[1:],
            transform=transform,
            invert=True
        )
        
        feature_pixels = feature_array[:, mask].T
        
        label_pixels = np.full(feature_pixels.shape[0], class_label)
        X.append(feature_pixels)
        y.append(label_pixels)
   
    X = np.concatenate(X, axis=0)
    y = np.concatenate(y, axis=0)
   
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    print(f"样本提取完成：总样本数{len(X)}，训练集{len(X_train)}，验证集{len(X_test)}")
    return X_train, X_test, y_train, y_test
# ===================== 4. 随机森林模型训练与精度评价 =====================
def train_rf_model(X_train, X_test, y_train, y_test):
   
    print("开始训练随机森林模型...")

    rf_model = RandomForestClassifier(**RF_PARAMS)
 
    rf_model.fit(X_train, y_train)
  
    y_pred = rf_model.predict(X_test)
   
   
    oa = accuracy_score(y_test, y_pred) * 100

    kappa = cohen_kappa_score(y_test, y_pred)
   
    conf_matrix = confusion_matrix(y_test, y_pred)
   
    class_report = classification_report(
        y_test, y_pred,
        target_names=[CLASS_DICT[i] for i in sorted(CLASS_DICT.keys()) if i != 0],
        output_dict=True
    )
 
    oob_score = rf_model.oob_score_ * 100
   
    print("=" * 50)
    print(f"总体精度(OA): {oa:.4f}%")
    print(f"Kappa系数: {kappa:.4f}")
    print(f"袋外分数(OOB): {oob_score:.4f}%")
    print("=" * 50)
    
    report_df = pd.DataFrame(class_report).T
    report_df.to_excel(os.path.join(ACCURACY_PATH, "分类精度报告.xlsx"))
    np.savetxt(os.path.join(ACCURACY_PATH, "混淆矩阵.csv"), conf_matrix, delimiter=",", fmt="%d")
   
    feature_names = BAND_NAMES + ["NDVI", "G-NDVI", "MNDWI", "NDBI", "NDWI"]
    feature_importance = pd.DataFrame({
        "特征名称": feature_names,
        "重要性得分": rf_model.feature_importances_
    }).sort_values(by="重要性得分", ascending=False)
    feature_importance.to_excel(os.path.join(ACCURACY_PATH, "特征重要性.xlsx"), index=False)
    print("特征重要性排名：")
    print(feature_importance)
    return rf_model, oa, kappa
# ===================== 5. 影像批量分类与结果保存 =====================
def predict_whole_image(rf_model, feature_array, profile, image_name):
    
    print("开始对整幅影像进行分类预测...")
   
    band_num, height, width = feature_array.shape
   
    img_reshape = feature_array.reshape(band_num, -1).T
   
    block_size = 100000
    pred_result = np.zeros(img_reshape.shape[0], dtype=np.uint8)
    for i in tqdm(range(0, img_reshape.shape[0], block_size), desc="分块预测"):
        end = min(i + block_size, img_reshape.shape[0])
        pred_result[i:end] = rf_model.predict(img_reshape[i:end])
    
    pred_img = pred_result.reshape(height, width)
   
    output_profile = profile.copy()
    output_profile.update({
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": 1,
        "dtype": "uint8",
        "compress": "lzw"  # 压缩保存，减少文件体积
    })
    output_file = os.path.join(OUTPUT_PATH, f"{image_name}_分类结果.tif")
    with rasterio.open(output_file, "w", **output_profile) as dst:
        dst.write(pred_img, 1)
    print(f"分类结果已保存至：{output_file}")
    return pred_img
# ===================== 6. 主函数执行全流程 =====================
if __name__ == "__main__":
    # 1. 读取遥感影像
    print("读取遥感影像...")
    with rasterio.open(IMAGE_PATH) as src:
        img_array = src.read()
        profile = src.profile
        transform = src.transform
        crs = src.crs
    image_name = os.path.basename(IMAGE_PATH).split(".")[0]
    print(f"影像读取完成：波段数{img_array.shape[0]}，高度{img_array.shape[1]}，宽度{img_array.shape[2]}")
    
    feature_array = calculate_spectral_indices(img_array)
    print(f"特征计算完成：总特征数{feature_array.shape[0]}")
   
    X_train, X_test, y_train, y_test = extract_sample_data(feature_array, SAMPLE_SHP_PATH, transform)
    
    rf_model, oa, kappa = train_rf_model(X_train, X_test, y_train, y_test)
    
    pred_img = predict_whole_image(rf_model, feature_array, profile, image_name)
    
    image_dir = r"./vsdf_fusion_image/"  # 20期时序影像文件夹
    image_list = [f for f in os.listdir(image_dir) if f.endswith(".tif")]
    for image_file in image_list:
        image_path = os.path.join(image_dir, image_file)
        image_name = os.path.basename(image_path).split(".")[0]
        with rasterio.open(image_path) as src:
            img_array = src.read()
            profile = src.profile
        feature_array = calculate_spectral_indices(img_array)
        predict_whole_image(rf_model, feature_array, profile, image_name)
    """
    print("=" * 50)
    print("江苏省滨海湿地随机森林分类全流程执行完成！")
    print("=" * 50)


