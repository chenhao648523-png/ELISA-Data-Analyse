import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import io
import scipy.stats as stats
from scipy.optimize import curve_fit
import numpy as np

# ==========================================
# 1. 网页基础配置
# ==========================================
st.set_page_config(page_title="ELISA 分析平台 v6.0", page_icon="🧬", layout="wide")

st.title("🧬 ELISA 实验数据全能分析平台 (免整理版)")
st.markdown("### 智能列映射 | 即使 Excel 格式乱七八糟也能用")

# 初始化 Session State (用于在内存中暂存数据，不需要反复上传)
if 'std_df' not in st.session_state:
    st.session_state['std_df'] = None
if 'sample_data' not in st.session_state:
    st.session_state['sample_data'] = {} # 格式: {'Group Name': df}

# ==========================================
# 2. 核心算法 (保持不变，极其健壮)
# ==========================================
# --- 线性模型 ---
def linear_fit(X, y):
    model = LinearRegression()
    model.fit(X, y)
    r2 = model.score(X, y)
    def predict_func(od_values): return model.predict(od_values.reshape(-1, 1))
    x_range = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
    y_range = model.predict(x_range)
    return predict_func, r2, x_range, y_range

# --- 4-PL 模型 ---
def four_pl_func(x, a, b, c, d): return d + (a - d) / (1.0 + (x / c) ** b)
def four_pl_inverse(y, a, b, c, d):
    try:
        if a > d: 
            if y >= a or y <= d: return np.nan
        else:
            if y <= a or y >= d: return np.nan
        return c * (((a - d) / (y - d)) - 1) ** (1 / b)
    except: return np.nan

def four_pl_fit(conc_std, od_std):
    try:
        p0 = [min(od_std), 1.0, np.median(conc_std), max(od_std)]
        popt, pcov = curve_fit(four_pl_func, conc_std, od_std, p0=p0, maxfev=10000)
        a, b, c, d = popt
        residuals = od_std - four_pl_func(conc_std, *popt)
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((od_std - np.mean(od_std))**2)
        r2 = 1 - (ss_res / ss_tot)
        def predict_func(od_values):
            results = [four_pl_inverse(y, a, b, c, d) for y in od_values.flatten()]
            return np.array(results)
        x_plot = np.logspace(np.log10(min(conc_std[conc_std>0])), np.log10(max(conc_std)), 100)
        y_plot = four_pl_func(x_plot, *popt)
        return predict_func, r2, y_plot.reshape(-1, 1), x_plot
    except: return None, 0, None, None

def fit_standard_curve(df, model_type):
    # 这里的df已经是清洗过的标准格式：['OD_Mean', 'Concentration']
    y_conc = df['Concentration'].values 
    x_od = df['OD_Mean'].values         

    if model_type.startswith("Linear"):
        return linear_fit(x_od.reshape(-1, 1), y_conc)
    else:
        return four_pl_fit(y_conc, x_od)

def get_significance_label(p_value):
    if p_value < 0.001: return "***"
    if p_value < 0.01: return "**"
    if p_value < 0.05: return "*"
    return "ns"

# ==========================================
# 3. 界面布局：分两步走
# ==========================================

# 创建两个大 Tab：数据准备 vs 数据分析
tab_setup, tab_analyze = st.tabs(["📥 第一步：数据导入 (Smart Import)", "📊 第二步：生成图表"])

with tab_setup:
    col_setup_L, col_setup_R = st.columns([1, 1])

    # --- 左侧：标准曲线输入 ---
    with col_setup_L:
        st.subheader("1. 准备标准曲线")
        st.info("直接在下方表格输入数据，或从 Excel 复制粘贴 (Ctrl+V)")
        
        # 默认给一个空模板
        default_std = pd.DataFrame({
            "Concentration": [0, 20, 40, 80, 160, 200],
            "OD1": [0.05, 0.18, 0.27, 0.48, 0.95, 1.12],
            "OD2": [0.05, 0.18, 0.30, 0.56, 1.00, 1.21],
            "OD3": [0.05, 0.18, 0.29, 0.56, 1.01, 1.21]
        })
        
        # 可编辑的表格
        edited_std = st.data_editor(default_std, num_rows="dynamic", use_container_width=True)
        
        if st.button("💾 确认保存标准曲线"):
            # 简单清洗
            try:
                # 自动计算均值，存入 session state
                cols = [c for c in edited_std.columns if 'OD' in c]
                if not cols:
                    st.error("表格里必须至少包含一列 OD 数据 (列名包含 'OD')")
                else:
                    edited_std['OD_Mean'] = edited_std[cols].mean(axis=1)
                    st.session_state['std_df'] = edited_std
                    st.success("✅ 标准曲线已保存！")
            except Exception as e:
                st.error(f"保存失败: {e}")

    # --- 右侧：实验样本导入 (核心：列映射) ---
    with col_setup_R:
        st.subheader("2. 导入实验样本 (任意格式)")
        raw_file = st.file_uploader("上传任意 Excel 文件", type=["xlsx"], key="sample_uploader")
        
        if raw_file:
            xls = pd.ExcelFile(raw_file)
            sheet_name = st.selectbox("选择包含数据的 Sheet", xls.sheet_names)
            
            # 读取原始数据
            raw_df = pd.read_excel(raw_file, sheet_name=sheet_name)
            st.write("👀 原始数据预览 (前3行):")
            st.dataframe(raw_df.head(3), height=100)
            
            # === ⚡️ 智能列映射区 ===
            st.markdown("#### 🔧 告诉程序哪一列是哪一列")
            all_cols = raw_df.columns.tolist()
            
            c1, c2, c3 = st.columns(3)
            # 用户自己选，不用改 Excel
            time_col = c1.selectbox("哪一列是时间/X轴?", all_cols, index=0)
            od_cols = c2.multiselect("哪几列是 OD 值?", all_cols, default=[c for c in all_cols if 'OD' in c or 'Abs' in c])
            group_name = c3.text_input("给这组数据起个名", value=sheet_name)
            
            if st.button("➕ 添加到分析列表"):
                if not od_cols:
                    st.error("请至少选择一列 OD 值！")
                else:
                    # 自动清洗并标准化
                    clean_df = pd.DataFrame()
                    clean_df['Time'] = raw_df[time_col]
                    
                    # 把用户选的 OD 列改名为 OD1, OD2... 以适配算法
                    for i, col in enumerate(od_cols):
                        clean_df[f'OD{i+1}'] = raw_df[col]
                    
                    # 存入 Session State
                    st.session_state['sample_data'][group_name] = clean_df
                    st.success(f"🎉 已添加组: {group_name}")

        # 展示已添加的组
        st.markdown("#### 📦 当前已缓存的实验组")
        if st.session_state['sample_data']:
            for name, df in st.session_state['sample_data'].items():
                st.text(f"✅ {name} (包含 {len(df)} 个数据点)")
            if st.button("🗑️ 清空所有数据"):
                st.session_state['sample_data'] = {}
                st.rerun()
        else:
            st.info("暂无数据，请在上方上传并添加。")

# ==========================================
# 4. 分析界面 (从 Session State 读取数据)
# ==========================================
with tab_analyze:
    # 检查数据是否就绪
    if st.session_state['std_df'] is None:
        st.warning("⚠️ 请先在【第一步】中保存标准曲线！")
    elif not st.session_state['sample_data']:
        st.warning("⚠️ 请先在【第一步】中添加至少一组实验数据！")
    else:
        # === 侧边栏移到这里，作为局部设置 ===
        st.markdown("### ⚙️ 分析参数配置")
        col_cfg1, col_cfg2, col_cfg3, col_cfg4 = st.columns(4)
        
        fit_model = col_cfg1.radio("拟合模型", ("Linear (fHb)", "4-PL (PAF)"), index=1)
        conc_unit = col_cfg2.text_input("浓度单位", "ng/mL")
        lod_val = col_cfg3.number_input("最低检测限 (LOD)", value=1.5, help="低于此值将切换为 OD 模式")
        
        # 拟合标准曲线
        std_df = st.session_state['std_df']
        predict_engine, r2, plot_x, plot_y = fit_standard_curve(std_df, fit_model)
        
        if predict_engine:
            col_cfg4.metric("标准曲线 R²", f"{r2:.4f}")
            
            # === 主分析区 ===
            st.markdown("---")
            st.subheader("⚔️ 多组对比分析")
            
            # 选择要分析的组
            avail_groups = list(st.session_state['sample_data'].keys())
            c_sel1, c_sel2 = st.columns(2)
            compare_groups = c_sel1.multiselect("选择参与对比的组", avail_groups, default=avail_groups)
            control_group = c_sel2.selectbox("选择 Control 组", compare_groups) if compare_groups else None
            
            # 图表设置
            c_opt1, c_opt2, c_opt3 = st.columns(3)
            show_stars = c_opt1.checkbox("显示显著性 (*)", True)
            use_log = c_opt2.checkbox("Log 坐标轴", False)
            manual_ylim = c_opt3.number_input("Y轴最大值 (0=自动)", 0.0)

            # 稀释设置
            if compare_groups:
                st.caption("设置各组稀释倍数 (如有)")
                dil_data = pd.DataFrame({
                    "Group": compare_groups,
                    "Dilution_Time_Point": [6.0]*len(compare_groups),
                    "Dilution_Factor": [1.0]*len(compare_groups)
                })
                edited_dil = st.data_editor(dil_data, num_rows="fixed", hide_index=True)

            if st.button("🔥 生成最终图表", type="primary"):
                combined_data = []
                trigger_od = False
                
                # --- 计算循环 ---
                for grp in compare_groups:
                    # 从缓存读取，这步不需要读 Excel 文件了
                    df_tmp = st.session_state['sample_data'][grp].copy()
                    
                    # 1. 找 OD 列 (列名包含 OD 的)
                    od_cols_local = [c for c in df_tmp.columns if c.startswith('OD')]
                    df_tmp['OD_Mean'] = df_tmp[od_cols_local].mean(axis=1)
                    
                    # 2. 预测
                    df_tmp['Concentration'] = predict_engine(df_tmp['OD_Mean'].values)
                    
                    # 3. 稀释
                    cfg = edited_dil[edited_dil['Group'] == grp].iloc[0]
                    d_time, d_fact = cfg['Dilution_Time_Point'], cfg['Dilution_Factor']
                    
                    if d_time in df_tmp['Time'].values:
                        mask = (df_tmp['Time'] == d_time) & (df_tmp['Concentration'].notna())
                        df_tmp.loc[mask, 'Concentration'] *= d_fact

                    # 4. 复孔数据
                    for i, col in enumerate(od_cols_local):
                        conc_col = f'Conc_{i+1}'
                        df_tmp[conc_col] = predict_engine(df_tmp[col].values)
                        if d_time in df_tmp['Time'].values:
                             mask_i = (df_tmp['Time'] == d_time) & (df_tmp[conc_col].notna())
                             df_tmp.loc[mask_i, conc_col] *= d_fact
                    
                    df_tmp['Group'] = grp
                    combined_data.append(df_tmp)
                    
                    # 5. LOD 检查
                    valid_concs = df_tmp['Concentration'].dropna()
                    if len(valid_concs) > 0 and (valid_concs < lod_val).any():
                        trigger_od = True
                    elif len(valid_concs) == 0:
                        trigger_od = True

                df_all = pd.concat(combined_data)
                
                # --- 绘图 ---
                if trigger_od and lod_val > 0:
                    st.warning(f"⚠️ 自动降级：部分样本 < LOD ({lod_val})，已切换至 OD 模式。")
                    plot_mode = 'OD'
                    y_label = "Optical Density (OD)"
                else:
                    plot_mode = 'Conc'
                    y_label = f"Concentration ({conc_unit})"

                time_points = sorted(df_all['Time'].unique())
                fig, ax = plt.subplots(figsize=(10, 6))
                bar_width = 0.8 / len(compare_groups)
                x_indices = np.arange(len(time_points))
                colors = ['#bdc3c7', '#3498db', '#e74c3c', '#2ecc71', '#9b59b6']
                max_y = 0

                for i, grp in enumerate(compare_groups):
                    means, errors = [], []
                    df_g = df_all[df_all['Group'] == grp]
                    
                    for t_idx, t in enumerate(time_points):
                        row = df_g[df_g['Time'] == t]
                        if not row.empty:
                            od_cols_local = [c for c in row.columns if c.startswith('OD') and c != 'OD_Mean']
                            conc_cols_local = [c for c in row.columns if c.startswith('Conc_')]
                            
                            vals = row[od_cols_local].values.flatten() if plot_mode == 'OD' else row[conc_cols_local].values.flatten()
                            vals = vals[~np.isnan(vals)]
                            
                            if len(vals) > 0:
                                m, s = np.mean(vals), np.std(vals, ddof=1) if len(vals)>1 else 0
                                means.append(m)
                                errors.append(s)
                                if m+s > max_y: max_y = m+s
                                
                                # 显著性
                                if show_stars and grp != control_group:
                                    df_ctrl = df_all[(df_all['Group'] == control_group) & (df_all['Time'] == t)]
                                    if not df_ctrl.empty:
                                        c_vals = df_ctrl[od_cols_local].values.flatten() if plot_mode == 'OD' else df_ctrl[conc_cols_local].values.flatten()
                                        c_vals = c_vals[~np.isnan(c_vals)]
                                        if len(vals)>1 and len(c_vals)>1:
                                            try:
                                                _, p = stats.ttest_ind(vals, c_vals)
                                                sig = get_significance_label(p)
                                                if sig != "ns":
                                                    xp = t_idx + (i - len(compare_groups)/2 + 0.5) * bar_width
                                                    yp = (m+s)*1.1 if use_log else m+s+(m*0.05)
                                                    ax.text(xp, yp, sig, ha='center', va='bottom', fontsize=12, fontweight='bold')
                                                    if yp > max_y: max_y = yp
                                            except: pass
                            else: means.append(0); errors.append(0)
                        else: means.append(0); errors.append(0)
                    
                    x_pos = x_indices + (i - len(compare_groups)/2 + 0.5) * bar_width
                    ax.bar(x_pos, means, yerr=errors, width=bar_width, label=grp, capsize=4, alpha=0.9, color=colors[i%5])

                if use_log: ax.set_yscale('log')
                else: 
                    if manual_ylim>0: ax.set_ylim(0, manual_ylim)
                    else: ax.set_ylim(0, max_y*1.25)
                
                ax.set_xticks(x_indices)
                ax.set_xticklabels(time_points)
                ax.set_ylabel(y_label)
                ax.set_title(f"Comparison Analysis (vs {control_group})")
                ax.legend(loc='upper left')
                if plot_mode=='OD':
                    ax.text(0.5, 0.98, f"Values < LOD ({lod_val}) detected. Shown as OD.", transform=ax.transAxes, ha='center', va='top', color='red')
                
                st.pyplot(fig)
                
                out = io.BytesIO()
                with pd.ExcelWriter(out, engine='openpyxl') as writer: df_all.to_excel(writer, index=False)
                st.download_button("下载结果 (.xlsx)", out.getvalue(), "Result.xlsx")

        else: st.error("标准曲线拟合失败，请检查数据")