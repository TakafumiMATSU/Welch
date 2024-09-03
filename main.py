#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import pandas as pd
from scipy.stats import ttest_ind
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import plotly.graph_objects as go

# Streamlitの設定
st.set_page_config(page_title="Welch's t-test App", layout="wide")

st.title("Welch's t-test Comparison Tool")

# サイドバーでファイルをアップロード
uploaded_file = st.sidebar.file_uploader("Excelファイルをアップロードしてください", type=["xlsx"])

# サイドバーで群の名前を入力
group_names = []
for i in range(5):
    group_name = st.sidebar.text_input(f'群 {i + 1} の名前', value=f'Group{i + 1}')
    group_names.append(group_name)

# サイドバーで閾値の設定
alpha = st.sidebar.slider('p-value 閾値 (alpha)', 0.001, 0.1, 0.05)
percentage_threshold = st.sidebar.slider('Fold Change 閾値 (百分率)', 10, 200, 50) / 100

if uploaded_file:
    # データの読み込み
    data = pd.read_excel(uploaded_file)
    
    st.sidebar.write("アップロードされたファイル:")
    st.sidebar.write(uploaded_file.name)
    
    # 各条件のデータを抽出
    components = data['Name']
    condition1 = data[[group_names[0], f'{group_names[0]}.1', f'{group_names[0]}.2']]
    condition2 = data[[group_names[1], f'{group_names[1]}.1', f'{group_names[1]}.2']]
    condition3 = data[[group_names[2], f'{group_names[2]}.1', f'{group_names[2]}.2']]
    condition4 = data[[group_names[3], f'{group_names[3]}.1', f'{group_names[3]}.2']]
    condition5 = data[[group_names[4], f'{group_names[4]}.1', f'{group_names[4]}.2']]

    # Welch's t-testの結果を格納するリストを初期化
    results = []

    # 各コンポーネントに対してWelch's t-testを実行
    for i, component in enumerate(components):
        c1 = condition1.iloc[i]
        c2 = condition2.iloc[i]
        c3 = condition3.iloc[i]
        c4 = condition4.iloc[i]
        c5 = condition5.iloc[i]

        ttest_c1_c2 = ttest_ind(c1, c2, equal_var=False)
        ttest_c1_c3 = ttest_ind(c1, c3, equal_var=False)
        ttest_c1_c4 = ttest_ind(c1, c4, equal_var=False)
        ttest_c1_c5 = ttest_ind(c1, c5, equal_var=False)
        ttest_c2_c3 = ttest_ind(c2, c3, equal_var=False)
        ttest_c2_c4 = ttest_ind(c2, c4, equal_var=False)
        ttest_c2_c5 = ttest_ind(c2, c5, equal_var=False)
        ttest_c3_c4 = ttest_ind(c3, c4, equal_var=False)
        ttest_c3_c5 = ttest_ind(c3, c5, equal_var=False)
        ttest_c4_c5 = ttest_ind(c4, c5, equal_var=False)

        results.append({
            'Component': component,
            f'{group_names[0]}_vs_{group_names[1]}': ttest_c1_c2.pvalue,
            f'{group_names[0]}_vs_{group_names[2]}': ttest_c1_c3.pvalue,
            f'{group_names[0]}_vs_{group_names[3]}': ttest_c1_c4.pvalue,
            f'{group_names[0]}_vs_{group_names[4]}': ttest_c1_c5.pvalue,
            f'{group_names[1]}_vs_{group_names[2]}': ttest_c2_c3.pvalue,
            f'{group_names[1]}_vs_{group_names[3]}': ttest_c2_c4.pvalue,
            f'{group_names[1]}_vs_{group_names[4]}': ttest_c2_c5.pvalue,
            f'{group_names[2]}_vs_{group_names[3]}': ttest_c3_c4.pvalue,
            f'{group_names[2]}_vs_{group_names[4]}': ttest_c3_c5.pvalue,
            f'{group_names[3]}_vs_{group_names[4]}': ttest_c4_c5.pvalue
        })

    # リストをデータフレームに変換
    results_df = pd.DataFrame(results)

    # p値を元のデータフレームに追加
    data = pd.concat([data, results_df.drop(columns=['Component'])], axis=1)

    # RSD（相対標準偏差）の計算関数
    def calculate_rsd(std, mean):
        return (std / mean) * 100

    # 統計量の計算
    stats = pd.DataFrame({
        'Component': components,
        f'{group_names[0]}_mean': condition1.mean(axis=1),
        f'{group_names[0]}_std': condition1.std(axis=1),
        f'{group_names[0]}_rsd': calculate_rsd(condition1.std(axis=1), condition1.mean(axis=1)),
        f'{group_names[1]}_mean': condition2.mean(axis=1),
        f'{group_names[1]}_std': condition2.std(axis=1),
        f'{group_names[1]}_rsd': calculate_rsd(condition2.std(axis=1), condition2.mean(axis=1)),
        f'{group_names[2]}_mean': condition3.mean(axis=1),
        f'{group_names[2]}_std': condition3.std(axis=1),
        f'{group_names[2]}_rsd': calculate_rsd(condition3.std(axis=1), condition3.mean(axis=1)),
        f'{group_names[3]}_mean': condition4.mean(axis=1),
        f'{group_names[3]}_std': condition4.std(axis=1),
        f'{group_names[3]}_rsd': calculate_rsd(condition4.std(axis=1), condition4.mean(axis=1)),
        f'{group_names[4]}_mean': condition5.mean(axis=1),
        f'{group_names[4]}_std': condition5.std(axis=1),
        f'{group_names[4]}_rsd': calculate_rsd(condition5.std(axis=1), condition5.mean(axis=1))
    })

    # 結果を表示
    st.write("Welch's t-test p-value:")
    st.dataframe(data)

    # 有意差のあるコンポーネントの数をカウント
    significant_c1_vs_c2 = (data[f'{group_names[0]}_vs_{group_names[1]}'] < alpha).sum()
    significant_c1_vs_c3 = (data[f'{group_names[0]}_vs_{group_names[2]}'] < alpha).sum()
    significant_c1_vs_c4 = (data[f'{group_names[0]}_vs_{group_names[3]}'] < alpha).sum()
    significant_c1_vs_c5 = (data[f'{group_names[0]}_vs_{group_names[4]}'] < alpha).sum()
    significant_c2_vs_c3 = (data[f'{group_names[1]}_vs_{group_names[2]}'] < alpha).sum()
    significant_c2_vs_c4 = (data[f'{group_names[1]}_vs_{group_names[3]}'] < alpha).sum()
    significant_c2_vs_c5 = (data[f'{group_names[1]}_vs_{group_names[4]}'] < alpha).sum()
    significant_c3_vs_c4 = (data[f'{group_names[2]}_vs_{group_names[3]}'] < alpha).sum()
    significant_c3_vs_c5 = (data[f'{group_names[2]}_vs_{group_names[4]}'] < alpha).sum()
    significant_c4_vs_c5 = (data[f'{group_names[3]}_vs_{group_names[4]}'] < alpha).sum()

    # マトリクスを生成
    matrix = pd.DataFrame({
        f'{group_names[0]}_vs_{group_names[1]}': [significant_c1_vs_c2],
        f'{group_names[0]}_vs_{group_names[2]}': [significant_c1_vs_c3],
        f'{group_names[0]}_vs_{group_names[3]}': [significant_c1_vs_c4],
        f'{group_names[0]}_vs_{group_names[4]}': [significant_c1_vs_c5],
        f'{group_names[1]}_vs_{group_names[2]}': [significant_c2_vs_c3],
        f'{group_names[1]}_vs_{group_names[3]}': [significant_c2_vs_c4],
        f'{group_names[1]}_vs_{group_names[4]}': [significant_c2_vs_c5],
        f'{group_names[2]}_vs_{group_names[3]}': [significant_c3_vs_c4],
        f'{group_names[2]}_vs_{group_names[4]}': [significant_c3_vs_c5],
        f'{group_names[3]}_vs_{group_names[4]}': [significant_c4_vs_c5]
    }, index=['Significant Components'])

    # マトリクスを表示
    st.write("有意差のあるコンポーネントの数:")
    st.dataframe(matrix)

    # 統計量の表示
    st.write("各群ごとの統計量:")
    st.dataframe(stats)

    # ボルケーノプロットの作成関数を修正
    def plot_volcano(data, p_col, condition1, condition2, percentage_threshold, alpha, title):
        mean1 = data[[col for col in data.columns if condition1 in col]].mean(axis=1)
        mean2 = data[[col for col in data.columns if condition2 in col]].mean(axis=1)
        data['log2FC'] = np.log2(mean1 / mean2)
        data['-log10p'] = -np.log10(data[p_col])

        fc_threshold_upper = np.log2(1 + percentage_threshold)
        fc_threshold_lower = np.log2(1 - percentage_threshold)

        data['significant'] = (data[p_col] < alpha) & ((data['log2FC'] > fc_threshold_upper) | (data['log2FC'] < fc_threshold_lower))

        fig = px.scatter(
            data,
            x='log2FC',
            y='-log10p',
            color='significant',
            color_discrete_map={True: 'red', False: 'gray'},
            hover_name='Name',
            labels={'log2FC': f'log2 Fold Change ({condition1}/{condition2})', '-log10p': '-log10 p-value'},
            title=title
        )

        # 閾値外のポイントに名前を追加
        for i, row in data[data['significant']].iterrows():
            fig.add_annotation(
                x=row['log2FC'],
                y=row['-log10p'],
                text=row['Name'],
                showarrow=False,
                yshift=10
            )

        fig.add_hline(y=-np.log10(alpha), line_dash="dash", line_color="red")
        fig.add_vline(x=fc_threshold_upper, line_dash="dash", line_color="blue")
        fig.add_vline(x=fc_threshold_lower, line_dash="dash", line_color="blue")

        st.plotly_chart(fig)

    # ボルケーノプロットの表示
    st.write("ボルケーノプロット:")

    for i in range(len(group_names)):
        for j in range(i + 1, len(group_names)):
            st.write(f"{group_names[i]} vs {group_names[j]}")
            plot_volcano(data, f'{group_names[i]}_vs_{group_names[j]}', group_names[i], group_names[j], percentage_threshold, alpha, f'Volcano Plot: {group_names[i]} vs {group_names[j]}')

    # RSDのボックスプロットの作成
    st.write("RSDのボックスプロット:")
    data_rsd = pd.melt(stats, id_vars=['Component'], value_vars=[f'{group_names[i]}_rsd' for i in range(len(group_names))], var_name='Condition', value_name='RSD')
    fig = go.Figure()
    for condition, color in zip([f'{group_names[i]}_rsd' for i in range(len(group_names))], ['blue', 'green', 'red', 'purple', 'orange']):
        fig.add_trace(go.Box(
            y=data_rsd[data_rsd['Condition'] == condition]['RSD'],
            name=condition.split('_')[0],
            marker_color=color,
            boxpoints='outliers',  # ここで外れ値のみ表示
            jitter=0.3,
            pointpos=0  # ポイントの位置を中央に設定
        ))
    fig.update_layout(title='Box Plot of RSD', yaxis_title='RSD (%)')
    st.plotly_chart(fig)

    # データを標準化（オートスケール）
    all_conditions = pd.concat([condition1, condition2, condition3, condition4, condition5], axis=1)
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(all_conditions.T)

    # PCAの実行
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(scaled_data)
    
    # PCA結果をデータフレームに格納
    pca_df = pd.DataFrame(pca_result, columns=['PC1', 'PC2'])
    pca_df['Condition'] = [group_names[i] for i in range(len(group_names)) for _ in range(condition1.shape[1])]
    
    # PCAプロットの作成
    st.write("PCAプロット:")
    colors = {group_names[i]: color for i, color in enumerate(['blue', 'green', 'red', 'purple', 'orange'])}
    fig = px.scatter(
        pca_df,
        x='PC1',
        y='PC2',
        color='Condition',
        color_discrete_map=colors,
        labels={'PC1': 'Principal Component 1', 'PC2': 'Principal Component 2'},
        title='PCA Plot'
    )

    for condition in pca_df['Condition'].unique():
        indices = pca_df['Condition'] == condition
        cov = np.cov(pca_df.loc[indices, ['PC1', 'PC2']].T)
        center = pca_df.loc[indices, ['PC1', 'PC2']].mean()
        eigvals, eigvecs = np.linalg.eigh(cov)
        order = eigvals.argsort()[::-1]
        eigvals, eigvecs = eigvals[order], eigvecs[:, order]
        theta = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))
        width, height = 2 * 1.96 * np.sqrt(eigvals)

        fig.add_shape(
            type="circle",
            xref="x", yref="y",
            x0=center['PC1'] - width/2, y0=center['PC2'] - height/2,
            x1=center['PC1'] + width/2, y1=center['PC2'] + height/2,
            line_color=colors[condition],
            fillcolor=colors[condition],
            opacity=0.2
        )

    st.plotly_chart(fig)

    # 結果をExcelファイルとして出力
    output_path = 'welch_test_results.xlsx'
    with pd.ExcelWriter(output_path) as writer:
        data.to_excel(writer, sheet_name='Data with p-values', index=False)
        matrix.to_excel(writer, sheet_name='Significant Counts')
        stats.to_excel(writer, sheet_name='Statistics')
        pca_df.to_excel(writer, sheet_name='PCA Results', index=False)

    # ダウンロードリンクを作成
    with open(output_path, "rb") as file:
        btn = st.download_button(
            label="結果をダウンロード",
            data=file,
            file_name=output_path,
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

