# 聚丙烯改性分类预测与文献依据网站 v2

本项目是本地可运行的 Streamlit 网站，含配方试算、预测依据文献、别名检索和模型适用范围。网站显示的相近文献是用于核查的实测类比记录；模型预测由整个训练集共同学习，不归因于单篇文献。

## 一、放到 D:\ppdatabase 并用 VS Code 打开

1. 下载 `聚丙烯改性分类预测网站_v2.zip`。
2. 在文件资源管理器中打开“下载”文件夹，右键 ZIP 文件，选择“全部解压…”。
3. 将解压目标设为：

   `D:\ppdatabase\pp_taxonomy_prediction_v2`

   如果 Windows 提示要建立新文件夹，选择“是”。解压后，`app.py` 应直接位于 `D:\ppdatabase\pp_taxonomy_prediction_v2\app.py`，不要多套一层同名目录。

4. 打开 VS Code，选择“文件”→“打开文件夹…”，选择：

   `D:\ppdatabase\pp_taxonomy_prediction_v2`

5. 左侧文件列表应能看到 `app.py`、`requirements.txt`、`README_CN.md`、`data`、`outputs` 和 `scripts`。

## 二、安装依赖并启动

在 VS Code 菜单选择“终端”→“新建终端”。先确认路径：

```powershell
Get-Location
```

输出应为 `D:\ppdatabase\pp_taxonomy_prediction_v2`。随后逐条执行：

```powershell
python -m venv .venv
```

```powershell
.\.venv\Scripts\python.exe -m pip install -r .\requirements.txt
```

```powershell
.\.venv\Scripts\python.exe -m streamlit run .\app.py
```

不需要运行 `Activate.ps1`。如果终端报告找不到文件，请先运行：

```powershell
Set-Location D:\ppdatabase\pp_taxonomy_prediction_v2
```

首次运行若 Streamlit 问是否接收邮件，直接按 Enter 留空即可。浏览器打开终端中显示的网址，通常为 `http://localhost:8501`。网站运行期间不要关闭终端；结束时按 `Ctrl+C`。

## 三、预测结果怎样显示文献依据

点击“检查配方并预测”后，页面会显示：

1. 拉伸强度实验性估计值。
2. 模型训练范围、验证 MAE 和适用分组。
3. 与本次配方最相近的 3 条训练文献实测记录。
4. 每条记录可展开查看题名、DOI、样品配方、实测拉伸强度、标准、测试/加工条件、文献文件名，以及强度、配方和方法在复核表中的原文页码定位。

排序先在同一模型适用分组内进行，再按规范物质组成、物质家族、组分功能、加工路线、测试标准及有报告的数值条件计算距离。页面展示“较接近/有一定相似/扩展参考”，不把距离说成概率或置信度。文献标题和 DOI 可供用户打开核对。

**解释边界：**模型综合 113 条记录、20 篇文献学习，不能诚实地说某一个估计值“由某一篇文献产生”。页面列出的 3 篇是最相似的实测证据，帮助用户对照输入与已有实验，不代表单篇文献归因。若同一添加物只出现在薄膜等其他形态，网页会另列为“未参与当前模型训练”的文献并说明形态差异。

## 四、本轮模型验证摘要

| 训练分组 | 数据量 | 均值基线 MAE | 随机森林 MAE | 随机森林 RMSE | 随机森林 R² |
|---|---:|---:|---:|---:|---:|
| 形态复核后通用块状候选 | 139 条 / 26 篇 | 6.43 MPa | 5.68 MPa | 7.80 MPa | 0.048 |
| 形态未报告/未确认且均聚或未明确 PP | 113 条 / 20 篇 | 6.83 MPa | 5.49 MPa | 6.87 MPa | 0.214 |
| 已明确注塑/模压块材 | 21 条 / 5 篇 | 6.35 MPa | 9.21 MPa | 11.17 MPa | -0.924 |

验证使用留一文献交叉验证，同一篇文献的记录不同时进入训练和测试。形态复核时将 SRC_P119 的 3 条薄膜数据和 SRC_P169 的 4 条无纺布/纤维复合数据移出通用块材范围。网页只对第二行所列分组提供实验性试算；形态明确、共聚/特殊 PP、其他性能暂时只支持检索。

## 五、文件说明与公开部署

- 正常运行网页需要 `app.py`、`requirements.txt`、`data` 和 `outputs`。
- ZIP 不附带正式 SQLite 数据库及 Excel 复核表，避免误传到公开仓库。它们只在本地重新训练/重建数据时需要，可从正式项目复制到 `data`；不要公开提交这两类原始文件。
- `data/literature_measurements.csv` 含整理后的测量、配方、文献和测试字段，文献检索会向网站访问者展示这些信息。推送到公开 GitHub / Streamlit 前，请确认你允许公开这些整理数据。
- 若将来重新训练，必须重新运行整篇文献留出验证；不要仅替换模型文件。
- 当前数据库快照的 `reviewed` 标记全为 false，程序没有擅自修改。若人工复核结论尚未写回数据库标记，网页会照实展示数据库原标记。

详细数值和限制见 `MODEL_EVALUATION_CN.md`。`outputs/taxonomy_model_metrics.csv` 和 `outputs/taxonomy_oof_predictions.csv` 保存按分组验证的指标与逐条留出预测。
