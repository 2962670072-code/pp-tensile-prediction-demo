from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"
MODEL_PATH = OUTPUT_DIR / "experimental_ridge_model.joblib"
FEATURES_PATH = DATA_DIR / "feature_manifest.json"
TRAINING_DATA_PATH = DATA_DIR / "model_input_features_v0.csv"
METRICS_PATH = OUTPUT_DIR / "metrics.json"

st.set_page_config(
    page_title="聚丙烯拉伸强度预测原型",
    page_icon="🧪",
    layout="wide",
)

NUMERIC_LABELS = {
    "test_speed_mm_min": "拉伸测试速度（mm/min）",
    "extrusion_temp_min_C": "挤出温度下限（°C）",
    "extrusion_temp_max_C": "挤出温度上限（°C）",
    "extruder_screw_speed_rpm": "挤出机螺杆转速（rpm）",
    "injection_temp_min_C": "注塑温度下限（°C）",
    "injection_temp_max_C": "注塑温度上限（°C）",
    "mold_temp_C": "模具温度（°C）",
    "post_anneal_temp_C": "退火温度（°C）",
    "post_anneal_time_h": "退火时间（h）",
    "conditioning_temp_C": "试样调节温度（°C）",
    "conditioning_rh_pct": "试样调节相对湿度（%）",
    "conditioning_time_h": "试样调节时间（h）",
    "mixing_temp_min_C": "混合温度下限（°C）",
    "mixing_temp_max_C": "混合温度上限（°C）",
    "mixer_speed_rpm": "混合设备转速（rpm）",
    "mixing_time_min": "混合时间（min）",
    "injection_pressure_bar": "注射压力（bar）",
    "holding_time_s": "保压时间（s）",
}

CATEGORY_LABELS = {
    "processing_route": "成型方式",
    "compounding_route": "共混/造粒方式",
    "processing_variant": "具体改性/加工路线",
    "test_standard": "拉伸测试标准",
    "test_temperature_status": "测试温度报告情况",
}

CATEGORY_TRANSLATIONS = {
    "not_reported": "未报告/未提供",
    "injection_molding": "常规注塑",
    "micro_injection_molding": "微注塑",
    "twin_screw_compounding": "双螺杆共混",
    "twin_screw_extrusion": "双螺杆挤出",
    "carbon_fiber_interphase_treatment_pair": "碳纤维界面处理对照组",
    "direct_compounding_no_compatibilizer": "直接共混，不加相容剂",
    "direct_compounding_with_PP_g_m_TMI": "直接共混，加入 PP-g-m-TMI",
    "micro_injection_then_annealing": "微注塑后退火",
    "neat_control": "纯 PP 对照",
    "standard_extruded_compound": "常规挤出共混",
    "three_step_masterbatch_no_PP_g_m_TMI": "三步母粒法，不加 PP-g-m-TMI",
    "three_step_masterbatch_with_PP_g_m_TMI": "三步母粒法，加入 PP-g-m-TMI",
    "reported_room_temperature": "报告为室温",
    "ASTM D638": "ASTM D638",
    "ASTM D638-2014": "ASTM D638-2014",
    "GB/T 1040.2-2006": "GB/T 1040.2-2006",
    "GB/T 1040—2018（按原文；分册未报告）": "GB/T 1040—2018（原文未报告分册）",
    "ISO 527-1:2019": "ISO 527-1:2019",
}

COMPONENT_LABELS = {
    "Antioxidant B215": "抗氧剂 B215（Antioxidant B215）",
    "PP-g-m-TMI": "PP-g-m-TMI 相容剂",
    "Al effect pigment, grade/particle size not identified": "铝效应颜料（牌号/粒径未报告）",
    "Barium sulfate": "硫酸钡（BaSO₄）",
    "Calcium carbonate": "碳酸钙（CaCO₃）",
    "Fe3+/硬脂酸改性蒙脱土 SMMT": "Fe³⁺/硬脂酸改性蒙脱土 SMMT",
    "Na-MMT PGW": "钠基蒙脱土 Na-MMT PGW",
    "Talc": "滑石粉",
    "纸浆粉": "纸浆粉",
    "Bromine-phosphorus-nitrogen flame-retardant blend": "溴-磷-氮复合阻燃剂",
    "Calcium stearate": "硬脂酸钙",
    "EBS lubricant": "EBS 润滑剂",
    "PP BX3900": "PP BX3900（聚丙烯基体）",
    "PP HP500N-Z homopolymer": "均聚 PP HP500N-Z（聚丙烯基体）",
    "PP M1100": "PP M1100（聚丙烯基体）",
    "均聚PP基体": "均聚 PP（具体牌号未报告）",
    "嵌段共聚PP BB110": "嵌段共聚 PP BB110（聚丙烯基体）",
    "等规聚丙烯 T30S": "等规 PP T30S（聚丙烯基体）",
    "TMB-5": "成核剂 TMB-5",
    "Antioxidant 1010": "抗氧剂 1010",
    "T-700 desized carbon fiber": "脱浆 T-700 碳纤维",
    "n-butylphenyl-functionalized T-700 carbon fiber": "正丁基苯基官能化 T-700 碳纤维",
    "Antioxidant 168": "抗氧剂 168",
}

ROLE_LABELS = {
    "matrix": "PP基体",
    "filler": "填料",
    "reinforcement": "增强材料",
    "flame retardant": "阻燃剂",
    "compatibilizer": "相容剂",
    "antioxidant": "抗氧剂/稳定剂",
    "primary antioxidant": "主抗氧剂",
    "secondary antioxidant": "辅助抗氧剂",
    "lubricant": "润滑剂",
    "nucleating_agent": "成核剂",
    "effect pigment": "颜料",
}


@st.cache_data
def load_inputs():
    manifest = json.loads(FEATURES_PATH.read_text(encoding="utf-8"))
    training = pd.read_csv(TRAINING_DATA_PATH, encoding="utf-8-sig")
    metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8")) if METRICS_PATH.exists() else {}
    return manifest, training, metrics


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


def category_display(value: str) -> str:
    return CATEGORY_TRANSLATIONS.get(value, value)


def parse_optional_number(raw: str, label: str):
    raw = raw.strip()
    if not raw:
        return np.nan, None
    try:
        value = float(raw.replace(",", "."))
    except ValueError:
        return np.nan, f"“{label}”需要填写数字，例如 200 或 10.5。"
    if not np.isfinite(value):
        return np.nan, f"“{label}”不是有效数字。"
    return value, None


def component_group(role: str) -> str:
    if role == "matrix":
        return "PP基体"
    if role in ("filler", "reinforcement"):
        return "填料与增强材料"
    return "助剂与其他组分"


st.title("聚丙烯改性配方 → 拉伸强度预测")
st.caption("本地运行的研究原型｜当前只预测拉伸强度，不预测冲击强度、弯曲性能等其他指标")

st.warning(
    "这是27条记录、6篇文献训练出的试验模型。留一文献验证中，Ridge 的 MAE 为 5.53 MPa，"
    "高于简单平均值基线的 4.68 MPa，且 R² 为负。界面显示的数值目前不能作为配方设计、"
    "采购或生产决策依据，也不代表单个预测的误差范围。"
)

if not MODEL_PATH.exists():
    st.error("没有找到已训练模型文件。请先在本文件夹运行 scripts/train_baseline.py。")
    st.stop()
if not FEATURES_PATH.exists() or not TRAINING_DATA_PATH.exists():
    st.error("没有找到 data 文件夹中的特征清单或训练数据。请确认 app.py 放在试跑项目根目录。")
    st.stop()

try:
    manifest, training, metrics = load_inputs()
    model = load_model()
except Exception as exc:
    st.error(f"读取模型或训练文件失败：{exc}")
    st.info("请使用同一个虚拟环境重新运行训练脚本，再启动本程序。")
    st.stop()

numeric_features = manifest["numeric_features"]
categorical_features = manifest["categorical_features"]
component_map = manifest["component_feature_map"]
component_columns = [item["feature_column"] for item in component_map]
process_numeric_features = [c for c in numeric_features if c not in component_columns]

with st.sidebar:
    st.subheader("当前模型范围")
    st.write("目标属性：拉伸强度（MPa）")
    st.write("训练范围：27条注塑体相 PP 拉伸记录，来自6篇文献")
    st.write("材料选项：仅限本原型下拉框列出的组分")
    if metrics:
        ridge = metrics.get("ridge", {})
        baseline = metrics.get("mean_baseline", {})
        st.metric("留一文献 MAE", f"{ridge.get('MAE_MPa', float('nan')):.2f} MPa")
        st.metric("简单基线 MAE", f"{baseline.get('MAE_MPa', float('nan')):.2f} MPa")
        st.caption("MAE 是整体交叉验证误差，不是本次输入的置信区间。")

st.markdown(
    "先按**质量百分数（wt%）**填写配方。PP 基体和所有添加剂的质量百分数合计必须为 **100%**。"
    "再填写尽可能完整的加工、成型和测试条件。"
)

with st.form("recipe_form"):
    st.header("1. 配方组成")
    st.caption(
        "当前原型只认识本试跑数据中已有的23种组分。若你的物质不在列表中，先不要用相似名称替代；"
        "需要先把该物质的规范名称映射并加入训练数据。"
    )

    percentage_inputs = {}
    groups = ["PP基体", "填料与增强材料", "助剂与其他组分"]
    for group_name in groups:
        with st.expander(group_name, expanded=(group_name == "PP基体")):
            group_items = [
                item for item in component_map
                if component_group(item.get("role", "")) == group_name
            ]
            cols = st.columns(2)
            for idx, item in enumerate(group_items):
                colname = item["feature_column"]
                raw_name = item["component_name"]
                display_name = COMPONENT_LABELS.get(raw_name, raw_name)
                with cols[idx % 2]:
                    percentage_inputs[colname] = st.number_input(
                        display_name,
                        min_value=0.0,
                        max_value=100.0,
                        value=0.0,
                        step=0.1,
                        key=f"pct_{colname}",
                    )

    total_pct = float(sum(percentage_inputs.values()))
    matrix_pct = sum(
        percentage_inputs[item["feature_column"]]
        for item in component_map if item.get("role") == "matrix"
    )
    st.write(f"当前配方合计：**{total_pct:.1f} wt%**；PP基体：**{matrix_pct:.1f} wt%**")

    st.header("2. 加工与成型条件")
    st.caption("未报告或暂时不知道的项目可以留空；模型会用训练数据中的中位数补齐，并提示缺失情况。")
    cat_input = {}
    cat_col1, cat_col2 = st.columns(2)
    for idx, field in enumerate(("compounding_route", "processing_route", "processing_variant")):
        values = sorted(training[field].dropna().astype(str).unique().tolist())
        if "not_reported" not in values:
            values.insert(0, "not_reported")
        with (cat_col1 if idx % 2 == 0 else cat_col2):
            cat_input[field] = st.selectbox(
                CATEGORY_LABELS[field],
                options=values,
                index=values.index("not_reported") if "not_reported" in values else 0,
                format_func=category_display,
                key=f"cat_{field}",
            )

    process_groups = {
        "挤出/共混条件": [
            "extrusion_temp_min_C", "extrusion_temp_max_C",
            "extruder_screw_speed_rpm", "mixing_temp_min_C", "mixing_temp_max_C",
            "mixer_speed_rpm", "mixing_time_min",
        ],
        "注塑条件": [
            "injection_temp_min_C", "injection_temp_max_C", "mold_temp_C",
            "injection_pressure_bar", "holding_time_s",
        ],
        "退火与试样调节条件": [
            "post_anneal_temp_C", "post_anneal_time_h",
            "conditioning_temp_C", "conditioning_rh_pct", "conditioning_time_h",
        ],
    }
    numeric_raw = {}
    for group_name, fields in process_groups.items():
        with st.expander(group_name, expanded=False):
            cols = st.columns(2)
            for idx, field in enumerate(fields):
                with cols[idx % 2]:
                    numeric_raw[field] = st.text_input(
                        NUMERIC_LABELS[field],
                        value="",
                        placeholder="可留空；例如 200",
                        key=f"num_{field}",
                    )

    st.header("3. 拉伸测试条件")
    test_cat1, test_cat2 = st.columns(2)
    for idx, field in enumerate(("test_standard", "test_temperature_status")):
        values = sorted(training[field].dropna().astype(str).unique().tolist())
        if "not_reported" not in values:
            values.insert(0, "not_reported")
        with (test_cat1 if idx == 0 else test_cat2):
            cat_input[field] = st.selectbox(
                CATEGORY_LABELS[field],
                options=values,
                index=values.index("not_reported") if "not_reported" in values else 0,
                format_func=category_display,
                key=f"cat_{field}",
            )

    test_speed_col, _ = st.columns(2)
    with test_speed_col:
        numeric_raw["test_speed_mm_min"] = st.text_input(
            NUMERIC_LABELS["test_speed_mm_min"],
            value="",
            placeholder="可留空；例如 10",
            key="num_test_speed_mm_min",
        )

    submitted = st.form_submit_button("计算试验性预测", type="primary", use_container_width=True)

if submitted:
    errors = []
    if abs(total_pct - 100.0) > 0.05:
        errors.append(f"所有组分合计为 {total_pct:.1f} wt%，需要合计为100 wt%。")
    if matrix_pct <= 0:
        errors.append("PP基体比例必须大于0 wt%。")

    numeric_values = {}
    for field, raw in numeric_raw.items():
        value, error = parse_optional_number(raw, NUMERIC_LABELS[field])
        numeric_values[field] = value
        if error:
            errors.append(error)

    if errors:
        for error in errors:
            st.error(error)
        st.stop()

    row = {field: np.nan for field in numeric_features}
    for item in component_map:
        row[item["feature_column"]] = float(percentage_inputs[item["feature_column"]])
    row.update(numeric_values)
    row.update({field: "not_reported" for field in categorical_features})
    for field, value in cat_input.items():
        # If the training set never contained an unreported value for this
        # category, pass a true missing value so the saved imputer uses the
        # training mode instead of creating an unseen category.
        if value == "not_reported" and not (training[field].astype(str) == "not_reported").any():
            row[field] = np.nan
        else:
            row[field] = value

    input_frame = pd.DataFrame([row], columns=numeric_features + categorical_features)

    warnings = []
    if any(pd.isna(value) for value in numeric_values.values()):
        missing_count = sum(pd.isna(value) for value in numeric_values.values())
        warnings.append(
            f"有 {missing_count} 项加工/测试数值条件留空，模型会用训练数据中位数补齐。"
        )

    for item in component_map:
        colname = item["feature_column"]
        value = float(percentage_inputs[colname])
        if value <= 0:
            continue
        observed = pd.to_numeric(training[colname], errors="coerce").dropna()
        positive = observed[observed > 0]
        if len(positive) < 3:
            warnings.append(
                f"{COMPONENT_LABELS.get(item['component_name'], item['component_name'])} "
                f"在训练集中仅有 {len(positive)} 条非零记录，相关预测特别不稳定。"
            )
        if len(positive) and (value < positive.min() or value > positive.max()):
            warnings.append(
                f"{COMPONENT_LABELS.get(item['component_name'], item['component_name'])} "
                f"的输入比例 {value:g}% 超出训练数据的非零范围 "
                f"{positive.min():g}%–{positive.max():g}%。"
            )

    for field, value in numeric_values.items():
        if pd.isna(value):
            continue
        observed = pd.to_numeric(training[field], errors="coerce").dropna()
        if len(observed) and (value < observed.min() or value > observed.max()):
            warnings.append(
                f"{NUMERIC_LABELS[field]}={value:g} 超出训练数据范围 "
                f"{observed.min():g}–{observed.max():g}。"
            )

    try:
        prediction = float(model.predict(input_frame)[0])
    except Exception as exc:
        st.error(f"模型计算失败：{exc}")
        st.stop()

    st.divider()
    st.header("预测结果")
    st.metric("试验性预测拉伸强度", f"{prediction:.2f} MPa")
    st.caption(
        "这是当前小样本模型给出的单个数值；系统没有经过校准的单次预测置信区间。"
        "留一文献 MAE 只能描述整体验证误差，不能直接当成本次预测的误差范围。"
    )

    recipe_rows = [
        {
            "组分": COMPONENT_LABELS.get(item["component_name"], item["component_name"]),
            "质量百分数（wt%）": float(percentage_inputs[item["feature_column"]]),
        }
        for item in component_map
        if float(percentage_inputs[item["feature_column"]]) > 0
    ]
    st.subheader("本次输入的配方")
    st.dataframe(pd.DataFrame(recipe_rows), hide_index=True, use_container_width=True)

    if warnings:
        st.warning("本次输入的适用性提示：\n\n" + "\n\n".join(f"- {message}" for message in warnings))
    else:
        st.info(
            "输入值没有触发简单的范围检查，但这不代表模型已经可靠；"
            "当前模型在总体验证中仍未优于简单平均值基线。"
        )

    export_rows = [
        {"类别": "配方", "项目": r["组分"], "输入值": r["质量百分数（wt%）"]}
        for r in recipe_rows
    ]
    for field, value in numeric_values.items():
        export_rows.append({"类别": "加工/测试数值条件", "项目": NUMERIC_LABELS[field], "输入值": value})
    for field, value in cat_input.items():
        export_rows.append({"类别": "加工/测试分类条件", "项目": CATEGORY_LABELS[field], "输入值": category_display(value)})
    export_rows.append({"类别": "预测结果", "项目": "拉伸强度（MPa）", "输入值": prediction})
    st.download_button(
        "下载本次输入与预测结果 CSV",
        data=pd.DataFrame(export_rows).to_csv(index=False, encoding="utf-8-sig"),
        file_name="pp_tensile_prediction_experimental.csv",
        mime="text/csv",
    )
