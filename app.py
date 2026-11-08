from __future__ import annotations
import difflib, json, re, unicodedata
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import streamlit as st

ROOT=Path(__file__).resolve().parent
DATA=ROOT/'data'; OUT=ROOT/'outputs'
st.set_page_config(page_title='聚丙烯改性预测与文献依据',page_icon='🧪',layout='wide')

STANDARD_LABELS={
    'not_reported':'未报告',
    'ASTM D638':'ASTM D638（塑料拉伸测试标准）',
    'GB/T 1040-2006':'GB/T 1040-2006（塑料拉伸测试标准）',
    'GB/T 1040.2-2022':'GB/T 1040.2-2022（塑料拉伸测试标准）',
    'GB/T 1040.1-2006':'GB/T 1040.1-2006（塑料拉伸测试标准）',
    'GB/T 1040.1-2018':'GB/T 1040.1-2018（塑料拉伸测试标准）',
    'GB/T 1040-1992':'GB/T 1040-1992（塑料拉伸测试标准）',
    'EN ISO 527 (tensile) / ASTM D790-17 (flexural)':'EN ISO 527（拉伸）/ ASTM D790-17（弯曲）',
    'GB/T 1040.1—2018 / ISO 527-1:2012':'GB/T 1040.1—2018 / ISO 527-1:2012（塑料拉伸测试标准）',
    'ISO 527-2:2018':'ISO 527-2:2018（塑料拉伸测试标准）',
    'ISO 527-1':'ISO 527-1（塑料拉伸测试标准）',
    'ISO 527-1/3':'ISO 527-1/3（塑料拉伸测试标准）',
}

def display_standard(value):
    raw=str(value or '').strip()
    if not raw or raw.lower() in {'not_reported','not reported','none','nan'}:return '未报告'
    return STANDARD_LABELS.get(raw,raw)

def display_route(value):
    raw=str(value or '').strip()
    if not raw or raw.lower() in {'not_reported_or_unclassified','not reported or unclassified','none','nan'}:return '未报告或未分类'
    names={'injection_molding':'注塑成型','extrusion':'挤出成型','melt_blending':'熔融共混','compression_molding':'模压成型'}
    return ' + '.join(names.get(part.strip(),part.strip().replace('_',' ')) for part in raw.split('+'))


@st.cache_data
def load_data():
    manifest=json.loads((OUT/'taxonomy_model_manifest.json').read_text(encoding='utf-8'))
    metrics=json.loads((OUT/'taxonomy_model_metrics.json').read_text(encoding='utf-8'))
    refs=pd.read_csv(DATA/'model_reference_records.csv',encoding='utf-8-sig',low_memory=False).fillna('')
    catalog=json.loads((DATA/'substance_catalog.json').read_text(encoding='utf-8'))
    measurements=pd.read_csv(DATA/'literature_measurements.csv',encoding='utf-8-sig',low_memory=False).fillna('')
    inventory=pd.read_csv(DATA/'property_inventory.csv',encoding='utf-8-sig').fillna('')
    taxonomy=json.loads((DATA/'classification_taxonomy_v1.json').read_text(encoding='utf-8'))
    return manifest,metrics,refs,catalog,measurements,inventory,taxonomy

@st.cache_resource
def load_model():
    return joblib.load(OUT/'experimental_taxonomy_rf.joblib')

def norm(s):
    s=unicodedata.normalize('NFKC',str(s or '')).strip().lower()
    return re.sub(r'[\s_\-‐‑–—·,，。()（）\[\]{}]','',s)

def alias_index(catalog):
    out={}
    for item in catalog:
        for value in [item.get('id'),item.get('name'),item.get('formula'),*item.get('aliases',[])]:
            if value: out.setdefault(norm(value),item)
    return out

def match_substance(value,index):
    key=norm(value)
    if key in index:return index[key],'exact'
    close=difflib.get_close_matches(key,list(index),n=4,cutoff=.72) if key else []
    return ([index[k] for k in close],'suggestions') if close else (None,'none')

def parse_optional(value):
    if not str(value or '').strip():return np.nan
    x=float(str(value).replace(',','.'))
    if not np.isfinite(x):raise ValueError
    return x

def format_formula(comps):
    out=[]
    for c in comps:
        name=c.get('material_name') or c.get('substance_name') or c.get('substance_id') or '未知物质'
        wt=c.get('wt_pct')
        out.append(f"{name} {float(wt):g} wt%" if wt not in (None,'') else name)
    return '；'.join(out)

def rank_literature(input_comps,conditions,refs,role_map):
    def vec(comps):
        sid={};family={};function={}
        for c in comps:
            wt=float(c.get('wt_pct') or 0)
            k=str(c.get('substance_id') or 'unknown');sid[k]=sid.get(k,0)+wt
            k='family_wt__'+str(c.get('substance_family') or 'unknown');family[k]=family.get(k,0)+wt
            role=' '.join(str(c.get('role') or 'unknown').replace('_',' ').split()).lower()
            fn=role_map.get(role,role if role in role_map.values() else 'other_function')
            k='function_wt__'+fn;function[k]=function.get(k,0)+wt
        return sid,family,function
    def l1(a,b):return min(1.0,sum(abs(a.get(k,0)-b.get(k,0)) for k in set(a)|set(b))/200.0)
    us,uf,un=vec(input_comps)
    condition_cols=[('test_standard','test_standard','cat'),('processing_route','processing_route','cat'),
        ('test_speed_mm_min','test_speed_mm_min','num'),('test_temperature_C','test_temperature_C','num'),
        ('relative_humidity_percent','relative_humidity_percent','num'),('process_temperature_min_C','process_temperature_min_C','num'),
        ('process_temperature_max_C','process_temperature_max_C','num'),('process_screw_speed_rpm','process_screw_speed_rpm','num'),
        ('process_duration_min_max','process_duration_min_max','num')]
    ranges={}
    for _,col,kind in condition_cols:
        if kind=='num' and col in refs:
            v=pd.to_numeric(refs[col],errors='coerce').dropna()
            ranges[col]=max(float(v.max()-v.min()),1e-9) if len(v)>1 else 1.0
    results=[]
    for _,r in refs.iterrows():
        try:comps=json.loads(r.get('components_json') or '[]')
        except Exception:comps=[]
        rs,rf,rn=vec(comps)
        comp_dist=.5*l1(us,rs)+.3*l1(uf,rf)+.2*l1(un,rn)
        dcond=[];exact=0;coverage=0
        for ik,rk,kind in condition_cols:
            a=conditions.get(ik);b=r.get(rk)
            if kind=='cat':
                if not a or not b:continue
                coverage+=1;same=str(a).strip().lower()==str(b).strip().lower();dcond.append(0 if same else 1);exact+=int(same)
            else:
                try:a=float(a);b=float(b)
                except (ValueError,TypeError):continue
                if not np.isfinite(a) or not np.isfinite(b):continue
                coverage+=1;dcond.append(min(1,abs(a-b)/ranges.get(rk,1.0)))
        condition_dist=float(np.mean(dcond)) if dcond else np.nan
        total=.75*comp_dist+.25*(condition_dist if np.isfinite(condition_dist) else comp_dist)
        shared_ids=(set(us)&set(rs)) - {'SUB_PP'}
        shared=[c.get('substance_name') or c.get('substance_id') for c in input_comps if c.get('substance_id') in shared_ids]
        item=r.to_dict();item['_components']=comps;item['_composition_distance']=comp_dist
        item['_condition_distance']=condition_dist;item['_condition_coverage']=coverage;item['_condition_exact_matches']=exact;item['_rank_distance']=total
        item['_shared_additives']=list(dict.fromkeys(shared));item['_has_exact_additive']=bool(shared_ids)
        results.append(item)
    results.sort(key=lambda x:(not x['_has_exact_additive'],-len(x['_shared_additives']),x['_rank_distance']))
    # Show distinct papers rather than several formulation rows from a single paper.
    unique=[];seen=set()
    for item in results:
        source=str(item.get('source_id') or '')
        if source in seen:continue
        unique.append(item);seen.add(source)
        if len(unique)==3:break
    return unique

def show_evidence(r,n):
    title=str(r.get('paper_title') or r.get('paper_title_db') or '题名未收录')
    sample=str(r.get('sample_label') or r.get('source_sample') or '样品编号未报告')
    value=r.get('strength_mpa',r.get('value',''))
    doi=str(r.get('doi') or r.get('doi_db') or '').strip()
    d=r['_rank_distance']
    if r.get('_has_exact_additive'):label='含相同添加物的文献'
    else:label='未含相同添加物，按类别/比例作参考'
    with st.expander(f"{n}. {title}｜{sample}｜实测 {value} MPa｜{label}",expanded=st.session_state.get('show_related_literature',False)):
        st.markdown(f"**DOI：** [{doi}](https://doi.org/{doi})" if doi else '**DOI：** 数据库未收录')
        st.write(f"**记录：** 文献 {r.get('source_id','')}；测量 {r.get('measurement_id','')}")
        st.write(f"**配方：** {format_formula(r['_components'])}")
        st.write('**与本次输入相同的添加物：** '+('、'.join(r['_shared_additives']) if r.get('_shared_additives') else '无；此条是模型特征距离较近的扩展参考，不是同一添加物的直接文献证据。'))
        st.write(f"**实测拉伸强度：** {value} {r.get('unit','MPa')}；标准：{display_standard(r.get('test_standard') or r.get('standard'))}；速度：{r.get('test_speed_mm_min') or '未报告'} mm/min；测试温度：{r.get('test_temperature_C') or '未报告'} °C")
        st.write(f"**加工/试样：** {display_route(r.get('processing_route'))}；加工温度 {r.get('process_temperature_min_C') or '未报告'}–{r.get('process_temperature_max_C') or '未报告'} °C；试样 {r.get('specimen_json') or r.get('specimen_description') or '未报告'}")
        locs=[('文献文件',r.get('source_locator') or r.get('数据库文献定位')),('强度原文定位',r.get('拉伸强度原文定位') or r.get('evidence_locator') or r.get('review_evidence')),('配方原文定位',r.get('配方原文定位')),('加工/测试方法定位',r.get('加工/制样定位') or r.get('condition_override_evidence'))]
        for label,value in locs:
            if value:st.write(f'**{label}：** {value}')
        st.caption(f"排序信息：配方组成距离 {r['_composition_distance']:.3f}（越低越近）；可比较条件 {r['_condition_coverage']} 项，分类条件一致 {r['_condition_exact_matches']} 项。距离不是预测概率或置信度。")

def find_same_material_outside_scope(input_components,allmeas,training_ids):
    wanted={c.get('substance_id') for c in input_components if c.get('substance_id') and c.get('substance_id')!='SUB_PP'}
    if not wanted:return []
    tensile=allmeas[allmeas.property.astype(str).isin(['tensile_strength','tensile_strength_reported'])]
    out=[]
    for _,r in tensile.iterrows():
        if str(r.get('measurement_id')) in training_ids:continue
        try:comps=json.loads(r.get('components_json') or '[]')
        except Exception:comps=[]
        ids={c.get('substance_id') for c in comps}
        overlap=wanted&ids
        if not overlap:continue
        input_amount={}
        ref_amount={}
        for c in input_components:
            sid=str(c.get('substance_id') or '')
            if sid and sid!='SUB_PP':input_amount[sid]=input_amount.get(sid,0)+float(c.get('wt_pct') or 0)
        for c in comps:
            sid=str(c.get('substance_id') or '')
            if sid and sid!='SUB_PP':ref_amount[sid]=ref_amount.get(sid,0)+float(c.get('wt_pct') or 0)
        comp_distance=min(1.0,sum(abs(input_amount.get(k,0)-ref_amount.get(k,0)) for k in set(input_amount)|set(ref_amount))/200.0)
        item=r.to_dict();item['_components']=comps;item['_overlap']=[c.get('substance_name') or c.get('substance_id') for c in input_components if c.get('substance_id') in overlap];item['_distance']=comp_distance
        sid=str(item.get('source_id') or '');title=str(item.get('paper_title') or '').lower()
        if sid=='SRC_P119' or ('silica' in title and 'film' in title) or ('硅' in title and '薄膜' in title):item['_scope_note']='薄膜/片材；形态与当前模型组不同'
        elif sid=='SRC_P169' or 'nonwoven' in title or '无纺布' in title:item['_scope_note']='无纺布/纤维复合；形态与当前模型组不同'
        else:item['_scope_note']='未进入当前模型训练组；请核对试样形态和测试条件'
        out.append(item)
    out.sort(key=lambda z:(-len(z['_overlap']),z['_distance']))
    unique=[];seen=set()
    for item in out:
        source=str(item.get('source_id') or '')
        if source in seen:continue
        unique.append(item);seen.add(source)
        if len(unique)==3:break
    return unique

def show_outside_evidence(r,n):
    title=str(r.get('paper_title') or '题名未收录'); doi=str(r.get('doi') or '').strip()
    with st.expander(f"{n}. {title}｜{r.get('source_sample') or '样品未标明'}｜{r.get('_scope_note')}",expanded=st.session_state.get('show_related_literature',False)):
        if doi:st.markdown(f"**DOI：** [{doi}](https://doi.org/{doi})")
        st.write(f"**同一物质：** {'、'.join(r['_overlap'])}；**文献/测量：** {r.get('source_id')} / {r.get('measurement_id')}")
        st.write(f"**文献实测：** {r.get('value')} {r.get('unit')}；**配方：** {format_formula(r['_components'])}")
        st.write(f"**测试标准：** {display_standard(r.get('standard'))}；**条件：** {r.get('conditions_json') or '未报告'}")
        st.write(f"**试样信息：** {r.get('specimen_json') or '未报告'}")
        if r.get('source_locator'):st.write(f"**文献文件：** {r.get('source_locator')}")
        if r.get('evidence_locator'):st.write(f"**原文数值定位：** {r.get('evidence_locator')}")
        st.caption('此文献只作为同物质的外部参考，没有用于本次模型拟合或上方相近记录排序。')

st.title('聚丙烯改性配方预测与文献依据')
st.caption('分类先行的研究原型｜当前仅对限定分组试算拉伸强度；其他性能支持检索')
manifest_path=OUT/'taxonomy_model_manifest.json'
needed=[OUT/'experimental_taxonomy_rf.joblib',manifest_path,OUT/'taxonomy_model_metrics.json',DATA/'input_ranges.json',DATA/'model_reference_records.csv',DATA/'literature_measurements.csv',DATA/'substance_catalog.json',DATA/'property_inventory.csv',DATA/'classification_taxonomy_v1.json']
if not all(p.exists() for p in needed):st.error('网站文件不完整；请确认完整解压 ZIP。');st.stop()
manifest,metrics,refs,catalog,allmeas,inventory,tax=load_data();estimator=load_model();idx=alias_index(catalog);v=manifest['validation_metrics']
ranges=json.loads((DATA/'input_ranges.json').read_text(encoding='utf-8'))
# Count how often each exact substance occurs in the active model cohort.
training_substance_sources={}
for _,record in refs.iterrows():
    try:record_components=json.loads(record.get('components_json') or '[]')
    except (TypeError,json.JSONDecodeError):record_components=[]
    for component in record_components:
        sid=str(component.get('substance_id') or '')
        if sid:training_substance_sources.setdefault(sid,set()).add(str(record.get('source_id') or ''))
catalog_by_id={str(item['id']):item for item in catalog}

def substance_status(item):
    sid=str(item.get('id') or '')
    direct=len(training_substance_sources.get(sid,set()))
    family=str(item.get('family') or 'unknown')
    family_stat=ranges['ranges'].get('family_wt__'+family,{})
    support=int(family_stat.get('n_nonzero_papers') or 0)
    if direct:
        return f'训练组出现该物质 {direct} 篇｜模型按类别特征计算'
    if support:
        return f'仅有同家族训练数据 {support} 篇｜不是该物质专属模型'
    return '仅数据库可检索｜当前模型训练组不支持'

substance_labels={sid:(f"{item.get('name',sid)}"+(f"（{item.get('formula')}）" if item.get('formula') else '')+(f"｜别名：{'、'.join(str(a) for a in item.get('aliases',[])[:6])}" if item.get('aliases') else '')+f"　[{substance_status(item)}]") for sid,item in catalog_by_id.items()}
addon_ids=['']+[sid for sid,item in catalog_by_id.items() if sid!='SUB_PP']
base_ids=[sid for sid,item in catalog_by_id.items() if ('聚丙烯' in str(item.get('name','')) or 'pp' in str(item.get('name','')).lower()) and '蜡' not in str(item.get('name','')) and '母粒' not in str(item.get('name',''))]
if 'SUB_PP' in catalog_by_id:base_ids=['SUB_PP']+[sid for sid in base_ids if sid!='SUB_PP']
base_labels={sid:(f"{catalog_by_id[sid].get('name',sid)}　[当前模型适用]" if sid=='SUB_PP' else f"{catalog_by_id[sid].get('name',sid)}　[数据库可查；当前模型不支持该基体类型]") for sid in base_ids}

st.caption(f"实验性模型｜训练数据 {manifest['n_rows']} 条 / {manifest['n_papers']} 篇文献｜留一文献验证 MAE {v['model_MAE']:.2f} MPa，R² {v['model_R2']:.3f}。不是单次预测误差范围。")

input_widget_keys=['target_property','specimen_form','pp_matrix_class','matrix_substance_id','matrix_pct','test_standard','processing_route','test_speed','test_temperature','relative_humidity','process_temperature','process_screw_speed','process_duration']
for i in range(1,7):input_widget_keys.extend([f'substance_id{i}',f'pct{i}',f'role{i}'])

if st.session_state.get('screen','input')=='result':
    payload=st.session_state.get('prediction_payload',{})
    st.title('预测结果与文献依据')
    back_col,new_col=st.columns(2)
    if back_col.button('← 返回修改输入',type='secondary'):
        for key,value in st.session_state.get('saved_input_state',{}).items():st.session_state[key]=value
        st.session_state['screen']='input'
        st.rerun()
    if new_col.button('开始新配方'):
        for key in input_widget_keys:st.session_state.pop(key,None)
        st.session_state.pop('prediction_payload',None);st.session_state.pop('saved_input_state',None)
        st.session_state['screen']='input'
        st.rerun()
    st.success(f"实验性拉伸强度估计：**{payload.get('prediction',float('nan')):.2f} MPa**")
    st.info(f"本模型在 {manifest['n_rows']} 条、{manifest['n_papers']} 篇文献的训练范围内，留一文献验证 MAE 为 {v['model_MAE']:.2f} MPa。该误差不是单次预测区间。")
    st.markdown('### 本次输入')
    st.write('配方：'+format_formula(payload.get('components',[])))
    conditions=payload.get('conditions',{})
    condition_items=[('测试标准',display_standard(conditions.get('test_standard'))),('加工路线',display_route(conditions.get('processing_route'))),('拉伸速度（mm/min）',conditions.get('test_speed_mm_min')),('测试温度（°C）',conditions.get('test_temperature_C')),('加工温度（°C）',conditions.get('process_temperature_min_C')),('螺杆转速（rpm）',conditions.get('process_screw_speed_rpm')),('加工时间（min）',conditions.get('process_duration_min_max'))]
    shown=[{'条件':label,'输入值':str(value)} for label,value in condition_items if value is not None and str(value).lower()!='nan' and str(value).strip()]
    if shown:st.dataframe(pd.DataFrame(shown),hide_index=True,use_container_width=True)
    literature_open=st.session_state.get('show_related_literature',False)
    if st.button('查看相关文献' if not literature_open else '收起相关文献',key='toggle_related_literature'):
        st.session_state['show_related_literature']=not literature_open
        st.rerun()
    if st.session_state.get('show_related_literature',False):
        st.markdown('### 最相近的文献实测依据')
        st.caption('模型由整个训练集综合学习，以下文献是与本次输入较接近的实测记录，不表示预测由其中某篇文献单独决定。')
        for i,ref in enumerate(payload.get('nearest',[]),1):show_evidence(ref,i)
        if payload.get('outside'):
            st.markdown('### 同一添加物的其他文献（未参与当前模型训练）')
            st.caption('这些记录只作检索参考，不参与本次模型拟合。')
            for i,ref in enumerate(payload['outside'],1):show_outside_evidence(ref,i)
else:
    tab1,tab2,tab3=st.tabs(['配方预测与文献依据','数据库检索','分类与验证范围'])
    with tab1:
        st.subheader('预测设置')
        a,b,c=st.columns(3)
        target=a.selectbox('预测性能',['拉伸强度（MPa）—实验性试算','冲击/弯曲/伸长率—暂仅检索','热学/电学/表面性能—暂仅检索'],key='target_property')
        form=b.selectbox('试样/产品形态',['文献未报告/未确认（当前试算分组）','注塑/模压块材','薄膜/片材','纤维/无纺布','多孔膜/泡沫','电缆/涂层/其他专用形态'],key='specimen_form')
        matrix=c.selectbox('PP 基体类型',['均聚 PP 或原文未明确（当前试算分组）','共聚/抗冲 PP','再生 PP','接枝/交联等特殊 PP'],key='pp_matrix_class')
        if target!='拉伸强度（MPa）—实验性试算':st.info('该性能目前仅支持数据库检索，暂不输出预测。')
        if form!='文献未报告/未确认（当前试算分组）':st.info('该试样形态目前不在可预测范围内。')
        if matrix!='均聚 PP 或原文未明确（当前试算分组）':st.info('该 PP 基体类型目前不在可预测范围内。')

        family_display={'inorganic_filler':'无机填料','mineral_filler':'矿物填料','oxide_filler':'氧化物填料','polyolefin':'聚烯烃','polyolefin_elastomer':'聚烯烃弹性体','natural_fiber':'天然纤维','reinforcement':'增强材料','flame_retardant':'阻燃剂','organic_modifier':'有机改性剂','polymer_blend':'聚合物共混物','stabilizer':'稳定剂','silane_coupling_agent':'硅烷偶联剂'}
        def stat_text(feature, title):
            stat=ranges['ranges'].get(feature,{})
            lo=stat.get('positive_min')
            hi=stat.get('positive_max')
            n_papers=stat.get('n_nonzero_papers',0)
            if lo is None or hi is None or n_papers==0:
                return f'{title}：当前训练组没有可用的正含量记录。'
            return f'{title}：训练中出现 {lo:g}–{hi:g} wt%（{n_papers} 篇文献）。'

        fn_label={v:k for k,v in tax['function_categories'].items()}
        role_options=[label for label in fn_label if fn_label[label]!='matrix']
        st.markdown('**1. PP 基体**')
        matrix_id=st.selectbox('PP 基体物质（可搜索）',base_ids,index=base_ids.index('SUB_PP') if 'SUB_PP' in base_ids else 0,format_func=lambda sid:base_labels.get(sid,sid),key='matrix_substance_id')
        matrix_item=catalog_by_id.get(matrix_id,{})
        matrix_item_name=str(matrix_item.get('name') or '')
        matrix_supported=(matrix_id=='SUB_PP')
        if not matrix_supported:
            st.info('数据库收录了该 PP 基体，但当前训练模型只适用于普通 PP 或原文未明确的基体分组；可以检索文献，暂不能用它进行预测。')
        matrix_label='PP 基体质量比例（wt%）'
        if matrix_supported and st.session_state.get('matrix_pct',0)>0:
            matrix_stat=ranges['ranges'].get('function_wt__matrix',{})
            if matrix_stat.get('positive_min') is not None:
                matrix_label=f"PP 基体质量比例（wt%；训练范围 {matrix_stat['positive_min']:g}–{matrix_stat['positive_max']:g}）"
        matrix_pct=st.number_input(matrix_label,min_value=0.0,max_value=100.0,value=0.0,step=.1,key='matrix_pct',disabled=not matrix_supported)
        if matrix_pct>0 and matrix_supported:
            family=str(matrix_item.get('family') or 'unknown')
            st.caption(stat_text('family_wt__'+family,'基体所属家族：'+family_display.get(family,family))+' '+stat_text('function_wt__matrix','PP 基体功能'))

        st.markdown('**2. 填料和其他改性组分**')
        st.caption('依次填写填料、弹性体、相容剂、阻燃剂等；PP 基体已单独填写。总配方质量比例需合计 100 wt%。')
        rows=[(matrix_item_name,matrix_pct,'matrix')]
        def add_component_row(i):
            cols=st.columns([3,2.2,1.6])
            selected_id=cols[0].selectbox(f'改性物质 {i}（可搜索）',addon_ids,index=0,format_func=lambda sid:'请选择物质…' if not sid else substance_labels.get(sid,sid),key=f'substance_id{i}')
            item=catalog_by_id.get(selected_id) if selected_id else None
            name=str(item.get('name') or '') if item else ''
            role_label=cols[1].selectbox(f'组分功能 {i}',role_options,index=0,key=f'role{i}')
            role=fn_label[role_label]
            kind='exact' if item else 'none'
            family=str(item.get('family') or 'unknown') if item else 'unknown'
            fam_stat=ranges['ranges'].get('family_wt__'+family,{}) if item else {}
            role_stat=ranges['ranges'].get('function_wt__'+role,{})
            limit=100.0;pct_label='含量 wt%'
            family_supported=('family_wt__'+family) in manifest['numeric_features'] and int(fam_stat.get('n_nonzero_papers') or 0)>0 if item else False
            if item and family_supported:
                caps=[float(x['positive_max']) for x in (fam_stat,role_stat) if x.get('positive_max') is not None]
                if caps:
                    limit=min(100.0,*caps)
                    pct_label=f'含量 wt%（类别上限 {limit:g}）'
            pct_key=f'pct{i}'
            previous_pct=float(st.session_state.get(pct_key,0.0) or 0.0)
            if previous_pct>limit:
                st.session_state[pct_key]=limit
                st.caption(f'该物质类别上限为 {limit:g} wt%，已将本行输入限制到该值。')
            pct=cols[2].number_input(pct_label,min_value=0.0,max_value=limit,value=0.0,step=.1,key=pct_key,disabled=bool(item and not family_supported))
            if item:
                st.caption(substance_status(item))
                if family_supported:
                    st.caption(stat_text('family_wt__'+family,'物质家族：'+family_display.get(family,family))+' '+stat_text('function_wt__'+role,'所选组分功能'))
                else:
                    st.caption('该物质保留在下拉列表中供查看和检索；当前预测模型不支持此物质家族。')
            return name,pct,role

        for i in range(1,4):rows.append(add_component_row(i))
        with st.expander('添加更多组分（第 4–6 项）'):
            for i in range(4,7):rows.append(add_component_row(i))

        any_composition_input=matrix_pct>0 or any(n.strip() or p>0 for n,p,_ in rows[1:])
        current_total=sum(float(p) for n,p,_ in rows if n.strip())
        if any_composition_input:
            composition_preview=' + '.join(f"{name.strip()} {float(pct):g}%" for name,pct,_ in rows if name.strip() and float(pct)>0)
            if composition_preview:st.caption('配方预览：'+composition_preview)
            total_col,help_col=st.columns([1,3])
            total_col.metric('当前配方合计',f'{current_total:.1f} wt%')
            if abs(current_total-100)>.11:
                help_col.warning('配方总量目前不是 100 wt%；调整各组分比例后再预测。')
            else:
                help_col.success('配方总量为 100 wt%。')

        with st.expander('实验条件（选填；未知时留空）',expanded=False):
            def condition_label(label,features,key):
                if not str(st.session_state.get(key,'')).strip():return label
                stats=[ranges['ranges'].get(feature,{}) for feature in features]
                stats=[item for item in stats if item.get('min') is not None]
                if not stats:return label
                lo=min(item['min'] for item in stats);hi=max(item['max'] for item in stats)
                n=max(item['n_rows'] for item in stats);papers=max(item['n_papers'] for item in stats)
                return f"{label}（训练范围 {lo:g}–{hi:g}；{n} 条/{papers} 篇文献）"
            standards=sorted(refs.test_standard.astype(str).replace('',np.nan).dropna().unique().tolist())
            routes=sorted(refs.processing_route.astype(str).replace('',np.nan).dropna().unique().tolist())
            test_col,process_col=st.columns(2)
            test_col.markdown('**力学测试条件**')
            standard=test_col.selectbox('测试标准',standards,index=standards.index('not_reported') if 'not_reported' in standards else 0,format_func=display_standard,key='test_standard')
            speed=test_col.text_input(condition_label('拉伸速度（mm/min）',['test_speed_mm_min'],'test_speed'),key='test_speed')
            test_temp=test_col.text_input(condition_label('测试温度（°C）',['test_temperature_C'],'test_temperature'),key='test_temperature')
            humidity=test_col.text_input(condition_label('相对湿度（%）',['relative_humidity_percent'],'relative_humidity'),key='relative_humidity')
            process_col.markdown('**加工条件**')
            route=process_col.selectbox('加工/成型路线',routes,index=routes.index('not_reported_or_unclassified') if 'not_reported_or_unclassified' in routes else 0,format_func=display_route,key='processing_route')
            process_temp=process_col.text_input(condition_label('加工温度（°C）',['process_temperature_min_C','process_temperature_max_C'],'process_temperature'),key='process_temperature')
            screw=process_col.text_input(condition_label('螺杆转速（rpm）',['process_screw_speed_rpm'],'process_screw_speed'),key='process_screw_speed')
            duration=process_col.text_input(condition_label('加工时间（min）',['process_duration_min_max'],'process_duration'),key='process_duration')

        # Live checks run as fields change, before prediction; range issues disable the button.
        live_range_issues=[];live_fam={};live_func={};live_input_problem=False
        for name,pct,role in rows:
            if pct>0 and not name.strip():
                live_input_problem=True
                continue
            if name.strip() and pct>0:
                item,kind=match_substance(name,idx)
                if kind!='exact':
                    live_input_problem=True
                    continue
                family=str(item.get('family') or 'unknown')
                fkey='family_wt__'+family
                fstat=ranges['ranges'].get(fkey,{})
                if fkey not in manifest['numeric_features'] or int(fstat.get('n_nonzero_papers') or 0)==0:
                    live_input_problem=True
                    continue
                live_fam[fkey]=live_fam.get(fkey,0)+float(pct)
                live_func['function_wt__'+role]=live_func.get('function_wt__'+role,0)+float(pct)
        live_values={**live_fam,**live_func}
        for value,features,title in [
            (speed,['test_speed_mm_min'],'拉伸速度'),(test_temp,['test_temperature_C'],'测试温度'),
            (humidity,['relative_humidity_percent'],'相对湿度'),
            (process_temp,['process_temperature_min_C','process_temperature_max_C'],'加工温度'),
            (screw,['process_screw_speed_rpm'],'螺杆转速'),(duration,['process_duration_min_max'],'加工时间')]:
            if str(value).strip():
                try:
                    parsed=parse_optional(value)
                    if not np.isfinite(parsed):continue
                    for feature in features:live_values[feature]=parsed
                except ValueError:
                    live_input_problem=True
                    live_range_issues.append(f'{title}需填写有效数字。')
        role_labels=tax['function_categories']
        condition_labels={'test_speed_mm_min':'拉伸速度','test_temperature_C':'测试温度','relative_humidity_percent':'相对湿度','process_temperature_min_C':'加工温度','process_temperature_max_C':'加工温度','process_screw_speed_rpm':'螺杆转速','process_duration_min_max':'加工时间'}
        for feature,value in live_values.items():
            stats=ranges['ranges'].get(feature,{})
            if stats.get('min') is None or stats.get('max') is None:continue
            if float(value)<float(stats['min'])-1e-9 or float(value)>float(stats['max'])+1e-9:
                if feature.startswith('family_wt__'):
                    code=feature.removeprefix('family_wt__');label='物质家族 '+family_display.get(code,code)
                elif feature.startswith('function_wt__'):
                    code=feature.removeprefix('function_wt__');label='组分功能 '+role_labels.get(code,code)
                else:label=condition_labels.get(feature,feature)
                live_range_issues.append(f'{label}累计输入 {float(value):g} 超出训练观察范围 {float(stats["min"]):g}–{float(stats["max"]):g}。')
        total_is_valid=abs(current_total-100)<=.11
        if not any_composition_input:st.caption('先填写 PP 基体和改性组分比例；配方合计达到 100 wt% 后即可预测。')
        scope_is_valid=(target=='拉伸强度（MPa）—实验性试算' and form=='文献未报告/未确认（当前试算分组）' and matrix=='均聚 PP 或原文未明确（当前试算分组）' and matrix_supported)
        if live_range_issues:
            st.warning('输入范围提醒：'+'；'.join(live_range_issues)+' 请在上方对应输入栏调整后再预测。')
        if any_composition_input and not total_is_valid:
            st.caption('配方总量需调整到 100 wt% 后才能预测。')
        submitted=st.button('检查配方并预测',type='primary',disabled=bool(live_range_issues or live_input_problem or not total_is_valid or matrix_pct<=0 or not scope_is_valid))
        if submitted:
            errors=[];total=sum(float(p) for n,p,r in rows if n.strip())
            if abs(total-100)>.11:errors.append(f'配方合计 {total:.1f} wt%，需要为 100 wt%。')
            if target!='拉伸强度（MPa）—实验性试算':errors.append('该性能暂无可试算模型。')
            if form!='文献未报告/未确认（当前试算分组）':errors.append('该形态暂无合格预测模型。')
            if matrix!='均聚 PP 或原文未明确（当前试算分组）' or not matrix_supported:errors.append('该 PP 基体类型暂无合格预测模型。')
            components=[];fam={};func={};matrix_pct=0.0
            for name,pct,role in rows:
                if pct<=0 and not name.strip():continue
                if pct<=0:continue
                item,kind=match_substance(name,idx)
                if kind!='exact':
                    if kind=='suggestions':errors.append(f"“{name}”未精确匹配。候选："+'；'.join(f"{q['name']}（{q.get('formula') or q['id']}）" for q in item[:3])+"。请用完整规范名称或别名重填。")
                    else:errors.append(f'数据库中未找到“{name}”。')
                    continue
                family=str(item.get('family') or 'unknown');fc='family_wt__'+family
                if fc not in manifest['numeric_features']:errors.append(f"{item['name']} 的物质家族 {family} 不在当前模型训练范围内。")
                else:fam[fc]=fam.get(fc,0)+float(pct)
                fn='function_wt__'+role
                if fn in manifest['numeric_features']:func[fn]=func.get(fn,0)+float(pct)
                if role=='matrix':matrix_pct+=float(pct)
                components.append({'substance_id':item['id'],'substance_name':item['name'],'substance_family':family,'wt_pct':float(pct),'role':role,'formula':item.get('formula')})
            if matrix_pct<=0:errors.append('至少需要一个 PP 基体组分，并选择“PP基体”功能。')
            try:nums={'test_speed_mm_min':parse_optional(speed),'test_temperature_C':parse_optional(test_temp),'relative_humidity_percent':parse_optional(humidity),'process_temperature_min_C':parse_optional(process_temp),'process_temperature_max_C':parse_optional(process_temp),'process_screw_speed_rpm':parse_optional(screw),'process_duration_min_max':parse_optional(duration)}
            except ValueError:nums={};errors.append('条件字段必须是有效数字，未知时请留空。')
            if errors:
                for err in errors:st.error(err)
            else:
                vals={c:(0.0 if c.startswith(('family_wt__','function_wt__')) else np.nan) for c in manifest['numeric_features']};vals.update(fam);vals.update(func)
                for k,val in nums.items():
                    if k in vals:vals[k]=val
                cats={'test_standard':standard,'test_temperature_status':'numeric_reported' if np.isfinite(nums['test_temperature_C']) else 'not_reported','test_state':'not_reported','processing_route':route,'specimen_form':'not_reported','specimen_form_class':'unknown_or_not_reported','pp_matrix_class':'pp_homopolymer_or_unspecified'}
                row={**vals,**cats};X=pd.DataFrame([row])[manifest['numeric_features']+manifest['categorical_features']]
                prediction=float(estimator.predict(X)[0])
                conditions={**nums,'test_standard':standard,'processing_route':route}
                nearest=rank_literature(components,conditions,refs,manifest['role_map'])
                training_ids=set(refs.measurement_id.astype(str))
                outside=find_same_material_outside_scope(components,allmeas,training_ids)
                st.session_state['prediction_payload']={'prediction':prediction,'components':components,'conditions':conditions,'nearest':nearest,'outside':outside}
                st.session_state['saved_input_state']={key:st.session_state[key] for key in input_widget_keys if key in st.session_state}
                st.session_state['show_related_literature']=False
                st.session_state['screen']='result'
                st.rerun()

    with tab2:
        st.subheader('数据库性能数据检索')
        st.caption(f"当前整理检索表：{len(allmeas):,} 条性能测量记录。")
        a,b,c=st.columns([2,1.5,1.5]);query=a.text_input('物质名 / 化学式 / 别名',placeholder='SiO2、二氧化硅');props=sorted(allmeas.property.astype(str).unique());prop=b.selectbox('性能指标',['全部指标']+props);review=c.selectbox('数据库 reviewed 状态',['全部','reviewed=true','reviewed=false'])
        result=allmeas.copy()
        if prop!='全部指标':result=result[result.property.astype(str)==prop]
        if review!='全部':
            flags=result.reviewed.astype(str).str.lower().isin(['1','true']);result=result[flags if review.endswith('true') else ~flags]
        if query.strip():
            item,kind=match_substance(query,idx)
            if kind=='exact':result=result[result.components_search.astype(str).str.contains(re.escape(item['id']),case=False,regex=True)];st.caption(f"规范物质：{item['name']}（{item.get('formula') or item['id']}）")
            elif kind=='suggestions':st.info('没有精确命中，建议候选：'+'；'.join(f"{q['name']}（{q.get('formula') or q['id']}）" for q in item[:4]));result=result.iloc[:0]
            else:st.info('未找到该物质，请换用化学式或规范名。');result=result.iloc[:0]
        st.write(f'检索命中 {len(result):,} 条')
        if len(result):
            view=result.head(300).copy();view['配方']=view.components_json.map(lambda s:format_formula(json.loads(s)) if isinstance(s,str) and s.startswith('[') else '')
            show=[x for x in ['property','value','unit','paper_title','doi','source_sample','配方','standard','conditions_json','evidence_locator','source_locator','reviewed'] if x in view]
            st.dataframe(view[show],width='stretch',hide_index=True)
            if len(result)>300:st.caption('最多显示前300条；请增加筛选条件。')

    with tab3:
        st.subheader('分类结构与模型验证')
        with st.expander('查看模型输入范围和数据覆盖'):
            st.caption(f"范围由当前适用组 {ranges['n_rows']} 条记录、{ranges['n_papers']} 篇文献计算。完整观测范围以外会阻止预测；字段范围不代表任意配方组合都经过验证。")
            family_names={'polyolefin':'聚烯烃','inorganic_filler':'无机填料','mineral_filler':'矿物填料','oxide_filler':'氧化物填料','polyolefin_elastomer':'聚烯烃弹性体','natural_fiber':'天然纤维','reinforcement':'增强材料','flame_retardant':'阻燃剂','organic_modifier':'有机改性剂','polymer_blend':'聚合物共混物','stabilizer':'稳定剂','silane_coupling_agent':'硅烷偶联剂'}
            role_names={v:k for k,v in tax['function_categories'].items()}
            condition_names={'test_speed_mm_min':'拉伸速度（mm/min）','test_temperature_C':'测试温度（°C）','relative_humidity_percent':'相对湿度（%）','process_temperature_min_C':'加工温度下限（°C）','process_temperature_max_C':'加工温度上限（°C）','process_screw_speed_rpm':'螺杆转速（rpm）','process_duration_min_max':'加工时间（min）'}
            range_rows=[]
            for feature,stats in ranges['ranges'].items():
                if feature.startswith('family_wt__'):
                    code=feature.removeprefix('family_wt__');label='物质家族：'+family_names.get(code,code)
                elif feature.startswith('function_wt__'):
                    code=feature.removeprefix('function_wt__');label='组分功能：'+role_names.get(code,code)
                else:label=condition_names.get(feature,feature)
                range_rows.append({'输入特征':label,'训练最小值':stats['min'],'常见区间 P05–P95':f"{stats['p05']:.2f}–{stats['p95']:.2f}" if stats['p05'] is not None else '无数据','训练最大值':stats['max'],'有效记录':stats['n_rows'],'该特征出现记录':stats['n_nonzero_rows'],'出现文献数':stats['n_nonzero_papers']})
            st.dataframe(pd.DataFrame(range_rows),use_container_width=True,hide_index=True)
        st.dataframe(pd.DataFrame([{'维度':a['label'],'类别':'、'.join(a['categories'])} for a in tax['axes']]),width='stretch',hide_index=True)
        st.markdown('#### 留一文献验证结果')
        result=pd.DataFrame(metrics['results'])
        st.dataframe(result[['scope','model','n_rows','n_papers','baseline_MAE','model_MAE','model_RMSE','model_R2','folds_model_beats_baseline_MAE','folds_total']],width='stretch',hide_index=True)
        st.markdown('#### 数据库已有性能记录');st.dataframe(inventory,width='stretch',hide_index=True)
        st.caption('性能记录数不等于可训练模型数；没有经过分组验证的性能类别仅供检索，暂不预测。')
