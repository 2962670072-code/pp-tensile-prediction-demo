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
    with st.expander(f"{n}. {title}｜{sample}｜实测 {value} MPa｜{label}",expanded=(n==1)):
        st.markdown(f"**DOI：** [{doi}](https://doi.org/{doi})" if doi else '**DOI：** 数据库未收录')
        st.write(f"**记录：** 文献 {r.get('source_id','')}；测量 {r.get('measurement_id','')}")
        st.write(f"**配方：** {format_formula(r['_components'])}")
        st.write('**与本次输入相同的添加物：** '+('、'.join(r['_shared_additives']) if r.get('_shared_additives') else '无；此条是模型特征距离较近的扩展参考，不是同一添加物的直接文献证据。'))
        st.write(f"**实测拉伸强度：** {value} {r.get('unit','MPa')}；标准：{r.get('test_standard') or r.get('standard') or '未报告'}；速度：{r.get('test_speed_mm_min') or '未报告'} mm/min；测试温度：{r.get('test_temperature_C') or '未报告'} °C")
        st.write(f"**加工/试样：** {r.get('processing_route') or '未报告'}；加工温度 {r.get('process_temperature_min_C') or '未报告'}–{r.get('process_temperature_max_C') or '未报告'} °C；试样 {r.get('specimen_json') or r.get('specimen_description') or '未报告'}")
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
    with st.expander(f"{n}. {title}｜{r.get('source_sample') or '样品未标明'}｜{r.get('_scope_note')}"):
        if doi:st.markdown(f"**DOI：** [{doi}](https://doi.org/{doi})")
        st.write(f"**同一物质：** {'、'.join(r['_overlap'])}；**文献/测量：** {r.get('source_id')} / {r.get('measurement_id')}")
        st.write(f"**文献实测：** {r.get('value')} {r.get('unit')}；**配方：** {format_formula(r['_components'])}")
        st.write(f"**测试标准：** {r.get('standard') or '未报告'}；**条件：** {r.get('conditions_json') or '未报告'}")
        st.write(f"**试样信息：** {r.get('specimen_json') or '未报告'}")
        if r.get('source_locator'):st.write(f"**文献文件：** {r.get('source_locator')}")
        if r.get('evidence_locator'):st.write(f"**原文数值定位：** {r.get('evidence_locator')}")
        st.caption('此文献只作为同物质的外部参考，没有用于本次模型拟合或上方相近记录排序。')

st.title('聚丙烯改性配方预测与文献依据')
st.caption('分类先行的研究原型｜当前仅对限定分组试算拉伸强度；其他性能支持检索')
manifest_path=OUT/'taxonomy_model_manifest.json'
needed=[OUT/'experimental_taxonomy_rf.joblib',manifest_path,OUT/'taxonomy_model_metrics.json',DATA/'model_reference_records.csv',DATA/'literature_measurements.csv',DATA/'substance_catalog.json',DATA/'property_inventory.csv',DATA/'classification_taxonomy_v1.json']
if not all(p.exists() for p in needed):st.error('网站文件不完整；请确认完整解压 ZIP。');st.stop()
manifest,metrics,refs,catalog,allmeas,inventory,tax=load_data();estimator=load_model();idx=alias_index(catalog);v=manifest['validation_metrics']
st.warning(f"模型综合使用整个训练集的 {manifest['n_rows']} 条记录、{manifest['n_papers']} 篇文献。留一文献验证 MAE {v['model_MAE']:.2f} MPa，R² {v['model_R2']:.3f}。这些是整体验证统计，不是某篇单独文献的归因，也不是本次预测的误差区间。")

tab1,tab2,tab3=st.tabs(['配方预测与文献依据','数据库检索','分类与验证范围'])
with tab1:
    a,b,c=st.columns(3)
    target=a.selectbox('预测性能',['拉伸强度（MPa）—实验性试算','冲击/弯曲/伸长率—暂仅检索','热学/电学/表面性能—暂仅检索'])
    form=b.selectbox('试样/产品形态',['文献未报告/未确认（当前试算分组）','注塑/模压块材','薄膜/片材','纤维/无纺布','多孔膜/泡沫','电缆/涂层/其他专用形态'])
    matrix=c.selectbox('PP 基体类型',['均聚 PP 或原文未明确（当前试算分组）','共聚/抗冲 PP','再生 PP','接枝/交联等特殊 PP'])
    if target!='拉伸强度（MPa）—实验性试算':st.info('该性能目前没有通过验证门槛的模型，可到“数据库检索”查看已收录测量。')
    if form!='文献未报告/未确认（当前试算分组）':st.info('该试样形态目前数据不足或未优于基线，暂不输出预测值。')
    if matrix!='均聚 PP 或原文未明确（当前试算分组）':st.info('该 PP 基体类型暂不在已验证模型范围内。')
    mech=next(x for x in tax['axes'] if x['id']=='mechanism')
    st.multiselect('改性机制（可多选，便于分类记录）',mech['categories'])
    st.subheader('配方组成')
    st.caption('按总配方质量百分数填写，PP 和改性组分合计必须为 100 wt%。物质名称支持数据库别名，如 SiO2 / 二氧化硅。')
    fn_label={v:k for k,v in tax['function_categories'].items()}
    options=list(fn_label)
    with st.form('formulation'):
        rows=[]
        for i in range(8):
            cols=st.columns([3,1,2])
            name=cols[0].text_input(f'物质 {i+1}',value='聚丙烯' if i==0 else '',key=f'name{i}',placeholder='例如 聚丙烯、SiO2、滑石粉')
            pct=cols[1].number_input(f'wt% {i+1}',min_value=0.0,max_value=100.0,value=95.0 if i==0 else 0.0,step=.1,key=f'pct{i}')
            role=cols[2].selectbox(f'组分功能 {i+1}',options,index=0 if i==0 else min(1,len(options)-1),key=f'role{i}')
            rows.append((name,pct,fn_label[role]))
        submitted=st.form_submit_button('检查配方并预测',type='primary')
    st.subheader('加工与测试条件')
    standards=sorted(refs.test_standard.astype(str).replace('',np.nan).dropna().unique().tolist())
    routes=sorted(refs.processing_route.astype(str).replace('',np.nan).dropna().unique().tolist())
    x,y,z=st.columns(3)
    standard=x.selectbox('测试标准',standards,index=standards.index('not_reported') if 'not_reported' in standards else 0)
    route=y.selectbox('加工/成型路线',routes,index=routes.index('not_reported_or_unclassified') if 'not_reported_or_unclassified' in routes else 0)
    speed=x.text_input('拉伸速度（mm/min）'); test_temp=y.text_input('测试温度（°C）'); humidity=z.text_input('相对湿度（%）')
    process_temp=z.text_input('加工温度（°C）'); screw=y.text_input('螺杆转速（rpm）'); duration=x.text_input('加工时间（min）')
    if submitted:
        errors=[];total=sum(float(p) for n,p,r in rows if n.strip())
        if abs(total-100)>.11:errors.append(f'配方合计 {total:.1f} wt%，需要为 100 wt%。')
        if target!='拉伸强度（MPa）—实验性试算':errors.append('该性能暂无可试算模型。')
        if form!='文献未报告/未确认（当前试算分组）':errors.append('该形态暂无合格预测模型。')
        if matrix!='均聚 PP 或原文未明确（当前试算分组）':errors.append('该 PP 基体类型暂无合格预测模型。')
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
            prediction=float(estimator.predict(X)[0]);st.success(f'实验性拉伸强度估计：**{prediction:.2f} MPa**')
            st.info(f"本模型在 {manifest['n_rows']} 条、{manifest['n_papers']} 篇文献的训练范围内，留一文献验证 MAE 为 {v['model_MAE']:.2f} MPa。该误差不是单次预测区间。")
            st.markdown('### 本次输入最接近的文献实测依据')
            st.caption('模型由整个训练集综合学习，以下列出的是同一适用分组中配方及条件最接近的实测记录，不表示预测由其中某一篇文献单独决定。每条记录可展开查看 DOI、配方、实测值、测试条件与原文定位。')
            conditions={**nums,'test_standard':standard,'processing_route':route}
            for i,ref in enumerate(rank_literature(components,conditions,refs,manifest['role_map']),1):show_evidence(ref,i)
            training_ids=set(refs.measurement_id.astype(str))
            outside=find_same_material_outside_scope(components,allmeas,training_ids)
            if outside:
                st.markdown('#### 同一添加物的其他文献（未参与当前模型训练）')
                st.caption('这些记录可能属于薄膜、无纺布等不同试样形态，或未达到本模型训练条件。它们不能解释上方预测值，但可以帮助你判断模型缺少了哪些同类训练文献。')
                for i,ref in enumerate(outside,1):show_outside_evidence(ref,i)

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
    st.dataframe(pd.DataFrame([{'维度':a['label'],'类别':'、'.join(a['categories'])} for a in tax['axes']]),width='stretch',hide_index=True)
    st.markdown('#### 留一文献验证结果')
    result=pd.DataFrame(metrics['results'])
    st.dataframe(result[['scope','model','n_rows','n_papers','baseline_MAE','model_MAE','model_RMSE','model_R2','folds_model_beats_baseline_MAE','folds_total']],width='stretch',hide_index=True)
    st.markdown('#### 数据库已有性能记录');st.dataframe(inventory,width='stretch',hide_index=True)
    st.caption('性能记录数不等于可训练模型数；没有经过分组验证的性能类别仅供检索，暂不预测。')
