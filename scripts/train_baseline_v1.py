from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
import sklearn
from joblib import dump
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT=Path(__file__).resolve().parents[1]

def make_pipeline(nums,cats,alpha=10.0):
    numeric=Pipeline([('imputer',SimpleImputer(strategy='median',keep_empty_features=True)),('scale',StandardScaler())])
    categorical=Pipeline([('imputer',SimpleImputer(strategy='most_frequent',keep_empty_features=True)),('onehot',OneHotEncoder(handle_unknown='ignore'))])
    prep=ColumnTransformer([('numeric',numeric,nums),('categorical',categorical,cats)])
    return Pipeline([('preprocess',prep),('ridge',Ridge(alpha=alpha))])

def main():
    ap=argparse.ArgumentParser(description='比较均值基线与Ridge；整篇文献留出验证。')
    ap.add_argument('--data',type=Path,default=ROOT/'data'/'model_input_features_v1.csv')
    ap.add_argument('--manifest',type=Path,default=ROOT/'data'/'feature_manifest_v1.json')
    ap.add_argument('--outdir',type=Path,default=ROOT/'outputs')
    ap.add_argument('--alpha',type=float,default=10.0)
    args=ap.parse_args(); args.outdir.mkdir(parents=True,exist_ok=True)
    df=pd.read_csv(args.data,encoding='utf-8-sig')
    if 'include_in_general_model' in df:
        df=df[pd.to_numeric(df['include_in_general_model'],errors='coerce').fillna(0).astype(int)==1].copy()
    meta=json.loads(args.manifest.read_text(encoding='utf-8'))
    nums=[c for c in meta['numeric_features'] if c in df.columns]
    cats=[c for c in meta['categorical_features'] if c in df.columns]
    if len(df)<10 or df['source_id'].nunique()<3:
        raise ValueError('训练样本或独立文献数量过少，无法进行有意义的留一文献验证。')
    y=pd.to_numeric(df['strength_mpa'],errors='raise').to_numpy(float)
    groups=df['source_id'].astype(str).to_numpy()
    X=df[nums+cats].copy()
    for c in cats: X[c]=X[c].where(X[c].notna(),'not_reported').astype(str)
    logo=LeaveOneGroupOut(); pr=np.full(len(df),np.nan); pm=np.full(len(df),np.nan); folds=[]
    for tr,te in logo.split(X,y,groups):
        model=make_pipeline(nums,cats,args.alpha)
        model.fit(X.iloc[tr],y[tr]); pr[te]=model.predict(X.iloc[te]); pm[te]=float(np.mean(y[tr]))
        folds.append({'held_out_source_id':groups[te[0]],'n_test':int(len(te)),
          'mae_mean_baseline_mpa':float(mean_absolute_error(y[te],pm[te])),
          'mae_ridge_mpa':float(mean_absolute_error(y[te],pr[te])),
          'rmse_mean_baseline_mpa':float(np.sqrt(mean_squared_error(y[te],pm[te]))),
          'rmse_ridge_mpa':float(np.sqrt(mean_squared_error(y[te],pr[te])))})
    if np.isnan(pr).any(): raise RuntimeError('交叉验证有未生成的预测值。')
    pd.DataFrame({'measurement_id':df['measurement_id'],'source_id':groups,'sample_label':df['sample_label'],
      'actual_strength_mpa':y,'mean_baseline_oof_mpa':pm,'ridge_oof_mpa':pr,
      'mean_baseline_abs_error_mpa':np.abs(y-pm),'ridge_abs_error_mpa':np.abs(y-pr)}).to_csv(args.outdir/'cross_validation_predictions_v1.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(folds).sort_values('held_out_source_id').to_csv(args.outdir/'metrics_by_held_out_paper_v1.csv',index=False,encoding='utf-8-sig')
    mean_metrics={'MAE_MPa':float(mean_absolute_error(y,pm)),'RMSE_MPa':float(np.sqrt(mean_squared_error(y,pm))),'R2_oof':float(r2_score(y,pm))}
    ridge_metrics={'MAE_MPa':float(mean_absolute_error(y,pr)),'RMSE_MPa':float(np.sqrt(mean_squared_error(y,pr))),'R2_oof':float(r2_score(y,pr))}
    summary={'purpose':'Exploratory v1 model comparison; not production-ready','n_rows':len(df),'n_papers':int(pd.Series(groups).nunique()),
      'validation':'Leave-one-paper-out; source_id used only as group','model':f'Ridge(alpha={args.alpha}) inside preprocessing Pipeline',
      'scikit_learn_version':sklearn.__version__,'features':{'numeric':len(nums),'categorical':len(cats)},
      'mean_baseline':mean_metrics,'ridge':ridge_metrics,'folds':folds,
      'recommend_deploy':bool(ridge_metrics['MAE_MPa'] < mean_metrics['MAE_MPa'] and ridge_metrics['RMSE_MPa'] <= mean_metrics['RMSE_MPa']),
      'interpretation_rule':'A lower MAE/RMSE than the mean baseline is necessary for progress, but is not by itself sufficient for a public prediction claim.',
      'caution':'Initial cohort includes only proposed general bulk PP modification candidates. Results apply only to this data distribution and do not guarantee accuracy for arbitrary new ingredients or processes.'}
    (args.outdir/'metrics_v1.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    model_path=args.outdir/'experimental_ridge_model_v1.joblib'
    if summary['recommend_deploy']:
        final=make_pipeline(nums,cats,args.alpha); final.fit(X,y); dump(final,model_path)
        (args.outdir/'MODEL_NOT_READY.txt').unlink(missing_ok=True)
    else:
        model_path.unlink(missing_ok=True)
        (args.outdir/'MODEL_NOT_READY.txt').write_text('留一文献验证未同时优于均值基线；当前不生成可接入网站的模型文件。先检查数据分组、特征和条件可比性。\n',encoding='utf-8')
    lines=['# 聚丙烯拉伸强度 v1 首轮验证结果','',f"- 通用块状候选：{len(df)}条；独立文献：{pd.Series(groups).nunique()}篇。",'- 验证方式：每次整篇留出一篇文献。','- 来源ID只用于分组验证，没有作为预测特征。','- 对照为训练集均值；模型为Ridge回归，所有预处理在每个训练折内部完成。','',
      '| 方法 | MAE (MPa) | RMSE (MPa) | R²（辅助） |','|---|---:|---:|---:|',
      f"| 均值基线 | {mean_metrics['MAE_MPa']:.2f} | {mean_metrics['RMSE_MPa']:.2f} | {mean_metrics['R2_oof']:.3f} |",
      f"| Ridge | {ridge_metrics['MAE_MPa']:.2f} | {ridge_metrics['RMSE_MPa']:.2f} | {ridge_metrics['R2_oof']:.3f} |",'',
      '## 使用限制','', '- 这是探索性模型。它用于判断新数据是否改善了初版流程，不应直接作为任意配方的可靠预测器。',
      '- 训练集按初筛规则排除了电缆绝缘、纤维产品、多孔膜、开放网格、再生PP料流及专用橡胶复合应用；这些数据仍保留在全量候选表。',
      '- 成分输入以规范物质ID聚合；具体牌号、原始用量和出处仍保存在components_json与长表中。',
      '- 测试速度、温度和加工条件未报告时保持缺失，模型在训练折内处理缺失值。',
      '- 后续接入网站前需确认输入变量与网站表单一致，并用未参与训练的新文献做外部验证。','']
    (args.outdir/'pilot_report_v1.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__=='__main__': main()
