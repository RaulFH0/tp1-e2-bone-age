"""Figuras descritivas das previsões OOF existentes; não ajusta modelos.

Usa somente os IDs congelados de treino e não comprova independência por paciente.
"""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parents[1]
AGE_BANDS = ['0–5', '6–10', '11–15', '16–19']


def load_predictions(path, splits_dir, family='HOG', model='GradientBoosting'):
    data = pd.read_csv(path, dtype={'id': str})
    data = data[(data.family == family) & (data.model == model)].copy()
    ids = pd.read_csv(Path(splits_dir) / 'train_ids.csv', dtype={'id': str}).id.tolist()
    if len(ids) != len(set(ids)) or data.id.duplicated().any() or set(data.id) != set(ids):
        raise ValueError('Previsões devem cobrir uma vez somente os IDs de treino')
    if not np.isfinite(data[['boneage', 'y_pred', 'fold']].to_numpy(dtype=float)).all():
        raise ValueError('Previsões não finitas')
    if (data.fold < 1).any() or not np.equal(data.fold, np.floor(data.fold)).all():
        raise ValueError('Dobras inválidas')
    if not data.boneage.between(0, 240, inclusive='left').all():
        raise ValueError('Idade fora das faixas documentadas')
    data = data.set_index('id').loc[ids].reset_index()
    data['difference'] = data.y_pred - data.boneage
    data['abs_error'] = data.difference.abs()
    return data


def summarize_predictions(data):
    difference = data.y_pred.to_numpy() - data.boneage.to_numpy()
    if len(difference) < 2 or not np.isfinite(difference).all():
        raise ValueError('Necessárias pelo menos duas previsões finitas')
    bias, sd = float(difference.mean()), float(difference.std(ddof=1))
    grouped = data.copy()
    grouped['band'] = pd.cut(grouped.boneage, [0, 72, 132, 192, 240], right=False, labels=AGE_BANDS)
    by_age = []
    for band in AGE_BANDS:
        rows = grouped[grouped.band == band]
        if len(rows):
            by_age.append({'band': band, 'n': len(rows),
                           'MAE_meses': float(np.abs(rows.y_pred - rows.boneage).mean())})
    return {'n': len(data), 'MAE_meses': float(mean_absolute_error(data.boneage, data.y_pred)),
            'RMSE_meses': float(np.sqrt(mean_squared_error(data.boneage, data.y_pred))),
            'R2': float(r2_score(data.boneage, data.y_pred)),
            'bias_meses': bias, 'sd_difference_meses': sd,
            'lower_limit_meses': bias - 1.96 * sd, 'upper_limit_meses': bias + 1.96 * sd,
            'by_age': by_age, 'partition': 'OOF no treino',
            'patient_independence_verified': False,
            'note': 'Métricas agrupadas OOF; distintas da média por dobra usada na comparação.'}


def main(args):
    data = load_predictions(args.predictions, args.splits_dir)
    summary = summarize_predictions(data)
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.size': 10, 'pdf.fonttype': 42})
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.7), layout='constrained')
    axes[0].scatter((data.y_pred + data.boneage) / 2, data.difference, s=7, alpha=.25)
    for field, color, name in [('bias_meses', '#333333', 'Viés'),
                                ('lower_limit_meses', '#a63603', 'Limite inferior'),
                                ('upper_limit_meses', '#a63603', 'Limite superior')]:
        value = summary[field]
        axes[0].axhline(value, color=color, linestyle='--', label=f'{name}: {value:.2f}')
    axes[0].set(xlabel='Média entre referência e previsão (meses)',
                ylabel='Predito − referência (meses)', title='(a) Bland–Altman descritivo OOF')
    axes[0].legend(fontsize=8)
    values = summary['by_age']
    bars = axes[1].bar([v['band'] for v in values], [v['MAE_meses'] for v in values], color='#377eb8')
    axes[1].bar_label(bars, labels=[f"{v['MAE_meses']:.2f}\nn={v['n']}" for v in values], padding=3, fontsize=9)
    axes[1].set(xlabel='Faixa da idade de referência (anos)', ylabel='MAE agrupado (meses)',
                title='(b) Erro OOF por idade', ylim=(0, max(v['MAE_meses'] for v in values) * 1.25))
    fig.suptitle(f'HOG + Gradient Boosting com sexo — {len(data)} previsões OOF no treino')
    for ext in ['png', 'pdf']:
        fig.savefig(out / f'figura_erro_oof_HOG_GB.{ext}', dpi=200)
    plt.close(fig)
    fig, axis = plt.subplots(figsize=(4.5, 4.2), layout='constrained')
    axis.scatter(data.boneage, data.y_pred, s=8, alpha=.25)
    axis.plot([0, 240], [0, 240], linestyle='--', color='#333333', label='Identidade')
    axis.set(xlabel='Referência (meses)', ylabel='Predição OOF (meses)',
             title='HOG + Gradient Boosting com sexo', xlim=(0, 240), ylim=(0, 240))
    axis.legend(fontsize=9)
    for ext in ['png', 'pdf']:
        fig.savefig(out / f'figura_dispersao_oof_HOG_GB.{ext}', dpi=200)
    plt.close(fig)
    selected = pd.concat([
        data.sort_values(['abs_error', 'id'], ascending=[True, True], kind='stable').head(4),
        data.sort_values(['abs_error', 'id'], ascending=[False, True], kind='stable').head(4)])
    selected.to_csv(out / 'casos_extremos_oof.csv', index=False)
    summary['predictions_sha256'] = hashlib.sha256(args.predictions.read_bytes()).hexdigest()
    (out / 'analise_oof.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'Análise OOF concluída: {len(data)} IDs de treino; nenhum novo treinamento.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--predictions', type=Path, required=True)
    parser.add_argument('--splits-dir', type=Path, default=ROOT / 'data/splits')
    parser.add_argument('--out-dir', type=Path, default=Path('results/analise_oof'))
    main(parser.parse_args())
