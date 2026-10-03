"""Reprodução da CV interna a partir da base original, sem ZIPs de correção.

Reutiliza os IDs congelados. Não realiza avaliação nem seleção no teste externo.
--audit-only gera apenas a auditoria. A execução completa pode demorar horas.
"""
import argparse
from collections import defaultdict
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
DATASET = 'kmader/rsna-bone-age/versions/2'


def find_image_directory(dataset_dir, ids):
    required = set(map(str, ids))
    folders = defaultdict(set)
    for image in Path(dataset_dir).rglob('*.png'):
        folders[image.parent].add(image.stem)
    matches = [folder for folder, present in folders.items() if required <= present]
    if len(matches) != 1:
        raise ValueError('Esperada uma única pasta contendo todos os IDs dos rótulos')
    return matches[0]


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'src' / f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run(command):
    subprocess.run([sys.executable, *map(str, command)], cwd=ROOT, check=True)


def create_texture(images, labels, output):
    import numpy as np
    import pandas as pd
    from image_preprocessing import preprocess_image
    cv = load_script('08_cross_validation')
    ids = cv.frozen_ids(ROOT / 'data/splits')
    columns = cv.COMPARISON.TEXTURE_COLUMNS
    indexed = labels.set_index('id')
    output.mkdir(parents=True, exist_ok=True)
    processed = 0
    for split in ('train', 'val', 'test'):
        rows = []
        for image_id in ids[split]:
            values = cv.TEXTURE.extract_features(preprocess_image(images / f'{image_id}.png'))
            rows.append([image_id, *[values[c] for c in columns]])
            processed += 1
            if processed % 100 == 0:
                print(f'Textura: {processed}/4000 imagens ({processed / 40:.1f}%)', flush=True)
        matrix = pd.DataFrame(rows, columns=['id', *columns])
        if not np.isfinite(matrix.drop(columns='id').to_numpy()).all():
            raise ValueError('Textura contém valores não finitos')
        matrix.to_csv(output / f'texture_{split}.csv', index=False)
        metadata = indexed.loc[ids[split], ['boneage', 'male']].reset_index()
        metadata.to_csv(output / f'metadata_{split}.csv', index=False)
        cv.COMPARISON.load_texture_split(output, split)


def create_radiograph_figures(images, predictions, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from image_preprocessing import preprocess_image
    analysis = load_script('09_oof_error_analysis')
    data = analysis.load_predictions(predictions, ROOT / 'data/splits')
    ordered = data.sort_values(['abs_error', 'id'], kind='stable')
    young = data[data.boneage < 72].sort_values(['abs_error', 'id'], ascending=[False, True], kind='stable')
    older = data[data.boneage >= 192].sort_values(['abs_error', 'id'], ascending=[False, True], kind='stable')
    four = [*ordered.head(2).id.tolist(), young.iloc[0].id, older.iloc[0].id]
    worst = data.sort_values(['abs_error', 'id'], ascending=[False, True], kind='stable')
    eight = [*ordered.head(4).id.tolist(), *worst.head(4).id.tolist()]
    output.mkdir(parents=True, exist_ok=True)
    for name, selected, nrows in [('casos_oof', four, 1), ('radiografias_oof_8_casos', eight, 2)]:
        fig, axes = plt.subplots(nrows, 4, figsize=(9, 3.1 * nrows))
        for axis, image_id in zip(axes.flat, selected):
            row = data[data.id == image_id].iloc[0]
            axis.imshow(preprocess_image(images / f'{image_id}.png'), cmap='gray', vmin=0, vmax=1)
            axis.set_title(f"ID {image_id} · dobra {int(row.fold)}\nR {row.boneage:.0f} · P {row.y_pred:.2f}\nE {row.abs_error:.2f}", fontsize=11)
            axis.axis('off')
        fig.tight_layout(pad=.25)
        for ext in ('png', 'pdf'):
            fig.savefig(output / f'{name}.{ext}', dpi=220, bbox_inches='tight')
        plt.close(fig)
        data.set_index('id').loc[selected].reset_index().to_csv(output / f'{name}.csv', index=False)
    evidence = {image_id: hashlib.sha256((images / f'{image_id}.png').read_bytes()).hexdigest() for image_id in set(eight + four)}
    (output / 'proveniencia_imagens.json').write_text(json.dumps(evidence, indent=2), encoding='utf-8')


def main(args):
    import pandas as pd
    started = time.time()
    if not args.audit_only and not args.allow_image_folds and not args.groups_csv:
        raise ValueError('Informar --groups-csv ou registrar o protocolo provisório com --allow-image-folds')
    print('0% — Preparando base e IDs congelados.', flush=True)
    if args.dataset_dir:
        dataset = args.dataset_dir.resolve()
    else:
        import kagglehub
        dataset = Path(kagglehub.dataset_download(DATASET))
    csvs = list(dataset.rglob('boneage-training-dataset.csv'))
    if len(csvs) != 1:
        raise ValueError('Esperado um boneage-training-dataset.csv na base')
    csv_path = csvs[0]
    labels = pd.read_csv(csv_path, dtype={'id': str})
    if len(labels) != 12611 or labels.id.duplicated().any():
        raise ValueError('Base diferente da versão auditada de 12.611 rótulos')
    images = find_image_directory(dataset, labels.id.tolist())
    args.out_dir.mkdir(parents=True, exist_ok=True)
    audit = args.out_dir / 'data_audit.json'
    run(['src/00_audit_data.py', '--csv', csv_path, '--images-dir', images, '--out', audit])
    report = json.loads(audit.read_text(encoding='utf-8'))
    if report['images']['missing']:
        raise ValueError('Imagens ausentes na auditoria')
    print('10% — Auditoria real concluída.', flush=True)
    if args.audit_only:
        print(f'100% da auditoria — {audit}', flush=True)
        return
    texture = args.out_dir / 'texture'
    create_texture(images, labels, texture)
    print('25% — Textura refeita dos pixels originais e IDs conferidos.', flush=True)
    cv_dir = args.out_dir / 'cv'
    command = ['src/08_cross_validation.py', '--images-dir', images, '--csv', csv_path,
               '--texture-dir', texture, '--out-dir', cv_dir, '--folds', '3', '--n-estimators', '300']
    if args.groups_csv:
        command += ['--groups-csv', args.groups_csv]
    else:
        command += ['--allow-image-folds']
        print('PROTOCOLO PROVISÓRIO: independência por paciente não verificada.', flush=True)
    print('35% — Iniciando extração e treinamentos da CV; acompanhar as dobras no log.', flush=True)
    run(command)
    print('85% — CV e ablação concluídas; gerando figuras.', flush=True)
    run(['src/09_oof_error_analysis.py', '--predictions', cv_dir / 'predictions_oof.csv', '--out-dir', args.out_dir / 'figuras'])
    create_radiograph_figures(images, cv_dir / 'predictions_oof.csv', args.out_dir / 'figuras')
    frozen = json.loads((cv_dir / 'metrics_cross_validation.json').read_text())
    reference = json.loads((ROOT / 'docs/resultados_oof/metrics_cross_validation.json').read_text())
    differences = []
    for family, models in reference['results'].items():
        for model, values in models.items():
            for metric in ('MAE_meses', 'RMSE_meses', 'R2'):
                for statistic in ('mean', 'std'):
                    differences.append(abs(values[metric][statistic] - frozen['results'][family][model][metric][statistic]))
    result = {'python': platform.python_version(), 'dataset': DATASET, 'source': 'imagens originais',
              'n_labels': len(labels), 'n_train': frozen['n_train'], 'protocol': frozen['protocol'],
              'val_test_used_for_selection': frozen['val_test_used_for_selection'],
              'max_absolute_metric_difference': max(differences), 'elapsed_seconds': time.time() - started,
              'audit_sha256': hashlib.sha256(audit.read_bytes()).hexdigest(),
              'source_hashes': frozen['source_hashes'], 'split_hashes': frozen['split_hashes']}
    (args.out_dir / 'verificacao_reproducibilidade.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    archive = shutil.make_archive(str(args.out_dir.parent / 'verificacao_limpa_TP1'), 'zip', args.out_dir)
    print(f'100% desta execução — {archive}', flush=True)
    print('O pacote não encerra afiliação, contribuição assinada, agrupamento ou avaliação final externa.', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-dir', type=Path)
    parser.add_argument('--out-dir', type=Path, default=ROOT / 'results/reproducibilidade')
    parser.add_argument('--audit-only', action='store_true')
    parser.add_argument('--allow-image-folds', action='store_true')
    parser.add_argument('--groups-csv', type=Path)
    main(parser.parse_args())
