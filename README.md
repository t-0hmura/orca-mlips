# mlips4orca

ORCA (`ProgExt` / ExtTool) から MLIP を直接呼ぶためのプラグイン集です。

提供プラグイン:
- `plugins/uma_orca.py` (FAIR-Chem / UMA)
- `plugins/orbmol_orca.py` (orb-models / OrbMol)
- `plugins/mace_orca.py` (MACE)

## 1. 仕様調査の根拠

この実装は以下の仕様に合わせています。
- ORCA External interface (`basename_EXT.extinp.tmp` -> `basename_EXT.engrad`):
  - https://github.com/faccts/orca-external-tools
  - https://www.faccts.de/docs/orca/6.1/manual/contents/essentialelements/externaloptimizer.html
- UMA モデル一覧・ロード API:
  - https://github.com/facebookresearch/fairchem
- OrbMol モデル一覧・ロード API:
  - https://github.com/orbital-materials/orb-models
- MACE モデル一覧・ロード API:
  - https://github.com/ACEsuit/mace

## 2. インストール

```bash
cd /data2/tohmura/pdb2reaction_workspace/mlips4orca
python3 -m pip install -r requirements.txt
```

バックエンド別に追加インストール:
- UMA: `pip install fairchem-core`
- OrbMol: `pip install orb-models`
- MACE: `pip install mace-torch`

依存衝突がある場合は、バックエンドごとに仮想環境を分ける運用を推奨します。

## 3. ORCA入力例

```text
! SP

%pal
  nprocs 8
end

%method
  ProgExt "/data2/tohmura/pdb2reaction_workspace/mlips4orca/plugins/uma_orca.py"
  Ext_Params "--model uma-s-1p1 --task omol --device auto"
end

* xyz 0 1
O  0.000000  0.000000  0.000000
H  0.758602  0.000000  0.504284
H -0.758602  0.000000  0.504284
*
```

OrbMol / MACE も `ProgExt` のスクリプトを置き換えるだけで使えます。

## 4. モデル指定

### UMA
- 利用可能モデル確認:
```bash
python3 plugins/uma_orca.py --list-models
python3 plugins/uma_orca.py --list-tasks
```
- デフォルト: `uma-s-1p1`
- `--list-models` は、インストール済み `fairchem` の `available_models` を優先し、未導入時はリポジトリ由来のフォールバック一覧を返します。
- 代表例: `uma-s-1p1`, `uma-m-1p1`, `esen-sm-conserving-all-omol` など
- タスク例: `--task omol|omat|oc20|oc25|odac|omc`

### OrbMol
- 利用可能モデル確認:
```bash
python3 plugins/orbmol_orca.py --list-models
```
- デフォルト: `orb_v3_conservative_omol`
- 代表例: `orb-v3-conservative-omol`, `orb_v3_conservative_omol`, `orb-v3-direct-omol`, `orb-v3-conservative-inf-omat` など（`-` と `_` の両形式を受理）

### MACE
- 利用可能モデル確認:
```bash
python3 plugins/mace_orca.py --list-models
```
- デフォルト: `MACE-OMOL-0` (内部的に `omol:extra_large` と同義)
- 指定形式:
  - `MACE-OMOL-0` (OMOL-0のデフォルトエイリアス)
  - `mp:<alias>` / `<alias>` 例 `mp:medium-mpa-0`, `medium-mpa-0`
  - `off:<alias>` 例 `off:medium`
  - `off-small|off-medium|off-large`
  - `omol:extra_large`
  - `anicc`
  - ローカルモデルパス / URL

## 5. Hessianモード

ORCA ExtTool は標準では `engrad` (エネルギー + 勾配) を受け取る仕様です。
このため Hessian は ORCA 側へ直接返しませんが、プラグイン内部では以下を選択可能です。

- `--hessian-mode Analytical`
- `--hessian-mode Numerical`

必要なら `--dump-hessian <path>` を指定すると、その点での Hessian 行列 (eV/Å²) を保存できます。

## 6. ジョブ投入

`run.sh` を編集して `qsub run.sh` で投入できます。

```bash
cd /data2/tohmura/pdb2reaction_workspace/mlips4orca
qsub run.sh
```

`run.sh` には以下を含めています。
- `. /home/apps/Modules/init/profile.sh`
- `module load gaussian16.C02`
- `module load orca/6.1.1`
