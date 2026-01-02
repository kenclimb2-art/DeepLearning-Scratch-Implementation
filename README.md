```markdown
# カタカナ画像識別プロジェクト - E資格認定課題

![Python](https://img.shields.io/badge/Python-3.x-blue?style=flat-square&logo=python)
![NumPy](https://img.shields.io/badge/Library-NumPy_Only-orange?style=flat-square)
![Accuracy](https://img.shields.io/badge/Test_Accuracy-98.57%25-brightgreen?style=flat-square)

## 1. 概要 (Overview)
本プロジェクトは、手書きカタカナ15文字（アイウエオカキクケコサシスセソ）の画像データを識別するDeep Learningモデルの実装です。
**フレームワーク（PyTorch/TensorFlow等）を一切使用せず、NumPyのみでCNN（畳み込みニューラルネットワーク）をスクラッチ実装**し、高精度な識別を実現しました。

---

## 2. 成果 (Results)
* **最高精度 (Test Accuracy):** **98.57%** (2026.1.1時点)
* **コア・アプローチ:**
    * データ拡張による回転・ズレ耐性の強化
    * アーキテクチャ最適化による微細な特徴抽出の維持

---

## 3. 技術的アプローチと工夫 (Technical Strategy)
「回転した文字」や「類似した文字（カ/ク、ソ/ン）」の誤認を防ぐため、以下の独自チューニングを施しています。

### A. 動的データ拡張 (Aggressive Data Augmentation)
テストデータに含まれる多様なバリエーションに対応するため、OpenCVを用いて学習時に以下の処理を適用。
* **シフト:** 上下左右へのピクセル移動
* **回転:** ±15〜20度のランダム回転を適用
* **効果:** 「ソ」と「ン」のように、傾きによって混同しやすい文字の識別境界を明確化。

### B. 解像度維持アルゴリズム (Resolution Preservation)
通常のCNN設計を疑い、あえてセオリーから外れることで精度を向上させました。
* **課題:** 3層のプーリングを重ねると特徴マップが $3 \times 3$ まで縮小され、「カ」のハネなどの微細な情報が消失する。
* **解決策:** **3層目のプーリング処理を削除**。
* **結果:** $7 \times 7$ (49ピクセル) の高解像度情報を全結合層へ伝達。
    * `Before`: $3 \times 3$ map → 「カ」と「ク」の区別が困難
    * `After`: $7 \times 7$ map → 微細な形状を維持し、誤認率を劇的に低減



---

## 4. ファイル構成 (File Structure)

| ファイル名 | 役割 |
| :--- | :--- |
| `train_katakana.py` | 学習用メインスクリプト。CNN定義、データ拡張、学習、保存を一括実行。 |
| `analyze_models.py` | 検証・分析用。混同行列を作成し、誤認ワーストランキングを出力。 |
| `pickle_light.py` | モデル軽量化。推論に不要な勾配情報等を削除し、サイズを圧縮。 |
| `putil.py` | データ読み込みや精度計算用のユーティリティ関数群。 |
| `requirements.txt` | 依存ライブラリ一覧（NumPy, OpenCV等）。 |

---

## 5. 実行手順 (Usage)

### 前提条件
* Python 3.x
* NumPy
* OpenCV (`opencv-python`)

```bash
# インストール
pip install -r requirements.txt

```

### Step 1: モデルの学習

データ拡張を適用しながらモデルを最適化します。

```bash
python train_katakana.py

```

### Step 2: 精度評価・エラー分析

どの文字を間違えたか、弱点を可視化します。

```bash
python analyze_models.py

```

### Step 3: モデルの軽量化（提出用）

推論に必要なパラメータのみを抽出して圧縮します。

```bash
python pickle_light.py

```

---

## 6. 更新履歴 (Changelog)

* **2026.01.01**: `train_katakana_dropout_3.py` にて精度 **98.57%** を達成。
* **2025.12.31**: ドロップアウト導入により 98.4% 〜 98.5% へ向上。
* **初期リリース**: 3層目プーリング削除の採用により 98.12% 達成。

---

## 7. Author / Notes

* 本コードはE資格課題の要件（フレームワーク不使用）に基づき作成されました。
* 学習済みモデルは `pickle` 形式で管理しています。

```
