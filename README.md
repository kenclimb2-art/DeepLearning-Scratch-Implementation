【カタカナ画像識別プロジェクト - E資格認定課題】

1. 概要 (Overview)
このプロジェクトは、手書きカタカナ15文字（アイウエオカキクケコサシスセソ）の画像データを識別するDeep Learningモデルの実装です。 
フレームワーク（PyTorch/TensorFlow等）を使用せず、NumPyのみでCNN（畳み込みニューラルネットワーク）をスクラッチ実装し、
高精度な識別を実現しました。

2. 成果 (Results)
・Test Accuracy: 98.12%
・アプローチ: データ拡張による回転耐性の強化と、プーリング層の調整による解像度維持。

3. 技術的アプローチと工夫 (Technical Strategy)
本プロジェクトでは、「回転した文字」と「類似した文字（カ/ク、ソ/ン）」の誤認を防ぐため、以下の独自チューニングを行いました。

A. データ拡張 (Aggressive Data Augmentation)
テストデータに含まれる「回転」や「ズレ」に対応するため、学習データに対して以下の拡張を動的に適用しました。

・シフト: 上下左右へのピクセル移動
・回転: OpenCVを使用し、±15〜20度のランダム回転を適用
・これにより、「ソ」と「ン」のような傾きによって混同しやすい文字の識別能力を向上させました。

B. モデル構造の最適化 (Resolution Preservation)
通常のCNN（3層畳み込み）では、プーリングを重ねることで特徴マップが 3x3 まで縮小され、「カ」のハネなどの微細な特徴が消失する問題がありました。 
これを解決するため、3層目のプーリング処理を意図的に削除し、7x7 (49ピクセル) の高解像度情報を全結合層へ渡すアーキテクチャを採用しました。

・Before: 3x3 map -> 「カ」と「ク」の区別がつかない
・After: 7x7 map -> 微細な形状を維持し、誤認率をほぼゼロ化

4. ファイル構成
・train_katakana.py: 学習用メインスクリプト
    CNNクラス定義(Three_ConvNet)、データ拡張、学習ループ、モデル保存までを一括で行います。
・analyze_models.py: 検証・分析スクリプト。
    学習済みモデルを読み込み、混同行列（Confusion Matrix）を作成。
    「どの文字を間違えたか」をランキング形式で出力し、弱点を分析します。
・pickle_light.py: モデル軽量化スクリプト。
   推論に不要な中間データ（勾配情報やバッファ）を削除し、ファイルサイズを圧縮します。
   util.pyデータ読み込みや精度計算のためのユーティリティ関数群。
・requirements.txt: 必要なライブラリ一覧

5. 実行手順 (Usage)
前提条件 (Requirements)
・Python 3.x
・NumPy
・OpenCV (opencv-python)

インストール:
Bash
pip install -r requirements.txt

Step 1: モデルの学習
データ拡張を行いながらモデルを学習させます。
Bash
python train_katakana.py

・出力: katakana_model.pickle (Best Model), katakana_model_end.pickle(Final Model)

Step 2: 精度評価・エラー分析
学習したモデルが「どの文字を間違えているか」を分析します。
Bash
python analyze_models.py

・出力例: 正解「ケ」→ 誤認「サ」: 2回 のようにワーストランキングを表示。

Step 3: モデルの軽量化（提出用）
提出用にファイルサイズを圧縮します。

Bash
python pickle_light.py

・出力: katakana_model_light.pickle

6. Author / Notes
本コードはE資格課題の要件に基づき作成されました。
学習済みモデルは pickle 形式で保存されます。

7. その後
・2025.12.31
    train_katakana_dropout.py で98.4%達成
    train_katakana_dropout_2.py で98.5%達成
