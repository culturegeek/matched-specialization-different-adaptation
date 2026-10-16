# 形成中の勾配と二次モーメント

二つの対象から来る更新信号が、形成中にどれだけ打ち消し合い、その履歴がAdamの二次モーメントへどう残るかを調べた。元の40 seed、両方向、両経路を条件を変えずに再実行し、160個の終点状態がすべて保存値と一致した。

取得経路では、形成500更新で蓄積した重み付き個別勾配二乗項の合計を、負の交差項が第1層で94.2%、第2層で93.0%相殺した。更新経路では28.4%、25.2%だった。最終二次モーメントの取得／更新比は0.242、0.340だった。初期状態から減衰して残る量も含めて、保存された二次モーメントを再構成できた。

対象は二対象tanh基準系、減衰率0.999、形成500更新である。これは実際に生じた系列の探索的分解であり、相殺を変えた別の学習の因果効果ではない。初期の記録用係数の誤りは独立照合で検出し、修正履歴を残して全件を再計算した。詳細な定義・環境・実行手順はREADME.md、固定計画はprotocol.md、正本はruns/とresults/にある。32万行の逐次更新と、解析的連鎖律による3,840記録の検査、160終点の一致を保持する。

元の保存先はIV-F内のformation_logging_20261017、公開資料ではexperiments/iv-formation-loggingに対応する。個々の実施報告PDFは作らず、主要実験の104報告へ統合した。

## 別環境での再現

160終点のbitwise一致は、記録されたCPU環境（Python 3.12.14、PyTorch 2.13.0）で観測した結果である。別のライブラリ版や機械での一致を保証するものではない。元の固定済みrun_logging.pyを保持し、再現用reproduce_logging.pyを追加した。

既定値は厳密一致で、`--tolerance 1e-6`を明示すると許容比較を使う。重みとmでは最大絶対差が tolerance × (1 + 参照tensorの最大絶対値) 以下、vでは 1e-12 + tolerance × 参照tensor最大絶対値 以下を要求する。形状、型、有限値、step、設定も検査する。実際のbitwise一致、最大差、許容幅を記録する。

```sh
python experiments/iv-formation-logging/reproduce_logging.py --archive-root . --seeds 20000 --out /tmp/formation-strict
python experiments/iv-formation-logging/reproduce_logging.py --archive-root . --seeds 20000 --out /tmp/formation-tolerant --tolerance 1e-6
```

記録環境で両モードのseed 20000再実行は厳密一致した。異版ライブラリでの試験は本リリースでは行っていない。許容判定の通過は、指定幅での数値一致を表し、後続の全推定値が同じという保証ではない。保存結果と監査は変更していない。公式の説明は[PyTorch再現性文書](https://docs.pytorch.org/docs/stable/notes/randomness.html)を参照。
