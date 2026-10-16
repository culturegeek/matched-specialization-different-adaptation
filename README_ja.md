# 揃えた特化、異なる適応

高橋徹による論文「Matched Specialization, Different Adaptation: Second-Moment Structure in Shared Representations」のコード・生成データ。2026年10月17日、v1.5.0。

現在の特化が近くても、同じ次の学習への応答は異なりうる。Adamが保持する勾配二乗の履歴を交換すると応答が変わり、平均倍率を揃えても座標配置の効果が残った。形成中の記録と、新しい40 seedでの勾配射影・状態移植によって、保持状態の作られ方と応答への作用を結びつけた。

三Gaussian対象では終点一致を伴う比較が成立した一方、実測センサと射影後の形成では終点一致が未達だった。初期化同等性、符号反転、長期間での成立範囲も結果に残す。速い再配分が最終性能の改善を意味するとは確認していない。

[英語の案内と検証コマンド](README.md)・[論文PDF](paper/preprint.pdf)・[論文とデータの対応](docs/PAPER_TO_DATA.md)・[再現手順](docs/REPRODUCING.md)

論文は17ページ、付録五区分へ再構成した。主張に必要な比較を選び、終点診断を形成機構へ統合し、三対象比較の表も統合した。研究全体の旧実験、全候補、未達、生記録はこのリポジトリに保持する。

今回の形成介入は [iv-cancellation](experiments/iv-cancellation/README.md)、形成記録は [iv-formation-logging](experiments/iv-formation-logging/README_ja.md) にある。全92信頼区間を日英で正本JSONと照合し、主結果27平均、ファイル完全性、新実験の解析的勾配・射影・二次モーメント更新を検査した。検査範囲は [検証記録](provenance/cancellation_validation.json) に示す。

再実行は新しい出力先へ行う。記録環境は新解析でPython 3.12.14、NumPy 2.4.6、PyTorch 2.13.0、CPU単一thread。厳密一致を既定とし、別環境向けに明示的な許容比較を別runnerへ追加した。許容幅、最大差、実際の一致状態を記録し、保存結果を変更しない。

編集可能なTeX・参考文献・様式は paper/tex、図は paper/figures。公開英語稿はmain_arxiv.tex、匿名英語稿はmain.tex、日本語稿はmain_ja.texを組版する。

コードはMIT、生成データ・図・説明資料はCC BY 4.0。UCI温湿度データの元の出典とライセンスを保持する。引用にはCITATION.cffと使用版を用いる。旧公開タグは固定のままである。生成AIは設計・実装・解析・文献調査・執筆を支援し、著者が研究と公開に責任を負う。
