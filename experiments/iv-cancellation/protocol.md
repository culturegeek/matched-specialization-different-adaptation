# 形成中の相殺への射影介入と状態交換

2026-10-17。局所固定後に未使用40 seed（38000–38039）で実行する、既存知見に基づく機構追試。smokeは37999。新規の公開レジストリ事前登録ではない。全40を保持し、結果依存の停止・候補追加・校正をしない。

問い：取得経路の形成中の勾配衝突を射影で除くと、第二モーメントと追加学習応答はどう変わるか。その第二モーメントだけを元の重みに移しても応答は変わるか。

基準は元の二対象tanh、幅32・bottleneck2、相関0.35、初期100・形成500、Adam beta=(0.9,0.999)、取得経路はfavored-preciseで等重み、更新経路はbalanced-preciseで0.8/0.2。全条件で祖先、データ、batch、方向、スケジュールを共有する。形成はnative acquisition、native update、projected acquisitionの3枝。

射影はYu et al.2020の二対象PCGradを共有encoder座標に限定して適用する。重み付きhead勾配u,vの内積が負なら、u'=u-(u·v)/(v·v)v、v'=v-(u·v)/(u·u)u。両式は元のu,vを用いる。encoderはu'+v'、headは元の重み付き勾配のまま。weight decayはAdamがその後に加える。射影は角度・大きさ・後の重みを変えるため、相殺率だけを独立に操作する介入とは呼ばない。射影でvが増えることを前提にしない。

形成後の比較はnative acquisitionとprojected acquisitionのrecipientをそれぞれ固定し、v donorをnative/projectedで交差する2×2。同じrecipient内では重み、m、stepを完全一致させvだけを交換する。native update ownを応答差の共通参照とする。pulseは従来のcombined opposing conditionで50更新。0/1/5/10/20/50を保存し、10を主評価、50を副評価とする。

主推定：(1)形成終点のencoder v総量と平均逆倍率のprojected/native比、(2)native recipientでv donorをprojectedにしたときの10更新応答の対応差、(3)projected own minus native own応答（形成操作の総効果）。反対recipientの交換効果、0/1/5/20/50と各層を補助として全保存。応答はaligned_s(0)-aligned_s(t)。両方向内平均後seed単位でbootstrap10,000回、乱数2026101704、個別95%区間。比は方向内対数平均→seed間幾何平均。

既存の五つの終点基準（aligned_s,target_error,other_error,total_error,low_precision_s）をnative acquisition/updateとprojected acquisition/updateで併記する。許容幅はいずれも0.02。projectionで一致しなかった場合は同等終点の応答比較と呼ばない。v-only比較は同一recipient固定なので、元の経路一致とは別の介入比較として解釈する。

記録：全stepの射影前後のencoder勾配二乗・内積・cos・v・実更新、step0/100/250/500の重み/m/v、全終点とpulse、端点評価。独立監査で射影式・衝突符号・不変head・保存祖先・交換時の重み/m/stepを検査し、代表snapshotは既存NumPy解析連鎖律と照合する。

本実験は相殺→v→応答の自然な媒介割合を同定しない。v-only効果があれば、射影形成で得られたvが固定recipientの応答を変えられることを示す。総効果と移植効果が整合しない場合も削除せず、単純な一本の説明の不成立として残す。

原論文：https://arxiv.org/abs/2001.06782
