KalmanFilter_tutorial
=====================

To deepen my understanding of Kalman Filter

- Linear Kalman Filter
- Adaptive Kalman Filter

## 使い方と結果

#### Extended Kalman Filter

```bash
python ExtendedKalmanFilter/ekf.py
```

![Alt Text](https://github.com/eisoku9618/KalmanFilter_tutorial/raw/master/image/EKF/ekf.png)

- 三点測量の観測地点で覆われる三角形の外に出るとCorrectionが弱くなって収束しない，ということに気づいた

![Alt Text](https://github.com/eisoku9618/KalmanFilter_tutorial/raw/master/image/EKF/ekf_not-good.png)

- 三点測量の地点を外側にしたらしっかりと収束した．この例では観測の方のヤコビアンの値が大事だった．

![Alt Text](https://github.com/eisoku9618/KalmanFilter_tutorial/raw/master/image/EKF/ekf_good.png)




#### Linear Kalman Filter

###### Adaptive Kalman Filter

- トンネルを通過する車を想定
- 速度一定という条件
   - 入力は加速度だが，今回は加速度が0なので無視できる
- 状態量は位置と速度
- 観測できるのは位置と速度
- ただし突然速度の観測器にトラブルが発生してノイジーになったり，トンネルに入ってなぜかxの位置がノイジーになったりする
   - 本当はトンネルに入ったらGPSをOFFにすべきだが，めんどくさいので未実装
   - 具体的にはzが4次元から2次元，Hが4x4から2x4，Rが4x4から2x2に変わるような実装にする必要あり

```bash
./LinearKalmanFilter/ConstantVelocity_AKF_with_GPS.py -h
```

- Rを可変にした場合（速度一定という知識とその速度が既知という知識を使ってリアルタイムにGPSのRとスピードメータのRを更新している）
   - これがいわゆるAKFらしい

```bash
./LinearKalmanFilter/ConstantVelocity_AKF_with_GPS.py
```

![Alt Text](https://github.com/eisoku9618/KalmanFilter_tutorial/raw/master/image/LKF/ConstantVelocity_AKF_with_GPS.png)

- Rを固定した場合
   - これは普通のLKF

```bash
./LinearKalmanFilter/ConstantVelocity_AKF_with_GPS.py --non-adaptive
```

![Alt Text](https://github.com/eisoku9618/KalmanFilter_tutorial/raw/master/image/LKF/ConstantVelocity_LKF_with_GPS.png)

- LKFの方がAKFより推定値の分散低い？
   - 「これは観測器がノイジーになった」という部分の実装を標準誤差を増やす，という実装にしており，平均値は正確な値が出ているから？
      - 平均値もずらさないといけない？
   - LKFの方はノイジーになった近辺で誤差が大きくなっている気もする？


---

- トンネルに入った車を想定
- 速度一定という条件
   - 入力は加速度だが，今回は加速度が0なので無視できる
- 状態量は位置と速度
- 観測できるのは速度
- ただし突然観測器にトラブルが発生してノイジーになった

```bash
./LinearKalmanFilter/ConstantVelocity_AKF.py -h
```

- Rを可変にした場合（速度一定という知識を使ってリアルタイムにRを更新している）
   - これが，いわゆるAKFらしい

```bash
./LinearKalmanFilter/ConstantVelocity_AKF.py -t 0.5 -r 1.0 -N 200 --vx 20 --vy 40 --noise 50
```

![Alt Text](https://github.com/eisoku9618/KalmanFilter_tutorial/raw/master/image/LKF/ConstantVelocity_AKF.png)

- Rが変わらない場合
   - これは普通のカルマンフィルタ（LKF）

```bash
./LinearKalmanFilter/ConstantVelocity_AKF.py -t 0.5 -r 1.0 -N 200 --vx 20 --vy 40 --noise 50 --non-adaptive
```

![Alt Text](https://github.com/eisoku9618/KalmanFilter_tutorial/raw/master/image/LKF/ConstantVelocity_LKF.png)

---

- GPSを搭載した台車ロボットを想定
- 速度一定という条件
   - 入力は速度
- 状態量は位置
- 観測できるのも位置
- 緑枠は分散

```bash
./LinearKalmanFilter/all-1.py
```

![Alt Text](https://github.com/eisoku9618/KalmanFilter_tutorial/raw/master/image/LKF/all-1.png)


## reference

#### Kalman Filter
- http://biorobotics.ri.cmu.edu/papers/sbp_papers/integrated3/kleeman_kalman_basics.pdf

#### Adaptive Kalman Filter
- http://nbviewer.ipython.org/github/balzer82/Kalman/blob/master/Adaptive-Kalman-Filter-CV.ipynb

#### robot application example
- http://aisl.cs.tut.ac.jp/~jun/pdffiles/moon-jrsj99.pdf
- http://d.hatena.ne.jp/meison_amsl/20130413/1365826157
- http://d.hatena.ne.jp/meison_amsl/20140614/1402731732

#### error ellipse or confidence ellipse
- http://stackoverflow.com/questions/20126061/creating-a-confidence-ellipses-in-a-sccatterplot-using-matplotlib
- https://github.com/joferkington/oost_paper_code/blob/master/error_ellipse.py
- http://d.hatena.ne.jp/meison_amsl/20140621/1403336277
