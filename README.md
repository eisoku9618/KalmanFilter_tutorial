KalmanFilter_tutorial
=====================

To deepen my understanding of Kalman Filter

- Linear Kalman Filter
- Adaptive Kalman Filter

## How to use

#### Linear Kalman Filter

###### Adaptive Kalman Filter

```bash
./LinearKalmanFilter/ConstantVelocity_AKF.py
```

- Adaptive R version

```bash
./ConstantVelocity_AKF.py -t 0.5 -r 1.0 -N 200 --vx 20 --vy 40 --noise 50
```

![Alt Text](https://github.com/eisoku9618/KalmanFilter_tutorial/raw/master/image/LKF/ConstantVelocity_AKF.png)

- R is constant

```bash
./ConstantVelocity_AKF.py -t 0.5 -r 1.0 -N 200 --vx 20 --vy 40 --noise 50 --non-adaptive
```

![Alt Text](https://github.com/eisoku9618/KalmanFilter_tutorial/raw/master/image/LKF/ConstantVelocity_LKF.png)

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
