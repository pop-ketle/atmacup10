# atmacup10

主に1サブ1コミットで実験をしています。コミットログから、CVとPublic LBのスコアをみれます。

---

https://www.guruguru.science/competitions/16

## 概要
>美術作品の属性情報から、その作品がどのぐらい人々に評価されるのか?を予測します。

## Timeline
> 開催日時: 2021-03-05 18:00 〜 2021-03-13 18:00 (Asia/Tokyo)

## 評価指標
> root mean squared logarithmic error によって評価します。  
$RMSLE = \sqrt{\frac{1}{n} \sum^n_{i=1} (log(t_i - 1) - log(y_i +1))^2}$

## 取り組み
はてなブログにまとめました。
- [atmaCup#10参加記（pop-ketle版）](https://pop-ketle.hatenablog.com/entry/2021/03/17/203931)