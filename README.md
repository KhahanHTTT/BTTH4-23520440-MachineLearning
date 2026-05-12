# Báo Cáo Kết Quả: Wine Quality Classification

## 1. Thông tin dữ liệu

| | |
|---|---|
| Dataset | Wine Quality (Red + White) |
| Tổng số mẫu | 6497 |
| Tập huấn luyện | 5197 mẫu |
| Tập kiểm tra | 1300 mẫu |
| Nhãn | 0 (quality < 6), 1 (quality ≥ 6) |
| Phân phối nhãn | 2384 (nhãn 0) / 4113 (nhãn 1) |

## 2. Tiền xử lý

- Ghép dữ liệu Red Wine và White Wine, thêm cột `type` (0 = red, 1 = white)
- Binarize nhãn: `quality >= 6 → 1`, ngược lại `→ 0`
- Chia train/test theo tỉ lệ 80/20 với `stratify=y` để đảm bảo phân phối nhãn đồng đều
- Chuẩn hoá dữ liệu bằng Z-score (tính trên tập train, áp dụng cho cả test)

## 3. Kết quả

### Assignment 1 – Decision Tree (NumPy)

| | F1 Score |
|---|---|
| Train | 0.8276 |
| Test | 0.8017 |

> Cài đặt Decision Tree thuần NumPy với Gini impurity.

---

### Assignment 2 – Random Forest (NumPy)

| | F1 Score |
|---|---|
| Train | 0.8914 |
| Test | 0.8344 |

> Cài đặt Random Forest thuần NumPy sử dụng Bootstrap Sampling + Majority Voting. F1 Test cải thiện **+0.033** so với Decision Tree đơn lẻ.

---

### Assignment 3 – Decision Tree (sklearn)

| | F1 Score |
|---|---|
| Train | 0.8276 |
| Test | 0.8027 |

> Kết quả gần giống Decision Tree NumPy → chứng tỏ cài đặt thủ công ở Assignment 1 là chính xác.

---

### Assignment 3 – Random Forest (sklearn)

| | F1 Score |
|---|---|
| Train | 0.8477 |
| Test | 0.8273 |

> Random Forest sklearn cho kết quả tương đương Random Forest NumPy (0.8273 vs 0.8344).

## 4. So sánh tổng hợp

| Mô hình | F1 Train | F1 Test |
|---|---|---|
| Decision Tree (NumPy) | 0.8276 | 0.8017 |
| Random Forest (NumPy) | 0.8914 | 0.8344 |
| Decision Tree (sklearn) | 0.8276 | 0.8027 |
| Random Forest (sklearn) | 0.8477 | 0.8273 |

## 5. Nhận xét

- **Random Forest vượt trội hơn Decision Tree** nhờ cơ chế Bagging giảm variance
- **Cài đặt NumPy cho kết quả tương đương sklearn** → implement đúng thuật toán
- Random Forest NumPy đạt F1 Test **0.8344**, cao hơn cả Random Forest sklearn (0.8273)
