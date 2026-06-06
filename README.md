# VRPPD - Vehicle Routing Problem with Pickup and Delivery

Dự án này cài đặt và so sánh 4 chiến lược giải bài toán định tuyến xe có lấy và giao hàng (Vehicle Routing Problem with Pickup and Delivery - VRPPD):

- Greedy
- Divide & Conquer
- Branch & Bound
- Genetic Algorithm

Bộ dữ liệu sử dụng là các instance benchmark Li & Lim cho bài toán PDPTW. Trong phiên bản hiện tại, chương trình chỉ xét ràng buộc pickup-delivery và sức chứa xe; các cột time window và service time được đọc từ file nhưng không được đưa vào hàm mục tiêu/ràng buộc.

## Mục Tiêu Bài Toán

Mỗi yêu cầu vận chuyển gồm một cặp node:

- Pickup: node lấy hàng, có `demand > 0`.
- Delivery: node giao hàng tương ứng, có `demand < 0`.

Lời giải hợp lệ cần thỏa mãn:

- Mỗi xe xuất phát từ depot `0` và quay lại depot khi tính chi phí.
- Pickup của mỗi yêu cầu phải xuất hiện trước delivery tương ứng.
- Pickup và delivery của cùng một yêu cầu phải nằm trong cùng một tuyến.
- Tải trọng trên xe không được âm và không được vượt quá sức chứa xe.
- Mỗi node khác depot phải được phục vụ đúng một lần.

Mục tiêu là tối thiểu hóa tổng khoảng cách Euclidean của tất cả tuyến xe hợp lệ.

## Cấu Trúc Dự Án

```text
DAA/
|-- main.py                  # Chạy nhanh 4 thuật toán trên file mặc định lc103.txt
|-- Experiment.py            # Chạy thực nghiệm trên nhiều file và ghi CSV
|-- models.py                # Node, Request, Vehicle, Route, Solution
|-- utils.py                 # Đọc và parse file benchmark
|-- greedy.py                # Thuật toán Greedy
|-- divide_and_conquer.py    # K-Means clustering + greedy routing + 2-opt
|-- branch_and_bound.py      # Branch & Bound có giới hạn thời gian
|-- geneticAlgorithm.py      # Genetic Algorithm + local search
|-- visualize.py             # GUI trực quan hóa lộ trình (Tkinter + Matplotlib)
|-- experiment_results.csv   # Kết quả thực nghiệm gần nhất
|-- data/
|   |-- pdp_100/             # 56 file benchmark
|   |-- pdp_200/             # 60 file benchmark
|   `-- pdp_400/             # 60 file benchmark
`-- README.md
```

## Yêu Cầu Môi Trường

Khuyến nghị dùng Python 3.10 trở lên.

Cài các thư viện phụ thuộc:

```bash
pip install numpy scikit-learn matplotlib
```

Ghi chú:

- `numpy` và `scikit-learn` được dùng cho K-Means trong Divide & Conquer.
- `matplotlib` được import trong module trực quan hóa của Divide & Conquer.
- Greedy và Branch & Bound về cơ bản chỉ cần thư viện chuẩn, nhưng khi chạy `main.py` vẫn cần các dependency trên vì chương trình import cả Divide & Conquer.

## Cách Chạy

### 1. Chạy so sánh nhanh

```bash
python main.py
```

`main.py` tự tìm file theo thứ tự:

```text
lc103.txt
data/pdp_100/lc103.txt
```

Chương trình sẽ in thông tin instance, chi tiết từng tuyến xe và bảng so sánh 4 chiến lược theo:

- Tổng chi phí
- Số tuyến
- Tính hợp lệ
- Thời gian chạy

### 2. Chạy thực nghiệm nhiều file

```bash
python Experiment.py
```

`Experiment.py` mặc định chạy 20 file trong mỗi nhóm `pdp_100` và `pdp_200`, mỗi file chạy cả 4 thuật toán. Kết quả được ghi vào:

```text
experiment_results.csv
```

Cần lưu ý: cấu hình mặc định có thể tốn nhiều thời gian. Branch & Bound có giới hạn 60 giây/file, Genetic Algorithm có giới hạn 60 giây/file nhưng thực tế có thể vượt mức này vì một thế hệ GA phải chạy xong mới kiểm tra timeout.

### 3. Trực quan hóa lộ trình

```bash
python visualize.py
```

Module `visualize.py` cung cấp giao diện đồ họa để chạy thuật toán và xem kết quả trực quan trên bản đồ 2D.

#### Quy trình sử dụng

1. **Cửa sổ chọn cấu hình** — Khi chạy, một dialog Tkinter hiện ra cho phép chọn:
   - **Bộ dữ liệu**: thư mục con trong `data/` (ví dụ: `pdp_100`, `pdp_200`, `pdp_400`).
   - **File**: file benchmark cụ thể trong bộ dữ liệu đã chọn.
   - **Thuật toán**: một trong 4 chiến lược — Greedy, Divide & Conquer, Branch & Bound, Genetic Algorithm.

2. **Nhấn "▶ Chạy & Trực quan hóa"** — Chương trình chạy thuật toán được chọn trên file đã chọn.

3. **Cửa sổ kết quả** — Hiển thị bản đồ lộ trình và bảng thông tin chi tiết.

#### Bản đồ lộ trình (panel trái)

Bản đồ hiển thị toàn bộ tuyến xe trên hệ tọa độ 2D với các quy ước:

| Ký hiệu | Ý nghĩa |
|---|---|
| ★ đỏ | Depot (node 0) |
| △ tam giác | Node pickup (demand > 0) |
| □ vuông | Node delivery (demand < 0) |
| → mũi tên trên đoạn thẳng | Hướng di chuyển của xe |

- Mỗi tuyến xe được vẽ bằng một **màu riêng** (tối đa 20 màu phân biệt).
- Mỗi node được ghi nhãn số ID.
- Legend ở góc dưới trái liệt kê tất cả tuyến xe kèm số node và chi phí.

#### Bảng thông tin (panel phải)

Hiển thị các chỉ số chính:

- **Thuật toán** và **file** đang xem.
- **Tổng chi phí** (tổng khoảng cách Euclidean).
- **Số tuyến xe**, **tổng node**, **số yêu cầu**, **sức chứa xe**.
- **Feasible** — tải trọng và thứ tự pickup-delivery hợp lệ.
- **Complete** — tất cả node được phục vụ đúng một lần, không thiếu, không trùng.
- **Valid** — kết hợp cả feasible và complete.
- **Thời gian chạy** (ms).
- **Chi tiết tuyến xe** — liệt kê từng xe với số node, chi phí và trạng thái (nếu ≤ 15 tuyến).

#### Cấu hình visualizer

Trong `visualize.py`:

```python
B_AND_B_TIME_LIMIT = 30   # Giới hạn thời gian Branch & Bound (giây)
GA_TIME_LIMIT      = 60   # Giới hạn thời gian Genetic Algorithm (giây)
```

#### Yêu cầu thêm

- Cần cài `matplotlib` và backend `TkAgg` (đi kèm Tkinter trong bản Python chuẩn).
- Cửa sổ dialog sử dụng theme tối (Catppuccin Mocha), cửa sổ biểu đồ sử dụng theme tối GitHub.

## Cấu Hình Chính

Trong `main.py`:

```python
B_AND_B_TIME_LIMIT_SECONDS = 30
```

Trong `Experiment.py`:

```python
FILES_PER_GROUP = 20
B_AND_B_TIME_LIMIT = 60
GA_TIME_LIMIT = 60
```

Trong `geneticAlgorithm.py`:

```python
POPULATION_SIZE   = 50
MAX_GENERATIONS   = 200
NO_IMPROVE_LIMIT  = 40
ELITE_RATIO       = 0.10
TOURNAMENT_K      = 3
CROSSOVER_PROB    = 0.80
MUTATION_PROB     = 0.30
LOCAL_SEARCH_PROB = 0.20
```

## Tóm Tắt Các Thuật Toán

| Thuật toán | File | Ý tưởng chính | Điểm mạnh | Hạn chế |
|---|---|---|---|---|
| Greedy | `greedy.py` | Mỗi bước chọn node gần nhất có thể đi tiếp mà vẫn thỏa ràng buộc. | Rất nhanh, kết quả ổn định. | Dễ kẹt ở cực trị cục bộ. |
| Divide & Conquer | `divide_and_conquer.py` | Gom cụm request bằng K-Means, giải từng cụm bằng greedy, cải thiện bằng 2-opt. | Cân bằng tốc độ và chất lượng. | Phụ thuộc vào chất lượng chia cụm. |
| Branch & Bound | `branch_and_bound.py` | Duyệt cây trạng thái, tính lower bound và cắt nhánh có chi phí kém. | Có khả năng tìm nghiệm rất tốt trên bài nhỏ. | Không gian tìm kiếm tăng rất nhanh, phụ thuộc timeout. |
| Genetic Algorithm | `geneticAlgorithm.py` | Tiến hóa hoán vị node bằng selection, crossover, mutation và local search. | Chất lượng tốt trên nhóm nhỏ. | Chậm hơn, nhạy cảm với tham số và kích thước instance. |

## Định Dạng Dữ Liệu Đầu Vào

Mỗi file benchmark có dạng:

```text
<num_vehicles> <capacity> <speed>
<id> <x> <y> <demand> <earliest> <latest> <service_time> <pickup_idx> <delivery_idx>
...
```

Ví dụ:

```text
25  200  1
0   40  50   0    0  1236   0   0   0
1   45  68   10   0  1127  90   0  75
75  45  65  -10   0  1130  90   1   0
```

Ý nghĩa các cột chính:

| Cột | Ý nghĩa |
|---|---|
| `id` | Mã node, trong đó `0` là depot. |
| `x`, `y` | Tọa độ 2D của node. |
| `demand` | Dương là pickup, âm là delivery, `0` là depot. |
| `earliest`, `latest` | Time window, hiện không được sử dụng. |
| `service_time` | Thời gian phục vụ, hiện không được sử dụng. |
| `pickup_idx` | ID pickup tương ứng của một delivery. |
| `delivery_idx` | ID delivery tương ứng của một pickup. |

## Kết Quả Thực Nghiệm Gần Nhất

Số liệu dưới đây được tổng hợp từ `experiment_results.csv` hiện có trong repo.

### pdp_100, 20 files

| Thuật toán | Valid | Chi phí TB | Thời gian TB | Số tuyến TB | Wins |
|---|---:|---:|---:|---:|---:|
| Genetic Algorithm | 20/20 | 701.41 | 61,258.1 ms | 1.0 | 18 |
| Greedy | 20/20 | 833.01 | 8.8 ms | 1.0 | 2 |
| Branch & Bound | 20/20 | 952.73 | 60,016.6 ms | 4.0 | 0 |
| Divide & Conquer | 20/20 | 961.75 | 412.2 ms | 5.0 | 0 |

### pdp_200, 20 files

| Thuật toán | Valid | Chi phí TB | Thời gian TB | Số tuyến TB | Wins |
|---|---:|---:|---:|---:|---:|
| Greedy | 20/20 | 2,255.27 | 48.6 ms | 1.0 | 10 |
| Divide & Conquer | 20/20 | 2,371.67 | 294.0 ms | 5.0 | 7 |
| Branch & Bound | 20/20 | 2,506.99 | 60,118.9 ms | 4.4 | 1 |
| Genetic Algorithm | 20/20 | 2,839.38 | 312,413.0 ms | 1.0 | 2 |

Nhận xét ngắn:

- Trên `pdp_100`, Genetic Algorithm cho chi phí trung bình tốt nhất nhưng thời gian chạy cao.
- Trên `pdp_200`, Greedy đang có tỷ lệ thắng cao nhất trong kết quả hiện tại, đồng thời nhanh nhất.
- Branch & Bound thường bị chi phối bởi timeout nên kết quả là nghiệm tốt nhất tìm được trong giới hạn thời gian, không đảm bảo tối ưu cho instance lớn.
- Divide & Conquer chạy nhanh hơn các phương pháp tìm kiếm nặng và có kết quả khá ổn định, nhưng chất lượng phụ thuộc cách gom cụm.

## Mô Hình Dữ Liệu

Các lớp chính nằm trong `models.py`:

| Lớp | Vai trò |
|---|---|
| `Node` | Biểu diễn depot, pickup hoặc delivery. |
| `Request` | Một cặp pickup-delivery. |
| `Vehicle` | Xe với ID và sức chứa. |
| `Route` | Tuyến của một xe, gồm danh sách node và các hàm kiểm tra feasibility. |
| `Solution` | Tập hợp các route, tính tổng chi phí và kiểm tra tính khả thi. |

Một solution được coi là hợp lệ khi:

- `Solution.is_feasible()` trả về `True` cho ràng buộc tải trọng và thứ tự pickup-delivery.
- Tất cả node khác depot được phục vụ đúng một lần, không thiếu và không lặp.

## Ghi Chú Kỹ Thuật

- Chi phí di chuyển được tính bằng khoảng cách Euclidean.
- `models.cdist` có cache khoảng cách để giảm tính toán lặp lại.
- `Experiment.py` xóa cache khoảng cách trước mỗi lần chạy để tránh bộ nhớ tăng quá nhiều.
- Thư mục `.venv`, `.idea` và `__pycache__` là file môi trường/phát sinh, không phải thành phần logic của thuật toán.
