Numpy là một thư viện toán học của Python
các thư viện Pandas, Matplotlib ... được xây dựng trên thư viện NumPy này

# Hàm numpy.zeros()

## 1. Giới thiệu

`numpy.zeros()` là một hàm trong thư viện NumPy dùng để tạo mảng với tất cả phần tử là số 0.

## 2. Cú pháp

```python
numpy.zeros(shape, dtype=float, order='C')
```

Trong đó:

- **shape**: kích thước mảng (tuple hoặc int)
- **dtype**: kiểu dữ liệu của các phần tử (mặc định là float)
- **order**: thứ tự lưu trữ trong bộ nhớ ('C' hoặc 'F')

## 3. Các ví dụ cơ bản

### 3.1 Tạo mảng 1 chiều

```python
import numpy as np

# Tạo mảng 5 phần tử
arr1 = np.zeros(5)
print(arr1)  # [0. 0. 0. 0. 0.]

# Chỉ định kiểu dữ liệu int
arr2 = np.zeros(3, dtype=int)
print(arr2)  # [0 0 0]
```

### 3.2 Tạo mảng 2 chiều

```python
# Tạo ma trận 2x3
arr3 = np.zeros((2, 3))
print(arr3)
# [[0. 0. 0.]
#  [0. 0. 0.]]

# Tạo ma trận 3x2 kiểu int
arr4 = np.zeros((3, 2), dtype=int)
print(arr4)
# [[0 0]
#  [0 0]
#  [0 0]]
```

### 3.3 Tạo mảng 3 chiều

```python
# Tạo mảng 2x3x2
arr5 = np.zeros((2, 3, 2))
print(arr5)
# [[[0. 0.]
#   [0. 0.]
#   [0. 0.]]
#  [[0. 0.]
#   [0. 0.]
#   [0. 0.]]]
```

## 4. Các kiểu dữ liệu phổ biến

```python
# Float (mặc định)
arr_float = np.zeros(3)                 # [0. 0. 0.]

# Integer
arr_int = np.zeros(3, dtype=int)        # [0 0 0]

# Boolean
arr_bool = np.zeros(3, dtype=bool)      # [False False False]

# Complex
arr_complex = np.zeros(3, dtype=complex) # [0.+0.j 0.+0.j 0.+0.j]
```

## 5. Ứng dụng thực tế

### 5.1 Khởi tạo ma trận trọng số

```python
# Khởi tạo ma trận trọng số cho neural network
weights = np.zeros((input_size, output_size))
```

### 5.2 Tạo mask cho xử lý ảnh

```python
# Tạo mask đen (0) cho xử lý ảnh
mask = np.zeros((height, width), dtype=np.uint8)
```

### 5.3 Khởi tạo mảng tích lũy

```python
# Mảng đếm tần suất
frequency = np.zeros(10, dtype=int)
```

## 6. Lưu ý quan trọng

1. **Bộ nhớ**:

   - Zeros() tạo ra một mảng contiguous trong bộ nhớ
   - Tiết kiệm bộ nhớ hơn so với tạo list Python

2. **Hiệu suất**:

   - Nhanh hơn so với tạo list Python
   - Phù hợp cho khởi tạo mảng lớn

3. **Kiểu dữ liệu**:
   - Mặc định là float64
   - Nên chỉ định dtype phù hợp để tối ưu bộ nhớ

## 7. So sánh với các hàm tương tự

```python
# np.zeros(): Tạo mảng toàn số 0
arr1 = np.zeros(3)  # [0. 0. 0.]

# np.ones(): Tạo mảng toàn số 1
arr2 = np.ones(3)   # [1. 1. 1.]

# np.empty(): Tạo mảng không khởi tạo giá trị
arr3 = np.empty(3)  # [giá_trị_ngẫu_nhiên]

# np.full(): Tạo mảng với giá trị tùy chọn
arr4 = np.full(3, 5)  # [5. 5. 5.]
```

# Hàm numpy.ones()

## 1. Giới thiệu

`numpy.ones()` là hàm trong thư viện NumPy dùng để tạo mảng với tất cả phần tử là số 1.

## 2. Cú pháp

```python
numpy.ones(shape, dtype=float, order='C')
```

Trong đó:

- **shape**: kích thước mảng (tuple hoặc int)
- **dtype**: kiểu dữ liệu của các phần tử (mặc định là float)
- **order**: thứ tự lưu trữ trong bộ nhớ ('C' hoặc 'F')

## 3. Các ví dụ cơ bản

### 3.1 Tạo mảng 1 chiều

```python
import numpy as np

# Tạo mảng 5 phần tử
arr1 = np.ones(5)
print(arr1)  # [1. 1. 1. 1. 1.]

# Chỉ định kiểu dữ liệu int
arr2 = np.ones(3, dtype=int)
print(arr2)  # [1 1 1]
```

### 3.2 Tạo mảng 2 chiều

```python
# Tạo ma trận 2x3
arr3 = np.ones((2, 3))
print(arr3)
# [[1. 1. 1.]
#  [1. 1. 1.]]

# Tạo ma trận 3x2 kiểu int
arr4 = np.ones((3, 2), dtype=int)
print(arr4)
# [[1 1]
#  [1 1]
#  [1 1]]
```

### 3.3 Tạo mảng 3 chiều

```python
# Tạo mảng 2x2x3
arr5 = np.ones((2, 2, 3))
print(arr5)
# [[[1. 1. 1.]
#   [1. 1. 1.]]
#  [[1. 1. 1.]
#   [1. 1. 1.]]]
```

## 4. Các kiểu dữ liệu phổ biến

```python
# Float (mặc định)
arr_float = np.ones(3)                 # [1. 1. 1.]

# Integer
arr_int = np.ones(3, dtype=int)        # [1 1 1]

# Boolean
arr_bool = np.ones(3, dtype=bool)      # [True True True]

# Complex
arr_complex = np.ones(3, dtype=complex) # [1.+0.j 1.+0.j 1.+0.j]
```

## 5. Ứng dụng thực tế

### 5.1 Tạo ma trận đơn vị (Identity Matrix)

```python
# Tạo ma trận đơn vị từ ones()
n = 3
identity = np.ones((n, n)) * np.eye(n)
print(identity)
# [[1. 0. 0.]
#  [0. 1. 0.]
#  [0. 0. 1.]]
```

### 5.2 Tạo mask cho xử lý ảnh

```python
# Tạo mask trắng (1) cho xử lý ảnh
mask = np.ones((height, width), dtype=np.uint8) * 255
```

### 5.3 Khởi tạo trọng số Neural Network

```python
# Khởi tạo trọng số ban đầu là 1
weights = np.ones((input_size, hidden_size))
```

## 6. Kết hợp với các phép toán

### 6.1 Tạo mảng với giá trị tùy chọn

```python
# Tạo mảng toàn số 5
arr = np.ones(4) * 5
print(arr)  # [5. 5. 5. 5.]

# Tạo mảng với các giá trị âm
neg_arr = -np.ones(3)
print(neg_arr)  # [-1. -1. -1.]
```

### 6.2 Kết hợp với phép toán ma trận

```python
# Tạo ma trận A
A = np.ones((2, 2))
# Tạo ma trận B
B = np.ones((2, 2)) * 2
# Phép nhân ma trận
C = np.dot(A, B)
print(C)
# [[2. 2.]
#  [2. 2.]]
```

## 7. Lưu ý quan trọng

1. **Hiệu suất**:

   - Nhanh hơn so với tạo list Python
   - Tối ưu cho tính toán vector hóa

2. **Bộ nhớ**:

   - Sử dụng bộ nhớ liên tục
   - Hiệu quả cho mảng lớn

3. **So sánh với các hàm tương tự**:

```python
# np.ones(): Tạo mảng toàn số 1
arr1 = np.ones(3)  # [1. 1. 1.]

# np.zeros(): Tạo mảng toàn số 0
arr2 = np.zeros(3) # [0. 0. 0.]

# np.full(): Tạo mảng với giá trị tùy chọn
arr3 = np.full(3, 5) # [5. 5. 5.]
```

## 8. Các biến thể của ones

### 8.1 ones_like()

```python
# Tạo mảng mới giống kích thước mảng có sẵn
original = np.array([[1, 2], [3, 4]])
ones_array = np.ones_like(original)
print(ones_array)
# [[1 1]
#  [1 1]]
```

### 8.2 Kết hợp reshape

```python
# Tạo và reshape cùng lúc
arr = np.ones(6).reshape(2, 3)
print(arr)
# [[1. 1. 1.]
#  [1. 1. 1.]]
```

# Hàm numpy.full()

## 1. Giới thiệu

`numpy.full()` là hàm trong NumPy dùng để tạo mảng với tất cả phần tử là một giá trị được chỉ định.

## 2. Cú pháp

```python
numpy.full(shape, fill_value, dtype=None, order='C')
```

Trong đó:

- **shape**: kích thước mảng (tuple hoặc int)
- **fill_value**: giá trị để điền vào mảng
- **dtype**: kiểu dữ liệu của các phần tử
- **order**: thứ tự lưu trữ ('C' hoặc 'F')

## 3. Các ví dụ cơ bản

### 3.1 Tạo mảng 1 chiều

```python
import numpy as np

# Tạo mảng 5 phần tử giá trị 7
arr1 = np.full(5, 7)
print(arr1)  # [7 7 7 7 7]

# Chỉ định kiểu dữ liệu float
arr2 = np.full(3, 3.14, dtype=float)
print(arr2)  # [3.14 3.14 3.14]
```

### 3.2 Tạo mảng 2 chiều

```python
# Tạo ma trận 2x3 với giá trị 5
arr3 = np.full((2, 3), 5)
print(arr3)
# [[5 5 5]
#  [5 5 5]]

# Tạo ma trận 3x2 với giá trị -1
arr4 = np.full((3, 2), -1)
print(arr4)
# [[-1 -1]
#  [-1 -1]
#  [-1 -1]]
```

### 3.3 Tạo mảng 3 chiều

```python
# Tạo mảng 2x2x2 với giá trị 9
arr5 = np.full((2, 2, 2), 9)
print(arr5)
# [[[9 9]
#   [9 9]]
#  [[9 9]
#   [9 9]]]
```

## 4. Các kiểu dữ liệu và giá trị đặc biệt

```python
# Số thực
arr_float = np.full(3, 3.14)  # [3.14 3.14 3.14]

# Số nguyên âm
arr_neg = np.full(3, -5)      # [-5 -5 -5]

# Boolean
arr_bool = np.full(3, True)   # [True True True]

# Số phức
arr_complex = np.full(3, 1+2j) # [1.+2.j 1.+2.j 1.+2.j]
```

## 5. Ứng dụng thực tế

### 5.1 Khởi tạo ma trận bias

```python
# Khởi tạo bias cho neural network
bias = np.full((1, neurons), 0.01)
```

### 5.2 Tạo mask cho xử lý ảnh

```python
# Tạo mask với giá trị alpha
alpha_mask = np.full((height, width), 128, dtype=np.uint8)
```

### 5.3 Khởi tạo mảng với giá trị mặc định

```python
# Mảng điểm số với giá trị mặc định -1
scores = np.full((num_students, num_subjects), -1)
```

## 6. Kết hợp với các phép toán

### 6.1 Thay đổi giá trị có điều kiện

```python
# Tạo mảng ban đầu
arr = np.full((3, 3), 5)
# Thay đổi giá trị theo điều kiện
arr[arr > 3] = 0
```

### 6.2 Tính toán với mảng khác

```python
# Tạo hai mảng
A = np.full((2, 2), 4)
B = np.full((2, 2), 2)
# Phép nhân element-wise
C = A * B
print(C)
# [[8 8]
#  [8 8]]
```

## 7. Các biến thể và hàm liên quan

### 7.1 full_like()

```python
# Tạo mảng mới với kích thước của mảng có sẵn
original = np.array([[1, 2], [3, 4]])
new_arr = np.full_like(original, 7)
print(new_arr)
# [[7 7]
#  [7 7]]
```

### 7.2 Kết hợp với reshape

```python
# Tạo và reshape
arr = np.full(6, 3).reshape(2, 3)
print(arr)
# [[3 3 3]
#  [3 3 3]]
```

## 8. So sánh với các hàm tương tự

```python
# np.full(): Giá trị tùy chọn
arr1 = np.full(3, 5)    # [5 5 5]

# np.zeros(): Toàn số 0
arr2 = np.zeros(3)      # [0. 0. 0.]

# np.ones(): Toàn số 1
arr3 = np.ones(3)       # [1. 1. 1.]

# np.empty(): Không khởi tạo giá trị
arr4 = np.empty(3)      # [giá_trị_ngẫu_nhiên]
```

## 9. Lưu ý quan trọng

1. **Hiệu suất**:

   - Hiệu quả cho khởi tạo mảng lớn
   - Tối ưu hơn so với vòng lặp Python

2. **Bộ nhớ**:

   - Sử dụng bộ nhớ liên tục
   - Quản lý bộ nhớ hiệu quả

3. **Các trường hợp đặc biệt**:

   - Có thể dùng để tạo mảng với giá trị NaN hoặc inf

   ```python
   nan_array = np.full(3, np.nan)
   inf_array = np.full(3, np.inf)
   ```

4. **Thực hành tốt**:
   - Luôn chỉ định dtype khi cần thiết
   - Kiểm tra kích thước mảng trước khi tạo
   - Xem xét sử dụng full_like() cho mảng có sẵn

# Hàm numpy.arange()

## 1. Giới thiệu

`numpy.arange()` là hàm trong NumPy dùng để tạo mảng với dãy số theo một khoảng cách đều nhau, tương tự range() của Python nhưng trả về mảng NumPy.

## 2. Cú pháp

```python
numpy.arange([start,] stop[, step,], dtype=None)
```

Trong đó:

- **start**: giá trị bắt đầu (mặc định là 0)
- **stop**: giá trị kết thúc (không bao gồm)
- **step**: bước nhảy (mặc định là 1)
- **dtype**: kiểu dữ liệu của các phần tử

## 3. Các ví dụ cơ bản

### 3.1 Chỉ có stop

```python
import numpy as np

# Tạo mảng từ 0 đến 5
arr1 = np.arange(5)
print(arr1)  # [0 1 2 3 4]

# Chỉ định kiểu dữ liệu float
arr2 = np.arange(4, dtype=float)
print(arr2)  # [0. 1. 2. 3.]
```

### 3.2 Có start và stop

```python
# Tạo mảng từ 2 đến 8
arr3 = np.arange(2, 8)
print(arr3)  # [2 3 4 5 6 7]

# Mảng số thực
arr4 = np.arange(1.5, 5.5)
print(arr4)  # [1.5 2.5 3.5 4.5]
```

### 3.3 Đầy đủ start, stop và step

```python
# Tạo mảng với bước nhảy 2
arr5 = np.arange(0, 10, 2)
print(arr5)  # [0 2 4 6 8]

# Mảng số âm giảm dần
arr6 = np.arange(0, -10, -2)
print(arr6)  # [ 0 -2 -4 -6 -8]
```

## 4. Các trường hợp đặc biệt

### 4.1 Số thực làm bước nhảy

```python
# Bước nhảy là số thực
arr = np.arange(0, 2, 0.3)
print(arr)  # [0.  0.3 0.6 0.9 1.2 1.5 1.8]
```

### 4.2 Mảng rỗng

```python
# Khi start >= stop với step dương
empty1 = np.arange(5, 0)
print(empty1)  # []

# Khi start <= stop với step âm
empty2 = np.arange(0, 5, -1)
print(empty2)  # []
```

## 5. Ứng dụng thực tế

### 5.1 Tạo vector chỉ số

```python
# Tạo chỉ số cho mảng
indices = np.arange(len(data))
```

### 5.2 Tạo dãy thời gian

```python
# Tạo timestamps theo giây
timestamps = np.arange(0, 3600, 60)  # mỗi phút
```

### 5.3 Tạo giá trị x cho đồ thị

```python
# Tạo giá trị x cho plot
x = np.arange(-5, 5, 0.1)
y = np.sin(x)
```

## 6. Kết hợp với các hàm khác

### 6.1 Reshape mảng

```python
# Tạo ma trận 3x4
matrix = np.arange(12).reshape(3, 4)
print(matrix)
# [[ 0  1  2  3]
#  [ 4  5  6  7]
#  [ 8  9 10 11]]
```

### 6.2 Thao tác toán học

```python
# Phép toán với mảng
arr = np.arange(5)
print(arr * 2)        # [0 2 4 6 8]
print(arr ** 2)       # [0 1 4 9 16]
print(np.sin(arr))    # [0. 0.84147098 0.90929743 0.14112001 -0.7568025]
```

## 7. So sánh với các hàm tương tự

### 7.1 arange vs linspace

```python
# arange: chỉ định bước nhảy
arr1 = np.arange(0, 1, 0.2)
print(arr1)  # [0.  0.2 0.4 0.6 0.8]

# linspace: chỉ định số lượng điểm
arr2 = np.linspace(0, 1, 5)
print(arr2)  # [0.   0.25 0.5  0.75 1.  ]
```

### 7.2 arange vs range

```python
# range: tạo iterator
r = range(5)
print(list(r))  # [0, 1, 2, 3, 4]

# arange: tạo mảng numpy
a = np.arange(5)
print(a)  # [0 1 2 3 4]
```

## 8. Lưu ý quan trọng

1. **Vấn đề với số thực**:

```python
# Có thể gặp vấn đề với số thực do làm tròn
arr = np.arange(0, 1, 0.1)
print(arr)  # [0.  0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9]
# Số phần tử có thể không như mong đợi do sai số
```

2. **Bộ nhớ**:

```python
# Cẩn thận với mảng lớn
large_arr = np.arange(10**7)  # Tạo 10 triệu phần tử
```

3. **Hiệu suất**:

```python
# arange nhanh hơn tạo list thủ công
arr = np.arange(1000)  # hiệu quả
# list(range(1000))    # chậm hơn
```

## 9. Các trường hợp sử dụng phổ biến

### 9.1 Tạo chỉ số cho vòng lặp

```python
for i in np.arange(start, stop, step):
    # xử lý với i
    pass
```

### 9.2 Tạo dữ liệu huấn luyện

```python
# Tạo dữ liệu training
X = np.arange(-10, 10, 0.1)
y = X**2 + np.random.normal(0, 1, len(X))
```

### 9.3 Tạo ma trận mesh

```python
x = np.arange(-5, 5, 0.5)
y = np.arange(-5, 5, 0.5)
X, Y = np.meshgrid(x, y)
```

# Hàm numpy.linspace()

## 1. Giới thiệu

`numpy.linspace()` là hàm trong NumPy dùng để tạo mảng với n phần tử được chia đều trong một khoảng cho trước.

## 2. Cú pháp

```python
numpy.linspace(start, stop, num=50, endpoint=True, retstep=False, dtype=None, axis=0)
```

Trong đó:

- **start**: giá trị bắt đầu
- **stop**: giá trị kết thúc
- **num**: số lượng phần tử (mặc định là 50)
- **endpoint**: bao gồm điểm cuối hay không (mặc định True)
- **retstep**: trả về khoảng cách giữa các điểm (mặc định False)
- **dtype**: kiểu dữ liệu của mảng

## 3. Các ví dụ cơ bản

### 3.1 Tạo mảng cơ bản

```python
import numpy as np

# Tạo 5 số từ 0 đến 1
arr1 = np.linspace(0, 1, 5)
print(arr1)  # [0.   0.25 0.5  0.75 1.  ]

# Tạo 4 số từ 0 đến 10
arr2 = np.linspace(0, 10, 4)
print(arr2)  # [ 0.  3.33333333  6.66666667 10.  ]
```

### 3.2 Không bao gồm điểm cuối

```python
# 5 số từ 0 đến 1, không bao gồm 1
arr = np.linspace(0, 1, 5, endpoint=False)
print(arr)  # [0.  0.2 0.4 0.6 0.8]
```

### 3.3 Lấy thêm khoảng cách giữa các điểm

```python
# Lấy thêm bước nhảy
arr, step = np.linspace(0, 1, 5, retstep=True)
print(f"Array: {arr}")  # Array: [0.   0.25 0.5  0.75 1.  ]
print(f"Step size: {step}")  # Step size: 0.25
```

## 4. Ứng dụng thực tế

### 4.1 Vẽ đồ thị hàm số

```python
# Tạo điểm cho đồ thị sin
x = np.linspace(-np.pi, np.pi, 100)
y = np.sin(x)

# Dùng với matplotlib
import matplotlib.pyplot as plt
plt.plot(x, y)
plt.show()
```

### 4.2 Tạo vector cho machine learning

```python
# Tạo dữ liệu training đều nhau
X = np.linspace(-10, 10, 1000)
y = X**2 + np.random.normal(0, 1, 1000)
```

### 4.3 Tạo thang đo logarit

```python
# Tạo thang đo logarit
log_scale = np.logspace(0, 2, 5, base=10)
print(log_scale)  # [1. 3.16227766 10. 31.6227766 100.]
```

## 5. So sánh với arange()

### 5.1 Độ chính xác

```python
# linspace() chính xác hơn với số điểm
arr1 = np.linspace(0, 1, 5)  # [0.   0.25 0.5  0.75 1.  ]

# arange() có thể gặp vấn đề với số thực
arr2 = np.arange(0, 1.1, 0.25)  # [0.  0.25 0.5  0.75 1.  ]
```

### 5.2 Kiểm soát số lượng điểm

```python
# linspace() kiểm soát số lượng điểm
arr1 = np.linspace(0, 1, 5)  # 5 điểm

# arange() kiểm soát khoảng cách
arr2 = np.arange(0, 1, 0.2)  # không chắc chắn số điểm
```

## 6. Các trường hợp đặc biệt

### 6.1 Một điểm

```python
# Tạo mảng một điểm
arr = np.linspace(5, 5, 1)
print(arr)  # [5.]
```

### 6.2 Số phức

```python
# Làm việc với số phức
arr = np.linspace(0, 1+1j, 5)
print(arr)  # [0.+0.j   0.25+0.25j 0.5+0.5j   0.75+0.75j 1.+1.j]
```

## 7. Các ứng dụng nâng cao

### 7.1 Tạo ma trận mesh

```python
# Tạo lưới 2D
x = np.linspace(-2, 2, 5)
y = np.linspace(-2, 2, 5)
X, Y = np.meshgrid(x, y)
```

### 7.2 Tạo phân phối xác suất

```python
# Tạo phân phối chuẩn
x = np.linspace(-3, 3, 100)
normal_dist = 1/np.sqrt(2*np.pi) * np.exp(-x**2/2)
```

## 8. Lưu ý quan trọng

1. **Hiệu suất và Bộ nhớ**:

```python
# Cẩn thận với số lượng điểm lớn
large_arr = np.linspace(0, 1, 10**6)  # Tốn nhiều bộ nhớ
```

2. **Độ chính xác số học**:

```python
# linspace() cho kết quả chính xác và dễ đoán hơn
# với số thực so với arange()
```

3. **Khi nào dùng linspace()**:

- Khi cần số lượng điểm cụ thể
- Khi làm việc với khoảng số thực
- Khi vẽ đồ thị hàm số
- Khi cần chia đều một khoảng

## 9. Ví dụ thực tế

### 9.1 Tạo thang màu

```python
# Tạo gradient màu
colors = np.linspace(0, 255, 10, dtype=int)
```

### 9.2 Tạo tín hiệu âm thanh

```python
# Tạo tín hiệu sin
t = np.linspace(0, 1, 44100)  # 1 giây ở 44.1kHz
signal = np.sin(2 * np.pi * 440 * t)  # Tần số 440Hz
```

### 9.3 Animation frames

```python
# Tạo frames cho animation
frames = np.linspace(0, 2*np.pi, 60)  # 60 frames/vòng
```

# Các hàm Random trong NumPy

## 1. Giới thiệu

NumPy cung cấp module `numpy.random` để tạo số ngẫu nhiên với nhiều phân phối khác nhau.

## 2. Các hàm cơ bản

### 2.1 rand() - Phân phối đều [0,1)

```python
import numpy as np

# Tạo số ngẫu nhiên từ 0 đến 1
x = np.random.rand()      # một số
arr1 = np.random.rand(4)  # mảng 4 số
arr2 = np.random.rand(2,3)  # ma trận 2x3

print(x)     # ví dụ: 0.4353
print(arr1)  # ví dụ: [0.12 0.54 0.87 0.36]
print(arr2)
# ví dụ:
# [[0.12 0.54 0.87]
#  [0.36 0.92 0.45]]
```

### 2.2 randn() - Phân phối chuẩn (Gaussian)

```python
# Tạo số ngẫu nhiên theo phân phối chuẩn
x = np.random.randn()      # một số
arr1 = np.random.randn(4)  # mảng 4 số
arr2 = np.random.randn(2,3)  # ma trận 2x3

print(x)     # ví dụ: -0.2153
print(arr1)  # ví dụ: [ 0.12 -1.54  0.87 -0.36]
print(arr2)
# ví dụ:
# [[ 0.12 -1.54  0.87]
#  [-0.36  0.92  0.45]]
```

### 2.3 randint() - Số nguyên ngẫu nhiên

```python
# Tạo số nguyên ngẫu nhiên
x = np.random.randint(10)        # từ 0 đến 9
y = np.random.randint(5, 10)     # từ 5 đến 9
arr = np.random.randint(1, 10, size=(3,3))  # ma trận 3x3

print(x)    # ví dụ: 7
print(y)    # ví dụ: 6
print(arr)
# ví dụ:
# [[3 7 2]
#  [8 1 5]
#  [4 9 6]]
```

## 3. Các hàm phân phối xác suất

### 3.1 normal() - Phân phối chuẩn với tham số

```python
# Phân phối chuẩn với mean=0, std=1
arr1 = np.random.normal(0, 1, 1000)

# Phân phối chuẩn với mean=100, std=15
arr2 = np.random.normal(100, 15, 1000)

print(np.mean(arr1))    # gần 0
print(np.std(arr1))     # gần 1
print(np.mean(arr2))    # gần 100
print(np.std(arr2))     # gần 15
```

### 3.2 uniform() - Phân phối đều với tham số

```python
# Phân phối đều từ low đến high
arr = np.random.uniform(0, 10, 1000)
print(np.min(arr))   # gần 0
print(np.max(arr))   # gần 10
```

### 3.3 choice() - Chọn ngẫu nhiên từ mảng

```python
# Chọn ngẫu nhiên từ mảng
arr = np.array([1, 2, 3, 4, 5])
x = np.random.choice(arr)             # một phần tử
samples = np.random.choice(arr, 3)    # 3 phần tử
with_prob = np.random.choice(arr, 3, p=[0.1, 0.2, 0.3, 0.2, 0.2])  # với xác suất

print(x)            # ví dụ: 3
print(samples)      # ví dụ: [1 5 2]
print(with_prob)    # ví dụ: [3 3 2]
```

## 4. Các hàm xáo trộn

### 4.1 shuffle() - Xáo trộn mảng

```python
# Xáo trộn mảng tại chỗ
arr = np.array([1, 2, 3, 4, 5])
np.random.shuffle(arr)
print(arr)  # ví dụ: [3 1 5 2 4]

# Xáo trộn ma trận
matrix = np.arange(9).reshape(3,3)
np.random.shuffle(matrix)  # chỉ xáo trộn theo hàng
```

### 4.2 permutation() - Tạo hoán vị ngẫu nhiên

```python
# Tạo hoán vị mới (không thay đổi mảng gốc)
arr = np.array([1, 2, 3, 4, 5])
perm = np.random.permutation(arr)
print(arr)   # [1 2 3 4 5]
print(perm)  # ví dụ: [3 1 5 2 4]
```

## 5. Kiểm soát ngẫu nhiên

### 5.1 seed() - Đặt hạt giống

```python
# Đặt seed để tái tạo kết quả
np.random.seed(42)
print(np.random.rand())  # luôn cho kết quả giống nhau
print(np.random.randint(100))  # luôn cho kết quả giống nhau
```

## 6. Ứng dụng thực tế

### 6.1 Tạo dữ liệu giả

```python
# Tạo dataset cho machine learning
X = np.random.normal(0, 1, (100, 2))  # 100 samples, 2 features
y = np.random.randint(0, 2, 100)      # nhãn nhị phân
```

### 6.2 Mô phỏng Monte Carlo

```python
# Mô phỏng tung xúc xắc
rolls = np.random.randint(1, 7, 1000)
print(np.mean(rolls))  # gần 3.5
```

### 6.3 Bootstrap sampling

```python
# Lấy mẫu bootstrap
data = np.array([1, 2, 3, 4, 5])
bootstrap_samples = np.random.choice(data, size=(1000, len(data)), replace=True)
```

## 7. Các phân phối xác suất khác

### 7.1 poisson() - Phân phối Poisson

```python
# Mô phỏng số sự kiện trong khoảng thời gian
events = np.random.poisson(lam=5, size=1000)  # trung bình 5 sự kiện
```

### 7.2 binomial() - Phân phối nhị thức

```python
# Mô phỏng tung đồng xu n lần
flips = np.random.binomial(n=10, p=0.5, size=1000)
```

## 8. Lưu ý quan trọng

1. **Seed và tái tạo**:

```python
# Đặt seed cho tái tạo kết quả
np.random.seed(42)
# Kết quả giống nhau mỗi lần chạy
```

2. **Hiệu suất**:

```python
# Tạo nhiều số ngẫu nhiên cùng lúc hiệu quả hơn
arr = np.random.rand(1000000)  # tốt
# so với vòng lặp tạo từng số
```

3. **Bộ nhớ**:

```python
# Cẩn thận với mảng lớn
large_arr = np.random.rand(10**8)  # cần nhiều bộ nhớ
```

# Indexing và Slicing trong NumPy Arrays

## 1. Giới thiệu

Indexing và Slicing là các phương pháp để truy cập và trích xuất dữ liệu từ mảng NumPy.

## 2. Indexing cơ bản

### 2.1 Mảng 1 chiều

```python
import numpy as np

arr = np.array([1, 2, 3, 4, 5])

# Index dương (từ trái sang phải)
print(arr[0])     # 1 (phần tử đầu tiên)
print(arr[2])     # 3
print(arr[-1])    # 5 (phần tử cuối cùng)
print(arr[-2])    # 4 (phần tử áp cuối)
```

### 2.2 Mảng 2 chiều

```python
arr_2d = np.array([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9]])

# Truy cập theo hàng, cột
print(arr_2d[0, 0])    # 1 (hàng 0, cột 0)
print(arr_2d[1, 2])    # 6 (hàng 1, cột 2)
print(arr_2d[-1, -1])  # 9 (hàng cuối, cột cuối)

# Truy cập từng hàng
print(arr_2d[0])    # [1 2 3] (hàng đầu tiên)
print(arr_2d[-1])   # [7 8 9] (hàng cuối)
```

### 2.3 Mảng 3 chiều

```python
arr_3d = np.array([[[1, 2], [3, 4]],
                   [[5, 6], [7, 8]]])

print(arr_3d[0, 0, 0])    # 1
print(arr_3d[1, 1, 1])    # 8
```

## 3. Slicing cơ bản

### 3.1 Mảng 1 chiều

```python
arr = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

# Cú pháp: arr[start:end:step]
print(arr[2:5])     # [2 3 4] (từ index 2 đến 4)
print(arr[:4])      # [0 1 2 3] (từ đầu đến index 3)
print(arr[6:])      # [6 7 8 9] (từ index 6 đến cuối)
print(arr[1:7:2])   # [1 3 5] (từ 1 đến 6, bước 2)
print(arr[::-1])    # [9 8 7 6 5 4 3 2 1 0] (đảo ngược)
```

### 3.2 Mảng 2 chiều

```python
arr_2d = np.array([[1, 2, 3, 4],
                   [5, 6, 7, 8],
                   [9, 10, 11, 12]])

# Lấy các hàng
print(arr_2d[0:2])    # 2 hàng đầu
# [[1 2 3 4]
#  [5 6 7 8]]

# Lấy các cột
print(arr_2d[:, 1:3])  # cột 1 và 2
# [[ 2  3]
#  [ 6  7]
#  [10 11]]

# Lấy submatrix
print(arr_2d[0:2, 1:3])
# [[2 3]
#  [6 7]]
```

### 3.3 Slicing với bước nhảy

```python
arr_2d = np.array([[1, 2, 3, 4],
                   [5, 6, 7, 8],
                   [9, 10, 11, 12],
                   [13, 14, 15, 16]])

# Lấy mỗi hàng thứ 2
print(arr_2d[::2])
# [[ 1  2  3  4]
#  [ 9 10 11 12]]

# Lấy mỗi cột thứ 2
print(arr_2d[:, ::2])
# [[ 1  3]
#  [ 5  7]
#  [ 9 11]
#  [13 15]]
```

## 4. Advanced Indexing

### 4.1 Boolean Indexing

```python
arr = np.array([1, 2, 3, 4, 5])

# Tạo mask boolean
mask = arr > 2
print(mask)  # [False False True True True]

# Lọc các phần tử thỏa điều kiện
print(arr[mask])  # [3 4 5]

# Kết hợp nhiều điều kiện
print(arr[(arr > 2) & (arr < 5)])  # [3 4]
```

### 4.2 Fancy Indexing

```python
arr = np.array([10, 20, 30, 40, 50])

# Lấy các phần tử theo index
indices = [0, 2, 4]
print(arr[indices])  # [10 30 50]

# Với mảng 2D
arr_2d = np.array([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9]])

# Lấy hàng cụ thể
rows = np.array([0, 2])
print(arr_2d[rows])
# [[1 2 3]
#  [7 8 9]]

# Lấy các phần tử cụ thể
print(arr_2d[rows, 1])  # [2 8]
```

## 5. Ví dụ ứng dụng thực tế

### 5.1 Xử lý ảnh

```python
# Giả sử có ảnh RGB
image = np.random.randint(0, 256, (100, 100, 3))

# Lấy kênh màu đỏ
red_channel = image[:, :, 0]

# Cắt một phần ảnh
region = image[20:50, 30:60]
```

### 5.2 Xử lý dữ liệu

```python
data = np.array([[1, 2, 3],
                 [4, 5, 6],
                 [7, 8, 9],
                 [10, 11, 12]])

# Lấy cột đầu tiên làm features
features = data[:, 0]

# Lọc dữ liệu theo điều kiện
filtered_data = data[data[:, 1] > 5]
```

## 6. Lưu ý quan trọng

1. **View vs Copy**

```python
arr = np.array([1, 2, 3, 4, 5])

# Slicing tạo view
view = arr[1:4]
view[0] = 10  # thay đổi arr gốc

# Fancy indexing tạo copy
copy = arr[[1, 2, 3]]
copy[0] = 10  # không thay đổi arr gốc
```

2. **Tránh lỗi phổ biến**

```python
# Index ngoài range
# arr[10]  # IndexError

# Số chiều không khớp
arr_2d = np.array([[1, 2], [3, 4]])
# arr_2d[0, 0, 0]  # IndexError
```

3. **Best Practices**

```python
# Kiểm tra shape trước khi indexing
print(arr.shape)

# Sử dụng negative indexing cẩn thận
# Ưu tiên sử dụng positive indexing khi có thể
```

# Array Concatenation và Splitting trong NumPy

## 1. Concatenation (Nối mảng)

### 1.1 np.concatenate()

```python
import numpy as np

# Nối mảng 1 chiều
arr1 = np.array([1, 2, 3])
arr2 = np.array([4, 5, 6])
arr3 = np.array([7, 8, 9])

# Nối theo trục mặc định (axis=0)
concat1 = np.concatenate((arr1, arr2, arr3))
print(concat1)  # [1 2 3 4 5 6 7 8 9]

# Nối mảng 2 chiều
a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6], [7, 8]])

# Nối theo hàng (axis=0)
concat_row = np.concatenate((a, b), axis=0)
print(concat_row)
# [[1 2]
#  [3 4]
#  [5 6]
#  [7 8]]

# Nối theo cột (axis=1)
concat_col = np.concatenate((a, b), axis=1)
print(concat_col)
# [[1 2 5 6]
#  [3 4 7 8]]
```

### 1.2 np.vstack() - Nối theo chiều dọc

```python
# Nối mảng 1 chiều thành mảng 2 chiều
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
vertical = np.vstack((a, b))
print(vertical)
# [[1 2 3]
#  [4 5 6]]

# Nối mảng 2 chiều
m1 = np.array([[1, 2], [3, 4]])
m2 = np.array([[5, 6], [7, 8]])
v_stack = np.vstack((m1, m2))
print(v_stack)
# [[1 2]
#  [3 4]
#  [5 6]
#  [7 8]]
```

### 1.3 np.hstack() - Nối theo chiều ngang

```python
# Nối mảng 1 chiều
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
horizontal = np.hstack((a, b))
print(horizontal)  # [1 2 3 4 5 6]

# Nối mảng 2 chiều
m1 = np.array([[1, 2], [3, 4]])
m2 = np.array([[5, 6], [7, 8]])
h_stack = np.hstack((m1, m2))
print(h_stack)
# [[1 2 5 6]
#  [3 4 7 8]]
```

### 1.4 np.dstack() - Nối theo chiều sâu

```python
# Nối theo chiều thứ ba
a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6], [7, 8]])
depth = np.dstack((a, b))
print(depth)
# [[[1 5]
#   [2 6]]
#  [[3 7]
#   [4 8]]]
```

## 2. Splitting (Tách mảng)

### 2.1 np.split()

```python
arr = np.array([1, 2, 3, 4, 5, 6])

# Tách thành 3 phần bằng nhau
split1 = np.split(arr, 3)
print(split1)  # [array([1, 2]), array([3, 4]), array([5, 6])]

# Tách tại các vị trí cụ thể
split2 = np.split(arr, [2, 4])
print(split2)  # [array([1, 2]), array([3, 4]), array([5, 6])]
```

### 2.2 np.vsplit() - Tách theo chiều dọc

```python
arr_2d = np.array([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9],
                   [10, 11, 12]])

# Tách thành 2 phần bằng nhau
v_split = np.vsplit(arr_2d, 2)
print(v_split)
# [array([[1, 2, 3],
#         [4, 5, 6]]),
#  array([[ 7,  8,  9],
#         [10, 11, 12]])]

# Tách tại vị trí cụ thể
v_split2 = np.vsplit(arr_2d, [2])
print(v_split2)
# [array([[1, 2, 3],
#         [4, 5, 6]]),
#  array([[ 7,  8,  9],
#         [10, 11, 12]])]
```

### 2.3 np.hsplit() - Tách theo chiều ngang

```python
arr_2d = np.array([[1, 2, 3, 4],
                   [5, 6, 7, 8]])

# Tách thành 2 phần bằng nhau
h_split = np.hsplit(arr_2d, 2)
print(h_split)
# [array([[1, 2],
#         [5, 6]]),
#  array([[3, 4],
#         [7, 8]])]

# Tách tại vị trí cụ thể
h_split2 = np.hsplit(arr_2d, [2])
print(h_split2)
# [array([[1, 2],
#         [5, 6]]),
#  array([[3, 4],
#         [7, 8]])]
```

## 3. Ứng dụng thực tế

### 3.1 Xử lý ảnh

```python
# Ghép nhiều ảnh
image1 = np.random.randint(0, 255, (100, 100))
image2 = np.random.randint(0, 255, (100, 100))

# Ghép ảnh theo chiều ngang
combined = np.hstack((image1, image2))

# Tách ảnh RGB thành các kênh màu
rgb_image = np.random.randint(0, 255, (100, 100, 3))
r, g, b = np.dsplit(rgb_image, 3)
```

### 3.2 Xử lý dữ liệu

```python
# Ghép nhiều dataset
data1 = np.random.rand(100, 3)  # 100 mẫu, 3 features
data2 = np.random.rand(100, 2)  # 100 mẫu, 2 features

# Ghép features
combined_features = np.hstack((data1, data2))

# Chia train/test
data = np.random.rand(1000, 5)
train, test = np.split(data, [800])  # 800 mẫu train, 200 mẫu test
```

## 4. Lưu ý quan trọng

1. **Kiểm tra kích thước**

```python
# Đảm bảo kích thước phù hợp khi nối
a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6]])  # Khác kích thước với a
# np.vstack((a, b))  # OK
# np.hstack((a, b))  # Error
```

2. **Copy vs View**

```python
# Split tạo copy của dữ liệu
arr = np.array([1, 2, 3, 4])
split_arr = np.split(arr, 2)
split_arr[0][0] = 99  # Không ảnh hưởng arr gốc
```

3. **Xử lý lỗi kích thước**

```python
try:
    # Tách mảng không chia hết
    arr = np.array([1, 2, 3, 4, 5])
    np.split(arr, 2)
except ValueError as e:
    print(e)  # array split does not result in an equal division
```

# Độ lệch chuẩn và Phương sai trong NumPy

## 1. Phương sai (Variance)

### 1.1 np.var() - Tính phương sai

```python
import numpy as np

# Mảng 1 chiều
arr = np.array([1, 2, 3, 4, 5])
variance = np.var(arr)
print(f"Phương sai: {variance}")

# Giải thích công thức:
# 1. Tính giá trị trung bình: mean = (1+2+3+4+5)/5 = 3
# 2. Tính tổng bình phương độ lệch: sum((x - mean)^2)
# 3. Phương sai = tổng bình phương độ lệch / n
```

### 1.2 Các tham số quan trọng

```python
# ddof (delta degrees of freedom)
arr = np.array([1, 2, 3, 4, 5])

# Phương sai mẫu (Sample variance), ddof=1
var_sample = np.var(arr, ddof=1)
print(f"Phương sai mẫu: {var_sample}")

# Phương sai tổng thể (Population variance), ddof=0
var_pop = np.var(arr, ddof=0)
print(f"Phương sai tổng thể: {var_pop}")

# axis - tính theo trục
arr_2d = np.array([[1, 2, 3], [4, 5, 6]])
var_cols = np.var(arr_2d, axis=0)  # theo cột
var_rows = np.var(arr_2d, axis=1)  # theo hàng
print(f"Phương sai theo cột: {var_cols}")
print(f"Phương sai theo hàng: {var_rows}")
```

## 2. Độ lệch chuẩn (Standard Deviation)

### 2.1 np.std() - Tính độ lệch chuẩn

```python
# Mảng 1 chiều
arr = np.array([1, 2, 3, 4, 5])
std_dev = np.std(arr)
print(f"Độ lệch chuẩn: {std_dev}")

# Giải thích:
# Độ lệch chuẩn = căn bậc hai của phương sai
print(f"Kiểm tra: {np.sqrt(np.var(arr))}")
```

### 2.2 Tham số và ví dụ

```python
# Độ lệch chuẩn mẫu vs tổng thể
arr = np.array([1, 2, 3, 4, 5])

# Độ lệch chuẩn mẫu (Sample std), ddof=1
std_sample = np.std(arr, ddof=1)
print(f"Độ lệch chuẩn mẫu: {std_sample}")

# Độ lệch chuẩn tổng thể (Population std), ddof=0
std_pop = np.std(arr, ddof=0)
print(f"Độ lệch chuẩn tổng thể: {std_pop}")
```

## 3. Ứng dụng thực tế

### 3.1 Phân tích dữ liệu

```python
# Dữ liệu điểm số
scores = np.array([85, 90, 88, 92, 95, 87, 89, 91, 86, 93])

# Thống kê cơ bản
mean = np.mean(scores)
std = np.std(scores, ddof=1)
var = np.var(scores, ddof=1)

print(f"Điểm trung bình: {mean:.2f}")
print(f"Độ lệch chuẩn: {std:.2f}")
print(f"Phương sai: {var:.2f}")

# Khoảng tin cậy
confidence_interval = (mean - 2*std, mean + 2*std)
print(f"Khoảng tin cậy 95%: {confidence_interval}")
```

### 3.2 Chuẩn hóa dữ liệu (Standardization)

```python
# Dữ liệu gốc
data = np.array([15, 25, 35, 45, 55])

# Chuẩn hóa
standardized = (data - np.mean(data)) / np.std(data)
print("Dữ liệu chuẩn hóa:", standardized)
print("Mean của dữ liệu chuẩn hóa:", np.mean(standardized))
print("Std của dữ liệu chuẩn hóa:", np.std(standardized))
```

### 3.3 Phân tích nhiều chiều

```python
# Ma trận dữ liệu
data_2d = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
])

# Phân tích theo cột
print("Độ lệch chuẩn theo cột:", np.std(data_2d, axis=0))
print("Phương sai theo cột:", np.var(data_2d, axis=0))

# Phân tích theo hàng
print("Độ lệch chuẩn theo hàng:", np.std(data_2d, axis=1))
print("Phương sai theo hàng:", np.var(data_2d, axis=1))
```

## 4. So sánh và lưu ý

### 4.1 So sánh ddof=0 và ddof=1

```python
data = np.array([1, 2, 3, 4, 5])

# ddof=0 (Population)
std_pop = np.std(data, ddof=0)
var_pop = np.var(data, ddof=0)

# ddof=1 (Sample)
std_sample = np.std(data, ddof=1)
var_sample = np.var(data, ddof=1)

print(f"Population std: {std_pop:.4f}, var: {var_pop:.4f}")
print(f"Sample std: {std_sample:.4f}, var: {var_sample:.4f}")
```

### 4.2 Xử lý giá trị NaN

```python
# Dữ liệu có giá trị NaN
data_with_nan = np.array([1, 2, np.nan, 4, 5])

# Sử dụng nanstd và nanvar
std = np.nanstd(data_with_nan)
var = np.nanvar(data_with_nan)

print(f"Std (bỏ qua NaN): {std}")
print(f"Var (bỏ qua NaN): {var}")
```

## 5. Best Practices

```python
# 1. Luôn chỉ định ddof rõ ràng
std = np.std(data, ddof=1)  # Sample std
var = np.var(data, ddof=1)  # Sample var

# 2. Kiểm tra dữ liệu trước khi tính
if np.any(np.isnan(data)):
    std = np.nanstd(data)
else:
    std = np.std(data)

# 3. Xử lý outliers nếu cần
def remove_outliers(data):
    mean = np.mean(data)
    std = np.std(data)
    return data[np.abs(data - mean) <= 2 * std]
```
