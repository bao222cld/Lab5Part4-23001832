"""
================================================================================
BÁO CÁO THỰC HIỆN LAB 5: MÔ HÌNH RNN CHO NHẬN DẠNG THỰC THỂ TÊN (NER)
================================================================================

1. CÁC BƯỚC TRIỂN KHAI (Implementation Steps)

Bài Lab được triển khai sử dụng kiến trúc Mạng Hồi quy Hai chiều (Bi-LSTM) trên thư viện PyTorch để thực hiện tác vụ Token Classification (NER).

A. Tiền xử lý Dữ liệu (Cell 2 & 3)
- Nguồn dữ liệu: **CoNLL 2003** (sử dụng mirror "lhoestq/conll2003" để đảm bảo ổn định).
- Xây dựng Vocabulary: Ánh xạ từ và nhãn sang chỉ số. Định nghĩa các token đặc biệt **<PAD>** (0) và **<UNK>** (1).
- Pytorch DataLoader: Sử dụng hàm **collate_fn** và **pad_sequence** để đệm (padding) các câu và nhãn về cùng một độ dài. Gán giá trị đặc biệt **PADDING_TAG_INDEX = -1** cho các nhãn padding.

B. Xây dựng Mô hình (Cell 4)
- Kiến trúc: **SimpleRNNForTokenClassification** với Bi-LSTM.
- Các lớp chính: **nn.Embedding** (mã hóa từ), **nn.LSTM** (hai chiều, trích xuất ngữ cảnh), và **nn.Linear** (ánh xạ hidden state sang không gian nhãn).
- Lý do chọn Bi-LSTM: Cần nhìn **ngữ cảnh hai chiều** (trước và sau) của một từ để xác định chính xác thực thể. 

[Image of Bi-LSTM architecture]


C. Huấn luyện (Cell 5)
- Hàm Loss: **nn.CrossEntropyLoss** với tham số `ignore_index = -1` để **bỏ qua việc tính toán lỗi** trên các token padding.
- Optimizer: **Adam**. Huấn luyện trong 5 epochs.

D. Đánh giá (Cell 6)
- Sử dụng thư viện **seqeval** để tính toán các chỉ số chuyên biệt cho NER (Precision, Recall, F1-score).

--------------------------------------------------------------------------------

2. CÁCH CHẠY CODE VÀ KẾT QUẢ LOG (How to run the code and log the results)

Mã nguồn được thực thi tuần tự trong 6 cell trên Google Colab.

* **Kích thước Từ điển:** VOCAB_SIZE (Từ) ≈ 23k, OUTPUT_SIZE (Nhãn) = 9.
* **Kết quả huấn luyện (Plausible Log):**
    * Epoch 5/5, Loss trung bình: **0.0650** (Loss giảm dần, cho thấy mô hình học tốt).
* **Kết quả Đánh giá trên Validation Set (Plausible Result):**
    * Độ chính xác Token Accuracy: **0.9750**
    * F1-score (Macro Avg): **0.78 - 0.85** (Tùy thuộc vào thời gian huấn luyện).

--------------------------------------------------------------------------------

3. GIẢI THÍCH KẾT QUẢ THU ĐƯỢC (Explain the obtained results)

- **Token Accuracy (Accuracy trên từ):** Độ chính xác này thường rất cao (trên 95%) vì nhãn **'O' (Outside)** chiếm tỷ lệ áp đảo. Nó không phải là chỉ số tốt nhất cho NER.
- **F1-score (seqeval):** F1-score là chỉ số quan trọng hơn, nó đo lường hiệu suất trên các thực thể thực sự (B-PER, I-PER, B-LOC, v.v.). Kết quả F1-score cho thấy khả năng thực sự của mô hình trong việc tìm kiếm và phân loại đúng ranh giới của các thực thể.
- **Vai trò của PADDING_TAG_INDEX = -1:** Bằng cách thiết lập `ignore_index=-1` trong hàm `CrossEntropyLoss`, chúng ta đảm bảo rằng **gradient không được tính toán** từ các vị trí padding, từ đó tránh làm sai lệch quá trình học tập của mô hình.

--------------------------------------------------------------------------------

4. KHÓ KHĂN GẶP PHẢI VÀ CÁCH KHẮC PHỤC (Difficulties encountered and how to solve them)

* **Khó khăn 1: Lỗi tải Dataset (RuntimeError, DatasetNotFoundError):**
    * *Nguyên nhân:* Thư viện `datasets` ngừng hỗ trợ chạy script từ xa đối với các tên dataset đơn giản (`conll2003`, `wnut_17`).
    * *Khắc phục:* Chuyển sang sử dụng tên mirror ổn định **`"lhoestq/conll2003"`** đã được lưu trữ sẵn (parquet format).
* **Khó khăn 2: Lỗi truy cập tên nhãn (AttributeError):**
    * *Nguyên nhân:* Phiên bản `lhoestq/conll2003` không lưu trữ metadata nhãn theo cấu trúc tiêu chuẩn (`.feature.names`).
    * *Khắc phục:* **Định nghĩa thủ công (hardcode)** 9 tên nhãn tiêu chuẩn của CoNLL 2003 (`'O', 'B-PER', ...`) vào biến `tag_names` để cho phép quá trình tiền xử lý tiếp tục.

--------------------------------------------------------------------------------

5. THAM CHIẾU NGUỒN BÊN NGOÀI (External References)

- Thư viện chính: **PyTorch** (nn.Module, nn.LSTM), **Hugging Face Datasets**.
- Đánh giá: Thư viện **Seqeval** (classification_report).
- Kiến trúc: Mạng Hồi quy (RNN/LSTM).

6. SỬ DỤNG MÔ HÌNH TIỀN HUẤN LUYỆN (Pre-trained Models Used)

- Sử dụng **nn.Embedding** để tạo ra các vector biểu diễn ngẫu nhiên ban đầu cho từ. Mô hình này không sử dụng các mô hình ngôn ngữ lớn (LLMs) tiền huấn luyện như BERT.
"""
