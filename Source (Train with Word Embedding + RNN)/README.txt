1. Các file có thể được sử dụng:
- config.json: file lưu các siêu tham số cần cho việc thực nghiệm.
- result.json: file cấu trúc lưu thông tin (giành cho quá trình thống kê tìm siêu tham số tốt).
2. Dữ liệu:
- train.pkl: là tập dữ liệu dùng để học chứa tổng cộng 4978 sentences.
- dicts.pkl: là tập bắt buộc cần có để đối chiếu từ (mapping word) trong Word Embedding.
- atis.pkl: là tập dữ liệu nén, gói tất cả các thông tin và dữ liệu cần cho quá trình học.
3. Huấn luyện mô hình:
- Chạy cmd: python Slot Filling.py
- File kết quả được lưu trong thư mục Result
4. Thư mục Result gồm:
- model_config.json và model_weight.h5: mô hình huấn luyện được.
Mục đích lưu file mô hình thành hai file riêng là lưu riêng bộ siêu tham số để thống kêvà áp dụng sau
này.
- valid.txt: kết quả dự đoán trên tập validation.
- result.json: cho biết thông tin độ lỗi đạt được.
- train.txt: kết quả dự đoán trên tập train.
Lưu ý: valid.txt, train.txt và result.json chỉ xuất hiện trong quá trình tìm bộ siêu tham số tốt.
5. Cấu trúc file test.txt
- Các câu được tách ra thành từng từ riêng biệt và được đặt trong cặp kí hiệu BOS và EOS.
- Với mỗi từ, ta sẽ có 2 nhãn: nhãn thứ nhất là nhãn mô hình dự đoán được và nhãn thứ hai là nhãn đúng.