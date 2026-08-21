# Giải mã kiến trúc Transformer: Từ nền tảng 2017 đến kỷ nguyên Mixture of Experts

## Nền tảng: Kiến trúc Transformer gốc (2017)

![Transformer Encoder-Decoder architecture diagram](https://upload.wikimedia.org/wikipedia/commons/thumb/b/b6/Artificial_Intelligence_Concept_Hypermap.pdf/page1-1280px-Artificial_Intelligence_Concept_Hypermap.pdf.jpg?utm_source=commons.wikimedia.org&utm_campaign=imageinfo&utm_content=thumbnail)
*Sơ đồ kiến trúc Transformer gốc bao gồm các khối Encoder và Decoder với cơ chế Multi-Head Attention.*

*Source: [Trebol6 — CC BY-SA 4.0](https://commons.wikimedia.org/wiki/File:Artificial_Intelligence_Concept_Hypermap.pdf)*

### Self‑Attention cơ bản  
Self‑Attention cho phép mỗi token trong chuỗi đầu vào tính toán một biểu diễn ngữ cảnh dựa trên toàn bộ chuỗi. Ba ma trận học được **Query (Q)**, **Key (K)** và **Value (V)** được tạo ra bằng cách nhân đầu vào \(X\) với các trọng số riêng:  

\(
Q = XW_Q,\; K = XW_K,\; V = XW_V
\)

Trọng số attention được tính bằng tích vô hướng giữa Q và K, sau đó chuẩn hoá bằng softmax:

\(
\text{Attention}(Q,K,V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V
\)

Kết quả là mỗi vị trí nhận được một tổng trọng số của các Value, trong đó trọng số phản ánh mức độ liên quan của các token khác. Điều này thay thế hoàn toàn các vòng lặp tuần tự của RNN/LSTM, cho phép tính toán song song trên toàn bộ chuỗi.

### Encoder‑Decoder gốc vs. xu hướng hiện đại  
Kiến trúc gốc gồm **Encoder** (N lớp self‑attention + feed‑forward) và **Decoder** (self‑attention + encoder‑decoder attention + feed‑forward). Encoder chỉ nhận đầu vào và tạo ra các biểu diễn ngữ cảnh, trong khi Decoder nhận cả biểu diễn encoder và các token đã sinh ra để dự đoán token tiếp theo. Các mô hình hiện đại (ví dụ: BERT, GPT‑3) đã tách rời cấu trúc này:  
- **Encoder‑only** (BERT) tập trung vào biểu diễn ngữ cảnh cho các nhiệm vụ hiểu ngôn ngữ.  
- **Decoder‑only** (GPT) loại bỏ hoàn toàn phần encoder, dùng attention tự‑động để sinh chuỗi.  
- **Encoder‑decoder hybrid** (T5, BART) vẫn giữ cấu trúc gốc nhưng bổ sung các kỹ thuật như relative positional encoding và sparse attention.

### Ưu điểm so với RNN/LSTM  
- **Parallelism**: Toàn bộ chuỗi được xử lý đồng thời, giảm thời gian đào tạo so với tính toán tuần tự của RNN/LSTM.  
- **Long‑range dependencies**: Attention có khả năng kết nối bất kỳ hai vị trí nào mà không bị suy giảm gradient, trong khi RNN gặp vanishing/exploding gradient.  
- **Scalability**: Khi tăng chiều sâu hoặc chiều rộng, hiệu suất tăng gần tuyến tính nhờ GPU/TPU tối ưu cho ma trận nhân.  
- **Flexibility**: Các head attention có thể học các loại quan hệ ngữ nghĩa khác nhau mà không cần thiết kế kiến trúc phức tạp.

### Vai trò của Multi‑Head Attention  
Multi‑Head Attention chia không gian Q/K/V thành \(h\) đầu, mỗi đầu thực hiện attention độc lập và sau đó hợp nhất kết quả. Điều này cho phép mô hình **nắm bắt đa chiều**: một head có thể tập trung vào quan hệ cú pháp, một head khác vào quan hệ ngữ nghĩa, và một head nữa vào vị trí tương đối. Kết quả là biểu diễn cuối cùng chứa thông tin phong phú hơn, cải thiện khả năng hiểu và sinh ngôn ngữ so với single‑head attention.

## Những cải tiến cốt lõi về hiệu năng (Modern Transformer)

Trong giai đoạn phát triển hiện nay, các kỹ thuật tối ưu hoá đã biến Transformer từ một mô hình “đủ dùng” thành một kiến trúc có thể mở rộng tới hàng trăm tỷ tham số mà vẫn duy trì ổn định và hiệu suất cao. Bốn cải tiến dưới đây là những yếu tố then chốt giúp đạt được mục tiêu này.

- **RMSNorm – chuẩn hoá ổn định cho quá trình huấn luyện**  
  RMSNorm (Root Mean Square Layer Normalization) thay thế LayerNorm truyền thống bằng việc chuẩn hoá dựa trên chuẩn độ lệch chuẩn của các giá trị đầu vào, bỏ qua việc tính trung bình. Điều này giảm thiểu chi phí tính toán và giảm độ nhạy của mô hình đối với các giá trị outlier, giúp gradient lan truyền mượt mà hơn trong các mô hình sâu và dài chuỗi. Nhờ không cần tính trung bình, RMSNorm còn giảm bộ nhớ tạm thời, cho phép tăng batch size mà không gây mất ổn định.

- **Rotary Position Embedding (RoPE) – vị trí xoay vòng**  
  RoPE nhúng thông tin vị trí bằng cách áp dụng phép quay (rotation) lên không gian embedding, cho phép mô hình học các quan hệ tương đối một cách tự nhiên. Khác với positional encoding tĩnh (sinusoidal) hoặc learned absolute embeddings, RoPE duy trì tính đồng nhất khi chuỗi dài hơn được mở rộng, vì các phép quay vẫn hợp lệ cho bất kỳ độ dài nào. Nhờ tính chất này, RoPE cải thiện đáng kể khả năng xử lý ngữ cảnh dài mà không cần tái huấn luyện lại các embedding vị trí.

- **Grouped-Query Attention (GQA) – giảm chi phí bộ nhớ so với Multi‑Head Attention**  
  GQA chia các query thành các nhóm, trong khi key và value vẫn được chia sẻ giữa các nhóm. Nhờ giảm số lượng query độc lập, GQA giảm đáng kể kích thước ma trận attention (Q·Kᵀ) và do đó giảm bộ nhớ GPU tiêu thụ, đặc biệt khi số head lớn. So với Multi‑Head Attention truyền thống, GQA duy trì khả năng biểu diễn đa dạng nhờ các head vẫn tồn tại, nhưng chi phí tính toán và băng thông bộ nhớ giảm tới 30‑40 % trong các mô hình LLM hiện đại.

- **FlashAttention – tối ưu hoá truy cập bộ nhớ GPU**  
  FlashAttention tái cấu trúc quá trình tính toán attention để thực hiện các phép nhân ma trận và softmax trong một kernel duy nhất, giảm việc ghi/đọc bộ nhớ trung gian. Kết quả là giảm latency và tăng throughput lên tới 2‑3× trên các GPU hiện đại, đồng thời cho phép xử lý chuỗi dài hơn mà không gặp lỗi out‑of‑memory. Nhờ việc tận dụng tối đa bộ nhớ cache của GPU, FlashAttention trở thành tiêu chuẩn cho các mô hình LLM có context length lớn.

## Cuộc cách mạng Mixture of Experts (MoE)

> **[IMAGE NOT FOUND]** Cơ chế định tuyến thưa thớt (Sparse Routing) trong kiến trúc Mixture of Experts (MoE).
>
> **Search:** Mixture of experts neural network routing architecture diagram


- **Cơ chế Sparse MoE**  
  Trong kiến trúc Transformer truyền thống, mỗi lớp Feed‑Forward (FFN) là một mạng dày đặc với cùng một tập tham số cho mọi token. MoE thay thế FFN bằng một tập hợp các *experts* – các mạng FFN nhỏ, độc lập và được kích hoạt một cách thưa thớt (sparse). Khi một token được đưa vào, chỉ một hoặc vài experts được chọn để tính toán, trong khi phần còn lại không tham gia, giảm đáng kể lượng phép tính cần thiết nhưng vẫn giữ được khả năng biểu diễn mạnh mẽ nhờ số lượng experts có thể mở rộng lên hàng nghìn.

- **Vai trò của Gating Network**  
  Gating network là một mô-đun nhẹ, thường là một linear layer theo sau một softmax, chịu trách nhiệm *định tuyến* mỗi token tới các experts phù hợp. Nó tính toán một vector trọng số cho tất cả experts và chỉ giữ lại top‑k giá trị (thường k = 1 hoặc 2), tạo ra một mask thưa thớt. Nhờ gating, các token có ngữ cảnh hoặc ngôn ngữ tương tự sẽ được xử lý bởi cùng một nhóm experts, giúp mô hình học các chuyên môn chuyên biệt mà không gây xung đột.

- **Lợi ích khi tăng số lượng tham số mà không tăng chi phí suy luận**  
  1. **Mở rộng tham số**: Thêm experts đồng nghĩa với việc tăng tổng số tham số lên hàng tỷ, nhưng do chỉ một phần nhỏ experts được kích hoạt, chi phí FLOPs và thời gian latency trong giai đoạn suy luận gần như không thay đổi.  
  2. **Hiệu suất mô hình**: Các nghiên cứu cho thấy MoE có thể đạt độ chính xác tương đương hoặc vượt trội so với các mô hình dense có cùng mức FLOPs, nhờ khả năng “điều chỉnh chuyên môn” cho từng token.  
  3. **Tiết kiệm tài nguyên**: Khi triển khai trên GPU/TPU, chỉ một phần nhỏ bộ nhớ được sử dụng cho các experts hoạt động, cho phép chạy các mô hình siêu lớn trên phần cứng hiện có mà không cần tăng băng thông bộ nhớ.

- **Thách thức: cân bằng tải (load balancing)**  
  Vì gating network quyết định experts nào sẽ được kích hoạt, một phân phối không đồng đều có thể dẫn tới một số experts bị “quá tải” trong khi các experts khác hầu như không được sử dụng. Điều này làm giảm hiệu quả tài nguyên và gây hiện tượng “expert collapse”. Các phương pháp cân bằng tải thường được đưa vào loss function, ví dụ như *auxiliary load‑balancing loss* dựa trên entropy của distribution gating, hoặc *router z‑loss* để khuyến khích sự đồng đều trong việc chọn experts.

## Tối ưu hóa suy luận (Inference): KV Cache và hơn thế nữa

> **[IMAGE NOT FOUND]** Minh họa cơ chế lưu trữ KV Cache và quản lý bộ nhớ PagedAttention giúp tối ưu hóa suy luận.
>
> **Search:** KV cache transformer attention mechanism diagram


### Cơ chế KV Cache và các biến thể
Trong giai đoạn decode, mỗi token mới chỉ cần truy cập các **key (K)** và **value (V)** đã được tính toán cho các token trước. KV Cache lưu trữ K và V ở mỗi lớp, cho phép mô hình tránh tính toán lại toàn bộ attention matrix cho toàn bộ chuỗi đầu vào. Khi một token mới được sinh ra, chỉ cần tính Q cho token đó và thực hiện dot‑product với K/V đã cache, giảm độ phức tạp từ \(O(N^2)\) xuống \(O(N)\) cho mỗi bước decode.

**PagedAttention** mở rộng KV Cache bằng cách chia bộ nhớ cache thành các “page” cố định kích thước. Khi chuỗi vượt quá khả năng lưu trữ một page, các page cũ được thay thế theo chiến lược LRU hoặc sliding‑window, cho phép mô hình xử lý ngữ cảnh dài mà không cần cấp phát bộ nhớ liên tục.

### So sánh Multi-Query Attention (MQA) và Grouped-Query Attention (GQA)
| Đặc điểm | MQA | GQA |
|----------|-----|-----|
| Số query per head | 1 (đồng nhất cho tất cả heads) | N > 1, chia heads thành nhóm, mỗi nhóm chia sẻ một query |
| K/V replication | K và V được chia sẻ giữa mọi heads | K/V vẫn được chia sẻ, nhưng mỗi nhóm có query riêng |
| Độ phức tạp tính toán | Giảm đáng kể vì chỉ một Q matrix | Giảm hơn so với full‑query, nhưng vẫn cao hơn MQA |
| Tác động bộ nhớ | Cache K/V duy nhất → giảm RAM | Cache K/V duy nhất, nhưng cần lưu trữ Q cho mỗi nhóm |

### Tác động đến độ trễ (latency) và thông lượng (throughput)
- **KV Cache**: Giảm latency mỗi token từ vài ms xuống < 1 ms trên GPU A100 khi batch size = 1, đồng thời tăng throughput lên gấp 3‑5× so với tính toán full attention.
- **PagedAttention**: Cho phép mở rộng context mà không làm tăng latency tuyến tính; các nghiên cứu cho thấy tăng độ dài context từ 8k → 32k token chỉ làm tăng latency < 15%.
- **MQA/GQA**: MQA giảm latency khoảng 20‑30% so với full‑query attention, trong khi GQA đạt mức giảm 15‑25% nhưng duy trì độ chính xác cao hơn MQA.

## Kết luận: Lựa chọn kiến trúc cho bài toán thực tế

- **Trade‑off độ chính xác vs chi phí tính toán**  
  - *Dense Transformers* (GPT‑style) đạt độ chính xác cao nhất vì mọi tham số đều được cập nhật cho mỗi mẫu, nhưng chi phí GPU/TPU và độ trễ tăng tuyến tính với kích thước mô hình.  
  - *Mixture of Experts (MoE)* giảm chi phí tính toán đáng kể bằng cách kích hoạt chỉ một phần nhỏ các chuyên gia (expert) cho mỗi token; độ trễ giảm, nhưng hiệu suất có thể giảm nếu dữ liệu không đồng nhất hoặc nếu routing không tối ưu.

- **Khi nào nên dùng Dense, MoE hay Hybrid**  
  - **Dense**: yêu cầu độ chính xác tối đa, môi trường có tài nguyên mạnh (đám mây GPU/TPU).
  - **MoE**: dịch vụ thời gian thực, chi phí vận hành hạn chế, hoặc khi mô hình cần mở rộng lên hàng trăm tỷ tham số.
  - **Hybrid**: dự án có cả yêu cầu độ chính xác cao và hạn chế latency/chi phí.