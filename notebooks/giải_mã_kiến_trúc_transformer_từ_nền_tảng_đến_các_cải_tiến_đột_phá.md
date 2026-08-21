# Giải mã kiến trúc Transformer: Từ nền tảng đến các cải tiến đột phá

## Nền tảng: Kiến trúc Transformer gốc (2017)

![Transformer encoder decoder architecture diagram](https://upload.wikimedia.org/wikipedia/commons/thumb/b/bf/Fault_diagnosis_for_wind_turbines_with_graph_neural_network_model_based_on_one-shot_learning.pdf/page1-960px-Fault_diagnosis_for_wind_turbines_with_graph_neural_network_model_based_on_one-shot_learning.pdf.jpg?utm_source=commons.wikimedia.org&utm_campaign=imageinfo&utm_content=thumbnail)
*Sơ đồ kiến trúc tổng quan của mạng Transformer gồm Encoder và Decoder.*

*Source: [Yang, Shuai; Zhou, Yifei; Chen, Xu; Li, Chuan; Song, Heng — CC BY 4.0](https://commons.wikimedia.org/wiki/File:Fault_diagnosis_for_wind_turbines_with_graph_neural_network_model_based_on_one-shot_learning.pdf)*

### Self‑Attention cơ bản  
Self‑Attention cho phép mỗi token trong chuỗi đầu vào tính toán một biểu diễn ngữ cảnh dựa trên toàn bộ chuỗi. Ba ma trận học được **Query (Q)**, **Key (K)** và **Value (V)** được tạo ra bằng cách nhân đầu vào \(X\) với các trọng số riêng:  

$$
Q = XW_Q,\; K = XW_K,\; V = XW_V
$$

Trọng số attention được tính bằng tích vô hướng giữa Q và K, sau đó chuẩn hoá bằng softmax:

$$
\text{Attention}(Q,K,V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V
$$

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

## Sự thống trị của kiến trúc Decoder‑only

- **Chuyển dịch từ mô hình dịch thuật sang mô hình tạo văn bản (Generative)**  
  Khi Transformer được giới thiệu vào 2017, kiến trúc Encoder‑Decoder được tối ưu cho nhiệm vụ dịch máy, nơi đầu vào và đầu ra có cấu trúc song song. Tuy nhiên, các mô hình ngôn ngữ lớn (LLM) cần dự đoán **token tiếp theo** dựa trên một chuỗi ngữ cảnh đang mở rộng, không yêu cầu mã hoá riêng biệt. Điều này đã dẫn đến việc chuyển sang **Decoder‑only**, một kiến trúc “autoregressive” cho phép mô hình học trực tiếp phân phối xác suất của token kế tiếp từ các token đã sinh ra trước đó.

- **Hiệu quả trong việc dự đoán token tiếp theo**  
  Decoder‑only chỉ cần một khối attention duy nhất (self‑attention) để tính toán các phụ thuộc thời gian, giảm độ phức tạp so với hai khối (encoder + decoder). Nhờ đó, chi phí tính toán cho mỗi bước sinh giảm đáng kể, đồng thời khả năng mở rộng lên hàng trăm nghìn token trở nên khả thi.

- **Đơn giản hoá thiết kế và tăng tốc độ huấn luyện**  
  Việc loại bỏ encoder không chỉ giảm số lượng tham số mà còn giảm độ sâu mạng, giúp **pipeline huấn luyện ngắn hơn** và tối ưu hoá bộ nhớ. Các cải tiến phần cứng như FlashAttention và Lean Attention tận dụng cấu trúc Decoder‑only để đạt tốc độ inference và training cao hơn.

- **Các mô hình tiêu biểu sử dụng Decoder‑only**  
  - **GPT‑3 / GPT‑4** (OpenAI)  
  - **LLaMA 2** (Meta)  
  - **PaLM 2** (Google)  
  - **Claude** (Anthropic)  
  - **Gemma** (Google DeepMind)  

Tất cả các mô hình trên đều dựa trên kiến trúc Decoder‑only, chứng tỏ đây là chuẩn mực cho các LLM hiện đại.

## Tối ưu hóa tính toán: FlashAttention và Kernel  

> **[IMAGE NOT FOUND]** So sánh luồng truy cập bộ nhớ giữa FlashAttention và Attention truyền thống.
>
> **Search:** FlashAttention GPU memory optimization diagram


**IO‑awareness của FlashAttention**  
FlashAttention được thiết kế để thực hiện toàn bộ quá trình Q·Kᵀ·V trong một kernel duy nhất, nhờ đó giảm số lần đọc/ghi vào bộ nhớ GPU. Thay vì lưu trữ ma trận attention trung gian trên DRAM, thuật toán tính toán trực tiếp trên các block dữ liệu đã được tải vào shared memory, đồng thời thực hiện “fusion” các bước softmax và weighted‑sum. Cơ chế này cho phép kernel “aware” các giới hạn băng thông I/O, tối thiểu hoá các truy cập bộ nhớ không cần thiết và đạt tốc độ gấp 2‑3 lần so với triển khai tiêu chuẩn.

**Kỹ thuật Tiling và Incremental Softmax**  
- *Tiling*: dữ liệu Q, K, V được chia thành các tile nhỏ (thường 64×64 hoặc 128×128). Mỗi tile được tải vào shared memory, tính toán attention cho tile đó, rồi ghi kết quả trở lại. Việc này giảm độ phụ thuộc vào kích thước toàn bộ sequence và cho phép sử dụng tối đa các warp và register của GPU.  
- *Incremental Softmax*: thay vì tính softmax trên toàn bộ ma trận attention, FlashAttention áp dụng softmax một cách tuần tự trên mỗi tile, duy trì giá trị max và sum một cách “incremental”. Điều này tránh việc phải lưu trữ toàn bộ logits và giảm chi phí tính toán O(N²) xuống O(N·B) với B là kích thước tile.

**Hiệu năng so với Attention tiêu chuẩn**  
| Kiến trúc | Thời gian (ms) cho seq‑len = 4096, batch = 1 | Tiêu thụ VRAM |
|----------|--------------------------------------------|--------------|
| Standard Scaled‑Dot‑Product (cuBLAS) | ~12 ms | ~8 GB |
| FlashAttention (kernel‑fused) | ~4 ms | ~3 GB |

**Ảnh hưởng của giảm chi phí bộ nhớ tới context window**  
Với việc giảm nhu cầu lưu trữ ma trận attention trung gian, các mô hình có thể mở rộng độ dài ngữ cảnh mà không cần tăng GPU memory. Ví dụ, một mô hình 7 B tham số có thể xử lý seq‑len = 32 k trên một card 24 GB khi dùng FlashAttention, trong khi phiên bản tiêu chuẩn chỉ hỗ trợ ≤ 8 k.

## Kiến trúc Mixture of Experts (MoE): Sự chuyển dịch sang tính thưa thớt

- **Thay thế lớp Feed‑Forward (FFN) bằng các chuyên gia**  
  Trong Transformer truyền thống, mỗi khối chứa một lớp Feed‑Forward (FFN) đồng nhất với kích thước cố định, tính toán trên mọi token. Kiến trúc Mixture of Experts (MoE) thay thế FFN bằng một tập hợp các “chuyên gia” (experts), mỗi chuyên gia là một mạng FFN riêng biệt nhưng có cùng cấu trúc. Khi một token đi qua lớp MoE, chỉ một hoặc một vài chuyên gia được kích hoạt, trong khi các chuyên gia còn lại không tham gia tính toán. Điều này cho phép mở rộng chiều rộng mô hình (số lượng tham số) mà không làm tăng chi phí tính toán trên mỗi token.

- **Cơ chế định tuyến (routing)**  
  MoE sử dụng một bộ định tuyến học được (learned router) để quyết định chuyên gia nào sẽ xử lý token hiện tại. Router thường là một mạng nơ-ron nhỏ tính toán trọng số cho mỗi chuyên gia dựa trên biểu diễn token. Hai chiến lược phổ biến là **top‑k routing** và **gating with load balancing**.

- **Lợi ích về hiệu năng suy luận so với mô hình Dense**  
  1. **Giảm FLOPs trên mỗi token**: Số phép tính thực tế giảm tới 70‑80 % so với mô hình Dense có cùng số tham số.  
  2. **Tăng khả năng mở rộng**: Thêm chuyên gia mới chỉ làm tăng bộ nhớ lưu trữ, không làm chậm thời gian suy luận.  
  3. **Cải thiện độ trễ**: MoE đạt độ trễ suy luận thấp hơn 30 % so với mô hình Dense khi chạy trên GPU đa‑luồng.

## Các tiêu chuẩn hóa trong kiến trúc hiện đại (2024-2025)

![Rotary Position Embedding RoPE visualization](https://upload.wikimedia.org/wikipedia/commons/thumb/9/98/Design_of_a_superelastic_alloy_actuator_for_a_minimally_invasive_surgical_manipulator_%28IA_designofsuperela1094542851%29.pdf/page1-960px-Design_of_a_superelastic_alloy_actuator_for_a_minimally_invasive_surgical_manipulator_%28IA_designofsuperela1094542851%29.pdf.jpg?utm_source=commons.wikimedia.org&utm_campaign=imageinfo&utm_content=thumbnail)
*Minh họa cơ chế quay vector trong không gian của Rotary Position Embedding (RoPE).*

*Source: [Parkhurst, William T. — Public domain](https://commons.wikimedia.org/wiki/File:Design_of_a_superelastic_alloy_actuator_for_a_minimally_invasive_surgical_manipulator_(IA_designofsuperela1094542851).pdf)*

- **Pre‑norm trở thành chuẩn mặc định**  
  Kể từ năm 2022, các mô hình LLM lớn chuyển sang áp dụng *pre‑norm* (LayerNorm trước khối attention và MLP) thay vì *post‑norm*. Điều này giảm thiểu hiện tượng gradient vanishing khi đào tạo sâu và cho phép tăng độ sâu lên 100‑200 lớp mà không mất ổn định.

- **RoPE (Rotary Positional Embeddings) cho ngữ cảnh dài**  
  RoPE nhúng vị trí bằng cách quay các vector embedding theo góc phụ thuộc vào vị trí, cho phép mô hình mở rộng một cách tự nhiên tới chuỗi dài mà không cần bổ sung các tham số vị trí tĩnh. Nhờ tính chất quay, RoPE duy trì tính đồng nhất khi kéo dài độ dài chuỗi.

- **Hàm kích hoạt GLU‑family trong MLP**  
  Các lớp MLP hiện đại thường thay thế ReLU/GELU bằng các biến thể của Gated Linear Unit (GLU), chẳng hạn GLU, SwiGLU, hoặc GEGLU. Các hàm này kết hợp một cổng gating để lọc thông tin, tăng khả năng biểu diễn phi tuyến và giảm độ phức tạp tính toán.

## Tối ưu hóa suy luận (Inference): KV Cache và hơn thế nữa

### Cơ chế KV Cache và các biến thể
Trong giai đoạn decode, mỗi token mới chỉ cần truy cập các **key (K)** và **value (V)** đã được tính toán cho các token trước. KV Cache lưu trữ K và V ở mỗi lớp, giảm độ phức tạp từ \(O(N^2)\) xuống \(O(N)\) cho mỗi bước decode.

**PagedAttention** mở rộng KV Cache bằng cách chia bộ nhớ cache thành các “page” cố định kích thước, cho phép mô hình xử lý ngữ cảnh dài (từ 8 k tới 100 k token) mà không cần cấp phát bộ nhớ liên tục.

### So sánh Multi-Query Attention (MQA) và Grouped-Query Attention (GQA)
| Đặc điểm | MQA | GQA |
|----------|-----|-----|
| Số query per head | 1 (đồng nhất cho tất cả heads) | N > 1, chia heads thành nhóm, mỗi nhóm chia sẻ một query |
| K/V replication | K và V được chia sẻ giữa mọi heads | K/V vẫn được chia sẻ, nhưng mỗi nhóm có query riêng |
| Độ phức tạp tính toán | Giảm đáng kể vì chỉ một Q matrix | Giảm hơn so với full‑query, nhưng vẫn cao hơn MQA |
| Tác động bộ nhớ | Cache K/V duy nhất → giảm RAM | Cache K/V duy nhất, nhưng cần lưu trữ Q cho mỗi nhóm |

## Hướng tới tương lai: Linear Attention và các mô hình thay thế

- **Linear Attention với độ phức tạp O(n)**  
  Các kiến trúc Linear Attention thay thế ma trận attention chuẩn (O(n²)) bằng các hàm tích chập hoặc kernel cho phép tính toán theo dạng tích vô hướng, nhờ đó chi phí thời gian và bộ nhớ giảm xuống O(n) cho chuỗi dài.

- **Ứng dụng thực tế trong xử lý chuỗi cực dài**  
  Linear Attention đã được tích hợp vào các hệ thống LLM để xử lý đầu vào lên tới hàng chục nghìn token mà không gặp lỗi bộ nhớ, duy trì độ chính xác khi chuỗi vượt quá 16 k token. Nhờ tính chất tuyến tính, các mô hình này dễ dàng triển khai trên phần cứng có bộ nhớ hạn chế và hỗ trợ inference thời gian thực.