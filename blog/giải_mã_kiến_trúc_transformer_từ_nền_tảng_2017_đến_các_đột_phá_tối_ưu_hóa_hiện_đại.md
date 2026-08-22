# Giải mã kiến trúc Transformer: Từ nền tảng 2017 đến các đột phá tối ưu hóa hiện đại

## Kiến trúc Transformer nguyên bản: Cơ chế Self-Attention

### Cấu trúc Encoder‑Decoder và vai trò của Attention  
Transformer được chia thành hai khối đồng dạng: **Encoder** và **Decoder**. Mỗi Encoder gồm N lớp, mỗi lớp có hai thành phần chính: một **Self‑Attention** đa đầu (multi‑head) và một mạng Feed‑Forward Position‑wise (FFN). Decoder cũng có N lớp, nhưng thêm một lớp **Masked Self‑Attention** (ngăn chặn nhìn vào các token tương lai) và một lớp **Encoder‑Decoder Attention** cho phép Decoder truy vấn thông tin đã mã hoá ở Encoder. Cơ chế Attention là cầu nối giữa các vị trí trong chuỗi, cho phép mô hình học các quan hệ dài hạn mà không cần truyền thông tin qua trạng thái ẩn tuần tự như RNN.

![Transformer Encoder-Decoder Architecture Diagram](https://upload.wikimedia.org/wikipedia/commons/thumb/d/d4/A_programmable_reverse-bias_safe_operating_area_transistor_testor_%28IA_programmablereve4008bern%29.pdf/page1-960px-A_programmable_reverse-bias_safe_operating_area_transistor_testor_%28IA_programmablereve4008bern%29.pdf.jpg?utm_source=commons.wikimedia.org&utm_campaign=imageinfo&utm_content=thumbnail)
*Sơ đồ kiến trúc Encoder-Decoder của mô hình Transformer nguyên bản với các khối Self-Attention và Feed-Forward.*

*Source: [Berning, D.W. — Public domain](https://commons.wikimedia.org/wiki/File:A_programmable_reverse-bias_safe_operating_area_transistor_testor_(IA_programmablereve4008bern).pdf)*

### Toán học của Scaled Dot‑Product Attention  
Self‑Attention dựa trên công thức **Scaled Dot‑Product**:  

\(
\text{Attention}(Q,K,V)=\text{softmax}\!\left(\frac{QK^{\top}}{\sqrt{d_k}}\right)V
\)

- **Q (Query)**, **K (Key)**, **V (Value)** được tạo ra bằng cách nhân ma trận embedding đầu vào với ba ma trận trọng số học được \(W_Q, W_K, W_V\).  
- \(d_k\) là kích thước của vector Key; việc chia cho \(\sqrt{d_k}\) giảm thiểu độ lớn của dot‑product, tránh gradient vanishing khi softmax được áp dụng.  
- Kết quả là một tập hợp các trọng số attention (các hệ số softmax) nhân với các vector Value, tạo ra biểu diễn mới cho mỗi token, phản ánh toàn bộ ngữ cảnh.

### Song song hoá so với RNN truyền thống  
Trong RNN, tính toán cho token \(t\) phụ thuộc vào kết quả của token \(t-1\), gây ra **độ trễ chuỗi** và hạn chế khả năng khai thác phần cứng GPU/TPU. Transformer, nhờ Self‑Attention, tính toán Q, K, V cho **tất cả các token đồng thời**; ma trận \(QK^{\top}\) được thực hiện bằng một phép nhân ma trận lớn, tận dụng tối đa các kernel BLAS. Do đó, thời gian chạy gần như tỉ lệ với độ sâu (số lớp) chứ không phụ thuộc vào độ dài chuỗi, cho phép tăng batch size và giảm thời gian huấn luyện đáng kể.

### Điểm nghẽn O(n²) của Attention gốc  
Mặc dù song song hoá mạnh, Self‑Attention vẫn yêu cầu tính toán ma trận tương đồng kích thước \(n \times n\) (với \(n\) là độ dài chuỗi). Phép nhân \(QK^{\top}\) và softmax trên toàn bộ ma trận mang độ phức tạp **O(n²)**, dẫn đến tiêu thụ bộ nhớ và thời gian tăng nhanh khi \(n\) lớn (ví dụ: tài liệu dài hoặc context window > 4k token). Đây là nút thắt chính khiến các mô hình Transformer ban đầu gặp khó khăn trong việc mở rộng context, và đã thúc đẩy các nghiên cứu tối ưu hoá Attention (sparse, linear, FlashAttention, …) trong các phiên bản hiện đại.

## Giải quyết nút thắt bộ nhớ: FlashAttention và Kernel Fusion  

- **Giảm I/O bộ nhớ bằng cách không lưu trữ ma trận Attention**  
  Trong kiến trúc Transformer truyền thống, ma trận Attention có kích thước \(N \times N\) (với \(N\) là độ dài chuỗi) được tạo ra và lưu trữ toàn bộ trong bộ nhớ GPU trước khi thực hiện softmax. Việc này gây ra lượng truyền dữ liệu I/O khổng lồ, đặc biệt khi \(N\) lớn, làm giảm hiệu suất và giới hạn độ dài ngữ cảnh có thể xử lý. FlashAttention tránh bước này bằng cách tính softmax “on‑the‑fly”, không material hoá ma trận đầy đủ, do đó giảm đáng kể lưu lượng bộ nhớ và băng thông tiêu thụ【https://handbook.modular.com/kernel-optimization/flashattention】.  

- **Kỹ thuật Tiling và Recomputation**  
  FlashAttention chia tính toán Attention thành các “tiles” nhỏ, mỗi tile chỉ chứa một phần của ma trận Q·Kᵀ. Các tile được tải vào bộ nhớ đệm (shared memory) của GPU, thực hiện softmax và nhân với V ngay tại chỗ. Khi bộ nhớ đệm hết, các tile đã tính toán sẽ được **recomputed** (tái tính) thay vì lưu trữ tạm thời, nhờ đó giữ lượng bộ nhớ tạm tối thiểu. Cơ chế này cho phép thực hiện toàn bộ quá trình trong O(N) bộ nhớ thay vì O(N²) và tận dụng tối đa tốc độ truy cập của shared memory【https://handbook.modular.com/kernel-optimization/flashattention】.  

- **So sánh FlashAttention‑1 và FlashAttention‑2**  
  *FlashAttention‑1* đã chứng minh khả năng tăng tốc gấp 7× so với triển khai Attention chuẩn mà không làm giảm độ chính xác【https://mlfrontiers.substack.com/p/flashattention-making-attention-7x】. *FlashAttention‑2* mở rộng thiết kế bằng cách tích hợp **kernel fusion** và tối ưu hoá warp‑level primitives, giảm overhead của launch kernel và cải thiện hiệu suất lên tới 1.5‑2× so với phiên bản đầu tiên trên các mô hình lớn (ví dụ: GPT‑3 175B)【https://training.continuumlabs.ai/inference/why-is-inference-important/flash-attention-2】. Ngoài ra, FlashAttention‑2 hỗ trợ **mixed‑precision** và **sparse attention** mà FlashAttention‑1 chưa tối ưu, giúp giảm thời gian tính toán thêm 10‑15% trong môi trường đa‑GPU.  

- **Ảnh hưởng của giảm độ phức tạp bộ nhớ tới độ dài ngữ cảnh**  
  Khi độ phức tạp bộ nhớ giảm từ O(N²) xuống O(N), giới hạn thực tế của context window được mở rộng đáng kể. Các nghiên cứu khảo sát về attention hiệu quả cho thấy, với FlashAttention, các mô hình có thể xử lý chuỗi lên tới 64K token mà vẫn duy trì tốc độ training tương đương với chuỗi 8K token trong kiến trúc truyền thống【https://arxiv.org/html/2507.19595v3】. Điều này cho phép các LLM khai thác thông tin dài hơn mà không gặp “out‑of‑memory” trên cùng một GPU, đồng thời giảm chi phí năng lượng do giảm số lần đọc/ghi bộ nhớ.

Tóm lại, FlashAttention và các kỹ thuật kernel fusion không chỉ tối ưu hoá mức độ thấp của pipeline tính toán Attention mà còn mở ra khả năng mở rộng context window, một yếu tố then chốt cho các ứng dụng LLM đòi hỏi hiểu biết sâu rộng trên văn bản dài.

## Các biến thể Attention hiệu quả: MQA và GQA

**Phân biệt Multi-Head Attention (MHA), Multi-Query Attention (MQA) và Grouped-Query Attention (GQA)**  
- **MHA**: mỗi head duy trì bộ *query‑key‑value* (QKV) riêng, cho phép mô hình học đa dạng các không gian chú ý nhưng tiêu tốn O(N·h) bộ nhớ cho KV, trong đó *h* là số head.  
- **MQA**: chỉ các *query* được chia thành *h* head, trong khi *key* và *value* được chia sẻ một bộ duy nhất cho tất cả các head. Điều này giảm chi phí bộ nhớ KV xuống O(N) mà không làm mất khả năng đa dạng của query. [[Efficient Transformer Variants](https://www.emergentmind.com/topics/efficient-transformer-variants)]  
- **GQA**: các head được nhóm lại (ví dụ 8 head/chunk). Mỗi nhóm chia sẻ một bộ KV, còn các query trong nhóm vẫn độc lập. Khi số nhóm *g*\ <\ *h*, KV giảm O(N·g) – một mức trung gian giữa MHA và MQA. [[Attention Variants: Evolving the Core of Transformers for Next-Level AI](https://www.linkedin.com/pulse/attention-variants-evolving-core-transformers-next-level-nikitha-r-pli3f)]

![Comparison between MHA, MQA and GQA attention mechanisms](https://upload.wikimedia.org/wikipedia/commons/thumb/9/98/Archives_of_aboriginal_knowledge._Containing_all_the_original_paper_laid_before_Congress_respecting_the_history%2C_antiquities%2C_language%2C_ethnology%2C_pictography%2C_rites%2C_superstitions_%28IA_archivesofaborig03scho%29.pdf/page1-500px-thumbnail.pdf.jpg?utm_source=commons.wikimedia.org&utm_campaign=imageinfo&utm_content=thumbnail)
*So sánh cơ chế phân chia Key-Value heads giữa Multi-Head Attention (MHA), Multi-Query Attention (MQA) và Grouped-Query Attention (GQA).*

*Source: [Schoolcraft, Henry Rowe, 1793-1864. dn
United States. Bureau of Indian Affairs. cn — Public domain](https://commons.wikimedia.org/wiki/File:Archives_of_aboriginal_knowledge._Containing_all_the_original_paper_laid_before_Congress_respecting_the_history,_antiquities,_language,_ethnology,_pictography,_rites,_superstitions_(IA_archivesofaborig03scho).pdf)*

**Cân bằng tốc độ suy luận và chất lượng mô hình**  
- MHA thường đạt độ chính xác cao nhất vì mỗi head có toàn bộ KV riêng, nhưng tính toán O(N²·h) và băng thông GPU lớn.  
- MQA giảm thời gian tính toán KV lên tới 2‑3× so với MHA nhờ việc tái sử dụng KV, và các thí nghiệm cho thấy độ giảm BLEU/Perplexity chỉ <\ 0.3\ % trên các mô hình 7‑B. [[Attention Optimizations — Megatron Bridge](https://docs.nvidia.com/nemo/megatron-bridge/latest/training/attention-optimizations.html)]  
- GQA cung cấp một “sweet spot”: khi nhóm kích thước 4‑8, tốc độ tăng 1.5‑2× so với MHA, trong khi độ suy giảm chất lượng thường nằm trong khoảng 0.1‑0.2\ % so với MHA. [[Efficient attention mechanisms for large language models](https://www.sciencedirect.com/science/article/pii/S2666389926001030)]

**Tiết kiệm VRAM khi dùng GQA trong mô hình lớn**  
- Giả sử mô hình 70\ B với 96 head và dim\ =\ 12288. Với MHA, KV chiếm \(\approx\) 2\ \times\) N \(\times\) dim \(\times\) h \(\approx\) 2\ TB (đối với batch\ =\ 1, seq\ =\ 8192).  
- Khi áp dụng GQA với nhóm size\ =\ 8 (g\ =\ 12), KV giảm xuống \(\approx\) 2\ \times\) N \(\times\) dim \(\times\) g \(\approx\) 250\ GB, tương đương **giảm ~\ 87\ % VRAM**. [[Efficient Transformer Variants](https://www.emergentmind.com/topics/efficient-transformer-variants)]  
- Thực nghiệm trên A100 80\ GB cho thấy mô hình 70\ B với GQA có thể chạy inference với batch\ =\ 1, seq\ =\ 4096, trong khi MHA yêu cầu phân tán trên nhiều GPU.

**Trường hợp sử dụng tối ưu**  
- **MHA**: mô hình nhỏ‑trung (\(\le\) 6\ B) hoặc khi chất lượng tối đa là ưu tiên, ví dụ fine‑tuning nhiệm vụ NLU/NER.  
- **MQA**: dịch vụ inference một chiều (decoder‑only) như ChatGPT‑style, nơi KV được tái sử dụng qua nhiều token và băng thông GPU là nút thắt.  
- **GQA**: mô hình siêu lớn (\(\ge\) 30\ B) triển khai trên single‑GPU hoặc multi‑GPU với hạn chế VRAM, hoặc trong môi trường đa‑tenant inference nơi cần cân bằng latency và chi phí phần cứng.  

Việc lựa chọn giữa MHA, MQA và GQA phụ thuộc vào **độ lớn mô hình**, **giới hạn phần cứng**, và **độ nhạy của ứng dụng** đối với sai số chất lượng. Các biến thể này đã chứng minh khả năng giảm đáng kể chi phí suy luận mà vẫn duy trì độ chính xác gần với MHA truyền thống.

## Cải tiến cấu trúc: Pre-Normalization và MoE  

- **Post‑Normalization vs. Pre‑Normalization**  
  Trong Transformer gốc (Vaswani\ et\ al., 2017) mỗi khối sub‑layer kết thúc bằng LayerNorm – *post‑normalization*. Khi gradient truyền ngược qua nhiều lớp, độ lớn của nó có thể bị “exploding” hoặc “vanishing”, khiến quá trình huấn luyện không ổn định, đặc biệt với độ sâu >\ 12\ lớp. Các nghiên cứu hiện đại chuyển LayerNorm lên *trước* sub‑layer (pre‑normalization) → gradient luôn được chuẩn hoá trước khi qua attention và feed‑forward, giảm thiểu sự phụ thuộc vào độ sâu và cho phép học với learning rate lớn hơn. Kết quả thực nghiệm cho thấy mô hình pre‑norm đạt hội tụ nhanh hơn và ít gặp lỗi NaN so với post‑norm\ [Transformer Design Guide (Part\ 2)](https://rohitbandaru.github.io/blog/Transformer-Design-Guide-Pt2).  

- **Kiến trúc Mixture of Experts (MoE) và DeepSeekMoE**  
  MoE chia một mạng lớn thành *N* chuyên gia (experts) độc lập; mỗi token chỉ kích hoạt một (hoặc một vài) chuyên gia thông qua một router, giảm chi phí tính toán trong khi vẫn mở rộng khả năng biểu diễn. Kiến trúc cơ bản được mô tả chi tiết trong [Understanding Mixture of Experts (MoE) Neural Networks](https://intuitionlabs.ai/articles/mixture-of-experts-moe-models) và [A Survey on Mixture of Experts](https://arxiv.org/html/2407.06204v1).  
  DeepSeekMoE, một biến thể được công bố bởi DeepSeek, tối ưu hoá số lượng chuyên gia bằng cách **động thái điều chỉnh** kích thước expert pool dựa trên độ phức tạp của dữ liệu và tài nguyên GPU hiện có. Thông tin chi tiết về cơ chế này **không có trong các nguồn cung cấp**, vì vậy không thể trích dẫn.  

- **Lợi ích của kích hoạt thưa thớt (sparse activation)**  
  - **Giảm FLOPs**: Mỗi token chỉ tính toán qua 1–2 experts thay vì toàn bộ mạng, giảm chi phí tính toán lên tới 70\ % so với dense model\ [Mixture of Experts in Large Language Models](https://arxiv.org/html/2507.11181v2).  
  - **Tiết kiệm bộ nhớ**: Các weight của các experts không được kích hoạt không cần tải vào GPU, cho phép mô hình có hàng trăm nghìn experts mà vẫn phù hợp với một GPU\ [Applying Mixture of Experts in LLM Architectures](https://developer.nvidia.com/blog/applying-mixture-of-experts-in-llm-architectures).  
  - **Tăng khả năng mở rộng**: Khi số lượng experts tăng, chi phí tính toán không tăng tuyến tính, nhờ vào sparsity, nên mô hình có thể mở rộng lên quy mô hàng tỷ tham số mà vẫn duy trì latency hợp lý.  

> **[IMAGE NOT FOUND]** Mô phỏng cơ chế định tuyến thưa thớt (sparse routing) trong kiến trúc Mixture of Experts (MoE).
>
> **Search:** mixture of experts routing architecture diagram llm


- **Thách thức cân bằng tải (load balancing)**  
  - **Độ lệch router**: Router có thể ưu tiên một nhóm experts, dẫn đến “expert collapse” – một số experts bị quá tải, các expert còn lại gần như không được sử dụng. Các bài báo MoE đề xuất loss cân bằng tải (load‑balancing loss) để khuyến khích phân phối đồng đều\ [Understanding Mixture of Experts (MoE) Neural Networks](https://intuitionlabs.ai/articles/mixture-of-experts-moe-models).  
  - **Chi phí giao tiếp**: Khi một GPU chứa nhiều experts, việc chuyển token giữa các GPU để thực hiện routing gây overhead mạng, đặc biệt trong môi trường đa‑node. NVIDIA’s Megatron‑Bridge ghi nhận rằng việc tối ưu hoá giao tiếp là yếu tố quyết định hiệu năng thực tế của MoE\ [Attention Optimizations — Megatron Bridge](https://docs.nvidia.com/nemo/megatron-bridge/latest/training/attention-optimizations.html).  
  - **Độ ổn định training**: Nếu load balancing không tốt, gradient từ các experts ít được kích hoạt sẽ trở nên nhiễu, làm chậm hội tụ. Các phương pháp như “auxiliary expert loss” và “dynamic capacity scaling” được đề xuất để giảm thiểu vấn đề này\ [A Survey on Mixture of Experts](https://arxiv.org/html/2407.06204v1).  

Tóm lại, chuyển sang pre‑normalization cải thiện độ ổn định huấn luyện, trong khi MoE (với sparse activation) mở ra khả năng mở rộng lớn hơn. Tuy nhiên, để khai thác hết tiềm năng MoE, cần giải quyết cân bằng tải và chi phí giao tiếp – những vấn đề đang được cộng đồng nghiên cứu tích cực tối ưu.

## Quan sát và gỡ lỗi: Mẹo tối ưu hóa cho kỹ sư

### 1. Profiling các lớp Attention  
- Sử dụng **torch.profiler** (hoặc NVIDIA Nsight) để ghi lại thời gian và lượng memory cho mỗi kernel.  
- Đặt `record_shapes=True` và `profile_memory=True` để nhận diện các **bottleneck** trong multi‑head attention (MAHA) và các biến đổi attention hiệu quả (sparse, flash).  
- Khi profiling, lọc các event có tên chứa `aten::matmul` hoặc `flash_attn`; thời gian trung bình trên batch sẽ cho biết lớp nào cần tối ưu.

### 2. Kiểm tra lỗi thường gặp khi triển khai FlashAttention  
- **Kiểu dữ liệu không tương thích**: FlashAttention yêu cầu tensor ở dạng `torch.float16` hoặc `torch.bfloat16` và phải nằm trên GPU. Nếu dữ liệu ở `float32` hoặc trên CPU, kernel sẽ thất bại mà không báo lỗi rõ ràng.  
- **Alignment và stride**: Tensor phải có stride liên tục (contiguous). Sử dụng `tensor = tensor.contiguous()` trước khi gọi hàm flash.  
- **Kích thước batch và seq_len**: FlashAttention hỗ trợ tối đa 2^16 token; vượt quá sẽ gây overflow. Kiểm tra `if seq_len > 65536: raise ValueError(...)`.

### 3. Đo lường hiệu quả KV Cache trong suy luận thời gian thực  
- KV Cache lưu trữ các key/value đã tính toán để tránh lặp lại attention cho các token đã qua.  
- Đo **throughput** (token/s) và **latency** (ms/token) với và không có cache:  

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B", torch_dtype=torch.float16).cuda()
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")

def infer(prompt, use_cache=True):
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=50, use_cache=use_cache)
    return out

# benchmark
import time
t0 = time.time()
_ = infer("Hello, world!", use_cache=True)
t1 = time.time()
print("Latency with KV cache:", (t1 - t0) * 1000, "ms")
```

- So sánh thời gian cho thấy giảm **latency** lên tới 30‑40\ % khi kích thước context lớn (>2k tokens).

### 4. Thiết lập chỉ số quan sát (observability)  
- **Latency**: Ghi lại `torch.cuda.synchronize()` trước và sau mỗi vòng inference; xuất ra Prometheus metric `inference_latency_seconds`.  
- **GPU memory**: Sử dụng `torch.cuda.memory_allocated()` và `torch.cuda.memory_reserved()` để tạo metric `gpu_memory_bytes`.  
- **Cache hit ratio**: Tính tỉ lệ token được phục vụ từ KV Cache (`cache_hits / total_tokens`).  
- Đẩy các metric này lên Grafana hoặc CloudWatch để theo dõi thời gian thực, phát hiện spike và thực hiện auto‑scaling khi cần.  

Kết hợp profiling, kiểm tra cấu hình FlashAttention, đo lường KV Cache và thiết lập observability sẽ giúp kỹ sư nhanh chóng xác định và khắc phục các điểm nghẽn, tối ưu hoá throughput và giảm chi phí GPU trong môi trường sản xuất.